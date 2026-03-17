"""
COCO Siamese Dataset
--------------------
Loads the entire COCO dataset (as compressed JPEG bytes) into RAM at startup,
then generates (template, search, heatmap) triples on the fly.

Reads Ultralytics-style YOLO labels (class cx cy w h, normalised) — no
pycocotools / annotation JSON required.

Crop convention
  Template : 128x128  —  object bbox + SiamFC context (context_amount=0.5)
  Search   : 256x256  —  2x template scale, center randomly jittered
  Heatmap  : 16x16    —  Gaussian blob at the object's projected location

NOTE: Kornia augmentations (color jitter, blur, etc.) are applied in the
training loop after the batch is moved to GPU, not here.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONTEXT_AMOUNT = 0.5   # SiamFC context padding factor
SEARCH_SCALE   = 2.0   # search crop covers 2x the template area in image space

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN.tolist(), std=_IMAGENET_STD.tolist()),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_canvas(s_int: int) -> np.ndarray:
    """Blank canvas filled with ImageNet mean colour (uint8, H×W×3)."""
    pad = (_IMAGENET_MEAN * 255).round().astype(np.uint8)
    canvas = np.empty((s_int, s_int, 3), dtype=np.uint8)
    canvas[:, :, 0] = pad[0]
    canvas[:, :, 1] = pad[1]
    canvas[:, :, 2] = pad[2]
    return canvas


def _crop_and_resize(img_np: np.ndarray, cx: float, cy: float,
                     s: float, out_size: int) -> Image.Image:
    """
    Crop a square of side s centred at (cx, cy) from img_np (H×W×3 uint8).
    Regions outside the image boundary are filled with ImageNet mean colour.
    Returns a PIL Image resized to out_size×out_size.
    """
    H, W = img_np.shape[:2]
    s_int = max(1, int(round(s)))
    half  = s_int / 2.0

    x1 = int(math.floor(cx - half))
    y1 = int(math.floor(cy - half))
    x2 = x1 + s_int
    y2 = y1 + s_int

    src_x1 = max(0, x1);  src_y1 = max(0, y1)
    src_x2 = min(W, x2);  src_y2 = min(H, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = _make_canvas(s_int)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _make_gaussian(cx: float, cy: float, size: int, sigma: float) -> np.ndarray:
    """
    size×size Gaussian heatmap with peak at (cx, cy).
    Returns float32 array with values in [0, 1].
    """
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class COCOSiameseDataset(Dataset):
    """
    Parameters
    ----------
    images_dir    : str  — path to JPEG images  (e.g. .../coco/images/train2017)
    labels_dir    : str  — path to YOLO labels  (e.g. .../coco/labels/train2017)
    template_size : int    128
    search_size   : int    256
    heatmap_size  : int    16     must equal search_size // backbone_stride
    sigma_k       : float  0.25   Gaussian sigma = max(min_sigma, k * sqrt(w_hm * h_hm))
    min_sigma     : float  1.0    floor on sigma in heatmap cells
    max_jitter    : float  0.25   max search-centre shift as fraction of s_x
    neg_ratio     : float  0.25   fraction of pairs that are cross-image negatives
    min_area      : int    400    minimum object area in pixels (filters tiny objects)
    """

    def __init__(
        self,
        images_dir: str = "",
        labels_dir: str = "",
        template_size: int = 128,
        search_size: int   = 256,
        heatmap_size: int  = 16,
        sigma_k: float     = 0.25,
        min_sigma: float   = 1.0,
        max_jitter: float  = 0.25,
        neg_ratio: float   = 0.25,
        min_area: int      = 400,
        _img_bytes: dict   = None,
        _instances: list   = None,
    ):
        self.template_size = template_size
        self.search_size   = search_size
        self.heatmap_size  = heatmap_size
        self.sigma_k       = sigma_k
        self.min_sigma     = min_sigma
        self.max_jitter    = max_jitter
        self.neg_ratio     = neg_ratio

        # Prebuilt path: reuse already-loaded data (e.g. for val split)
        if _img_bytes is not None and _instances is not None:
            self.img_bytes = _img_bytes
            self.instances = _instances
            print(f"Dataset ready: {len(self.instances):,} instances "
                  f"from {len(self.img_bytes):,} images")
            return

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        # ---- Phase 1: scan label files → normalised bboxes per image -------
        print("Scanning label files …")
        raw: dict[str, list] = {}   # img_name -> [[cx,cy,w,h], ...]
        for txt in labels_dir.glob("*.txt"):
            img_name = txt.stem + ".jpg"
            bboxes = []
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # class_id cx cy w h  (normalised)
                        bboxes.append([float(v) for v in parts[1:]])
            if bboxes:
                raw[img_name] = bboxes

        # ---- Phase 2: load images into RAM, convert to pixel coords --------
        print(f"Loading {len(raw)} images into RAM …")
        self.img_bytes: dict[str, bytes] = {}
        self.instances: list[dict]       = []

        for i, (img_name, norm_bboxes) in enumerate(raw.items()):
            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            with open(img_path, "rb") as f:
                data = f.read()

            # Get image dimensions
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]

            for cx_n, cy_n, w_n, h_n in norm_bboxes:
                cx = cx_n * W;  cy = cy_n * H
                w  = w_n  * W;  h  = h_n  * H
                if w * h >= min_area and w > 10 and h > 10:
                    self.instances.append({
                        "img_name": img_name,
                        "cx": cx, "cy": cy,
                        "w":  w,  "h":  h,
                    })

            self.img_bytes[img_name] = data

            if (i + 1) % 10_000 == 0:
                print(f"  {i + 1}/{len(raw)} images loaded")

        print(f"Dataset ready: {len(self.instances):,} instances "
              f"from {len(self.img_bytes):,} images")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.instances)

    def _get_crop_size(self, w: float, h: float):
        """SiamFC context formula — returns (s_z, s_x)."""
        context = CONTEXT_AMOUNT * (w + h)
        s_z = math.sqrt((w + context) * (h + context))
        return s_z, s_z * SEARCH_SCALE

    def _sigma_for(self, w: float, h: float, s_x: float) -> float:
        """Sigma scaled to the object's footprint in heatmap cells."""
        w_hm = w * (self.heatmap_size / s_x)
        h_hm = h * (self.heatmap_size / s_x)
        return max(self.min_sigma, self.sigma_k * math.sqrt(w_hm * h_hm))

    def __getitem__(self, idx: int):
        inst = self.instances[idx]
        cx, cy, w, h = inst["cx"], inst["cy"], inst["w"], inst["h"]

        img_np = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(self.img_bytes[inst["img_name"]], np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )

        # ---- template crop (always from the target instance) -----------
        s_z, s_x = self._get_crop_size(w, h)
        template  = _crop_and_resize(img_np, cx, cy, s_z, self.template_size)

        # ---- negative pair: search from a different image --------------
        if random.random() < self.neg_ratio:
            neg = inst
            while neg["img_name"] == inst["img_name"]:
                neg = self.instances[random.randrange(len(self.instances))]

            neg_img_np = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(self.img_bytes[neg["img_name"]], np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            _, neg_s_x  = self._get_crop_size(neg["w"], neg["h"])
            search      = _crop_and_resize(neg_img_np,
                                           neg["cx"], neg["cy"],
                                           neg_s_x, self.search_size)
            heatmap = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)

        # ---- positive pair: jittered search from the same image --------
        else:
            jitter_range = self.max_jitter * s_x
            search_cx = cx + random.uniform(-jitter_range, jitter_range)
            search_cy = cy + random.uniform(-jitter_range, jitter_range)

            scale       = self.search_size / s_x
            half_search = self.search_size / 2.0
            obj_x = (cx - search_cx) * scale + half_search
            obj_y = (cy - search_cy) * scale + half_search

            hm_scale = self.heatmap_size / self.search_size
            hm_cx    = obj_x * hm_scale
            hm_cy    = obj_y * hm_scale

            sigma   = self._sigma_for(w, h, s_x)
            search  = _crop_and_resize(img_np, search_cx, search_cy, s_x, self.search_size)
            heatmap = _make_gaussian(hm_cx, hm_cy, self.heatmap_size, sigma)

        return (
            _to_tensor(template),                        # (3, 128, 128)
            _to_tensor(search),                          # (3, 256, 256)
            torch.from_numpy(heatmap).unsqueeze(0),      # (1,  16,  16)
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloader(
    images_dir: str,
    labels_dir: str,
    batch_size: int  = 128,
    num_workers: int = 14,
    shuffle: bool    = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = COCOSiameseDataset(images_dir, labels_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=shuffle,   # only drop last for training
    )

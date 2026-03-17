"""
Fine-tuning script for CerberusSiamese — backbone unfrozen.
Resume from a frozen-backbone checkpoint and fine-tune end-to-end.
Run via %run train_finetune.py from a Jupyter notebook.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as KA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from livelossplot import PlotLosses

from Cerberus_Siamese import CerberusSiamese
from torch.utils.data import DataLoader
from dataset import COCOSiameseDataset


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def weighted_bce_loss(logits, targets):
    """BCE with per-pixel floor weighting + fg/bg split for logging."""
    weight = targets.clamp(min=0.2)
    bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss   = (bce * weight).mean()

    fg_mask = targets > 0.2
    fg_loss = bce[fg_mask].mean()   if fg_mask.any()   else bce.new_zeros(())
    bg_loss = bce[~fg_mask].mean()  if (~fg_mask).any() else bce.new_zeros(())
    return loss, fg_loss, bg_loss


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (used for GPU batch normalisation)
# ---------------------------------------------------------------------------
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Config — edit here
# ---------------------------------------------------------------------------
CFG = dict(
    images_dir     = "/workspace/data/coco/images/train2017",
    labels_dir     = "/workspace/data/coco/labels/train2017",

    epochs         = 50,
    batch_size     = 512,
    num_workers    = 12,
    lr             = 1e-3,   # lower than frozen run; backbone gets lr × 0.1 = 1e-4
    val_split      = 0.05,
    save_every     = 1,
    save_dir       = "snapshots",
    plot_every     = 20,

    freeze_backbone = False,
    resume          = "snapshots/cerberus_epoch11.pth",   # ← update as needed
    amp             = True,
    compile         = True,
)


# ---------------------------------------------------------------------------
# Augmentation  (applied on GPU in training loop only — val gets none)
# ---------------------------------------------------------------------------
def build_augmentations(device):
    aug_template = KA.AugmentationSequential(
        KA.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        KA.RandomGrayscale(p=0.1),
        same_on_batch=False,
    ).to(device)

    aug_search = KA.AugmentationSequential(
        KA.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        KA.RandomGrayscale(p=0.1),
        KA.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),
        KA.RandomMotionBlur(kernel_size=(7, 21), angle=(-45, 45), direction=(-1.0, 1.0), p=0.4),
        same_on_batch=False,
    ).to(device)

    return aug_template, aug_search


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- model -------------------------------------------------------------
    model = CerberusSiamese().to(device)

    if cfg["freeze_backbone"]:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen")
    else:
        print("Backbone unfrozen")

    if cfg["compile"]:
        model = torch.compile(model)
        print("torch.compile enabled")

    # ---- data --------------------------------------------------------------
    print("Loading dataset …")
    full_ds = COCOSiameseDataset(cfg["images_dir"], cfg["labels_dir"])

    random.shuffle(full_ds.instances)
    n_val = int(len(full_ds.instances) * cfg["val_split"])
    val_instances     = full_ds.instances[:n_val]
    full_ds.instances = full_ds.instances[n_val:]

    print(f"Split → train: {len(full_ds.instances):,}  val: {len(val_instances):,}")

    val_ds = COCOSiameseDataset(
        _img_bytes=full_ds.img_bytes,
        _instances=val_instances,
        neg_ratio=0.0,
    )

    train_loader = DataLoader(
        full_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )

    # ---- loss (weighted_bce_loss + soft_argmax coord loss) -----------------

    # ---- optimiser ---------------------------------------------------------
    if cfg["freeze_backbone"]:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=1e-4,
        )
    else:
        backbone_params = list(model.backbone.parameters())
        head_params     = [p for p in model.parameters()
                           if not any(p is bp for bp in backbone_params)]
        optimizer = AdamW([
            {"params": backbone_params, "lr": cfg["lr"] * 0.1},
            {"params": head_params,     "lr": cfg["lr"]},
        ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler    = torch.amp.GradScaler(enabled=cfg["amp"])

    aug_template, aug_search = build_augmentations(device)

    # Pre-build normalisation tensors on device
    norm_mean = _NORM_MEAN.to(device).view(1, 3, 1, 1)
    norm_std  = _NORM_STD.to(device).view(1, 3, 1, 1)

    start_epoch = 0

    # ---- resume ------------------------------------------------------------
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]

        ckpt_was_frozen = ckpt.get("backbone_frozen", False)
        if ckpt_was_frozen == cfg["freeze_backbone"]:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"Resumed from epoch {start_epoch}")
        else:
            state = "frozen → unfrozen" if ckpt_was_frozen else "unfrozen → frozen"
            print(f"Resumed from epoch {start_epoch} "
                  f"(backbone freeze state changed: {state} — optimizer reset)")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ---- livelossplot ------------------------------------------------------
    plotlosses = PlotLosses(groups={
        "loss":       ["loss", "val_loss"],
        "components": ["fg_loss", "bg_loss"],
    })
    last_val_loss = 0.0

    # ---- loop --------------------------------------------------------------
    epoch_bar = tqdm(range(start_epoch, start_epoch + cfg["epochs"]), desc="Epochs")

    for epoch in epoch_bar:

        # -- train -----------------------------------------------------------
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)
        window_loss    = 0.0
        window_fg_loss = 0.0
        window_bg_loss = 0.0
        for i, (template, search, heatmap_gt, _) in enumerate(train_bar):
            template   = template.to(device, non_blocking=True).float().div(255)
            search     = search.to(device, non_blocking=True).float().div(255)
            heatmap_gt = heatmap_gt.to(device, non_blocking=True)

            template = (template - norm_mean) / norm_std
            search   = (search   - norm_mean) / norm_std

            with torch.no_grad():
                template = aug_template(template)
                search   = aug_search(search)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=cfg["amp"]):
                pred = model(template, search)
                bce_loss, fg_l, bg_l = weighted_bce_loss(pred, heatmap_gt)
                loss = bce_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss   += loss.item()
            window_loss    += loss.item()
            window_fg_loss += fg_l.item()
            window_bg_loss += bg_l.item()
            train_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

            if (i + 1) % cfg["plot_every"] == 0:
                plotlosses.update({
                    "loss":     window_loss    / cfg["plot_every"],
                    "val_loss": last_val_loss,
                    "fg_loss":  window_fg_loss / cfg["plot_every"],
                    "bg_loss":  window_bg_loss / cfg["plot_every"],
                })
                plotlosses.send()
                window_loss    = 0.0
                window_fg_loss = 0.0
                window_bg_loss = 0.0

        avg_train_loss = running_loss / len(train_loader)

        # -- val -------------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val   {epoch+1}", leave=False)
            for template, search, heatmap_gt, _ in val_bar:
                template   = template.to(device, non_blocking=True).float().div(255)
                search     = search.to(device, non_blocking=True).float().div(255)
                heatmap_gt = heatmap_gt.to(device, non_blocking=True)

                template = (template - norm_mean) / norm_std
                search   = (search   - norm_mean) / norm_std

                with torch.amp.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred = model(template, search)
                    bce_l, _, _ = weighted_bce_loss(pred, heatmap_gt)
                    val_loss += bce_l.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        # -- logging ---------------------------------------------------------
        last_val_loss = avg_val_loss
        epoch_bar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}")
        leftover = len(train_loader) % cfg["plot_every"]
        step_loss    = (window_loss    / leftover) if leftover else avg_train_loss
        step_fg_loss = (window_fg_loss / leftover) if leftover else 0.0
        step_bg_loss = (window_bg_loss / leftover) if leftover else 0.0
        plotlosses.update({
            "loss":     step_loss,
            "val_loss": last_val_loss,
            "fg_loss":  step_fg_loss,
            "bg_loss":  step_bg_loss,
        })
        plotlosses.send()

        # -- checkpoint ------------------------------------------------------
        if (epoch + 1) % cfg["save_every"] == 0 or epoch + 1 == start_epoch + cfg["epochs"]:
            path = os.path.join(cfg["save_dir"], f"cerberus_epoch{epoch+1}.pth")
            torch.save({
                "epoch":           epoch + 1,
                "model":           model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "scheduler":       scheduler.state_dict(),
                "loss":            avg_train_loss,
                "val_loss":        avg_val_loss,
                "backbone_frozen": cfg["freeze_backbone"],
            }, path)
            print(f"Saved {path}")


if __name__ == "__main__":
    train(CFG)

"""
Training script for CerberusSiamese.
Run via %run train.py from a Jupyter notebook.
"""

import os

import torch
import torch.nn as nn
import kornia.augmentation as KA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from livelossplot import PlotLosses

from Cerberus_Siamese import CerberusSiamese
from dataset import build_dataloader


# ---------------------------------------------------------------------------
# Config — edit here
# ---------------------------------------------------------------------------
CFG = dict(
    images_dir     = "/workspace/data/coco/images/train2017",
    labels_dir     = "/workspace/data/coco/labels/train2017",
    val_images_dir = "/workspace/data/coco/images/val2017",
    val_labels_dir = "/workspace/data/coco/labels/val2017",

    epochs         = 50,
    batch_size     = 1024,
    num_workers    = 24,
    lr             = 4e-3,   # scaled linearly with batch size (128 → 512 = 4×)
    save_every     = 5,
    save_dir       = "checkpoints",
    plot_every     = 20,   # update livelossplot every N steps

    freeze_backbone = True,    # set False to unfreeze on next run
    resume          = None,    # e.g. "checkpoints/cerberus_epoch50.pth"
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
    print("Building train dataloader …")
    train_loader = build_dataloader(
        cfg["images_dir"],
        cfg["labels_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
    )

    print("Building val dataloader …")
    val_loader = build_dataloader(
        cfg["val_images_dir"],
        cfg["val_labels_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        neg_ratio=0.0,   # val: positives only so loss is comparable across epochs
    )

    # ---- loss --------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

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
    plotlosses = PlotLosses(groups={"loss": ["loss", "val_loss"]})
    last_val_loss = 0.0

    # ---- loop --------------------------------------------------------------
    epoch_bar = tqdm(range(start_epoch, cfg["epochs"]), desc="Epochs")

    for epoch in epoch_bar:

        # -- train -----------------------------------------------------------
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)
        window_loss = 0.0
        for i, (template, search, heatmap_gt) in enumerate(train_bar):
            template   = template.to(device, non_blocking=True)
            search     = search.to(device, non_blocking=True)
            heatmap_gt = heatmap_gt.to(device, non_blocking=True)

            with torch.no_grad():
                template = aug_template(template)
                search   = aug_search(search)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=cfg["amp"]):
                pred = model(template, search)
                loss = criterion(pred, heatmap_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            window_loss  += loss.item()
            train_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

            if (i + 1) % cfg["plot_every"] == 0:
                plotlosses.update({"loss": window_loss / cfg["plot_every"], "val_loss": last_val_loss})
                plotlosses.send()
                window_loss = 0.0

        avg_train_loss = running_loss / len(train_loader)

        # -- val -------------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val   {epoch+1}", leave=False)
            for template, search, heatmap_gt in val_bar:
                template   = template.to(device, non_blocking=True)
                search     = search.to(device, non_blocking=True)
                heatmap_gt = heatmap_gt.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred = model(template, search)
                    val_loss += criterion(pred, heatmap_gt).item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        # -- logging ---------------------------------------------------------
        last_val_loss = avg_val_loss
        epoch_bar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}")
        leftover = len(train_loader) % cfg["plot_every"]
        step_loss = (window_loss / leftover) if leftover else avg_train_loss
        plotlosses.update({"loss": step_loss, "val_loss": last_val_loss})
        plotlosses.send()

        # -- checkpoint ------------------------------------------------------
        if (epoch + 1) % cfg["save_every"] == 0 or epoch + 1 == cfg["epochs"]:
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

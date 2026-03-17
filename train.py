"""
Training script for CerberusSiamese.

Usage
-----
python train.py --images_dir data/coco/coco/images/train2017 \
                --labels_dir data/coco/coco/labels/train2017

Resume from checkpoint:
python train.py ... --resume cerberus_epoch10.pth
"""

import argparse
import os

import torch
import torch.nn as nn
import kornia.augmentation as KA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Cerberus_Siamese import CerberusSiamese
from dataset import build_dataloader


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# Applied on GPU after each batch is moved to device.
# Tensors arrive ImageNet-normalised — colour ops are directionally correct
# even on normalised values; magnitudes are approximate but sufficient for
# injecting variation.

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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- model -------------------------------------------------------------
    model = CerberusSiamese().to(device)
    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled")

    # ---- data --------------------------------------------------------------
    loader = build_dataloader(
        args.images_dir,
        args.labels_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- loss --------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    # ---- optimiser — lower LR on pretrained backbone ----------------------
    backbone_params = list(model.backbone.parameters()) if not args.compile else []
    head_params     = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ] if backbone_params else model.parameters()

    optimizer  = AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler     = torch.amp.GradScaler(enabled=args.amp)

    aug_template, aug_search = build_augmentations(device)

    start_epoch = 0

    # ---- resume ------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- loop --------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        for i, (template, search, heatmap_gt) in enumerate(loader):
            template   = template.to(device, non_blocking=True)
            search     = search.to(device, non_blocking=True)
            heatmap_gt = heatmap_gt.to(device, non_blocking=True)

            # GPU augmentations (independent per branch)
            with torch.no_grad():
                template = aug_template(template)
                search   = aug_search(search)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                pred = model(template, search)
                loss = criterion(pred, heatmap_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                avg = running_loss / (i + 1)
                print(f"epoch {epoch+1}/{args.epochs}  "
                      f"step {i+1}/{len(loader)}  "
                      f"loss {avg:.4f}  "
                      f"lr {optimizer.param_groups[-1]['lr']:.2e}")

        scheduler.step()

        avg_epoch_loss = running_loss / len(loader)
        print(f"--- epoch {epoch+1} done  avg_loss {avg_epoch_loss:.4f} ---")

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.save_dir, f"cerberus_epoch{epoch+1}.pth")
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss":      avg_epoch_loss,
            }, path)
            print(f"Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",    required=True)
    p.add_argument("--labels_dir",    required=True)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=14)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--log_interval",  type=int,   default=50)
    p.add_argument("--save_every",    type=int,   default=5)
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--resume",        default=None)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--compile",       action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

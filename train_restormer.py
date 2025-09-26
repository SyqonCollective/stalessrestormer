#!/usr/bin/env python
"""Training entrypoint for the Restormer-based starless model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from restormer_starless import (
    Restormer,
    PerceptualLoss,
    build_dataloader,
    ensure_dir,
    save_metrics,
)


def _build_dataloaders(
    train_dir: Optional[Path],
    val_dir: Optional[Path],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    train_input_dir: Optional[Path],
    train_target_dir: Optional[Path],
    val_input_dir: Optional[Path],
    val_target_dir: Optional[Path],
) -> tuple[DataLoader, Optional[DataLoader]]:
    train_root = train_dir
    if train_root is None:
        if train_input_dir is not None:
            train_root = Path(train_input_dir).parent
        elif train_target_dir is not None:
            train_root = Path(train_target_dir).parent
        else:
            raise ValueError("A training directory or explicit train input/target directories are required")

    train_loader = build_dataloader(
        root=train_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        random_crop=True,
        shuffle=True,
        input_dir=train_input_dir,
        target_dir=train_target_dir,
    )
    val_loader = None
    val_root: Optional[Path] = None
    if val_dir is not None and val_dir.exists():
        val_root = val_dir
    elif val_input_dir is not None:
        val_root = Path(val_input_dir).parent
    elif val_target_dir is not None:
        val_root = Path(val_target_dir).parent

    if val_root is not None:
        val_loader = build_dataloader(
            root=val_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=False,
            random_crop=False,
            shuffle=False,
            input_dir=val_input_dir,
            target_dir=val_target_dir,
        )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Restormer starless model")
    parser.add_argument("--config", type=Path, help="YAML config file with default arguments")
    parser.add_argument("--train-dir", type=Path, required=False, help="Root directory with train tiles")
    parser.add_argument("--val-dir", type=Path, help="Root directory with validation tiles")
    parser.add_argument("--output-dir", type=Path, default=Path("./restormer_logs"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-dim", type=int, default=48, help="Base channel dimension")
    parser.add_argument(
        "--dim-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to base channels (alpha)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--train-input-dir", type=Path, help="Explicit directory for training input tiles")
    parser.add_argument("--train-target-dir", type=Path, help="Explicit directory for training target tiles")
    parser.add_argument("--val-input-dir", type=Path, help="Explicit directory for validation input tiles")
    parser.add_argument("--val-target-dir", type=Path, help="Explicit directory for validation target tiles")
    return parser.parse_args()


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files. Install pyyaml or omit --config.")
    with args.config.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}
    for key, value in cfg.items():
        arg_key = key.replace("-", "_")
        if hasattr(args, arg_key) and value is not None:
            setattr(args, arg_key, value)
    if "augment" in cfg:
        args.no_augment = not bool(cfg["augment"])
    # Ensure required arguments present
    if not args.train_dir and cfg.get("train_dir") is not None:
        args.train_dir = Path(cfg.get("train_dir"))
    if args.val_dir is not None and not isinstance(args.val_dir, Path):
        args.val_dir = Path(args.val_dir)
    if args.output_dir is not None and not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)
    for attr in ("train_input_dir", "train_target_dir", "val_input_dir", "val_target_dir"):
        value = getattr(args, attr, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, attr, Path(value))
    return args


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion_l1: nn.Module,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = model(inputs)
            loss = criterion_l1(preds, targets)
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    args = _apply_config(args)
    if args.train_dir is not None and not isinstance(args.train_dir, Path):
        args.train_dir = Path(args.train_dir)
    if args.val_dir is not None and not isinstance(args.val_dir, Path):
        args.val_dir = Path(args.val_dir)
    if not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)

    if (args.train_input_dir is None) != (args.train_target_dir is None):
        raise ValueError("Provide both --train-input-dir and --train-target-dir when specifying explicit training directories")
    if (args.val_input_dir is None) != (args.val_target_dir is None):
        raise ValueError("Provide both --val-input-dir and --val-target-dir when specifying explicit validation directories")
    if args.train_dir is None and (args.train_input_dir is None or args.train_target_dir is None):
        raise ValueError("Provide --train-dir or both --train-input-dir/--train-target-dir")

    for name, path in (
        ("train", args.train_dir),
        ("validation", args.val_dir),
    ):
        if path is not None and not path.exists():
            raise ValueError(f"Specified {name} directory does not exist: {path}")

    for name, path in (
        ("train input", args.train_input_dir),
        ("train target", args.train_target_dir),
        ("validation input", args.val_input_dir),
        ("validation target", args.val_target_dir),
    ):
        if path is not None and not path.exists():
            raise ValueError(f"Specified {name} directory does not exist: {path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = args.mixed_precision and device.type == "cuda"
    ensure_dir(args.output_dir)

    train_loader, val_loader = _build_dataloaders(
        args.train_dir,
        args.val_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment=not args.no_augment,
        train_input_dir=args.train_input_dir,
        train_target_dir=args.train_target_dir,
        val_input_dir=args.val_input_dir,
        val_target_dir=args.val_target_dir,
    )

    model = Restormer(dim=args.base_dim, alpha=args.dim_scale).to(device)
    criterion_l1 = nn.L1Loss()
    perceptual = None
    if args.perceptual_weight > 0:
        perceptual = PerceptualLoss(["relu1_1", "relu2_1", "relu3_1"]).to(device)
        perceptual.eval()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler(device_type=amp_device_type, enabled=amp_enabled)

    best_val = math.inf
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        epoch_loss = 0.0
        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=scaler.is_enabled()):
                preds = model(inputs)
                loss = criterion_l1(preds, targets)
                if perceptual is not None:
                    loss = loss + args.perceptual_weight * perceptual(preds, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        save_metrics(
            args.output_dir / "train_metrics.jsonl",
            {"epoch": epoch, "loss": avg_loss, "step": global_step},
        )

        if val_loader is not None:
            val_loss = validate(model, val_loader, device, criterion_l1)
            save_metrics(
                args.output_dir / "val_metrics.jsonl",
                {"epoch": epoch, "loss": val_loss},
            )
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = args.output_dir / f"checkpoint_best_epoch_{epoch:04d}_l1_{val_loss:.4f}.pt"
                torch.save({
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
                torch.save(model.state_dict(), args.output_dir / "checkpoint_best.pt")
        if args.save_interval and epoch % args.save_interval == 0:
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)

    torch.save(model.state_dict(), args.output_dir / "restormer_final.pt")


if __name__ == "__main__":
    main()

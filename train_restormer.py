#!/usr/bin/env python
"""Training entrypoint for the Restormer-based starless model."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

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


PROJECT_ROOT = Path(__file__).resolve().parent


def _base_variants(path: Path) -> list[Path]:
    variants = [path]
    if not path.is_absolute():
        variants.append(Path.cwd() / path)
        variants.append(PROJECT_ROOT / path)
    return variants


def _resolve_dir(
    path: Optional[Path],
    *,
    expected_subdir: Optional[str] = None,
    default_names: Sequence[str] = (),
    fallback_roots: Sequence[Optional[Path]] = (),
) -> tuple[Optional[Path], Optional[Path]]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(candidate: Path) -> None:
        key = candidate.resolve(strict=False)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    def add_from_base(base: Path, include_base: bool) -> None:
        for variant in _base_variants(base):
            if include_base:
                add_candidate(variant)
            if expected_subdir is not None:
                add_candidate(variant / expected_subdir)

    original = Path(path) if path is not None else None

    if original is not None:
        add_from_base(original, include_base=True)
        if expected_subdir is not None:
            suffix = f"_{expected_subdir}"
            if original.name.endswith(suffix):
                base_name = original.name[: -len(suffix)]
                if base_name:
                    add_from_base(original.with_name(base_name), include_base=False)
    else:
        include_defaults = expected_subdir is None
        for name in default_names:
            add_from_base(Path(name), include_base=include_defaults)

    include_fallbacks = expected_subdir is None
    for root in fallback_roots:
        if root is None:
            continue
        add_from_base(Path(root), include_base=include_fallbacks)

    for candidate in candidates:
        if candidate.exists():
            return candidate, original

    return None, original


def _create_grad_scaler(device_type: str, enabled: bool):
    grad_scaler_ctor = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if grad_scaler_ctor is not None:
        try:
            return grad_scaler_ctor(device_type=device_type, enabled=enabled)
        except TypeError:
            return grad_scaler_ctor(enabled=enabled)
    cuda_amp = getattr(torch, "cuda", None)
    if cuda_amp is not None:
        cuda_grad_scaler = getattr(getattr(cuda_amp, "amp", None), "GradScaler", None)
        if cuda_grad_scaler is not None:
            return cuda_grad_scaler(enabled=enabled and device_type == "cuda")

    class _DisabledGradScaler:
        def is_enabled(self) -> bool:
            return False

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self) -> None:
            pass

    return _DisabledGradScaler()


def _autocast(device_type: str, enabled: bool):
    if not enabled:
        return nullcontext()

    amp_autocast = getattr(getattr(torch, "amp", None), "autocast", None)
    if amp_autocast is not None:
        try:
            return amp_autocast(device_type=device_type, enabled=enabled)
        except TypeError:
            return amp_autocast(enabled=enabled)

    cuda_amp = getattr(torch, "cuda", None)
    if device_type == "cuda" and cuda_amp is not None:
        cuda_autocast = getattr(getattr(cuda_amp, "amp", None), "autocast", None)
        if cuda_autocast is not None:
            return cuda_autocast(enabled=enabled)

    generic_autocast = getattr(torch, "autocast", None)
    if generic_autocast is not None:
        try:
            return generic_autocast(device_type, enabled=enabled)
        except TypeError:
            return generic_autocast(device_type)

    return nullcontext()


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

    resolved_train_dir, original_train_dir = _resolve_dir(
        args.train_dir,
        default_names=("train_tiles",),
    )
    resolved_val_dir, original_val_dir = _resolve_dir(
        args.val_dir,
        default_names=("val_tiles",),
    )

    train_input_dir, original_train_input = _resolve_dir(
        args.train_input_dir,
        expected_subdir="input",
        default_names=("train_tiles",),
        fallback_roots=(resolved_train_dir, PROJECT_ROOT / "train_tiles"),
    )
    train_target_dir, original_train_target = _resolve_dir(
        args.train_target_dir,
        expected_subdir="target",
        default_names=("train_tiles",),
        fallback_roots=(resolved_train_dir, PROJECT_ROOT / "train_tiles"),
    )

    if resolved_train_dir is None and train_input_dir is not None:
        resolved_train_dir = train_input_dir.parent
    if resolved_train_dir is None and train_target_dir is not None:
        resolved_train_dir = train_target_dir.parent

    val_input_dir, original_val_input = _resolve_dir(
        args.val_input_dir,
        expected_subdir="input",
        default_names=("val_tiles",),
        fallback_roots=(resolved_val_dir, PROJECT_ROOT / "val_tiles"),
    )
    val_target_dir, original_val_target = _resolve_dir(
        args.val_target_dir,
        expected_subdir="target",
        default_names=("val_tiles",),
        fallback_roots=(resolved_val_dir, PROJECT_ROOT / "val_tiles"),
    )

    if resolved_val_dir is None and val_input_dir is not None:
        resolved_val_dir = val_input_dir.parent
    if resolved_val_dir is None and val_target_dir is not None:
        resolved_val_dir = val_target_dir.parent

    if resolved_train_dir is None and (train_input_dir is None or train_target_dir is None):
        hint = original_train_dir or Path("train_tiles")
        raise ValueError(
            f"Training data directory not found. Provide --train-dir or explicit --train-input-dir/--train-target-dir (checked around {hint})."
        )

    if train_input_dir is None or not train_input_dir.exists():
        hint = original_train_input or (
            (resolved_train_dir / "input") if resolved_train_dir is not None else Path("train_tiles/input")
        )
        raise ValueError(f"Could not locate train input tiles directory near {hint}")

    if train_target_dir is None or not train_target_dir.exists():
        hint = original_train_target or (
            (resolved_train_dir / "target") if resolved_train_dir is not None else Path("train_tiles/target")
        )
        raise ValueError(f"Could not locate train target tiles directory near {hint}")

    if args.val_dir is not None and resolved_val_dir is None:
        raise ValueError(f"Specified validation directory does not exist: {args.val_dir}")

    if val_input_dir is not None and not val_input_dir.exists():
        hint = original_val_input or (
            (resolved_val_dir / "input") if resolved_val_dir is not None else Path("val_tiles/input")
        )
        raise ValueError(f"Could not locate validation input tiles directory near {hint}")

    if val_target_dir is not None and not val_target_dir.exists():
        hint = original_val_target or (
            (resolved_val_dir / "target") if resolved_val_dir is not None else Path("val_tiles/target")
        )
        raise ValueError(f"Could not locate validation target tiles directory near {hint}")

    args.train_dir = resolved_train_dir
    args.val_dir = resolved_val_dir
    args.train_input_dir = train_input_dir
    args.train_target_dir = train_target_dir
    args.val_input_dir = val_input_dir
    args.val_target_dir = val_target_dir
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

    scaler = _create_grad_scaler(amp_device_type, amp_enabled)

    best_val = math.inf
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        epoch_loss = 0.0
        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_device_type, scaler.is_enabled()):
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

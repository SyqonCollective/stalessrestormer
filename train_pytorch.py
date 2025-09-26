#!/usr/bin/env python
"""PyTorch training entrypoint for StarLess tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from pystarnet.configs import ExperimentConfig
from pystarnet.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyStarNet on StarLess tiles")
    parser.add_argument("--train-dir", type=Path, default=Path("../train_tiles"), help="Path to training tiles root (input/ and target/)")
    parser.add_argument("--val-dir", type=Path, default=Path("../val_tiles"), help="Path to validation tiles root")
    parser.add_argument("--output-dir", type=Path, default=Path("./pystarnet_logs"), help="Directory for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Optional number of steps per epoch (0 uses full dataset)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Generator learning rate (discriminator matches)")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--device", default="cuda", help="Training device")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.dataset.root = args.train_dir
    cfg.dataset.val_root = args.val_dir
    cfg.dataset.image_size = 256
    cfg.output_dir = args.output_dir

    cfg.trainer.max_epochs = args.epochs
    cfg.trainer.batch_size = args.batch_size
    cfg.trainer.steps_per_epoch = args.steps_per_epoch or None
    cfg.trainer.mixed_precision = args.mixed_precision
    cfg.trainer.device = args.device

    cfg.optimizer.lr_generator = args.lr
    cfg.optimizer.lr_discriminator = args.lr
    return cfg


def main() -> None:
    args = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()

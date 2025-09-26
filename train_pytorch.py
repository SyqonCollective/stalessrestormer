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
    parser.add_argument("--gan-weight", type=float, help="Override GAN loss weight")
    parser.add_argument("--gan-warmup-epochs", type=int, help="Override GAN warmup epochs")
    parser.add_argument("--gan-warmup-weight", type=float, help="Override GAN warmup starting weight")
    parser.add_argument("--r1-gamma", type=float, help="Override R1 regularization gamma")
    parser.add_argument("--d-reg-every", type=int, help="Override discriminator regularization interval")
    parser.add_argument("--l1-weight", type=float, help="Override L1 loss weight")
    parser.add_argument("--perceptual-weight", type=float, help="Override perceptual loss weight")
    parser.add_argument("--residual-weight", type=float, help="Weight for residual consistency loss")
    parser.add_argument("--ema-decay", type=float, help="Override EMA decay for generator")
    parser.add_argument("--no-gan", action="store_true", help="Disable GAN loss entirely")
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time augmentation")
    parser.add_argument("--supervised-epochs", type=int, help="Number of initial epochs with pure supervised training (no GAN)")
    parser.add_argument("--augment-start-epoch", type=int, help="Epoch from which to enable data augmentation")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint to resume or fine-tune from")
    parser.add_argument("--resume-load-optim", action="store_true", help="Load optimizer/scaler states when resuming")
    parser.add_argument("--star-mask-alpha", type=float, help="Weight for star-focused L1 penalty")
    parser.add_argument("--star-mask-threshold", type=float, help="Threshold on |input-target| to define star mask (0-2 range)")
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
    if args.ema_decay is not None:
        cfg.trainer.ema_decay = args.ema_decay
    if args.supervised_epochs is not None:
        cfg.trainer.supervised_epochs = max(0, args.supervised_epochs)
    if args.augment_start_epoch is not None:
        cfg.trainer.augment_start_epoch = max(0, args.augment_start_epoch)

    cfg.optimizer.lr_generator = args.lr
    cfg.optimizer.lr_discriminator = args.lr

    if args.gan_weight is not None:
        cfg.losses.gan_weight = args.gan_weight
    if args.gan_warmup_epochs is not None:
        cfg.losses.gan_warmup_epochs = args.gan_warmup_epochs
    if args.gan_warmup_weight is not None:
        cfg.losses.gan_warmup_weight = args.gan_warmup_weight
    if args.no_gan:
        cfg.losses.gan_weight = 0.0
        cfg.losses.gan_warmup_weight = 0.0
        cfg.losses.gan_warmup_epochs = 0
    if args.r1_gamma is not None:
        cfg.losses.r1_gamma = args.r1_gamma
    if args.d_reg_every is not None:
        cfg.losses.d_reg_every = args.d_reg_every
    if args.l1_weight is not None:
        cfg.losses.l1_weight = args.l1_weight
    if args.perceptual_weight is not None:
        cfg.losses.perceptual_weight = args.perceptual_weight
    if args.residual_weight is not None:
        cfg.losses.residual_weight = max(0.0, args.residual_weight)

    if args.star_mask_alpha is not None:
        cfg.losses.star_mask_alpha = max(0.0, args.star_mask_alpha)
    if args.star_mask_threshold is not None:
        cfg.losses.star_mask_threshold = max(0.0, args.star_mask_threshold)

    if args.no_augment:
        cfg.dataset.augment = False

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        cfg.resume_load_optim = args.resume_load_optim
    return cfg


def main() -> None:
    args = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()

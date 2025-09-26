"""Training script for PyStarNet."""

from __future__ import annotations

import math
from itertools import islice
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

from .configs import DatasetConfig, ExperimentConfig
from .data import TileDataset, build_dataloader
from .losses import (
    HingeGANLoss,
    PerceptualLoss,
    discriminator_loss,
    generator_loss,
)
from .models import build_models
from .utils import (
    MetricAverager,
    ensure_dir,
    save_checkpoint,
    save_metrics,
    save_preview,
    set_seed,
)


def _build_val_dataloader(dataset_cfg: DatasetConfig, trainer_cfg) -> Optional[torch.utils.data.DataLoader]:
    if dataset_cfg.val_root is None or not dataset_cfg.val_root.exists():
        return None
    val_cfg = DatasetConfig(
        root=dataset_cfg.val_root,
        val_root=None,
        image_size=dataset_cfg.image_size,
        random_crop=False,
        augment=False,
        valid_extensions=dataset_cfg.valid_extensions,
    )
    return build_dataloader(val_cfg, trainer_cfg, role="val")


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(4.0 / mse)


def train(config: ExperimentConfig) -> None:
    project_root = Path(__file__).resolve().parent
    cfg = config.resolve(project_root)
    ensure_dir(cfg.output_dir)
    set_seed(cfg.trainer.seed)

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")

    generator, discriminator = build_models(
        base_channels=cfg.model.base_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        attention=cfg.model.attention,
    )
    generator.to(device)
    discriminator.to(device)

    hinge = HingeGANLoss()
    perceptual = PerceptualLoss(cfg.losses.feature_layers).to(device)

    optim_g = Adam(
        generator.parameters(),
        lr=cfg.optimizer.lr_generator,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    optim_d = Adam(
        discriminator.parameters(),
        lr=cfg.optimizer.lr_discriminator,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )

    use_amp = cfg.trainer.mixed_precision and device.type == "cuda"
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)

    train_loader = build_dataloader(cfg.dataset, cfg.trainer, role="train")
    val_loader = _build_val_dataloader(cfg.dataset, cfg.trainer)

    global_step = 0

    for epoch in range(1, cfg.trainer.max_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_metrics = MetricAverager()

        data_iter = train_loader
        if cfg.trainer.steps_per_epoch:
            data_iter = islice(train_loader, cfg.trainer.steps_per_epoch)
            total_steps = cfg.trainer.steps_per_epoch
        else:
            total_steps = len(train_loader)

        progress = tqdm(
            data_iter,
            total=total_steps,
            desc=f"Epoch {epoch}/{cfg.trainer.max_epochs}",
            dynamic_ncols=True,
        )

        for batch_index, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Discriminator step
            optim_d.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake = generator(inputs)
                logits_real = discriminator(inputs, targets)
                logits_fake = discriminator(inputs, fake.detach())
                loss_d = discriminator_loss(logits_real, logits_fake, hinge)
            if use_amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optim_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optim_d.step()

            # Generator step
            optim_g.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake = generator(inputs)
                logits_fake = discriminator(inputs, fake)
                gen_losses = generator_loss(
                    logits_fake,
                    fake,
                    targets,
                    hinge,
                    perceptual,
                    cfg.losses.l1_weight,
                    cfg.losses.gan_weight,
                    cfg.losses.perceptual_weight,
                )
                loss_g = gen_losses["total"]
            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optim_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.trainer.gradient_clip)
                scaler_g.step(optim_g)
                scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.trainer.gradient_clip)
                optim_g.step()

            epoch_metrics.update(
                {
                    "loss_d": loss_d,
                    "loss_g": loss_g.detach(),
                    "loss_l1": gen_losses["l1"],
                    "loss_gan": gen_losses["gan"],
                    "loss_perc": gen_losses["perceptual"],
                }
            )

            global_step += 1
            if batch_index % cfg.trainer.log_interval == 0:
                averages = epoch_metrics.compute()
                progress.set_postfix({k: f"{v:.3f}" for k, v in averages.items()})

        averages = epoch_metrics.compute()
        save_metrics(
            cfg.output_dir / "train_metrics.jsonl",
            {"epoch": epoch, **averages},
        )

        if epoch % cfg.trainer.checkpoint_interval == 0:
            save_checkpoint(
                cfg.output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                epoch,
                global_step,
                generator,
                discriminator,
                optim_g,
                optim_d,
                scaler_g if use_amp else None,
                scaler_d if use_amp else None,
            )

        if val_loader and epoch % cfg.trainer.validation_interval == 0:
            validate(
                epoch,
                cfg,
                generator,
                val_loader,
                device,
            )

        if val_loader and epoch % cfg.trainer.preview_interval == 0:
            save_validation_preview(cfg, generator, val_loader, device, epoch)


def validate(
    epoch: int,
    cfg: ExperimentConfig,
    generator: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> None:
    generator.eval()
    metric = MetricAverager()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            outputs = generator(inputs)
            l1 = torch.mean(torch.abs(outputs - targets))
            psnr = _psnr(outputs, targets)
            metric.update({"l1": l1, "psnr": psnr})
    averages = metric.compute()
    save_metrics(cfg.output_dir / "val_metrics.jsonl", {"epoch": epoch, **averages})


def save_validation_preview(
    cfg: ExperimentConfig,
    generator: torch.nn.Module,
    dataloader,
    device: torch.device,
    epoch: int,
) -> None:
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = generator(inputs)
    save_preview(cfg.output_dir / "previews" / f"epoch_{epoch:04d}.png", inputs.cpu(), outputs.cpu(), targets.cpu())

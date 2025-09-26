"""Training script for PyStarNet."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Optional

import torch
from torch import amp
from torch.optim import Adam
from tqdm import tqdm

from .configs import DatasetConfig, ExperimentConfig
from .data import build_dataloader
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
    pred_01 = ((pred.clamp(-1.0, 1.0) + 1.0) / 2.0)
    target_01 = ((target.clamp(-1.0, 1.0) + 1.0) / 2.0)
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[PyStarNet] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def _should_use_amp(device: torch.device, requested: bool) -> bool:
    if not requested:
        return False
    if device.type != "cuda":
        return False
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev_index)
    arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    if arch not in supported_arches:
        print(
            f"[PyStarNet] CUDA arch {arch} not in PyTorch build ({supported_arches}). "
            "Disabling mixed precision to prevent instability."
        )
        return False
    return True


def _gan_weight_for_epoch(loss_cfg, epoch: int) -> float:
    warmup = max(loss_cfg.gan_warmup_epochs, 0)
    if warmup > 0 and epoch <= warmup:
        alpha = epoch / warmup
        return loss_cfg.gan_warmup_weight + alpha * (loss_cfg.gan_weight - loss_cfg.gan_warmup_weight)
    return loss_cfg.gan_weight


def _accumulate(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(target.parameters(), source.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1 - decay)
        for ema_buffer, buffer in zip(target.buffers(), source.buffers()):
            ema_buffer.copy_(buffer)


def _requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def _r1_regularization(discriminator, inputs, targets) -> torch.Tensor:
    inputs.requires_grad_(True)
    targets.requires_grad_(True)
    logits = discriminator(inputs, targets)
    grad_outputs = torch.ones_like(logits)
    (grad_targets,) = torch.autograd.grad(
        outputs=logits,
        inputs=targets,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    penalty = grad_targets.pow(2).view(grad_targets.size(0), -1).sum(dim=1).mean()
    return penalty


def train(config: ExperimentConfig) -> None:
    project_root = Path(__file__).resolve().parent
    cfg = config.resolve(project_root)
    ensure_dir(cfg.output_dir)
    set_seed(cfg.trainer.seed)

    device = _resolve_device(cfg.trainer.device)

    if cfg.trainer.augment_start_epoch <= cfg.trainer.supervised_epochs:
        cfg.trainer.augment_start_epoch = cfg.trainer.supervised_epochs + 1

    generator, discriminator = build_models(
        base_channels=cfg.model.base_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        attention=cfg.model.attention,
    )
    generator.to(device)
    discriminator.to(device)
    generator_ema = copy.deepcopy(generator).to(device)
    generator_ema.eval()

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
        lr=cfg.optimizer.lr_discriminator * 0.5,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )

    use_amp = _should_use_amp(device, cfg.trainer.mixed_precision)
    if use_amp:
        try:
            scaler_g = amp.GradScaler(device_type=device.type)
            scaler_d = amp.GradScaler(device_type=device.type)
        except TypeError:  # older torch: no device_type argument
            scaler_g = amp.GradScaler()
            scaler_d = amp.GradScaler()
    else:
        scaler_g = amp.GradScaler(enabled=False)
        scaler_d = amp.GradScaler(enabled=False)

    train_loader = build_dataloader(cfg.dataset, cfg.trainer, role="train")
    val_loader = _build_val_dataloader(cfg.dataset, cfg.trainer)
    base_augment = getattr(train_loader.dataset, "augment", False)

    global_step = 0
    best_val_l1 = math.inf

    if val_loader is not None:
        train_names = set(Path(p).name for p in getattr(train_loader.dataset, "inputs", []))
        val_names = set(Path(p).name for p in getattr(val_loader.dataset, "inputs", []))
        if train_names and val_names and train_names == val_names:
            print(
                "[PyStarNet] Warning: validation set appears identical to training set. "
                "Consider using separate tiles for meaningful metrics."
            )

    for epoch in range(1, cfg.trainer.max_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_metrics = MetricAverager()

        supervised_phase = epoch <= cfg.trainer.supervised_epochs or cfg.losses.gan_weight == 0
        if hasattr(train_loader.dataset, "augment"):
            train_loader.dataset.augment = base_augment and (epoch >= cfg.trainer.augment_start_epoch) and not supervised_phase
        if supervised_phase:
            current_gan_weight = 0.0
        else:
            current_gan_weight = _gan_weight_for_epoch(cfg.losses, epoch)
        enable_gan = current_gan_weight > 0
        augment_flag = getattr(train_loader.dataset, "augment", False)
        if supervised_phase:
            print(f"[PyStarNet] Epoch {epoch:03d}: Supervised warmup (GAN off, augmentation {augment_flag})")
        elif augment_flag:
            print(f"[PyStarNet] Epoch {epoch:03d}: GAN weight {current_gan_weight:.3f}, augmentation ON")
        else:
            print(f"[PyStarNet] Epoch {epoch:03d}: GAN weight {current_gan_weight:.3f}, augmentation OFF")

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

            if enable_gan:
                optim_d.zero_grad(set_to_none=True)
                _requires_grad(discriminator, True)
                _requires_grad(generator, False)
                autocast_cm = amp.autocast(device_type=device.type) if use_amp else nullcontext()
                with autocast_cm:
                    fake = generator(inputs)
                    fake = torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=-1.0)
                    logits_real = discriminator(inputs, targets)
                    logits_fake = discriminator(inputs, fake.detach())
                    logits_real = torch.clamp(logits_real, -10.0, 10.0)
                    logits_fake = torch.clamp(logits_fake, -10.0, 10.0)
                    loss_d = discriminator_loss(logits_real, logits_fake, hinge)
                if use_amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.unscale_(optim_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.trainer.gradient_clip)
                    scaler_d.step(optim_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.trainer.gradient_clip)
                    optim_d.step()

                if cfg.losses.d_reg_every > 0 and global_step % cfg.losses.d_reg_every == 0:
                    optim_d.zero_grad(set_to_none=True)
                    _requires_grad(generator, False)
                    autocast_cm = amp.autocast(device_type=device.type) if use_amp else nullcontext()
                    inputs.requires_grad_(True)
                    targets.requires_grad_(True)
                    with autocast_cm:
                        reg = _r1_regularization(discriminator, inputs, targets)
                    reg_loss = (cfg.losses.r1_gamma * 0.5) * reg
                    if use_amp:
                        scaler_d.scale(reg_loss).backward()
                        scaler_d.unscale_(optim_d)
                        scaler_d.step(optim_d)
                        scaler_d.update()
                    else:
                        reg_loss.backward()
                        optim_d.step()
                    inputs = inputs.detach()
                    targets = targets.detach()
                _requires_grad(discriminator, False)
            else:
                loss_d = torch.tensor(0.0, device=device)

            _requires_grad(generator, True)
            optim_g.zero_grad(set_to_none=True)
            autocast_cm = amp.autocast(device_type=device.type) if use_amp else nullcontext()
            with autocast_cm:
                fake = generator(inputs)
                fake = torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=-1.0)
                if enable_gan:
                    logits_fake = discriminator(inputs, fake)
                    logits_fake = torch.clamp(logits_fake, -10.0, 10.0)
                else:
                    logits_fake = torch.zeros(inputs.size(0), 1, device=device)
                gen_losses = generator_loss(
                    logits_fake,
                    fake,
                    targets,
                    hinge,
                    perceptual,
                    cfg.losses.l1_weight,
                    current_gan_weight,
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

            _accumulate(generator_ema, generator, cfg.trainer.ema_decay)

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
        print(
            "[PyStarNet] Epoch {:03d} train | L1: {:.4f} GAN: {:.4f} Perc: {:.4f} D: {:.4f}".format(
                epoch,
                averages.get("loss_l1", 0.0),
                averages.get("loss_gan", 0.0),
                averages.get("loss_perc", 0.0),
                averages.get("loss_d", 0.0),
            )
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
                generator_ema,
            )
            print(
                f"[PyStarNet] Saved checkpoint_epoch_{epoch:04d}.pt in {cfg.output_dir}" )

        if val_loader and epoch % cfg.trainer.validation_interval == 0:
            val_metrics = validate(
                epoch,
                cfg,
                generator_ema,
                val_loader,
                device,
            )
            if "l1" in val_metrics and val_metrics["l1"] < best_val_l1:
                best_val_l1 = val_metrics["l1"]
                save_checkpoint(
                    cfg.output_dir / "checkpoint_best.pt",
                    epoch,
                    global_step,
                    generator,
                    discriminator,
                    optim_g,
                    optim_d,
                    scaler_g if use_amp else None,
                    scaler_d if use_amp else None,
                    generator_ema,
                )
                print(
                    "[PyStarNet] New best checkpoint (val L1 {:.4f}) saved to {}".format(
                        best_val_l1,
                        cfg.output_dir / "checkpoint_best.pt",
                    )
                )

        if val_loader and epoch % cfg.trainer.preview_interval == 0:
            save_validation_preview(cfg, generator_ema, val_loader, device, epoch)


def validate(
    epoch: int,
    cfg: ExperimentConfig,
    generator: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    generator.eval()
    metric = MetricAverager()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            outputs = generator(inputs)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
            outputs = outputs.clamp(-1.0, 1.0)
            l1 = torch.mean(torch.abs(outputs - targets))
            psnr = _psnr(outputs, targets)
            metric.update({"l1": l1, "psnr": psnr})
    averages = metric.compute()
    save_metrics(cfg.output_dir / "val_metrics.jsonl", {"epoch": epoch, **averages})
    print(
        "[PyStarNet] Epoch {:03d} val   | L1: {:.4f} PSNR: {:.2f}".format(
            epoch,
            averages.get("l1", 0.0),
            averages.get("psnr", 0.0),
        )
    )
    return averages


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
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs = outputs.clamp(-1.0, 1.0)
    save_preview(cfg.output_dir / "previews" / f"epoch_{epoch:04d}.png", inputs.cpu(), outputs.cpu(), targets.cpu())

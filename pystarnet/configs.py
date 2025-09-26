"""Configuration dataclasses for PyStarNet."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DatasetConfig:
    root: Path = Path("../train_tiles")
    val_root: Optional[Path] = Path("../val_tiles")
    image_size: int = 256
    random_crop: bool = True
    augment: bool = True
    valid_extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass
class OptimizerConfig:
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass
class LossConfig:
    l1_weight: float = 10.0
    gan_weight: float = 1.0
    perceptual_weight: float = 0.1
    feature_layers: List[str] = field(
        default_factory=lambda: [
            "relu1_1",
            "relu2_1",
            "relu3_1",
            "relu4_1",
        ]
    )


@dataclass
class TrainerConfig:
    batch_size: int = 8
    num_workers: int = 8
    max_epochs: int = 200
    steps_per_epoch: Optional[int] = None
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_interval: int = 1
    validation_interval: int = 1
    preview_interval: int = 1
    log_interval: int = 50
    seed: int = 42
    device: str = "cuda"


@dataclass
class ModelConfig:
    base_channels: int = 64
    num_res_blocks: int = 2
    attention: bool = True


@dataclass
class ExperimentConfig:
    output_dir: Path = Path("./pystarnet_logs")
    dataset: DatasetConfig = DatasetConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    losses: LossConfig = LossConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: ModelConfig = ModelConfig()

    def resolve(self, project_root: Path) -> "ExperimentConfig":
        resolved = ExperimentConfig()
        resolved.dataset = DatasetConfig(
            root=(project_root / self.dataset.root).resolve(),
            val_root=(project_root / self.dataset.val_root).resolve()
            if self.dataset.val_root
            else None,
            image_size=self.dataset.image_size,
            random_crop=self.dataset.random_crop,
            augment=self.dataset.augment,
            valid_extensions=self.dataset.valid_extensions,
        )
        resolved.optimizer = self.optimizer
        resolved.losses = self.losses
        resolved.trainer = self.trainer
        resolved.model = self.model
        resolved.output_dir = (project_root / self.output_dir).resolve()
        return resolved

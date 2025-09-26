# Restormer Starless Training

This repository contains a plug-and-play pipeline based on **Restormer** to train a star-removal model directly on starless supervision pairs.

## Dataset Layout

Training and validation tiles should follow the existing layout:

```
train_tiles/
  input/
    *.png|jpg|tif
  target/
    *.png|jpg|tif
val_tiles/
  input/
  target/
```

Image pairs are expected to share filenames between the folders. All images are read in RGB, scaled to `[-1, 1]`, and cropped to the requested training size.

## Training

Install dependencies first (consider a fresh virtual environment):

```bash
pip install -r requirements.txt
```

Launch training with:

```bash
python train_restormer.py \
  --train-dir ./train_tiles \
  --val-dir ./val_tiles \
  --output-dir ./restormer_logs \
  --epochs 150 \
  --image-size 512 \
  --batch-size 4 \
  --lr 2e-4 \
  --perceptual-weight 0.1 \
  --mixed-precision
```

If you prefer a configuration file, edit `configs/restormer_default.yaml` and run:

```bash
python train_restormer.py --config configs/restormer_default.yaml
```

Key flags:

- `--no-augment` disables flips/rotations if you prefer deterministic crops.
- `--perceptual-weight` adds a VGG19-based perceptual loss (set to 0 for pure L1).
- Checkpoints are stored in `<output-dir>` every `--save-interval` epochs and whenever validation improves (`checkpoint_best.pt` + unique snapshots).

## Inference

```bash
python inference_restormer.py \
  --checkpoint ./restormer_logs/checkpoint_best.pt \
  --input /path/to/image.png \
  --output /path/to/image_starless.png \
  --tile 768 \
  --stride 384 \
  --device cuda \
  --show-progress
```

Tweak tile/stride for your GPU memory. The script automatically pads with reflection padding and blends overlapping predictions.

## Package

The `restormer_starless` package exposes:

- `Restormer`: the architecture definition
- `build_dataloader`: utility for the tile dataset
- `PerceptualLoss`: optional perceptual loss
- `ensure_dir`, `save_metrics`: simple helpers used by the scripts

Feel free to integrate the model into your own training loop if you need additional losses or logging.

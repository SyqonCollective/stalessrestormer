# PyStarNet

Implementazione PyTorch di StarNet++ con architettura UNet residua, self-attention e training moderno (hinge GAN, loss percettiva VGG e mixed precision opzionale).

## Caratteristiche principali

- Encoder-decoder con blocchi residuali e self-attention sul bottleneck
- PatchGAN con spectral norm e loss hinge per stabilità
- Loss percettiva basata su VGG19 pre-addestrata
- Mixed precision, gradient clipping, checkpointing e preview automatici
- Data pipeline per tile `input/` – `target/` (PNG/TIFF/JPEG) con augmentation

## Dipendenze

```
pip install -r requirements_pytorch.txt
```

Su RunPod (GPU NVIDIA, CUDA 12.x) puoi usare direttamente le build ufficiali di PyTorch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

## Training

```
python train_pytorch.py \
    --train-dir ../train_tiles \
    --val-dir ../val_tiles \
    --output-dir ./pystarnet_logs \
    --epochs 200 \
    --batch-size 8 \
    --mixed-precision
```

I checkpoint (es. `checkpoint_epoch_0020.pt`), i log JSONL e le preview `.png` vengono salvati in `pystarnet_logs/`.

## Inference

```
python inference_pytorch.py \
    --checkpoint pystarnet_logs/checkpoint_epoch_0200.pt \
    --input /path/to/input.tif \
    --output /path/to/output_starless.png \
    --tile-size 256 \
    --stride 128
```

Il modello lavora in floating point su range [-1, 1] e converte automaticamente in/out da immagini RGB.

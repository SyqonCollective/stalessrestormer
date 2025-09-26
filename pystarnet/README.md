# PyStarNet

Implementazione PyTorch di StarNet++ con architettura UNet residua, self-attention e training moderno (hinge GAN, loss percettiva VGG e mixed precision opzionale).

## Caratteristiche principali

- Encoder-decoder con blocchi residuali e self-attention sul bottleneck
- PatchGAN con spectral norm e loss hinge per stabilità
- Loss percettiva basata su VGG19 pre-addestrata
- Warmup supervisionato (solo L1+percettiva) nelle prime epoche, poi GAN a peso pieno con R1 regularization
- Mixed precision con fallback automatico se l'architettura CUDA non è supportata
- Gradient clipping su generator/discriminatore, EMA del generatore, monitor NaN automatico e checkpoint "best" basato sulla L1 di validazione
- Data pipeline per tile `input/` – `target/` (PNG/TIFF/JPEG) con le stesse augmentation di StarNet (rotazioni arbitrarie, resize, flip, channel shuffle, ecc.)

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
    --mixed-precision \
    --gan-warmup-epochs 5 \
    --gan-weight 0.2 \
    --r1-gamma 10 \
    --d-reg-every 16
```

I checkpoint per ogni epoca (`checkpoint_epoch_0020.pt`), il best model (`checkpoint_best.pt`), i log JSONL e le preview `.png` vengono salvati in `pystarnet_logs/`.

> **Nota su CUDA 12.0 (RTX 5090)**: se la build ufficiale di PyTorch non include ancora l'architettura `sm_120`, il training disattiverà automaticamente la mixed precision per evitare instabilità. Per sfruttare la GPU in fp16 conviene installare la nightly `cu124` da `https://download.pytorch.org/whl/nightly/cu124`.

Le prime epoche vengono dedicate al warmup supervisionato (solo L1 + percettiva) finché `gan_warmup_epochs` non è concluso; il peso avversario viene poi portato al valore definitivo (`gan_weight`) con R1 regularization (`r1_gamma`, `d_reg_every`) per mantenere il discriminatore stabile. Durante il training viene mantenuta anche una EMA del generatore utilizzata per validazione, preview e checkpoint.

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

## GUI

Per un'interfaccia grafica interattiva (selezione checkpoint, immagine e avanzamento), avvia:

```
python pystarnet_gui/inference_gui.py
```

Puoi scegliere tile size (default 512) e overlap; il risultato viene salvato accanto all'immagine originale con suffisso `_starless`.

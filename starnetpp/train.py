# This file is a part of StarNet code.
# https://github.com/nekitmm/starnet
#
# StarNet is a neural network that can remove stars from images leaving only background.
#
# Throughout the code all input and output images are 8 bits per channel tif images.
# This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
#
# Copyright (c) 2018 Nikita Misiura
# http://www.astrobin.com/users/nekitmm/
#
# This code is distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, express or implied.
# Please review LICENSE file before use.

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image as Image

import model
import starnet_utils

PANELS = 5                             # Number of panels in output pictures showcasing image transformations done by net.
MAX_TRAIN_IMGS = 30                    # Max number of training images loaded in each epoch. Increasing this value will increase memory
                                       # consumption, but will make outputs like losses and accuracy more smooth. 10 is default, but the
                                       # optimal value will depend on your machine and training image sizes.
L1_MULT = 100                          # L1 loss multiplier for output (just because it is much smaller than the most). Default is 100.
ACC_MULT = 100                         # Accuracy multiplier (to convert to percentage).
WINDOW_SIZE = 256                      # Size of the image fed to net. Do not change until you know what you are doing! Default is 256
                                       # and changing this will force you to train the net anew.

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
IMGS_DIR = LOGS_DIR
CHECKPOINT_PREFIX = BASE_DIR / "model.ckpt"
STEP_FILE = BASE_DIR / "step"
DEFAULT_TRAIN_DIR = BASE_DIR.parent / "train_tiles"
DEFAULT_VAL_DIR = BASE_DIR.parent / "val_tiles"


def _checkpoint_available() -> bool:
    return CHECKPOINT_PREFIX.with_suffix(".index").exists()


def _ensure_log_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    IMGS_DIR.mkdir(parents=True, exist_ok=True)


def _open_logs(resume: bool):
    _ensure_log_dirs()
    mode = "a" if resume else "w"
    l1 = (LOGS_DIR / "L1_loss.txt").open(mode)
    total = (LOGS_DIR / "total_loss.txt").open(mode)
    p_log = (LOGS_DIR / "perceptual_losses.txt").open(mode)
    acc_log = (LOGS_DIR / "accuracy.txt").open(mode)
    adv_log = (LOGS_DIR / "adversarial_losses.txt").open(mode)

    if not resume:
        l1.write(f"Epoch\tL1_loss (x{L1_MULT})\n")
        total.write("Epoch\tTotal_loss\n")
        p_log.write("Epoch\tP1\tP2\tP3\tP4\tP5\tP6\tP7\tP8\n")
        acc_log.write("Epoch\tAccuracy %\n")
        adv_log.write("Epoch\tGAN\tDiscriminative\n")
        for handle in (l1, total, p_log, acc_log, adv_log):
            handle.flush()

    return l1, total, p_log, acc_log, adv_log


def _close_logs(log_handles) -> None:
    for handle in log_handles:
        handle.flush()
        handle.close()


def _log_losses(log_handles, abs_epoch: float, losses: dict) -> None:
    l1, total, p_log, acc_log, adv_log = log_handles
    l1.write(f"{abs_epoch:.4f}\t{L1_MULT * losses['l1']:.5f}\n")
    total.write(f"{abs_epoch:.4f}\t{losses['total']:.5f}\n")
    p_log.write("{:.4f}\t".format(abs_epoch) + "\t".join(f"{loss:.5f}" for loss in losses["p"]) + "\n")
    acc_log.write(f"{abs_epoch:.4f}\t{ACC_MULT * losses['acc']:.5f}\n")
    adv_log.write(f"{abs_epoch:.4f}\t{losses['gan']:.5f}\t{losses['discrim']:.5f}\n")
    for handle in (l1, total, p_log, acc_log, adv_log):
        handle.flush()


def _load_step(default: int = 0) -> int:
    if STEP_FILE.exists():
        try:
            return int(Path(STEP_FILE).read_text().strip())
        except ValueError:
            return default
    return default


def _store_step(value: int) -> None:
    STEP_FILE.write_text(str(value))


def _format_losses(loss_values) -> dict:
    return {
        "discrim": float(loss_values[0]),
        "gan": float(loss_values[1]),
        "l1": float(loss_values[2]),
        "acc": float(loss_values[3]),
        "p": [float(x) for x in loss_values[4:12]],
        "total": float(loss_values[12]),
    }


def _print_losses(abs_epoch: float, step_index: int, losses: dict, verbose: bool) -> None:
    if verbose:
        print(
            f"Epoch {abs_epoch:.2f}: step {step_index}; discrim_loss: {losses['discrim']:.4f}; "
            f"gen_loss_GAN: {losses['gan']:.4f}; gen_loss_L1: {L1_MULT * losses['l1']:.4f}; "
            f"acc: {ACC_MULT * losses['acc']:.2f}"
        )
        print(
            "                   "
            + "; ".join(f"p{i + 1}_loss: {loss:.4f}" for i, loss in enumerate(losses["p"]))
        )
    else:
        print(
            f"Epoch {abs_epoch:.2f}: step {step_index}; L1 loss: {L1_MULT * losses['l1']:.4f}; "
            f"Total loss: {losses['total']:.4f}; acc: {ACC_MULT * losses['acc']:.2f}"
        )
    sys.stdout.flush()


def _save_preview(session, outputs, x_placeholder, y_placeholder, original, starless, panels, epoch_index: int) -> None:
    rows = []
    for _ in range(panels):
        x_sample, y_sample = starnet_utils.get_train_samples_with_augmentation(original, starless, 1)
        output = session.run(outputs, feed_dict={x_placeholder: x_sample, y_placeholder: y_sample})
        panel = (np.concatenate((x_sample[0], output[0], y_sample[0]), axis=1) + 1.0) / 2.0
        rows.append(panel)
        rows.append(np.zeros((1, WINDOW_SIZE * 3, 3), dtype=np.float32))
    preview = np.concatenate(rows, axis=0)
    image = Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8))
    image.save(IMGS_DIR / f"epoch_{epoch_index:04d}.png")


def _evaluate_validation(session, avers, x_placeholder, y_placeholder, val_dir: Path, val_list, batch: int, max_train_images: int):
    if not val_list:
        return None
    val_original, val_starless = starnet_utils.open_train_images(
        val_dir,
        val_list,
        min(max_train_images, len(val_list))
    )
    x_val, y_val = starnet_utils.get_train_samples_with_augmentation(val_original, val_starless, batch)
    val_losses = _format_losses(session.run(avers, feed_dict={x_placeholder: x_val, y_placeholder: y_val}))
    print(
        f"           val L1: {L1_MULT * val_losses['l1']:.4f}; val Total: {val_losses['total']:.4f}; "
        f"val acc: {ACC_MULT * val_losses['acc']:.2f}"
    )
    return val_losses


def train(
    epochs: int = 1,
    batch: int = 1,
    steps: int = 1000,
    output_freq: int = 50,
    verbose: bool = False,
    gen_plots: bool = True,
    images: bool = True,
    log_freq: int = 50,
    resume: bool = True,
    learning_rates=None,
    train_dir=DEFAULT_TRAIN_DIR,
    val_dir=DEFAULT_VAL_DIR,
    max_train_images: int = MAX_TRAIN_IMGS,
):
    if learning_rates is None:
        learning_rates = [0.002, 0.002]

    if gen_plots:
        import plot  # noqa: F401

    train_dir = Path(train_dir)
    val_dir = Path(val_dir) if val_dir else None

    train_list = starnet_utils.list_train_images(train_dir)
    if not train_list:
        raise ValueError(f"No training tiles found in {train_dir}")

    val_list = []
    if val_dir and val_dir.exists():
        val_list = starnet_utils.list_train_images(val_dir)

    X = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, WINDOW_SIZE, 3], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, WINDOW_SIZE, 3], name="Y")

    train_op, avers, outputs = model.model(X, Y, lr=learning_rates)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    resume_state = resume and _checkpoint_available()

    with tf.Session() as sess:
        sess.run(init)

        log_handles = _open_logs(resume_state)
        abs_step = _load_step(0) if resume_state else 0

        if resume_state:
            print("Restoring previous state of the model...")
            saver.restore(sess, str(CHECKPOINT_PREFIX))
            print("Done!")
        else:
            print("Starting new training run...")

        for epoch_index in range(epochs):
            start = time.time()
            original, starless = starnet_utils.open_train_images(
                train_dir,
                train_list,
                min(max_train_images, len(train_list))
            )

            for step_index in range(steps):
                abs_step += 1
                x_batch, y_batch = starnet_utils.get_train_samples_with_augmentation(original, starless, batch)
                _, loss_values = sess.run([train_op, avers], feed_dict={X: x_batch, Y: y_batch})
                losses = _format_losses(loss_values)
                abs_epoch = abs_step / steps

                if step_index % output_freq == 0:
                    _print_losses(abs_epoch, step_index, losses, verbose)

                if step_index % log_freq == 0:
                    _log_losses(log_handles, abs_epoch, losses)

            duration = float(time.time() - start)
            print(
                f"Epoch {abs_epoch - 1:.0f} took {duration:.1f} s; L1 loss: {L1_MULT * losses['l1']:.4f}; "
                f"Total loss: {losses['total']:.4f}; acc: {ACC_MULT * losses['acc']:.2f}"
            )

            _evaluate_validation(sess, avers, X, Y, val_dir, val_list, batch, max_train_images)
            saver.save(sess, str(CHECKPOINT_PREFIX))

            if images:
                try:
                    _save_preview(sess, outputs, X, Y, original, starless, PANELS, epoch_index)
                except Exception as exc:  # pragma: no cover - preview best effort
                    print(f"Preview generation failed: {exc}")

            _store_step(abs_step)

            if gen_plots:
                import plot as plot_module
                plot_module.plot(LOGS_DIR)

        _close_logs(log_handles)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train StarNet++ on tiled datasets.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run.")
    parser.add_argument("--batch", type=int, default=1, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=1000, help="Gradient steps per epoch.")
    parser.add_argument("--output-freq", type=int, default=50, dest="output_freq", help="Console logging frequency (steps).")
    parser.add_argument("--log-freq", type=int, default=50, dest="log_freq", help="Loss file logging frequency (steps).")
    parser.add_argument("--gen-lr", type=float, default=0.002, dest="gen_lr", help="Generator learning rate.")
    parser.add_argument("--disc-lr", type=float, default=0.002, dest="disc_lr", help="Discriminator learning rate.")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="Path to training tiles root.")
    parser.add_argument("--val-dir", default=str(DEFAULT_VAL_DIR), help="Path to validation tiles root.")
    parser.add_argument("--max-train-images", type=int, default=MAX_TRAIN_IMGS, dest="max_train_images", help="Max tiles to cache per epoch.")
    parser.add_argument("--no-images", action="store_true", help="Disable preview mosaic saving.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plot generation.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed loss breakdown.")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch even if checkpoints exist.")
    parser.add_argument("--resume", dest="resume_flag", action="store_true", help="Force resume even if checkpoints are missing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    resume_flag = args.resume_flag or not args.fresh
    train(
        epochs=args.epochs,
        batch=args.batch,
        steps=args.steps,
        output_freq=args.output_freq,
        verbose=args.verbose,
        gen_plots=not args.no_plots,
        images=not args.no_images,
        log_freq=args.log_freq,
        resume=resume_flag,
        learning_rates=[args.gen_lr, args.disc_lr],
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        max_train_images=args.max_train_images,
    )

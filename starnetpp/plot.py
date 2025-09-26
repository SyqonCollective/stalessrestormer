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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"                   # Output directory.

# list of nice colours
# thanks to this tutorial:
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = np.array([
    [31, 119, 180],
    [174, 199, 232],
    [255, 127, 14],
    [255, 187, 120],
    [44, 160, 44],
    [152, 223, 138],
    [214, 39, 40],
    [255, 152, 150],
    [148, 103, 189],
    [197, 176, 213],
    [140, 86, 75],
    [196, 156, 148],
    [227, 119, 194],
    [247, 182, 210],
    [127, 127, 127],
    [199, 199, 199],
    [188, 189, 34],
    [219, 219, 141],
    [23, 190, 207],
    [158, 218, 229],
], dtype=np.float32) / 255.0


def plot(logs_dir=None) -> None:
    logs_dir = Path(logs_dir) if logs_dir else LOGS_DIR
    plots = [
        ("perceptual_losses", "Perceptual losses", [f"P {i}" for i in range(1, 9)]),
        ("L1_loss", "L1 loss", [""]),
        ("accuracy", "Accuracy, %", [""] ),
        ("total_loss", "Total loss", [""] ),
        ("adversarial_losses", "Adversarial losses", ["Discriminative", "Generative"]),
    ]

    for filename, title, labels in plots:
        txt_path = logs_dir / f"{filename}.txt"
        if not txt_path.exists():
            continue
        try:
            data = np.loadtxt(txt_path, skiprows=1, delimiter="\t")
        except OSError:
            continue
        if data.size == 0:
            continue
        _create_plot(logs_dir, filename, title, labels, "Epoch", np.atleast_2d(data))


def _create_plot(output_dir: Path, file_name: str, plot_name: str, data_labels, x_label: str, data: np.ndarray) -> None:
    plt.figure(figsize=(16, 9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_x = int(data[-1, 0]) + 2
    y_data = data[:, 1:] if data.shape[1] > 1 else data[:, :1]
    max_y = float(np.max(y_data))
    if max_y == 0:
        max_y = 1.0
    plt.ylim(0, max_y * 1.2)
    plt.xlim(0, max_x)

    yaxis_range = np.linspace(0, max_y * 1.2, num=11)
    plt.yticks(yaxis_range, [f"{value:.2f}" for value in yaxis_range], fontsize=20)
    plt.xticks(fontsize=20)

    for y in yaxis_range:
        plt.plot(range(0, max_x + 1), [y] * (max_x + 1), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom=True, top=False, labelbottom=True, left=True, right=False, labelleft=True)

    ly_pos = np.linspace(max_y * 0.25, max_y, num=len(data_labels)) if data_labels[0] else []
    colour_indices = [0, 4, 16, 6, 3, 10, 8, 14]

    plt.text(max_x + 0.25, 0, x_label, fontsize=20, color="black")

    for index in range(1, data.shape[1]):
        colour = tableau20[colour_indices[index - 1]]
        plt.plot(data[:, 0], data[:, index], lw=1.5, color=colour)
        if ly_pos:
            plt.text(max_x + 0.5, ly_pos[index - 1], data_labels[index - 1], fontsize=20, color=colour)

    plt.text(max_x / 2, max_y * 1.05, plot_name, fontsize=25, ha="center")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{file_name}.png", bbox_inches="tight")
    plt.close()

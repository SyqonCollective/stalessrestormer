"""Simple Tkinter GUI for running Restormer starless inference."""

from __future__ import annotations

import threading
from pathlib import Path
import sys
from tkinter import StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import torch

# Ensure project root on sys.path when launched from subfolders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_restormer import (
    load_model,
    prepare_image,
    save_image,
    sliding_window,
)


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StarlessGUI:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Restormer Starless GUI")
        self.root.geometry("520x260")
        self.root.resizable(False, False)

        self.checkpoint_var = StringVar()
        self.image_var = StringVar()
        self.tile_var = StringVar(value="512")
        self.overlap_var = StringVar(value="128")
        self.mode_var = StringVar(value="tiles")
        self.status_var = StringVar(value="Pronto")
        self.progress_var = StringVar(value="0%")

        self.tile_entry: ttk.Entry | None = None
        self.overlap_entry: ttk.Entry | None = None

        self._build_ui()
        self._update_mode()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, **padding)

        ttk.Label(frame, text="Checkpoint").grid(row=0, column=0, sticky="w")
        checkpoint_entry = ttk.Entry(frame, textvariable=self.checkpoint_var, width=45)
        checkpoint_entry.grid(row=0, column=1, sticky="we", padx=(0, 5))
        ttk.Button(frame, text="Sfoglia", command=self._browse_checkpoint).grid(row=0, column=2)
        ttk.Button(frame, text="Usa best", command=self._choose_best).grid(row=0, column=3, padx=(5, 0))

        ttk.Label(frame, text="Immagine").grid(row=1, column=0, sticky="w")
        image_entry = ttk.Entry(frame, textvariable=self.image_var, width=45)
        image_entry.grid(row=1, column=1, sticky="we", padx=(0, 5))
        ttk.Button(frame, text="Sfoglia", command=self._browse_image).grid(row=1, column=2)

        ttk.Label(frame, text="Modalità").grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(
            frame,
            text="Sliding window",
            variable=self.mode_var,
            value="tiles",
            command=self._update_mode,
        ).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(
            frame,
            text="Intera immagine",
            variable=self.mode_var,
            value="full",
            command=self._update_mode,
        ).grid(row=2, column=2, sticky="w")

        ttk.Label(frame, text="Dimensione tile").grid(row=3, column=0, sticky="w")
        self.tile_entry = ttk.Entry(frame, textvariable=self.tile_var, width=10)
        self.tile_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(frame, text="Overlap (px)").grid(row=4, column=0, sticky="w")
        self.overlap_entry = ttk.Entry(frame, textvariable=self.overlap_var, width=10)
        self.overlap_entry.grid(row=4, column=1, sticky="w")

        self.run_button = ttk.Button(frame, text="Avvia", command=self.run_inference)
        self.run_button.grid(row=5, column=0, columnspan=4, pady=(10, 0))

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=4, sticky="we", pady=(10, 0))
        self.progress["maximum"] = 100

        ttk.Label(frame, textvariable=self.status_var).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Label(frame, textvariable=self.progress_var, anchor="e").grid(row=7, column=3, sticky="e", pady=(5, 0))

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=0)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title="Seleziona checkpoint", filetypes=[("File PyTorch", "*.pt *.pth"), ("Tutti i file", "*.*")])
        if path:
            self.checkpoint_var.set(path)

    def _choose_best(self) -> None:
        current = self.checkpoint_var.get().strip()
        initial_dir = PROJECT_ROOT
        if current:
            maybe = Path(current).expanduser()
            if maybe.is_dir():
                initial_dir = maybe
            elif maybe.parent.exists():
                initial_dir = maybe.parent
        if not initial_dir.exists():
            initial_dir = Path.cwd()

        path = filedialog.askopenfilename(
            title="Seleziona checkpoint_best",
            initialdir=str(initial_dir),
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(title="Seleziona immagine", filetypes=[("Immagini", "*.png *.jpg *.jpeg *.tif *.tiff"), ("Tutti i file", "*.*")])
        if path:
            self.image_var.set(path)

    def run_inference(self) -> None:
        checkpoint = Path(self.checkpoint_var.get()).expanduser()
        image_path = Path(self.image_var.get()).expanduser()

        mode = self.mode_var.get()

        tile = None
        overlap = None
        stride = None
        if mode == "tiles":
            try:
                tile = int(self.tile_var.get())
                overlap = int(self.overlap_var.get())
            except ValueError:
                messagebox.showerror("Errore", "Tile e overlap devono essere numeri interi.")
                return

            if tile <= 0:
                messagebox.showerror("Errore", "La dimensione del tile deve essere positiva.")
                return
            if not 0 <= overlap < tile:
                messagebox.showerror("Errore", "L'overlap deve essere compreso tra 0 e la dimensione del tile - 1.")
                return
            stride = tile - overlap

        if not checkpoint.exists():
            messagebox.showerror("Errore", f"Checkpoint non trovato: {checkpoint}")
            return
        if not image_path.exists():
            messagebox.showerror("Errore", f"Immagine non trovata: {image_path}")
            return
        output_path = image_path.with_name(f"{image_path.stem}_starless{image_path.suffix}")

        self.run_button.config(state="disabled")
        self.status_var.set("Elaborazione in corso...")
        self.progress_var.set("0%")
        self.progress["value"] = 0

        device = _default_device()

        def task() -> None:
            try:
                model = load_model(checkpoint, device)
                image = prepare_image(image_path)

                if mode == "tiles":
                    def _update_progress(value: float) -> None:
                        percent = int(value * 100)
                        self.root.after(0, self._set_progress, percent)

                    output = sliding_window(model, image, tile, stride, device, _update_progress)
                else:
                    self.root.after(0, self._set_progress, 50)
                    with torch.no_grad():
                        output = model(image.to(device)).detach().cpu()
                    self.root.after(0, self._set_progress, 100)
                save_image(output, output_path)
                self.root.after(0, self._on_success, output_path)
            except Exception as exc:  # pylint: disable=broad-except
                self.root.after(0, self._on_error, exc)

        threading.Thread(target=task, daemon=True).start()

    def _set_progress(self, percent: int) -> None:
        self.progress["value"] = percent
        self.progress_var.set(f"{percent}%")

    def _update_mode(self) -> None:
        state = "normal" if self.mode_var.get() == "tiles" else "disabled"
        if self.tile_entry is not None:
            self.tile_entry.config(state=state)
        if self.overlap_entry is not None:
            self.overlap_entry.config(state=state)

    def _on_success(self, output: Path) -> None:
        self.run_button.config(state="normal")
        self.status_var.set(f"Completato: {output}")
        self.progress["value"] = 100
        self.progress_var.set("100%")
        messagebox.showinfo("Fatto", f"Immagine salvata in {output}")

    def _on_error(self, exc: Exception) -> None:
        self.run_button.config(state="normal")
        self.status_var.set("Errore durante l'elaborazione")
        messagebox.showerror("Errore", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = StarlessGUI()
    app.run()


if __name__ == "__main__":
    main()

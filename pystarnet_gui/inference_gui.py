"""Tkinter GUI for interactive starless inference."""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue, Empty
from tkinter import Tk, StringVar, IntVar, DoubleVar, N, S, E, W, filedialog, messagebox
from tkinter import ttk

import torch

from inference_pytorch import (
    load_generator,
    prepare_image,
    save_image,
    sliding_window_inference,
)


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        checkpoint: Path,
        image_path: Path,
        tile_size: int,
        overlap: int,
        queue: Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.checkpoint = checkpoint
        self.image_path = image_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.queue = queue

    def run(self) -> None:
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                size_mb = self.checkpoint.stat().st_size / 1024 / 1024
            except OSError:
                size_mb = 0
            print(f"[GUI] Caricamento checkpoint: {self.checkpoint} ({size_mb:.2f} MB)")
            print(f"[GUI] Immagine: {self.image_path}")
            print(f"[GUI] Parametri: tile_size={self.tile_size}, overlap={self.overlap}, device={device}")

            self.queue.put(("status", f"Caricamento modello su {device}..."))
            generator = load_generator(self.checkpoint, device)
            generator.eval()

            self.queue.put(("status", "Caricamento immagine..."))
            image = prepare_image(self.image_path)

            self.queue.put(("status", "Inizio elaborazione..."))

            stride = max(1, self.tile_size - self.overlap)

            def progress_cb(value: float) -> None:
                self.queue.put(("progress", value))

            output = sliding_window_inference(
                generator,
                image,
                self.tile_size,
                stride,
                device,
                progress_cb,
            )

            target_path = self.image_path.with_name(f"{self.image_path.stem}_starless{self.image_path.suffix}")
            save_image(output, target_path)
            self.queue.put(("done", target_path))
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.queue.put(("error", str(exc)))


class InferenceGUI:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("PyStarNet Inference GUI")
        self.root.geometry("520x260")

        self.checkpoint_var = StringVar()
        self.image_var = StringVar()
        self.tile_var = IntVar(value=512)
        self.overlap_var = IntVar(value=64)
        self.status_var = StringVar(value="Pronto")
        self.progress_var = DoubleVar(value=0.0)

        self.queue: Queue = Queue()
        self.worker: InferenceWorker | None = None

        self._build_ui()
        self.root.grid_columnconfigure(1, weight=1)

    def _build_ui(self) -> None:
        padding = {"padx": 8, "pady": 4}

        ttk.Label(self.root, text="Checkpoint:").grid(row=0, column=0, sticky=E, **padding)
        entry_ckpt = ttk.Entry(self.root, textvariable=self.checkpoint_var)
        entry_ckpt.grid(row=0, column=1, sticky=W + E, **padding)
        ttk.Button(self.root, text="Scegli", command=self._browse_checkpoint).grid(row=0, column=2, sticky=W, **padding)

        ttk.Label(self.root, text="Immagine:").grid(row=1, column=0, sticky=E, **padding)
        entry_img = ttk.Entry(self.root, textvariable=self.image_var)
        entry_img.grid(row=1, column=1, sticky=W + E, **padding)
        ttk.Button(self.root, text="Scegli", command=self._browse_image).grid(row=1, column=2, sticky=W, **padding)

        ttk.Label(self.root, text="Tile size:").grid(row=2, column=0, sticky=E, **padding)
        ttk.Spinbox(self.root, from_=64, to=2048, textvariable=self.tile_var, increment=32).grid(row=2, column=1, sticky=W, **padding)

        ttk.Label(self.root, text="Overlap:").grid(row=3, column=0, sticky=E, **padding)
        ttk.Spinbox(self.root, from_=0, to=2047, textvariable=self.overlap_var, increment=16).grid(row=3, column=1, sticky=W, **padding)

        self.start_button = ttk.Button(self.root, text="Avvia", command=self._start_inference)
        self.start_button.grid(row=4, column=0, columnspan=3, sticky=W + E, padx=8, pady=12)

        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=1.0)
        self.progress.grid(row=5, column=0, columnspan=3, sticky=W + E, padx=8, pady=4)

        ttk.Label(self.root, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky=W, padx=8, pady=4)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona checkpoint",
            filetypes=(("Checkpoint", "*.pt"), ("Tutti i file", "*.*")),
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=(("Immagini", "*.png *.jpg *.jpeg *.tif *.tiff"), ("Tutti i file", "*.*")),
        )
        if path:
            self.image_var.set(path)

    def _start_inference(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("In corso", "Elaborazione già in corso")
            return

        checkpoint = Path(self.checkpoint_var.get())
        image_path = Path(self.image_var.get())
        tile_size = self.tile_var.get()
        overlap = self.overlap_var.get()

        if not checkpoint.exists():
            messagebox.showerror("Errore", f"Checkpoint non trovato:\n{checkpoint}")
            return
        if checkpoint.stat().st_size == 0:
            messagebox.showerror("Errore", f"Il checkpoint è vuoto (0 byte):\n{checkpoint}")
            return
        if not image_path.exists():
            messagebox.showerror("Errore", f"Immagine non trovata:\n{image_path}")
            return
        if tile_size <= 0:
            messagebox.showerror("Errore", "Tile size deve essere maggiore di 0")
            return
        if not (0 <= overlap < tile_size):
            messagebox.showerror("Errore", "Overlap deve essere compreso tra 0 e tile size - 1")
            return

        self.progress_var.set(0.0)
        self.status_var.set("Preparazione...")
        self.start_button.config(state="disabled")

        self.worker = InferenceWorker(checkpoint, image_path, tile_size, overlap, self.queue)
        self.worker.start()
        self.root.after(100, self._poll_queue)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind, payload = item
                if kind == "progress":
                    self.progress_var.set(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self.progress_var.set(1.0)
                    self.status_var.set(f"Completato: {payload}")
                    messagebox.showinfo("Completato", f"Immagine salvata in:\n{payload}")
                    self.start_button.config(state="normal")
                elif kind == "error":
                    self.status_var.set("Errore")
                    messagebox.showerror("Errore", payload)
                    self.start_button.config(state="normal")
        except Empty:
            pass

        if self.worker and self.worker.is_alive():
            self.root.after(100, self._poll_queue)
        elif self.start_button["state"] == "disabled":
            self.start_button.config(state="normal")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = InferenceGUI()
    gui.run()


if __name__ == "__main__":
    main()

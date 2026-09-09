"""
src/model/train.py
==================
Training pipeline for BioSoundCNN.

Usage
-----
    python src/model/train.py

Dataset structure expected (see config.py DATASET_DIR):
    data/raw/
        human/    (*.wav, *.mp3, *.flac, *.m4a)
        dog/
        cat/
        ...

The script:
1. Scans dataset folders
2. Computes mel-spectrograms and caches them
3. Splits into train/val/test
4. Trains BioSoundCNN with early stopping
5. Saves best model weights + label map
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SUPPORTED_CLASSES, DATASET_DIR, MODELS_DIR,
    CUSTOM_MODEL_PATH, LABEL_MAP_PATH,
    TARGET_SAMPLE_RATE, SEGMENT_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, FMIN, FMAX,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT,
)
from src.audio.preprocessing import preprocess
from src.audio.features import extract_mel_spectrogram, mel_to_fixed_size
from src.model.architecture import build_model, count_parameters


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def scan_dataset(dataset_dir: str, classes: list) -> list:
    """
    Scan dataset_dir for audio files organised in class sub-folders.

    Returns list of (file_path, class_index) tuples.
    """
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    samples = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARNING] Class folder not found: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS:
                samples.append((os.path.join(cls_dir, fname), idx))
    return samples


class AudioDataset(Dataset):
    """
    PyTorch Dataset that loads audio files, preprocesses them, and
    returns (mel_tensor, label) pairs.

    Mel-spectrograms are computed on-the-fly (or cached in memory).
    """

    def __init__(self, samples: list, classes: list, target_frames: int = 128):
        self.samples = samples
        self.classes = classes
        self.target_frames = target_frames
        self._cache = {}  # in-memory cache to avoid re-processing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if idx not in self._cache:
            try:
                y, sr, _ = preprocess(path, apply_noise_reduction=False)
                mel = extract_mel_spectrogram(y, sr)
                mel = mel_to_fixed_size(mel, self.target_frames)
                # Normalise to [0, 1]
                mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
            except Exception as exc:
                print(f"[WARNING] Skipping {path}: {exc}")
                mel = np.zeros((N_MELS, self.target_frames), dtype=np.float32)
            self._cache[idx] = mel

        mel = self._cache[idx]
        # Shape: (1, n_mels, n_frames) – 1 channel for CNN
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return tensor, label


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(dataset_dir: str = DATASET_DIR, classes: list = None):
    """
    Main training function.

    Parameters
    ----------
    dataset_dir : path to the root dataset directory
    classes     : list of class names (defaults to SUPPORTED_CLASSES)
    """
    if classes is None:
        classes = SUPPORTED_CLASSES

    print("=" * 60)
    print("  BioSound AI – Training Pipeline")
    print("=" * 60)

    # --- Scan dataset ---
    samples = scan_dataset(dataset_dir, classes)
    if len(samples) == 0:
        print(f"\n[ERROR] No audio files found in: {dataset_dir}")
        print("Please add audio files organised in class sub-folders.")
        print("Example structure:")
        for cls in classes:
            print(f"  {dataset_dir}/{cls}/  (*.wav, *.mp3, ...)")
        sys.exit(1)

    print(f"\nDataset: {len(samples)} audio files across {len(classes)} classes")
    for cls in classes:
        count = sum(1 for _, l in samples if l == classes.index(cls))
        print(f"  {cls:10s}: {count} files")

    # --- Create dataset and splits ---
    full_dataset = AudioDataset(samples, classes)
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nSplit: {n_train} train | {n_val} val | {n_test} test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Build model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    model = build_model(n_classes=len(classes), device=device)
    print(f"Parameters: {count_parameters(model):,}")

    # --- Optimizer & loss ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    criterion = nn.NLLLoss()

    # --- Training loop with early stopping ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()
                preds = output.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":    epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":  val_acc,
                "classes":  classes,
                "n_mels":   N_MELS,
                "n_frames": 128,
            }, CUSTOM_MODEL_PATH)
            # Save label map
            label_map = {str(i): cls for i, cls in enumerate(classes)}
            with open(LABEL_MAP_PATH, "w") as f:
                json.dump(label_map, f, indent=2)
            print(f"  *** New best model saved (val_acc={val_acc:.4f}) ***")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {CUSTOM_MODEL_PATH}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BioSoundCNN")
    parser.add_argument("--dataset", default=DATASET_DIR, help="Path to dataset directory")
    args = parser.parse_args()
    train(dataset_dir=args.dataset)

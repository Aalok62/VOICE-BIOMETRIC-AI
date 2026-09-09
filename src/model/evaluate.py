"""
src/model/evaluate.py
=====================
Evaluation of the trained BioSoundCNN on the held-out test set.

Computes:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Human-specific metrics (highlighted)
  - Confusion matrix

Usage
-----
    python src/model/evaluate.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SUPPORTED_CLASSES, DATASET_DIR, CUSTOM_MODEL_PATH,
    LABEL_MAP_PATH, METRICS_CACHE_PATH, MODELS_DIR,
    BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT,
)
from src.model.train import scan_dataset, AudioDataset
from src.model.architecture import BioSoundCNN


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------

def load_custom_model(model_path: str = CUSTOM_MODEL_PATH, device: str = "cpu"):
    """Load saved BioSoundCNN checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python src/model/train.py` first."
        )
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    n_mels  = checkpoint.get("n_mels", 128)
    n_frames = checkpoint.get("n_frames", 128)
    model = BioSoundCNN(n_classes=len(classes), n_mels=n_mels, n_frames=n_frames)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, classes


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(dataset_dir: str = DATASET_DIR, classes: list = None) -> dict:
    """
    Evaluate the trained model on the test split.

    Returns
    -------
    metrics : dict with accuracy, per-class metrics, confusion matrix
    """
    if classes is None:
        classes = SUPPORTED_CLASSES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trained_classes = load_custom_model(device=device)
    classes = trained_classes  # use classes the model was trained on

    # --- Reconstruct test split (same seed as training) ---
    samples = scan_dataset(dataset_dir, classes)
    full_dataset = AudioDataset(samples, classes)
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    _, _, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Collect predictions ---
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Overall accuracy ---
    overall_acc = float(accuracy_score(all_labels, all_preds))

    # --- Per-class metrics ---
    precision_per = precision_score(all_labels, all_preds, average=None, labels=list(range(len(classes))), zero_division=0)
    recall_per    = recall_score(   all_labels, all_preds, average=None, labels=list(range(len(classes))), zero_division=0)
    f1_per        = f1_score(       all_labels, all_preds, average=None, labels=list(range(len(classes))), zero_division=0)

    per_class = {}
    for idx, cls in enumerate(classes):
        cls_mask = all_labels == idx
        cls_acc  = float(accuracy_score(all_labels[cls_mask], all_preds[cls_mask])) if cls_mask.sum() > 0 else 0.0
        per_class[cls] = {
            "accuracy":  round(cls_acc, 4),
            "precision": round(float(precision_per[idx]), 4),
            "recall":    round(float(recall_per[idx]), 4),
            "f1":        round(float(f1_per[idx]), 4),
        }

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    # --- Human-specific metrics ---
    human_idx = classes.index("human") if "human" in classes else None
    human_metrics = per_class.get("human", {})

    metrics = {
        "overall_accuracy": round(overall_acc, 4),
        "per_class":        per_class,
        "human_metrics":    human_metrics,
        "confusion_matrix": cm.tolist(),
        "classes":          classes,
        "n_test_samples":   n_test,
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=classes, zero_division=0
        ),
    }

    # --- Cache metrics for fast loading in dashboard ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_CACHE_PATH, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    return metrics


def load_cached_metrics() -> dict | None:
    """Load previously computed metrics from cache (fast path for dashboard)."""
    if os.path.exists(METRICS_CACHE_PATH):
        with open(METRICS_CACHE_PATH, "r") as f:
            return json.load(f)
    return None


def plot_confusion_matrix(metrics: dict) -> plt.Figure:
    """Return a matplotlib Figure of the confusion matrix."""
    cm      = np.array(metrics["confusion_matrix"])
    classes = metrics["classes"]
    n       = len(classes)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([c.capitalize() for c in classes])
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", color="white")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate BioSoundCNN")
    parser.add_argument("--dataset", default=DATASET_DIR)
    args = parser.parse_args()

    print("Evaluating model on test set...")
    metrics = evaluate(dataset_dir=args.dataset)
    print(f"\nOverall Accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"\nHuman Metrics:")
    for k, v in metrics["human_metrics"].items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

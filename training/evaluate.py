"""
Evaluation script: confusion matrix, precision, recall, F1.
Run:  python -m training.evaluate
"""
from __future__ import annotations
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from pathlib import Path

from models.classifier import TetrominoCNN, NUM_CLASSES, PIECE_NAMES, get_device, MODEL_PATH
from training.train_model import MixedTetrominoDataset


def evaluate(model_path: Path = MODEL_PATH,
             syn_data_dir: Path = Path("data/dataset"),
             real_data_dir: Path = Path("data/real_dataset"),
             out_dir:  Path = Path("data")) -> dict:
    device = get_device()
    model  = TetrominoCNN(NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    ds     = MixedTetrominoDataset(syn_data_dir,real_data_dir, augment=False)
    loader = DataLoader(ds, batch_size=128)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            preds_all.extend(model(x.to(device)).argmax(1).cpu().tolist())
            labels_all.extend(y.tolist())

    print("\n=== Classification Report ===")
    report = classification_report(labels_all, preds_all, target_names=PIECE_NAMES)
    print(report)

    cm      = confusion_matrix(labels_all, preds_all)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=PIECE_NAMES, yticklabels=PIECE_NAMES,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")
    ax.set_title("Tetromino CNN — Normalised Confusion Matrix")
    plt.tight_layout()
    out_dir.mkdir(exist_ok=True)
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print(f"[Eval] Confusion matrix → {cm_path}")

    acc = float(np.mean(np.array(preds_all) == np.array(labels_all)))
    print(f"[Eval] Overall accuracy: {acc:.4f}")
    return {"accuracy": acc, "report": report}


if __name__ == "__main__":
    evaluate()

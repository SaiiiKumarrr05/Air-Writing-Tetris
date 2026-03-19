#!/usr/bin/env python3
"""
training/notebook.py
====================
Self-contained training notebook (plain Python — run directly or
open in Jupyter as a .py with Jupytext).

Sections
--------
1. Dataset generation
2. Dataloader + augmentation visualisation
3. Model training with live loss / accuracy plots
4. Evaluation: confusion matrix, classification report
5. Per-class sample visualisation
6. Inference smoke-test on held-out images
"""
# %% [markdown]
# # Tetromino CNN — Training Notebook

# %% 1. Imports & config
import sys, json, time, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.classifier import TetrominoCNN, NUM_CLASSES, PIECE_NAMES, get_device, MODEL_PATH
from training.generate_dataset import generate, OUT_DIR
from training.train_model import TetrominoDataset

DEVICE = get_device()
print(f"Device: {DEVICE}")

# %% 2. Generate dataset (skip if exists)
if not (OUT_DIR / "labels.json").exists():
    print("Generating synthetic dataset …")
    generate()
else:
    print(f"Dataset exists at {OUT_DIR}")

# %% 3. Dataset statistics
with open(OUT_DIR / "labels.json") as f:
    records = json.load(f)

counts = {n: sum(1 for r in records if r["label"] == n) for n in PIECE_NAMES}
print("\nSamples per class:")
for n, c in counts.items():
    print(f"  {n}: {c}")

# %% 4. Visualise samples
fig, axes = plt.subplots(7, 8, figsize=(14, 14))
for row_i, piece in enumerate(PIECE_NAMES):
    samples = [r for r in records if r["label"] == piece]
    chosen  = random.sample(samples, min(8, len(samples)))
    for col_i, item in enumerate(chosen):
        img = cv2.imread(str(OUT_DIR / item["file"]), cv2.IMREAD_GRAYSCALE)
        axes[row_i][col_i].imshow(img, cmap="hot")
        axes[row_i][col_i].axis("off")
        if col_i == 0:
            axes[row_i][col_i].set_title(piece, fontsize=12, color="white")
fig.patch.set_facecolor("#0d0d1a")
plt.tight_layout()
out = ROOT / "data" / "sample_grid.png"
plt.savefig(out, dpi=100, facecolor=fig.get_facecolor())
plt.close()
print(f"\nSample grid → {out}")

# %% 5. Dataloader
BATCH = 64
ds     = TetrominoDataset(OUT_DIR, augment=True)
n_val  = max(1, int(0.15 * len(ds)))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)
print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}")

# %% 6. Training loop
EPOCHS = 40
model     = TetrominoCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
opt       = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

history = {"loss": [], "val_acc": []}
best_acc = 0.0
MODEL_PATH.parent.mkdir(exist_ok=True)

for ep in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for x, y in tqdm(train_dl, desc=f"Ep {ep:02d}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        ep_loss += loss.item()
    sched.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    acc = correct / total
    history["loss"].append(ep_loss / len(train_dl))
    history["val_acc"].append(acc)
    print(f"  Ep {ep:02d}: loss={history['loss'][-1]:.4f}  val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest val accuracy: {best_acc:.4f}  →  {MODEL_PATH}")

# %% 7. Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4),
                                facecolor="#0d0d1a")
for ax in (ax1, ax2):
    ax.set_facecolor("#12122a");  ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#333355")

ax1.plot(history["loss"],    color="#00f0f0", lw=2)
ax1.set_title("Training Loss", color="white");  ax1.set_xlabel("Epoch", color="white")
ax2.plot(history["val_acc"], color="#00ff88", lw=2)
ax2.axhline(0.95, color="#ffcc00", lw=1, ls="--", label="95% target")
ax2.set_title("Val Accuracy",  color="white");  ax2.set_xlabel("Epoch", color="white")
ax2.legend(facecolor="#12122a", labelcolor="white")
plt.tight_layout()
curve_path = ROOT / "data" / "training_curves.png"
plt.savefig(curve_path, dpi=120)
plt.close()
print(f"Curves → {curve_path}")

# %% 8. Evaluation
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in DataLoader(TetrominoDataset(OUT_DIR, augment=False), batch_size=128):
        all_preds.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=PIECE_NAMES))

cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=PIECE_NAMES, yticklabels=PIECE_NAMES,
            cmap="Blues", ax=ax, vmin=0, vmax=1)
ax.set_xlabel("Predicted");  ax.set_ylabel("True")
ax.set_title("Normalised Confusion Matrix")
plt.tight_layout()
cm_path = ROOT / "data" / "confusion_matrix.png"
plt.savefig(cm_path, dpi=130)
plt.close()
print(f"Confusion matrix → {cm_path}")

# %% 9. Inference smoke-test
from vision.drawing_buffer import DrawingBuffer
from training.generate_dataset import TEMPLATES, augment, render_pts
import torch.nn.functional as F

buf = DrawingBuffer()
print("\n=== Inference smoke-test ===")
for piece in PIECE_NAMES:
    pts = augment(TEMPLATES[piece]())
    img = render_pts(pts, thickness=3)
    t   = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)
        conf, idx = probs.max(dim=1)
    pred = PIECE_NAMES[idx.item()]
    ok   = "✓" if pred == piece else "✗"
    print(f"  {ok} True={piece}  Pred={pred}  Conf={conf.item()*100:.1f}%")

"""
Training loop for TetrominoCNN (Synthetic + Real + Weighted Sampling)
Run:  python -m training.train_model
"""
from __future__ import annotations
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm

from models.classifier import (
    TetrominoCNN,
    NUM_CLASSES,
    PIECE_NAMES,
    get_device,
    MODEL_PATH
)

IMG_SIZE   = 64
SYNTH_DIR  = Path("data/dataset")
REAL_DIR   = Path("data/real_dataset")

BATCH      = 64
EPOCHS     = 40
LR         = 1e-3

REAL_WEIGHT  = 3.0
SYNTH_WEIGHT = 1.0

LABEL_MAP = {n: i for i, n in enumerate(PIECE_NAMES)}


# -----------------------------------------------------------------------
class MixedTetrominoDataset(Dataset):
    def __init__(self,
                 synth_dir: Path = SYNTH_DIR,
                 real_dir: Path = REAL_DIR,
                 augment: bool = False):

        self.samples = []
        self.is_real = []
        self.augment = augment

        # -----------------------------
        # Synthetic
        # -----------------------------
        lf = synth_dir / "labels.json"
        if lf.exists():
            with open(lf) as f:
                items = json.load(f)
            for it in items:
                self.samples.append((synth_dir / it["file"], it["label"]))
                self.is_real.append(0)

        # -----------------------------
        # Real
        # -----------------------------
        if real_dir.exists():
            for path in real_dir.glob("*.png"):
                label = path.name.split("_")[0]
                if label in LABEL_MAP:
                    self.samples.append((path, label))
                    self.is_real.append(1)

        print(f"Total samples: {len(self.samples)}")
        print(f"Real samples:  {sum(self.is_real)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.augment:
            # horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            # rotation
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))

            # gaussian noise
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        return tensor, LABEL_MAP[label_name]


# -----------------------------------------------------------------------
def train(model_out: Path = MODEL_PATH,
          epochs: int = EPOCHS) -> float:

    device = get_device()
    print(f"[Train] device={device}")

    dataset = MixedTetrominoDataset(augment=True)

    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # -----------------------------
    # Weighted Sampler (TRAIN ONLY)
    # -----------------------------
    weights = []
    for idx in train_ds.indices:
        if dataset.is_real[idx] == 1:
            weights.append(REAL_WEIGHT)
        else:
            weights.append(SYNTH_WEIGHT)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH
    )

    model = TetrominoCNN(NUM_CLASSES).to(device)

    if model_out.exists():
        model.load_state_dict(torch.load(model_out, map_location=device))
        print("Loaded existing model weights")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    opt       = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_dl, desc=f"Ep {ep:02d}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        sched.step()

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += len(y)

        acc = correct / total
        avg = total_loss / len(train_dl)

        print(f"  Ep {ep:02d}: loss={avg:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_out)

    print(f"[Train] Best val accuracy: {best_acc:.4f} → {model_out}")
    return best_acc


if __name__ == "__main__":
    train()
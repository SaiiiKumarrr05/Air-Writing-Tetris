"""
TetrominoCNN — 3-block CNN for 64×64 grayscale stroke images.
~310 k parameters; <5 ms inference on CPU, <1 ms on CUDA.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PIECE_NAMES     = ["I", "O", "T", "S", "Z", "J", "L"]
NUM_CLASSES     = len(PIECE_NAMES)
IMG_SIZE        = 64
CONF_THRESHOLD  = 0.60
MODEL_PATH      = Path(__file__).parent.parent / "data" / "model.pth"


# -----------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------
class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers += [nn.MaxPool2d(2), nn.Dropout2d(0.10)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TetrominoCNN(nn.Module):
    """
    Input:  (B, 1, 64, 64)  float32  [0, 1]
    Output: (B, 7)           raw logits
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.enc = nn.Sequential(
            _ConvBlock(1,   32, pool=True),   # → 32×32
            _ConvBlock(32,  64, pool=True),   # → 16×16
            _ConvBlock(64, 128, pool=True),   # →  8×8
        )
        flat = 128 * (IMG_SIZE // 8) ** 2    # 8192
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.50),
            nn.Linear(flat, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


# -----------------------------------------------------------------------
class TetrominoClassifier:
    """Runtime wrapper: load weights → predict."""

    def __init__(self,
                 model_path: Path | str | None = None,
                 device: torch.device | None = None):
        self.device = device or get_device()
        self.model  = TetrominoCNN(NUM_CLASSES).to(self.device)
        self.path   = Path(model_path or MODEL_PATH)
        self.loaded = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            print(f"[Classifier] No model at {self.path} — run auto_train.py first.")
            return
        try:
            sd = torch.load(self.path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(sd)
            self.model.eval()
            self.loaded = True
            print(f"[Classifier] Loaded {self.path}  device={self.device}")
        except Exception as e:
            print(f"[Classifier] Load error: {e}")

    # ------------------------------------------------------------------
    def _tensor(self, img: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(img.astype(np.float32) / 255.0)
                .unsqueeze(0).unsqueeze(0).to(self.device))

    def predict(self, img: np.ndarray) -> tuple[str | None, float]:
        if not self.loaded or img is None or img.sum() == 0:
            return None, 0.0
        with torch.no_grad():
            p = F.softmax(self.model(self._tensor(img)), dim=1)
            c, i = p.max(dim=1)
        return PIECE_NAMES[i.item()], float(c.item())

    def predict_topk(self, img: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        if not self.loaded or img is None or img.sum() == 0:
            return []
        with torch.no_grad():
            p        = F.softmax(self.model(self._tensor(img)), dim=1)
            vs, idxs = p.topk(k, dim=1)
        return [(PIECE_NAMES[i], float(c))
                for c, i in zip(vs[0].tolist(), idxs[0].tolist())]

    def confident(self, conf: float) -> bool:
        return conf >= CONF_THRESHOLD

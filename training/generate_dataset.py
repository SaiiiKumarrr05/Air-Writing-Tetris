"""
Synthetic tetromino stroke dataset generator.
Produces 64x64 grayscale PNGs + labels.json.

ALIGNED WITH DrawingBuffer.to_image()
- Preserves aspect ratio
- Black background (0)
- White strokes (255)
"""

from __future__ import annotations
import math
import random
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
IMG_SIZE    = 64
SAMPLES     = 900
OUT_DIR     = Path("data/dataset")


# -------------------------------------------------------------
# Stroke Interpolation
# -------------------------------------------------------------
def _seg(pts: list[tuple]) -> list[tuple]:
    out = []
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        end = (i == len(pts) - 2)

        for t in np.linspace(0, 1, 24, endpoint=end):
            out.append((ax + (bx - ax) * t,
                        ay + (by - ay) * t))
    return out


# -------------------------------------------------------------
# Tetromino Templates
# -------------------------------------------------------------
TEMPLATES = {
    "I": lambda: _seg([(0.1,0.5),(0.9,0.5)]),
    "O": lambda: _seg([(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8),(0.2,0.2)]),
    "T": lambda: _seg([(0.1,0.3),(0.9,0.3)]) + _seg([(0.5,0.3),(0.5,0.8)]),
    "S": lambda: (_seg([(0.5,0.25),(0.9,0.25)])
                + _seg([(0.5,0.25),(0.1,0.70)])
                + _seg([(0.1,0.70),(0.5,0.70)])),
    "Z": lambda: (_seg([(0.1,0.25),(0.5,0.25)])
                + _seg([(0.5,0.25),(0.9,0.70)])
                + _seg([(0.9,0.70),(0.5,0.70)])),
    "J": lambda: (_seg([(0.3,0.2),(0.7,0.2)])
                + _seg([(0.5,0.2),(0.5,0.75)])
                + _seg([(0.5,0.75),(0.25,0.75)])),
    "L": lambda: (_seg([(0.3,0.2),(0.7,0.2)])
                + _seg([(0.5,0.2),(0.5,0.75)])
                + _seg([(0.5,0.75),(0.75,0.75)])),
}


# -------------------------------------------------------------
# Geometric Augmentation
# -------------------------------------------------------------
def augment(pts,
            noise=0.03,
            scale_lo=0.7,
            scale_hi=1.3,
            ang_lo=-35.0,
            ang_hi=35.0):

    angle = math.radians(random.uniform(ang_lo, ang_hi))
    scale = random.uniform(scale_lo, scale_hi)

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    ca, sa = math.cos(angle), math.sin(angle)

    out = []
    for x, y in pts:
        x -= cx
        y -= cy
        x *= scale
        y *= scale

        xr = ca * x - sa * y
        yr = sa * x + ca * y

        xr += cx + random.gauss(0, noise)
        yr += cy + random.gauss(0, noise)

        out.append((xr, yr))

    return out


# -------------------------------------------------------------
# Rendering (Aspect-Ratio Preserving)
# -------------------------------------------------------------
def render_pts(pts, size=IMG_SIZE):
    thickness = random.randint(3, 5)
    img = np.zeros((size, size), dtype=np.uint8)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    mnx, mxx = min(xs), max(xs)
    mny, mxy = min(ys), max(ys)

    # IMPORTANT: preserve aspect ratio (match DrawingBuffer)
    r = max(mxx - mnx, mxy - mny, 1e-6)
    pad = 8

    norm = [
        (
            int(pad + (x - mnx) / r * (size - 2 * pad)),
            int(pad + (y - mny) / r * (size - 2 * pad))
        )
        for x, y in pts
    ]

    # Draw continuous stroke
    for i in range(1, len(norm)):
        if random.random() > 0.1:  # slight stroke dropout
            cv2.line(img, norm[i - 1], norm[i], 255, thickness)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Mild noise
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


# -------------------------------------------------------------
# Dataset Generation
# -------------------------------------------------------------
def generate(out_dir: Path = OUT_DIR, samples: int = SAMPLES):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for piece in PIECE_NAMES:
        pdir = out_dir / piece
        pdir.mkdir(exist_ok=True)
        template_fn = TEMPLATES[piece]

        for i in tqdm(range(samples), desc=f"{piece}", leave=False):
            pts = augment(template_fn())
            img = render_pts(pts)

            fname = f"{piece}_{i:04d}.png"
            cv2.imwrite(str(pdir / fname), img)

            records.append({
                "file": f"{piece}/{fname}",
                "label": piece
            })

    with open(out_dir / "labels.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"[Dataset] {len(records)} samples → {out_dir}")


if __name__ == "__main__":
    generate()
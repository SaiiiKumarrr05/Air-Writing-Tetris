# Air Writing Tetris 🎮

Play Tetris by **drawing tetromino shapes in the air** with your webcam.
Hand tracking via MediaPipe · Shape classification via PyTorch CNN · CUDA-accelerated.

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies  (CUDA wheels pulled automatically by torch)
pip install -r requirements.txt

# 3a. Run standalone OpenCV game  (primary entry point)
python game_main.py

# 3b.  — OR — run Streamlit web app
streamlit run app.py
```

On **first run** the model is auto-trained:
1. `auto_train.py` generates 6 300 synthetic stroke images (≈30 s)
2. Trains the CNN for 40 epochs (≈5 min CPU / ≈45 s CUDA)
3. Saves `data/model.pth`

To force retrain:
```bash
python auto_train.py
```

---

## How to Play

| Step | Action |
|------|--------|
| 1 | Point your index finger at the **highlighted drawing zone** |
| 2 | **Pinch** (thumb + index) to start drawing |
| 3 | Trace one of the 7 tetromino shapes |
| 4 | **Release** pinch to finish stroke |
| 5 | Press **SPACE** (or button) to confirm — piece spawns! |
| 6 | Use hand gestures to control the falling piece |

### Gesture Controls (while piece is falling)

| Gesture | Action |
|---------|--------|
| Swipe hand left | Move piece left |
| Swipe hand right | Move piece right |
| Swipe hand down | Hard drop |
| Draw a circle | Rotate piece |

### Keyboard Fallbacks

| Key | Action |
|-----|--------|
| `A` / `D` | Move left / right |
| `W` | Rotate |
| `S` | Soft drop |
| `E` | Hard drop |
| `SPACE` | Confirm drawing |
| `R` | Reset game |
| `Q` / `ESC` | Quit |

---

## Tetromino Shape Guide

```
I  ────────────    (horizontal line)

O  ████            (square / rectangle)
   ████

T  ────┬────       (T-shape, top bar then stem down)
       │

S   ─┐             (S-curve or Z drawn mirrored)
   ┌─┘

Z  ┌─              (Z-curve or S drawn mirrored)
    ─┘

J  ┌──────         (J-shape: top bar, stem down, hook left)
   └

L  ──────┐         (L-shape: top bar, stem down, hook right)
         └
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        AIR TETRIS                            │
│                                                              │
│  ┌─────────┐   BGR    ┌──────────────┐   landmarks          │
│  │  Webcam │─────────▶│ Hand Tracker │──────────────┐       │
│  │ OpenCV  │          │  MediaPipe   │              │       │
│  └─────────┘          └──────────────┘              │       │
│                                                     │       │
│                         ┌───────────────────────────┴──┐    │
│                         │                              │    │
│                  ┌──────▼──────┐         ┌─────────────▼──┐ │
│                  │  Drawing    │         │    Gesture     │ │
│                  │  Buffer     │         │  Recognizer    │ │
│                  │ (strokes)   │         │(swipe/circle)  │ │
│                  └──────┬──────┘         └──────┬─────────┘ │
│                         │ 64×64 image           │ gesture   │
│                  ┌──────▼──────┐                │           │
│                  │  Tetromino  │                │           │
│                  │    CNN      │                │           │
│                  │ (PyTorch)   │                │           │
│                  └──────┬──────┘                │           │
│                  piece + conf                   │           │
│                         └──────────┬────────────┘           │
│                                    │                        │
│                           ┌────────▼────────┐               │
│                           │  Tetris Engine  │               │
│                           │  (grid/physics) │               │
│                           └────────┬────────┘               │
│                                    │                        │
│                           ┌────────▼────────┐               │
│                           │    Renderer     │               │
│                           │   (OpenCV)      │               │
│                           └─────────────────┘               │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
air_tetris/
├── game_main.py              ← OpenCV standalone entry point
├── app.py                    ← Streamlit web entry point
├── auto_train.py             ← Bootstrap: generate data + train
├── requirements.txt
├── README.md
│
├── game/
│   ├── __init__.py
│   ├── pieces.py             ← 7 tetrominos, all rotations, colours
│   └── engine.py             ← Grid, collision, scoring, gravity
│
├── vision/
│   ├── __init__.py
│   ├── hand_tracker.py       ← MediaPipe wrapper
│   ├── drawing_buffer.py     ← Stroke capture, smoothing, rendering
│   └── gesture_recognizer.py ← Swipe + circle detection
│
├── models/
│   ├── __init__.py
│   └── classifier.py         ← TetrominoCNN + inference wrapper
│
├── training/
│   ├── __init__.py
│   ├── generate_dataset.py   ← Synthetic stroke image generation
│   ├── train_model.py        ← Training loop (AdamW + cosine LR)
│   ├── evaluate.py           ← Confusion matrix + classification report
│   └── notebook.py           ← End-to-end training notebook (Python)
│
├── utils/
│   ├── __init__.py
│   └── renderer.py           ← OpenCV game frame composer
│
└── data/
    ├── schema.json            ← Dataset schema specification
    ├── model.pth              ← Trained weights (auto-generated)
    ├── dataset/               ← Training images (auto-generated)
    │   ├── labels.json
    │   ├── I/  O/  T/  S/  Z/  J/  L/
    └── user_labeled/          ← Real samples saved from low-conf draws
        └── {CLASS}/
```

---

## CNN Architecture

```
Input (1×64×64)
   │
   ▼
ConvBlock-1  [Conv3×3→BN→ReLU] × 2  out=32   MaxPool→32×32
   │
ConvBlock-2  [Conv3×3→BN→ReLU] × 2  out=64   MaxPool→16×16
   │
ConvBlock-3  [Conv3×3→BN→ReLU] × 1  out=128  MaxPool→8×8
   │
Flatten → 8192
   │
Dropout(0.5) → Linear(8192→256) → ReLU
   │
Dropout(0.3) → Linear(256→128) → ReLU
   │
Linear(128→7) → softmax
```

- Parameters: ~310 k  
- Training: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, CrossEntropy + label smoothing 0.1  
- Target accuracy: ≥ 95 % on synthetic val set  
- Inference latency: < 5 ms CPU, < 1 ms CUDA  

---

## Confidence & Edge-Case Handling

| Condition | Behaviour |
|-----------|-----------|
| conf ≥ 0.60 | Auto-spawn detected piece |
| 0.45 ≤ conf < 0.60 | Show confidence bar warning, still auto-spawn |
| conf < 0.45 | **Ask user to label** — shows 7-choice picker |
| Stroke too short (< 15 pts) | Reject with message |
| Interrupted strokes (gap > 90 px) | Warn user to redraw |
| Multiple hands detected | Use first/dominant hand only |
| User labels low-conf sample | Saved to `data/user_labeled/` for retraining |

---

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| FPS | ≥ 25 | 30–60 (CUDA), 25–35 (CPU) |
| Inference latency | < 100 ms | < 5 ms |
| Val accuracy (synthetic) | ≥ 95 % | 96–98 % |
| Gesture debounce window | 320 ms | — |

---

## Retraining with User Data

After accumulating real user samples in `data/user_labeled/`:

```bash
# Copy user-labeled samples into the main dataset
python - << 'EOF'
import shutil, json
from pathlib import Path
base = Path("data/dataset")
user = Path("data/user_labeled")
records = json.load(open(base/"labels.json"))
for piece_dir in user.iterdir():
    for img_path in piece_dir.glob("*.png"):
        dst = base / piece_dir.name / img_path.name
        shutil.copy(img_path, dst)
        records.append({"file": f"{piece_dir.name}/{img_path.name}",
                        "label": piece_dir.name})
json.dump(records, open(base/"labels.json","w"), indent=2)
print(f"Merged. Total: {len(records)}")
EOF

# Retrain
python auto_train.py
```

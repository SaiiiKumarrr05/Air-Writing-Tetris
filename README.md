<div align="center">

# 🎮 Air Writing Tetris

**Play Tetris by drawing pieces in the air — no keyboard needed.**

Draw a shape with your finger → CNN classifies it → piece spawns → control it with hand gestures.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-0097A7?style=flat-square)](https://mediapipe.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## ✨ What Is This?

Air Writing Tetris is a computer vision project that replaces the keyboard entirely. Your webcam watches your hand, MediaPipe tracks your finger in real-time, and a custom-trained PyTorch CNN classifies whatever shape you drew in the air into one of the 7 classic Tetris pieces.

**Stack:** MediaPipe · PyTorch CNN · OpenCV · Streamlit · CUDA-accelerated

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SaiiiKumarrr05/Air-Writing-Tetris.git
cd Air-Writing-Tetris

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the standalone OpenCV game  ← recommended
python game_main.py

# 4b.  — OR — run the Streamlit web app
streamlit run app.py
```

> **First run?** The model auto-trains itself:
> 1. Generates ~6,300 synthetic stroke images (~30 sec)
> 2. Trains the CNN for 40 epochs (~5 min CPU / ~45 sec CUDA)
> 3. Saves weights to `data/model.pth`
>
> To force a retrain anytime: `python auto_train.py`

---

## 🕹️ How to Play

| Step | What to Do |
|------|-----------|
| 1 | Point your index finger at the **highlighted drawing zone** |
| 2 | **Pinch** (thumb + index together) to start drawing |
| 3 | Trace one of the 7 tetromino shapes in the air |
| 4 | **Release** the pinch to finish your stroke |
| 5 | Press **SPACE** (or the confirm button) — piece spawns! |
| 6 | Use hand gestures to control the falling piece |

### ✋ Gesture Controls

| Gesture | Action |
|---------|--------|
| Swipe hand **left** | Move piece left |
| Swipe hand **right** | Move piece right |
| Swipe hand **down** | Hard drop |
| Draw a **circle** | Rotate piece |

### ⌨️ Keyboard Fallbacks

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

## ✏️ Shape Guide

```
I  ────────────      horizontal line

O  ████              square / rectangle
   ████

T  ───┬───           T-shape
      │

S   ─┐               S-curve
  ┌─┘

Z  ┌─                Z-curve
    ─┘

J  ┌──────           top bar, stem down, hook left
   └

L  ──────┐           top bar, stem down, hook right
         └
```

---

## 🏗️ Architecture

```
Webcam (BGR frames)
    │
    ▼
Hand Tracker (MediaPipe)
    │
    ├──── index tip coords ────▶ Drawing Buffer
    │                                │
    │                          64×64 stroke image
    │                                │
    │                          Tetromino CNN (PyTorch)
    │                                │
    │                          piece + confidence
    │
    └──── wrist + tip coords ──▶ Gesture Recognizer
                                      │
                                  swipe / rotate
                                      │
                              ┌───────┴───────┐
                              │  Tetris Engine │
                              │ (grid/physics) │
                              └───────┬───────┘
                                      │
                                  Renderer (OpenCV)
```

---

## 🧠 CNN Architecture

```
Input: 1 × 64 × 64 (grayscale stroke)
  │
  ├─ ConvBlock 1: [Conv3×3 → BN → ReLU] × 2,  out=32,  MaxPool → 32×32
  ├─ ConvBlock 2: [Conv3×3 → BN → ReLU] × 2,  out=64,  MaxPool → 16×16
  ├─ ConvBlock 3: [Conv3×3 → BN → ReLU] × 1,  out=128, MaxPool →  8×8
  │
  ├─ Flatten → 8192
  ├─ Dropout(0.5) → Linear(8192→256) → ReLU
  ├─ Dropout(0.3) → Linear(256→128)  → ReLU
  └─ Linear(128→7) → Softmax
```

| Property | Value |
|----------|-------|
| Parameters | ~310K |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropy + label smoothing 0.1 |
| Val accuracy | 96–98% (synthetic) |
| Inference | < 5ms CPU / < 1ms CUDA |

---

## 🎯 Confidence Handling

| Confidence | Behaviour |
|-----------|-----------|
| ≥ 0.60 | Auto-spawn detected piece |
| 0.45 – 0.60 | Spawn with low-confidence warning |
| < 0.45 | Show 7-piece picker — you choose |
| Stroke < 15 pts | Rejected, redraw prompt |
| Gap > 90px in stroke | Warn to redraw continuously |

---

## 📁 Project Structure

```
Air-Writing-Tetris/
├── game_main.py              ← OpenCV standalone entry
├── app.py                    ← Streamlit web entry
├── auto_train.py             ← Bootstrap: generate data + train
├── requirements.txt
├── README.md
│
├── game/
│   ├── pieces.py             ← 7 tetrominos, rotations, colours
│   └── engine.py             ← Grid, collision, scoring, gravity
│
├── vision/
│   ├── hand_tracker.py       ← MediaPipe wrapper
│   ├── drawing_buffer.py     ← Stroke capture & smoothing
│   └── gesture_recognizer.py ← Swipe + circle detection
│
├── models/
│   └── classifier.py         ← TetrominoCNN + inference wrapper
│
├── training/
│   ├── generate_dataset.py   ← Synthetic stroke image generation
│   ├── train_model.py        ← Training loop
│   ├── evaluate.py           ← Confusion matrix + report
│   └── notebook.py           ← End-to-end training script
│
├── utils/
│   └── renderer.py           ← OpenCV frame composer
│
└── data/
    ├── model.pth              ← Trained weights (auto-generated)
    ├── dataset/               ← Training images (auto-generated)
    │   ├── labels.json
    │   └── I/ O/ T/ S/ Z/ J/ L/
    └── real_dataset/          ← Real samples from low-conf draws
```

---

## 🔁 Retrain with Your Own Data

After collecting real samples from low-confidence draws in `data/real_dataset/`:

```bash
# Merge user samples into the main dataset
python - << 'EOF'
import shutil, json
from pathlib import Path
base = Path("data/dataset")
user = Path("data/real_dataset")
records = json.load(open(base / "labels.json"))
for img_path in user.glob("*.png"):
    label = img_path.stem.split("_")[0]
    dst = base / label / img_path.name
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(img_path, dst)
    records.append({"file": f"{label}/{img_path.name}", "label": label})
json.dump(records, open(base / "labels.json", "w"), indent=2)
print(f"Merged. Total samples: {len(records)}")
EOF

# Retrain
python auto_train.py
```

---

## ⚡ Performance

| Metric | Target | Typical |
|--------|--------|---------|
| FPS | ≥ 25 | 30–60 (CUDA) / 25–35 (CPU) |
| Inference latency | < 100ms | < 5ms |
| Synthetic val accuracy | ≥ 95% | 96–98% |
| Gesture debounce | 320ms | — |

---

## 🛠️ Requirements

- Python 3.9+
- Webcam
- CUDA GPU *(optional, but recommended)*
- macOS / Linux / Windows

---

## 📄 License

MIT — do whatever you want with it.

---

<div align="center">
Built with 🖐️ + 🧠 + way too much caffeine · Woxsen University · ACV Project
</div>

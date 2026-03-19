"""
Auto-train bootstrap: generates dataset and trains model if model.pth absent.
Called automatically by game_main.py and app.py on first run.
"""
from pathlib import Path

MODEL_PATH   = Path("data/model.pth")
DATASET_PATH = Path("data/dataset")


def ensure_model(force: bool = False) -> bool:
    if not force and MODEL_PATH.exists():
        return True
    print("\n[AutoTrain] No trained model found — bootstrapping...\n")

    if not (DATASET_PATH / "labels.json").exists() or force:
        from training.generate_dataset import generate
        generate()

    from training.train_model import train
    acc = train()
    ok  = acc >= 0.75
    if not ok:
        print(f"[AutoTrain] WARNING: accuracy {acc:.4f} < 0.75 — consider re-running.")
    return ok


if __name__ == "__main__":
    ensure_model(force=True)

"""
FINAL Air-Writing Classifier Tester
Compatible with optimized HandTracker (returns results, small_shape)

Controls:
- Pinch (thumb + index) to draw
- Release to stop
- C = clear
- Q = quit
"""

import cv2
import torch
import numpy as np

from vision.drawing_buffer import DrawingBuffer
from vision.hand_tracker import HandTracker
from models.classifier import (
    TetrominoCNN,
    PIECE_NAMES,
    MODEL_PATH,
    get_device
)

# -------------------------------------------------
# Setup
# -------------------------------------------------
DEVICE = get_device()

model = TetrominoCNN(len(PIECE_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

tracker = HandTracker()
buf = DrawingBuffer()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

drawing = False


# -------------------------------------------------
# Inference
# -------------------------------------------------
def predict(img):
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return probs


# -------------------------------------------------
# Main Loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # IMPORTANT: pass BGR frame directly
    results, small_shape = tracker.process(frame)

    if results and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]

        tracker.draw(frame, lm)

        index_pt = tracker.index_tip(lm, small_shape)
        is_pinching = tracker.is_pinching(lm, small_shape)

        # Draw cursor
        cv2.circle(frame, index_pt, 6, (0, 255, 255), -1)

        if is_pinching:
            if not drawing:
                buf.start_stroke()
                drawing = True
            buf.add_point(index_pt)
        else:
            if drawing:
                buf.end_stroke()
                drawing = False

    # Draw accumulated strokes
    buf.draw_on_frame(frame)

    # Rasterized drawing for model
    drawing_img = buf.to_image()

    probs = None
    if buf.has_enough():
        probs = predict(drawing_img)

    # -------------------------------------------------
    # Sidebar UI
    # -------------------------------------------------
    h, w = frame.shape[:2]
    sidebar = np.zeros((h, 260, 3), dtype=np.uint8)
    sidebar[:] = (20, 20, 35)

    if probs is not None:
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[0]
        top_name = PIECE_NAMES[top_idx]
        top_conf = probs[top_idx]

        cv2.putText(sidebar,
                    f"Prediction: {top_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 150),
                    2)

        cv2.putText(sidebar,
                    f"{top_conf*100:.1f}%",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 255),
                    1)

        y = 90
        bar_w = 200

        for idx in sorted_idx:
            name = PIECE_NAMES[idx]
            p = probs[idx]
            fill = int(p * bar_w)

            cv2.rectangle(sidebar,
                          (20, y),
                          (20 + bar_w, y + 18),
                          (40, 40, 70),
                          -1)

            cv2.rectangle(sidebar,
                          (20, y),
                          (20 + fill, y + 18),
                          (0, 180, 255),
                          -1)

            cv2.putText(sidebar,
                        f"{name}",
                        (230, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1)

            y += 26

    combined = np.hstack((frame, sidebar))

    cv2.putText(combined,
                "Pinch to draw | C=clear | Q=quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 200),
                2)

    cv2.imshow("Air Model Test", combined)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('c'), ord('C')):
        buf.clear()

    elif key in (ord('q'), ord('Q')):
        break


cap.release()
tracker.close()
cv2.destroyAllWindows()
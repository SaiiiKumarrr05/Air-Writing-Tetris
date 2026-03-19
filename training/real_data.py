"""
FINAL Air-Writing Real Data Collector
Strong motion-aware augmentation
Compatible with optimized HandTracker

Run:
    python data_collection/collect_real_air_augmented.py
"""

import cv2
import numpy as np
import uuid
import random
from pathlib import Path
from scipy.ndimage import gaussian_filter

from vision.drawing_buffer import DrawingBuffer
from vision.hand_tracker import HandTracker
from models.classifier import PIECE_NAMES

# ------------------------------------------------------------
SAVE_DIR = Path("data/real_dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
AUG_PER_SAMPLE = 12
# ------------------------------------------------------------

tracker = HandTracker()
buf = DrawingBuffer()
cap = cv2.VideoCapture(0)

# Optional performance tuning
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

drawing = False

print("\nControls:")
print("Pinch to draw")
print("Press 1–7 to save (I,O,T,S,Z,J,L)")
print("C = clear | Q = quit\n")


# ============================================================
# STRONG AIR-WRITING AUGMENTATION
# ============================================================

def elastic_warp(img, alpha=5, sigma=4):
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*img.shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*img.shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(img.shape[1]),
                       np.arange(img.shape[0]))

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def random_motion_blur(img):
    if random.random() < 0.5:
        size = random.choice([3, 5])
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)
    return img


def random_perspective(img):
    shift = 5
    pts1 = np.float32([[0,0], [63,0], [0,63], [63,63]])
    pts2 = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [63-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), 63-random.randint(0,shift)],
        [63-random.randint(0,shift), 63-random.randint(0,shift)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (64, 64),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def random_thickness(img):
    k = random.choice([1,2,3])
    kernel = np.ones((k,k), np.uint8)
    if random.random() > 0.5:
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)


def random_erasing(img):
    if random.random() < 0.4:
        x = random.randint(10, 50)
        y = random.randint(10, 50)
        w = random.randint(4, 10)
        h = random.randint(4, 10)
        img[y:y+h, x:x+w] = 0
    return img


def contrast_jitter(img):
    alpha = random.uniform(0.7, 1.4)
    beta = random.randint(-25, 25)
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)


def gaussian_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def rotate(img):
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((32,32), angle, 1.0)
    return cv2.warpAffine(img, M, (64,64),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def strong_air_augment(img):
    img = rotate(img)
    img = elastic_warp(img)
    img = random_perspective(img)
    img = random_thickness(img)
    img = random_motion_blur(img)
    img = random_erasing(img)
    img = contrast_jitter(img)
    img = gaussian_noise(img)
    return img


# ============================================================
# Save Function
# ============================================================

def augment_and_save(img, label, base_id):

    variants = [img]

    for _ in range(AUG_PER_SAMPLE):
        aug = strong_air_augment(img.copy())
        variants.append(aug)

    for i, v in enumerate(variants):
        fname = f"{label}_{base_id}_{i}.png"
        cv2.imwrite(str(SAVE_DIR / fname), v)


# ============================================================
# Main Loop
# ============================================================

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

        cv2.circle(frame, index_pt, 6, (0,255,255), -1)

        if is_pinching:
            if not drawing:
                buf.start_stroke()
                drawing = True
            buf.add_point(index_pt)
        else:
            if drawing:
                buf.end_stroke()
                drawing = False

    buf.draw_on_frame(frame)

    cv2.putText(frame,
                "Pinch to draw | 1-7 save | C clear | Q quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,200),
                2)

    cv2.imshow("Air Collect", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in range(ord('1'), ord('8')):
        idx = key - ord('1')
        label = PIECE_NAMES[idx]

        img = buf.to_image()
        base_id = uuid.uuid4().hex[:8]

        augment_and_save(img, label, base_id)

        print(f"Saved augmented samples for {label}")
        buf.clear()

    elif key in (ord('c'), ord('C')):
        buf.clear()

    elif key in (ord('q'), ord('Q')):
        break


cap.release()
tracker.close()
cv2.destroyAllWindows()
"""
STRONG Offline Augmentation for Real Dataset
Run ONCE to expand dataset aggressively.

Usage:
    python tools/augment_real_dataset.py
"""

import cv2
import numpy as np
import random
from pathlib import Path

# ------------------------------------------------------------
DATA_DIR = Path("data/real_dataset")
IMG_SIZE = 64
AUG_PER_IMAGE = 15   # strong expansion
# ------------------------------------------------------------


# ------------------ BASIC TRANSFORMS ------------------------

def rotate(img):
    angle = random.uniform(-28, 28)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
    return cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def translate(img):
    tx = random.randint(-6, 6)
    ty = random.randint(-6, 6)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def scale(img):
    s = random.uniform(0.85, 1.2)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), 0, s)
    return cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ------------------ PERSPECTIVE -----------------------------

def perspective(img):
    shift = 6
    pts1 = np.float32([[0,0], [63,0], [0,63], [63,63]])
    pts2 = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [63-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), 63-random.randint(0,shift)],
        [63-random.randint(0,shift), 63-random.randint(0,shift)],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ------------------ MORPHOLOGY ------------------------------

def thickness_variation(img):
    k = random.choice([1,2,3])
    kernel = np.ones((k,k), np.uint8)
    if random.random() > 0.5:
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)

# ------------------ RANDOM ERASING --------------------------

def random_erasing(img):
    if random.random() < 0.4:
        x = random.randint(10, 50)
        y = random.randint(10, 50)
        w = random.randint(4, 12)
        h = random.randint(4, 12)
        img[y:y+h, x:x+w] = 0
    return img

# ------------------ MOTION BLUR -----------------------------

def motion_blur(img):
    if random.random() < 0.5:
        size = random.choice([3,5])
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)
    return img

# ------------------ CONTRAST & BRIGHTNESS -------------------

def contrast_brightness(img):
    alpha = random.uniform(0.7, 1.4)
    beta = random.randint(-30, 30)
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

# ------------------ NOISE -----------------------------------

def gaussian_noise(img):
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# ------------------ RE-CENTER RANDOM CROP -------------------

def random_crop_pad(img):
    margin = random.randint(2, 6)
    cropped = img[margin:64-margin, margin:64-margin]
    return cv2.resize(cropped, (64, 64))


# ------------------------------------------------------------
def strong_augment(img):
    img = rotate(img)
    img = translate(img)
    img = scale(img)

    if random.random() > 0.5:
        img = perspective(img)

    img = thickness_variation(img)
    img = random_crop_pad(img)
    img = random_erasing(img)
    img = motion_blur(img)
    img = contrast_brightness(img)
    img = gaussian_noise(img)

    return img


# ------------------------------------------------------------
def main():
    images = list(DATA_DIR.glob("*.png"))

    if not images:
        print("No images found.")
        return

    total_new = 0

    for img_path in images:
        if "_aug" in img_path.name:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        stem = img_path.stem

        for i in range(AUG_PER_IMAGE):
            aug = strong_augment(img.copy())
            new_name = f"{stem}_aug{i}.png"
            cv2.imwrite(str(DATA_DIR / new_name), aug)
            total_new += 1

    print(f"\nDone. Generated {total_new} augmented images.")
    print("Dataset now strongly expanded.\n")


if __name__ == "__main__":
    main()
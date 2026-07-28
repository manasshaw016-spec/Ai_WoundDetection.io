from __future__ import annotations

import datetime as dt
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = BASE_DIR / "wound_model.h5"
FALLBACK_MODEL_PATH = BASE_DIR / "wound_model.joblib"
HISTORY_CSV = BASE_DIR / "prediction_history.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_sample_dataset() -> None:
    """Create a lightweight sample dataset if the training folders are empty."""
    for split in ("train", "test"):
        for label in ("wound", "healthy"):
            folder = DATA_DIR / split / label
            folder.mkdir(parents=True, exist_ok=True)
            existing_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
            if existing_files:
                continue

            for idx in range(4 if split == "train" else 2):
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                if label == "wound":
                    cv2.circle(image, (112, 112), 65 + idx * 10, (0, 0, 255), -1)
                    cv2.rectangle(image, (70, 70), (154, 154), (255, 255, 255), 3)
                else:
                    cv2.rectangle(image, (40, 40), (184, 184), (0, 255, 0), 3)
                    cv2.circle(image, (112, 112), 40, (0, 0, 0), 2)

                noise = np.random.randint(0, 25, size=image.shape, dtype=np.uint8)
                image = cv2.add(image, noise)
                cv2.imwrite(str(folder / f"{label}_{idx}.png"), image)


def read_uploaded_image(uploaded_file):
    """Decode an uploaded image into a BGR OpenCV image."""
    if uploaded_file is None:
        return None

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image


def now_timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

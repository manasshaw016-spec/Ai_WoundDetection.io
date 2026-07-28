import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover - optional dependency may be unavailable
    tf = None
    Conv2D = Dense = Dropout = Flatten = MaxPooling2D = Sequential = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency may be unavailable
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = BASE_DIR / "wound_model.h5"
FALLBACK_MODEL_PATH = BASE_DIR / "wound_model.joblib"
IMAGE_SIZE = 224
FEATURE_SIZE = 64


def ensure_sample_dataset():
    """Create a tiny sample dataset if the expected folders are empty."""
    for split in ("train", "test"):
        for label in ("wound", "healthy"):
            folder = DATA_DIR / split / label
            folder.mkdir(parents=True, exist_ok=True)
            existing_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
            if existing_files:
                continue

            for idx in range(4 if split == "train" else 2):
                image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                if label == "wound":
                    cv2.circle(image, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), 65 + idx * 10, (0, 0, 255), -1)
                    cv2.rectangle(image, (70, 70), (IMAGE_SIZE - 70, IMAGE_SIZE - 70), (255, 255, 255), 3)
                else:
                    cv2.rectangle(image, (40, 40), (IMAGE_SIZE - 40, IMAGE_SIZE - 40), (0, 255, 0), 3)
                    cv2.circle(image, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), 40, (0, 0, 0), 2)

                noise = np.random.randint(0, 25, size=image.shape, dtype=np.uint8)
                image = cv2.add(image, noise)
                cv2.imwrite(str(folder / f"{label}_{idx}.png"), image)


def load_image_dataset():
    ensure_sample_dataset()
    images = []
    labels = []

    for label, class_dir in ((1, TRAIN_DIR / "wound"), (0, TRAIN_DIR / "healthy")):
        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)

    if len(images) < 4:
        raise ValueError("Not enough training images were found. Add images to data/train/wound and data/train/healthy.")

    train_images = np.array(images, dtype=np.float32)
    train_labels = np.array(labels, dtype=np.int32)

    test_images = []
    test_labels = []
    for label, class_dir in ((1, TEST_DIR / "wound"), (0, TEST_DIR / "healthy")):
        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test_images.append(image)
            test_labels.append(label)

    if len(test_images) < 2:
        raise ValueError("Not enough testing images were found. Add images to data/test/wound and data/test/healthy.")

    return train_images, train_labels, np.array(test_images, dtype=np.float32), np.array(test_labels, dtype=np.int32)


def preprocess_for_model(image):
    resized = cv2.resize(image, (FEATURE_SIZE, FEATURE_SIZE))
    resized = resized.astype(np.float32) / 255.0
    return resized.reshape(1, -1)


def build_tensorflow_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_model():
    train_images, train_labels, test_images, test_labels = load_image_dataset()

    if tf is not None and Sequential is not None:
        try:
            x_train = train_images / 255.0
            x_test = test_images / 255.0
            model = build_tensorflow_model()
            model.fit(x_train, train_labels, epochs=3, batch_size=16, validation_data=(x_test, test_labels), verbose=0)
            model.save(MODEL_PATH)
            return model, "tensorflow"
        except Exception:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

    if LogisticRegression is None or make_pipeline is None or StandardScaler is None:
        raise ImportError("Install scikit-learn to use the fallback model.")

    resized_train = []
    resized_test = []
    for image in train_images:
        resized_train.append(preprocess_for_model(image)[0])
    for image in test_images:
        resized_test.append(preprocess_for_model(image)[0])

    x_train = np.array(resized_train, dtype=np.float32)
    x_test = np.array(resized_test, dtype=np.float32)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(x_train, train_labels)
    predictions = model.predict(x_test)

    print("Fallback model accuracy:", accuracy_score(test_labels, predictions))
    print(classification_report(test_labels, predictions, target_names=["healthy", "wound"]))
    print(confusion_matrix(test_labels, predictions))

    import joblib

    joblib.dump(model, FALLBACK_MODEL_PATH)
    return model, "sklearn"


def load_model():
    if MODEL_PATH.exists() and tf is not None:
        try:
            from tensorflow.keras.models import load_model as keras_load_model

            return keras_load_model(MODEL_PATH), "tensorflow"
        except Exception:
            pass

    if FALLBACK_MODEL_PATH.exists():
        import joblib

        return joblib.load(FALLBACK_MODEL_PATH), "sklearn"

    return train_and_save_model()


def predict_image(image, model=None, model_type=None):
    if model is None or model_type is None:
        model, model_type = load_model()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model_type == "tensorflow":
        resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        prediction = float(model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0])
        label = "Wound" if prediction > 0.5 else "Healthy"
        return label, prediction

    feature_vector = preprocess_for_model(image_rgb)[0]
    probability = float(model.predict_proba(feature_vector.reshape(1, -1))[0][1])
    label = "Wound" if probability > 0.5 else "Healthy"
    return label, probability


if __name__ == "__main__":
    train_and_save_model()
    print("Training completed. Model saved to", MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH)

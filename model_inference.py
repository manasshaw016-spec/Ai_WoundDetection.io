from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from image_processing import detect_redness_mask, dominant_color_name, estimate_area, segment_wound_mask, to_rgb
from utils import DATA_DIR, FALLBACK_MODEL_PATH, MODEL_PATH, TRAIN_DIR, TEST_DIR, ensure_sample_dataset

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover - optional dependency
    tf = None
    Conv2D = Dense = Dropout = Flatten = MaxPooling2D = Sequential = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None

IMAGE_SIZE = 224
FEATURE_SIZE = 64


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


def infer_category(probability, wound_area_ratio, redness_ratio, dominant_color):
    if probability < 0.45:
        return "Healthy"

    if wound_area_ratio > 0.15 and redness_ratio > 0.12:
        return "Pressure Ulcer"
    if dominant_color in {"Red / pink", "Orange / warm"} and redness_ratio > 0.10:
        return "Burn"
    if redness_ratio > 0.14 and dominant_color in {"Purple / blue", "Dark / low contrast"}:
        return "Venous Ulcer"
    if wound_area_ratio < 0.08:
        return "Surgical Wound"
    return "Diabetic Foot Ulcer"


def infer_severity(wound_area_ratio, redness_ratio):
    if wound_area_ratio > 0.12 or redness_ratio > 0.18:
        return "Severe"
    if wound_area_ratio > 0.05 or redness_ratio > 0.09:
        return "Moderate"
    return "Mild"


def infer_risk(wound_area_ratio, redness_ratio, probability):
    if probability < 0.45:
        return "Low"
    if wound_area_ratio > 0.12 or redness_ratio > 0.16:
        return "High"
    if wound_area_ratio > 0.06 or redness_ratio > 0.08:
        return "Medium"
    return "Low"


def analyze_image(image, model=None, model_type=None):
    """Run a full analysis pipeline and return a structured result dictionary."""
    image_rgb = to_rgb(image)
    label, probability = predict_image(image, model=model, model_type=model_type)

    segmentation_mask = segment_wound_mask(image_rgb)
    redness_mask = detect_redness_mask(image_rgb)
    wound_pixels, wound_area_ratio = estimate_area(segmentation_mask)
    redness_pixels, redness_ratio = estimate_area(redness_mask)
    color_name = dominant_color_name(image_rgb, segmentation_mask)
    color_explanation = "Color alone does not confirm a medical condition and should be interpreted with other findings."
    color_explanation = "Color alone does not confirm a medical condition and should be interpreted with other findings."
    try:
        from image_processing import explain_color_meaning

        color_explanation = explain_color_meaning(color_name)
    except Exception:
        pass

    category = infer_category(probability, wound_area_ratio / 100.0, redness_ratio / 100.0, color_name)
    severity = infer_severity(wound_area_ratio / 100.0, redness_ratio / 100.0)
    risk = infer_risk(wound_area_ratio / 100.0, redness_ratio / 100.0, probability)
    confidence = round(min(99.0, max(55.0, 60.0 + (abs(probability - 0.5) * 80.0))), 1)

    if label == "Healthy":
        category = "Healthy"
        severity = "Mild"
        risk = "Low"

    educational_guidance = [
        "Keep the area clean and protect it from additional pressure or friction.",
        "Wash hands before and after touching the area and use clean dressings if appropriate.",
        "Monitor for increasing redness, warmth, swelling, or drainage and seek professional care if these worsen.",
    ]

    medical_evaluation = risk == "High" or severity == "Severe"

    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "category": category,
        "severity": severity,
        "risk": risk,
        "medical_evaluation_advisable": medical_evaluation,
        "wound_area_pixels": wound_pixels,
        "wound_area_percentage": wound_area_ratio,
        "redness_pixels": redness_pixels,
        "redness_ratio": redness_ratio,
        "segmentation_mask": segmentation_mask,
        "redness_mask": redness_mask,
        "dominant_color": color_name,
        "color_explanation": color_explanation,
        "educational_guidance": educational_guidance,
    }


if __name__ == "__main__":
    train_and_save_model()

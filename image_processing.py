from __future__ import annotations

import cv2
import numpy as np


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_bgr(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def segment_wound_mask(image_rgb):
    """Create a simple wound-region mask using HSV-based red-channel filtering."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_red = cv2.inRange(hsv, (0, 40, 40), (20, 255, 255))
    upper_red = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    mask = lower_red | upper_red
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if int(mask.sum() // 255) < 200:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, fallback = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(fallback, cv2.MORPH_OPEN, kernel)

    return mask


def detect_redness_mask(image_rgb):
    """Highlight inflamed or reddish regions in the image."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 40, 60), (20, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask


def estimate_area(mask):
    """Estimate wound area in pixels and as a percentage of the full frame."""
    pixel_count = int(np.count_nonzero(mask))
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = round((pixel_count / total_pixels) * 100, 2) if total_pixels else 0.0
    return pixel_count, percentage


def overlay_segmentation(image_bgr, mask, color=(0, 0, 255), alpha=0.45):
    """Overlay the segmentation mask on the original image."""
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :, 0] = color[0]
    colored[:, :, 1] = color[1]
    colored[:, :, 2] = color[2]
    overlay[mask > 0] = cv2.addWeighted(overlay[mask > 0], 1 - alpha, colored[mask > 0], alpha, 0)
    return overlay


def dominant_color_name(image_rgb, mask=None):
    """Estimate a dominant color label from the segmented region."""
    pixels = image_rgb.reshape(-1, 3)
    if mask is not None:
        pixels = pixels[mask.reshape(-1) > 0]
    if len(pixels) == 0:
        pixels = image_rgb.reshape(-1, 3)

    mean_color = np.mean(pixels, axis=0).astype(int)
    hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_RGB2HSV)[0][0]
    hue, sat, val = hsv

    if sat < 45:
        if val < 90:
            return "Dark / low contrast"
        return "Pale / neutral"
    if hue < 20 or hue > 160:
        return "Red / pink"
    if hue < 40:
        return "Orange / warm"
    if hue < 80:
        return "Yellow / green"
    if hue < 140:
        return "Green / teal"
    return "Purple / blue"


def explain_color_meaning(color_name):
    """Provide educational context for the dominant color."""
    mapping = {
        "Red / pink": "Red or pink tones can suggest inflammation, fresh tissue response, or increased blood flow.",
        "Orange / warm": "Warm tones often align with irritation or tissue response, though they can also appear with certain lighting conditions.",
        "Yellow / green": "Yellow-green tones may indicate old blood, debris, or a range of tissue changes and should be interpreted carefully.",
        "Purple / blue": "Purple or blue hues may reflect poor perfusion or a more chronic appearance in some cases.",
        "Pale / neutral": "Paler tones may suggest low contrast, reduced redness, or a less inflamed appearance.",
        "Dark / low contrast": "Dark or low-contrast regions can make assessment harder and may warrant careful review by a clinician.",
    }
    return mapping.get(color_name, "Color alone does not confirm a medical condition and should be interpreted with other findings.")


def generate_gradcam(model, image_bgr):
    """Create a basic Grad-CAM overlay when a TensorFlow/Keras model is available."""
    try:
        import tensorflow as tf
    except Exception:  # pragma: no cover - optional dependency
        return None

    if model is None or not hasattr(model, "layers"):
        return None

    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        return None

    last_conv_layer = conv_layers[-1]
    grad_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])
    image_rgb = to_rgb(image_bgr).astype(np.float32) / 255.0
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_rgb, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_tensor)
        class_idx = int(np.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0.0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

    heatmap = cv2.resize(heatmap.numpy(), (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.65, heatmap_colored, 0.35, 0)
    return overlay

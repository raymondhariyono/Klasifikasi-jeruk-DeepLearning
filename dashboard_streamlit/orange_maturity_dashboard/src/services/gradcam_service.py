import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.config.settings import IMAGE_SIZE


def find_mobilenet_layer(model):
    for layer in model.layers:
        if "MobileNetV3" in layer.name:
            return layer

    raise ValueError("Layer MobileNetV3 tidak ditemukan pada model.")


def find_last_4d_layer(base_model):
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("Layer 4D untuk Grad-CAM tidak ditemukan.")


def call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap(model, image_array, explain_class: str = "auto"):
    """
    explain_class:
    - "auto"         : menjelaskan kelas hasil prediksi
    - "matang"       : menjelaskan area yang mendorong prediksi matang
    - "belum_matang" : menjelaskan area yang mendorong prediksi belum matang
    """
    base_model = find_mobilenet_layer(model)
    last_conv_layer_name = find_last_4d_layer(base_model)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    base_grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(image_array, dtype=tf.float32)
        conv_outputs = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            if layer is base_model:
                conv_outputs, x = base_grad_model(x)
            else:
                x = call_layer(layer, x)

        predictions = x
        probability_matang = predictions[:, 0]

        if explain_class == "matang":
            class_score = probability_matang
        elif explain_class == "belum_matang":
            class_score = 1.0 - probability_matang
        else:
            class_score = tf.where(
                probability_matang >= 0.5,
                probability_matang,
                1.0 - probability_matang
            )

    if conv_outputs is None:
        raise ValueError(
            "Grad-CAM gagal dibuat karena output convolution layer tidak ditemukan."
        )

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError(
            "Grad-CAM gagal dibuat karena gradient tidak dapat dihitung."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy().astype(np.float32)


def create_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    image = ImageOps.exif_transpose(image)
    image_rgb = image.convert("RGB").resize(IMAGE_SIZE)
    image_np = np.asarray(image_rgb, dtype=np.uint8)

    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    heatmap_resized = cv2.resize(heatmap_array, IMAGE_SIZE)

    heatmap_uint8 = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
    heatmap_uint8 = np.ascontiguousarray(heatmap_uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # type: ignore[arg-type]
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_np,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return image_np, heatmap_resized, overlay
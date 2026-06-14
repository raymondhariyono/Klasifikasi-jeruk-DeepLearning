from PIL import Image, ImageOps
import numpy as np

from src.config.settings import IMAGE_SIZE


def normalize_image_orientation(image: Image.Image) -> Image.Image:
    """
    Menyamakan orientasi gambar berdasarkan metadata EXIF.
    Ini penting agar input model, tampilan gambar, dan Grad-CAM memiliki orientasi yang sama.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocessing gambar untuk MobileNetV3Large.
    Model memakai include_preprocessing=True, sehingga nilai pixel tetap 0-255.
    """
    image = normalize_image_orientation(image)
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def prepare_display_image(image: Image.Image) -> Image.Image:
    image = normalize_image_orientation(image)
    image = image.resize(IMAGE_SIZE)

    return image
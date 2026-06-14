from PIL import Image

from src.config.settings import (
    THRESHOLD_BELUM_MATANG,
    THRESHOLD_TRANSISI_UPPER,
    THRESHOLD_MATANG_SEMPURNA,
)
from src.utils.image_utils import preprocess_image


def interpret_prediction(probability_matang: float) -> dict:
    probability_belum_matang = 1.0 - probability_matang

    if probability_matang < THRESHOLD_BELUM_MATANG:
        interpreted_class = "Belum Matang"
        confidence_status = "Yakin"
        final_label = "belum_matang"
        confidence = probability_belum_matang

    elif probability_matang < THRESHOLD_TRANSISI_UPPER:
        interpreted_class = "Transisi / Tidak Yakin"
        confidence_status = "Ragu"
        final_label = "transisi"
        confidence = max(probability_matang, probability_belum_matang)

    elif probability_matang < THRESHOLD_MATANG_SEMPURNA:
        interpreted_class = "Matang Sedikit Kehijauan"
        confidence_status = "Cukup Yakin"
        final_label = "matang"
        confidence = probability_matang

    else:
        interpreted_class = "Matang Sempurna"
        confidence_status = "Yakin"
        final_label = "matang"
        confidence = probability_matang

    binary_class = "Matang" if probability_matang >= 0.5 else "Belum Matang"

    return {
        "binary_class": binary_class,
        "interpreted_class": interpreted_class,
        "confidence_status": confidence_status,
        "final_label": final_label,
        "probability_matang": probability_matang,
        "probability_belum_matang": probability_belum_matang,
        "confidence": confidence,
    }


def predict_image(model, image: Image.Image) -> dict:
    input_array = preprocess_image(image)

    probability_matang = float(model.predict(input_array, verbose=0)[0][0])

    result = interpret_prediction(probability_matang)
    result["input_array"] = input_array

    return result
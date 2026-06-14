from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "mobilenetv3_orange_maturity.keras"

CLASS_NAMES = ["belum_matang", "matang"]

DISPLAY_CLASS_NAMES = {
    "belum_matang": "Belum Matang",
    "matang": "Matang",
    "transisi": "Transisi / Tidak Yakin",
}

IMAGE_SIZE = (224, 224)

# Interpretasi tambahan berdasarkan probabilitas matang
THRESHOLD_BELUM_MATANG = 0.30
THRESHOLD_TRANSISI_UPPER = 0.70
THRESHOLD_MATANG_SEMPURNA = 0.90
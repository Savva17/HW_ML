from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent


MODEL_PATH = BASE_DIR / "mushroom_model.pkl"
BASE_URL = "http://127.0.0.1:8002"


# колонки, на которых обучали модель
FEATURE_COLUMNS = [
    "id",
    "cap-diameter",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-height",
    "stem-width",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season",
]
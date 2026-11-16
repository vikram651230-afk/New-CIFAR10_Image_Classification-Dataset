import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get("SECRET_KEY", "cifar10-pure-ann-secret-key-2026")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_KERAS_PATH = os.path.join(MODELS_DIR, "image_classification.keras")
    MODEL_H5_PATH = os.path.join(MODELS_DIR, "image_classification.h5")
    METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    CLASS_EMOJIS = {
        "airplane": "✈️",
        "automobile": "🚗",
        "bird": "🐦",
        "cat": "🐱",
        "deer": "🦌",
        "dog": "🐶",
        "frog": "🐸",
        "horse": "🐴",
        "ship": "🚢",
        "truck": "🚛",
    }

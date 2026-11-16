import os
import json
import threading
import tensorflow as tf
from keras.models import load_model
from app.config import Config

_model = None
_model_lock = threading.Lock()
_metrics_cache = None

def get_model():
    """
    Thread-safe singleton model loader.
    Loads the trained model once and keeps it in memory across requests.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        candidate_paths = [
            Config.MODEL_KERAS_PATH,
            Config.MODEL_H5_PATH,
            os.path.join(Config.BASE_DIR, "image_classification.h5"),
            os.path.join(Config.BASE_DIR, "image_classification.keras"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    _model = load_model(path, compile=False)
                    print(f"[ModelService] Loaded model from: {path}")
                    break
                except Exception as e:
                    print(f"[ModelService] Warning: Failed to load from {path}: {e}")

        return _model

def is_model_loaded():
    """Check if model is currently loaded in memory or available on disk."""
    global _model
    if _model is not None:
        return True
    return get_model() is not None

def get_metrics():
    """Retrieve evaluation metrics from models/metrics.json."""
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics_cache = json.load(f)
                return _metrics_cache
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {
        "status": "not_trained",
        "message": "Model not trained yet or metrics not exported.",
    }

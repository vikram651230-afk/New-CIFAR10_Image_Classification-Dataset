import os
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    send_from_directory,
    current_app,
)
from app.config import Config
from app.utils.validators import allowed_file, sanitize_filename
from app.services.model_service import is_model_loaded, get_metrics
from app.services.prediction_service import predict_image

main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------
# Web Interface Routes
# ---------------------------------------------------------

@main_bp.route("/", methods=["GET"])
def index():
    """Render the main CIFAR-10 image classification web interface."""
    model_ready = is_model_loaded()
    metrics = get_metrics()
    return render_template(
        "index.html",
        model_ready=model_ready,
        metrics=metrics,
        classes=Config.CLASS_NAMES,
        emojis=Config.CLASS_EMOJIS,
    )


# ---------------------------------------------------------
# REST API Endpoints (Phase 4 of readme.md)
# ---------------------------------------------------------

@main_bp.route("/api/health", methods=["GET"])
def health_check():
    """
    1. GET /api/health
    Return application status and model loading status.
    """
    model_ready = is_model_loaded()
    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "model_architecture": "Pure-ANN (Flatten + 3 Dense Relu + Dense Softmax)",
        "convolutional_layers": 0,
        "supported_classes": Config.CLASS_NAMES,
        "version": "1.0.0",
    }), 200


@main_bp.route("/api/classes", methods=["GET"])
def get_classes():
    """
    3. GET /api/classes
    Return the supported CIFAR-10 class labels.
    """
    classes_with_emojis = [
        {"index": idx, "name": name, "emoji": Config.CLASS_EMOJIS.get(name, "🖼️")}
        for idx, name in enumerate(Config.CLASS_NAMES)
    ]
    return jsonify({
        "total_classes": len(Config.CLASS_NAMES),
        "classes": Config.CLASS_NAMES,
        "detailed": classes_with_emojis,
    }), 200


@main_bp.route("/api/metrics", methods=["GET"])
def get_model_metrics():
    """Return model evaluation metrics, accuracy, and confusion matrix data."""
    metrics = get_metrics()
    return jsonify(metrics), 200


@main_bp.route("/api/predict", methods=["POST"])
def predict():
    """
    2. POST /api/predict
    Accept an uploaded image.
    Return:
    - Predicted class
    - Confidence percentage
    - Class probabilities
    - Prediction status
    - Processing information
    """
    if not is_model_loaded():
        return jsonify({
            "success": False,
            "error": "Model not loaded",
            "message": "The CIFAR-10 classification model is not loaded. Please train or provide model weights.",
        }), 503

    # Check for image file
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing image file",
            "message": "No 'image' key found in multipart/form-data request.",
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename",
            "message": "No file was selected for upload.",
        }), 400

    if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
        allowed = ", ".join(sorted(Config.ALLOWED_EXTENSIONS))
        return jsonify({
            "success": False,
            "error": "Invalid file format",
            "message": f"Unsupported file extension. Allowed formats: {allowed}",
        }), 400

    safe_name = sanitize_filename(file.filename)
    image_bytes = file.read()

    if len(image_bytes) == 0:
        return jsonify({
            "success": False,
            "error": "Empty file",
            "message": "Uploaded file is 0 bytes.",
        }), 400

    try:
        result = predict_image(image_bytes, filename=safe_name)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Prediction error",
            "message": str(e),
        }), 500


@main_bp.route("/api/sample/<class_name>", methods=["GET"])
def get_sample_image(class_name):
    """Serve real CIFAR-10 sample test images for quick UI testing."""
    samples_dir = os.path.join(Config.BASE_DIR, "app", "static", "samples")
    filename = f"{class_name.lower()}.png"
    if os.path.exists(os.path.join(samples_dir, filename)):
        return send_from_directory(samples_dir, filename)
    return jsonify({"error": "Sample image not found"}), 404


@main_bp.route("/api/artifacts/<filename>", methods=["GET"])
def get_artifact(filename):
    """Serve confusion matrix and training curves."""
    models_dir = Config.MODELS_DIR
    safe_name = sanitize_filename(filename)
    return send_from_directory(models_dir, safe_name)

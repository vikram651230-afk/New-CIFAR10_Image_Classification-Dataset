import time
import numpy as np
from app.config import Config
from app.services.model_service import get_model
from app.services.image_service import preprocess_image_for_model, create_data_uri

def predict_image(image_bytes_or_stream, filename="image.jpg"):
    """
    Execute full prediction pipeline matching Notebook Cell 48:
    1. Preprocess image: convert to RGB, resize to 32x32, normalize / 255.0, expand_dims
    2. Model forward pass: model.predict(image)
    3. Determine predicted index, class label, and confidence percentage
    4. Return structured response with full probabilities and top-3 candidates
    """
    model = get_model()
    if model is None:
        raise RuntimeError("Model is not loaded or not found in models/ directory.")

    start_time = time.time()
    input_tensor, pil_img = preprocess_image_for_model(image_bytes_or_stream)

    # Forward pass: shape (1, 32, 32, 3) -> shape (1, 10)
    predictions = model.predict(input_tensor, verbose=0)
    probabilities = predictions[0]

    # Matching Notebook Cell 48:
    # index = np.argmax(prediction)
    # confidence = np.max(prediction) * 100
    pred_index = int(np.argmax(probabilities))
    confidence_val = float(probabilities[pred_index])
    confidence_percent = round(confidence_val * 100.0, 2)
    predicted_class = Config.CLASS_NAMES[pred_index]
    emoji = Config.CLASS_EMOJIS.get(predicted_class, "🖼️")

    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    # All 10 class probabilities
    all_probabilities = {
        name: round(float(prob), 4)
        for name, prob in zip(Config.CLASS_NAMES, probabilities)
    }

    # Top 3 candidates
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = [
        {
            "class": Config.CLASS_NAMES[i],
            "emoji": Config.CLASS_EMOJIS.get(Config.CLASS_NAMES[i], "🖼️"),
            "probability": round(float(probabilities[i]), 4),
            "percentage": round(float(probabilities[i]) * 100.0, 2),
        }
        for i in top_indices
    ]

    # Generate preview data URI (255x255 display)
    preview_data_uri = create_data_uri(pil_img)

    return {
        "success": True,
        "filename": filename,
        "predicted_class": predicted_class,
        "predicted_index": pred_index,
        "emoji": emoji,
        "confidence_percentage": confidence_percent,
        "confidence_score": round(confidence_val, 4),
        "class_probabilities": all_probabilities,
        "top_3": top_3,
        "latency_ms": latency_ms,
        "preview_data_uri": preview_data_uri,
        "model_architecture": "Pure-ANN (No Convolutions)",
        "input_dimensions": "32x32x3",
    }

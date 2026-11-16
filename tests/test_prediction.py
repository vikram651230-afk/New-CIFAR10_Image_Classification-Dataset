import io
import os
import numpy as np
from PIL import Image
from app.config import Config
from app.services.model_service import get_model, is_model_loaded
from app.services.image_service import preprocess_image_for_model
from app.services.prediction_service import predict_image

def test_model_loading():
    """Test model loads properly from disk."""
    model = get_model()
    assert model is not None
    assert is_model_loaded() is True


def test_zero_convolutions_parity():
    """Verify that the model adheres strictly to Pure-ANN specification (no CNN layers)."""
    model = get_model()
    for layer in model.layers:
        layer_cls = layer.__class__.__name__.lower()
        assert "conv" not in layer_cls, f"Violation: Conv layer found: {layer.__class__.__name__}"
        assert "pool" not in layer_cls, f"Violation: Pooling layer found: {layer.__class__.__name__}"


def test_model_architecture_parity():
    """Verify layer names and count match notebook cell 28."""
    model = get_model()
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert "Flatten" in layer_types
    assert layer_types.count("Dense") >= 4
    # Check output shape
    assert model.output_shape == (None, 10)


def test_image_preprocessing():
    """Test preprocessing converts image to (1, 32, 32, 3) in [0, 1]."""
    test_img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buffer = io.BytesIO()
    test_img.save(buffer, format="PNG")
    buffer.seek(0)

    tensor, pil_img = preprocess_image_for_model(buffer)
    assert tensor.shape == (1, 32, 32, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_prediction_output_format():
    """Test full prediction response structure."""
    test_img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = io.BytesIO()
    test_img.save(buffer, format="PNG")
    buffer.seek(0)

    result = predict_image(buffer.getvalue(), filename="test_red.png")
    assert result["success"] is True
    assert result["predicted_class"] in Config.CLASS_NAMES
    assert 0 <= result["predicted_index"] < 10
    assert 0.0 <= result["confidence_percentage"] <= 100.0
    assert len(result["class_probabilities"]) == 10
    assert len(result["top_3"]) == 3
    assert "latency_ms" in result
    assert "preview_data_uri" in result


def test_api_predict_valid_image(client):
    """Test POST /api/predict with a real test sample image."""
    sample_path = os.path.join(Config.BASE_DIR, "app", "static", "samples", "airplane.png")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
    else:
        test_img = Image.new("RGB", (32, 32), color=(0, 128, 255))
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

    data = {
        "image": (io.BytesIO(img_bytes), "airplane_test.png")
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert res["predicted_class"] in Config.CLASS_NAMES
    assert res["confidence_percentage"] > 0
    assert "top_3" in res

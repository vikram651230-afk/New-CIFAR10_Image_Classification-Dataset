import io
from app.config import Config

def test_app_initialization(app):
    """Test Flask application initializes with correct configuration."""
    assert app is not None
    assert app.config["TESTING"] is True
    assert len(Config.CLASS_NAMES) == 10


def test_index_route(client):
    """Test GET / serves the HTML user interface."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "CIFAR-10" in html
    assert "Pure-ANN" in html


def test_api_health_endpoint(client):
    """Test GET /api/health returns application and model status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["convolutional_layers"] == 0
    assert len(data["supported_classes"]) == 10


def test_api_classes_endpoint(client):
    """Test GET /api/classes returns the supported CIFAR-10 class labels."""
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_classes"] == 10
    assert "airplane" in data["classes"]
    assert "cat" in data["classes"]
    assert "truck" in data["classes"]


def test_api_metrics_endpoint(client):
    """Test GET /api/metrics returns valid metrics data."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data


def test_predict_missing_image(client):
    """Test POST /api/predict handles missing image payload."""
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing image" in data["error"]


def test_predict_invalid_extension(client):
    """Test POST /api/predict rejects unsupported file types."""
    data = {
        "image": (io.BytesIO(b"fake txt content"), "test.txt")
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 400
    result = response.get_json()
    assert result["success"] is False
    assert "Invalid file format" in result["error"]


def test_predict_empty_file(client):
    """Test POST /api/predict rejects empty 0-byte files."""
    data = {
        "image": (io.BytesIO(b""), "empty.png")
    }
    response = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 400
    result = response.get_json()
    assert result["success"] is False


def test_not_found_handler(client):
    """Test 404 error handler returns clean JSON."""
    response = client.get("/api/nonexistent_endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False
    assert data["error"] == "Not Found"

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def create_test_image_bytes():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data

def test_models_status_endpoint():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "remoteclip" in data

def test_analyze_valid_image_endpoint():
    img_bytes = create_test_image_bytes()
    fake_image = ("test_sat.png", img_bytes, "image/png")
    response = client.post(
        "/api/analyze",
        data={"query": "What type of land cover is visible?"},
        files={"image1": fake_image}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "land-cover" in data["summary"].lower() or "vegetation" in data["answer"].lower()
    assert data["confidence"]["score"] >= 0.70
    assert len(data["execution_steps"]) > 0

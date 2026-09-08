import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def create_dummy_image_bytes(color=(100, 150, 200)):
    img = Image.new("RGB", (128, 128), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_e2e_single_image_analysis():
    img_bytes = create_dummy_image_bytes((30, 180, 50))
    response = client.post(
        "/api/analyze",
        data={"query": "What type of land cover is visible?"},
        files={"image1": ("sat1.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "cropland" in data["answer"].lower() or "vegetation" in data["answer"].lower() or "land" in data["answer"].lower()
    assert len(data["evidence"]) >= 1
    assert data["evidence"][0]["type"] == "heatmap"

def test_e2e_two_image_change_analysis():
    img1_bytes = create_dummy_image_bytes((100, 100, 100))
    img2_bytes = create_dummy_image_bytes((200, 200, 200))

    response = client.post(
        "/api/analyze",
        data={"query": "What changed between these two dates?"},
        files={
            "image1": ("before.png", img1_bytes, "image/png"),
            "image2": ("after.png", img2_bytes, "image/png")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["intent"] == "CHANGE_DETECTION"
    assert "change" in data["answer"].lower() or "surface" in data["answer"].lower()
    assert len(data["evidence"]) >= 1
    assert data["evidence"][0]["type"] == "change_mask"

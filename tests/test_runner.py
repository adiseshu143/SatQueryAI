import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

def create_test_image_bytes():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

class TestSatQueryAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("app_name", data)

    def test_models_status_endpoint(self):
        response = self.client.get("/api/models/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("remoteclip", data)

    def test_analyze_endpoint(self):
        img_bytes = create_test_image_bytes()
        fake_image = ("test_sat.png", img_bytes, "image/png")
        response = self.client.post(
            "/api/analyze",
            data={"query": "What type of land cover is visible?"},
            files={"image1": fake_image}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

if __name__ == "__main__":
    unittest.main()

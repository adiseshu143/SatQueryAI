import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.services.ai_service import ai_service
from backend.utils.exceptions import InvalidImageError

client = TestClient(app)

def create_test_image_bytes(format_str: str = "PNG", size: tuple = (100, 100)) -> bytes:
    img = Image.fromarray(np.random.randint(40, 200, (size[1], size[0], 3), dtype=np.uint8))
    bio = io.BytesIO()
    img.save(bio, format=format_str)
    return bio.getvalue()

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

def test_feature1_single_image_detection_api():
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/single",
        data={"query": "Describe this satellite image and classify main land cover."},
        files={"image": ("test_optical.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "IMAGE DETECTION"
    assert data["status"] == "success"
    assert "query" in data
    assert "answer" in data
    assert "confidence" in data
    assert len(data["evidence"]) > 0

def test_feature1_building_counting_query():
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/single",
        data={"query": "How many buildings are there in this satellite image?"},
        files={"image": ("test_optical.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "IMAGE DETECTION"
    assert "building_count" in data["statistics"]
    assert "building structure(s) were detected" in data["answer"]

def test_feature1_greenery_percentage_query():
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/single",
        data={"query": "What is the percentage of greenery in it?"},
        files={"image": ("test_optical.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "IMAGE DETECTION"
    assert data["statistics"]["intent"] == "VEGETATION_ANALYSIS"
    assert "vegetation_percentage" in data["statistics"]
    assert "Estimated greenery coverage:" in data["answer"]
    assert "building_count" not in data["statistics"]

def test_feature1_water_body_query():
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/single",
        data={"query": "Are there water bodies in this image?"},
        files={"image": ("test_optical.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "IMAGE DETECTION"
    assert data["statistics"]["intent"] == "WATER_BODY_ANALYSIS"
    assert "Water Body Analysis:" in data["answer"]
    assert "building_count" not in data["statistics"]

def test_feature2_change_analysis_api():
    img_a = create_test_image_bytes("PNG")
    img_b = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/change",
        data={
            "query": "What building construction or land cover conversion occurred between these dates?",
            "date_a": "2022-01-15",
            "date_b": "2025-04-20",
            "sensor_a": "Sentinel-2",
            "sensor_b": "Sentinel-2"
        },
        files={
            "image_a": ("before.png", img_a, "image/png"),
            "image_b": ("after.png", img_b, "image/png")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "CHANGE ANALYSIS"
    assert data["status"] == "success"
    assert "statistics" in data
    assert data["statistics"]["date_before"] == "2022-01-15"
    assert data["statistics"]["date_after"] == "2025-04-20"
    assert "bitemporal_ssim_similarity" in data["statistics"]
    assert "before_vegetation_pct" in data["statistics"]
    assert "after_vegetation_pct" in data["statistics"]
    assert "vegetation_net_change_pct" in data["statistics"]
    assert "before_building_coverage_pct" in data["statistics"]
    assert "after_building_coverage_pct" in data["statistics"]
    assert "building_net_change_pct" in data["statistics"]

def test_feature2_change_analysis_invalid_temporal_order():
    img_a = create_test_image_bytes("PNG")
    img_b = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/change",
        data={
            "query": "Check change analysis with invalid temporal order",
            "date_a": "2025-06-15",
            "date_b": "2022-05-10"
        },
        files={
            "image_a": ("after.png", img_a, "image/png"),
            "image_b": ("before.png", img_b, "image/png")
        }
    )
    assert response.status_code == 400
    data = response.json()
    assert "Before image must have an earlier acquisition date than After image" in data["message"]

def test_feature2_cross_sensor_warning():
    img_a = create_test_image_bytes("PNG")
    img_b = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze/change",
        data={
            "query": "Check change detection across different sensors",
            "date_a": "2022-01-01",
            "date_b": "2025-01-01",
            "sensor_a": "Sentinel-2",
            "sensor_b": "Landsat-9"
        },
        files={
            "image_a": ("before.png", img_a, "image/png"),
            "image_b": ("after.png", img_b, "image/png")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data.get("warnings", [])) > 0
    assert any("Different sensors detected" in w for w in data["warnings"])

def test_feature3_optical_sar_analysis_api():
    optical_bytes = create_test_image_bytes("PNG")
    sar_bytes = create_test_image_bytes("TIFF")
    response = client.post(
        "/api/analyze/optical-sar",
        data={"query": "Analyze optical multispectral and Sentinel-1 SAR together for BigEarthNet land cover."},
        files={
            "optical_image": ("optical.png", optical_bytes, "image/png"),
            "sar_image": ("sar.tif", sar_bytes, "image/tiff")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "OPTICAL + SAR ANALYSIS"
    assert data["status"] == "success"
    assert "bigearthnet_multi_labels" in data["statistics"]

def test_feature4_agent_route_api():
    # Priority 1: User explicit mode selection
    res1 = client.post("/api/agent/route", data={"query": "Anything", "image_count": 1, "mode_selected": "CHANGE ANALYSIS"})
    assert res1.json()["task"] == "CHANGE ANALYSIS"

    # Priority 2: One image provided
    res2 = client.post("/api/agent/route", data={"query": "Describe this image", "image_count": 1})
    assert res2.json()["task"] == "IMAGE DETECTION"

    # Priority 3: Optical + SAR keywords
    res3 = client.post("/api/agent/route", data={"query": "Compare Sentinel-1 SAR radar backscatter", "image_count": 2})
    assert res3.json()["task"] == "OPTICAL + SAR ANALYSIS"

    # Priority 4: Temporal change query
    res4 = client.post("/api/agent/route", data={"query": "What changed between T1 and T2?", "image_count": 2})
    assert res4.json()["task"] == "CHANGE ANALYSIS"

    # Priority 5: Ambiguous query
    res5 = client.post("/api/agent/route", data={"query": "Overview comparison", "image_count": 2})
    assert res5.json()["task"] == "AMBIGUOUS"

def test_legacy_analyze_api_preservation():
    img1 = create_test_image_bytes("PNG")
    response = client.post(
        "/api/analyze",
        data={"query": "General satellite land cover classification."},
        files={"image1": ("satellite.png", img1, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "answer" in data

def test_negative_invalid_image_validation():
    # Unsupported extension
    response = client.post(
        "/api/analyze/single",
        data={"query": "Describe image"},
        files={"image": ("invalid.pdf", b"%PDF-1.4...", "application/pdf")}
    )
    assert response.status_code == 400

    # Empty query
    response_empty = client.post(
        "/api/analyze/single",
        data={"query": "   "},
        files={"image": ("valid.png", create_test_image_bytes("PNG"), "image/png")}
    )
    assert response_empty.status_code == 400

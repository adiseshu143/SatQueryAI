import pytest
import numpy as np

from backend.preprocessing.dataset_loader import bigearthnet_loader
from backend.pipelines.sar_optical_fusion import sar_optical_pipeline, BIGEARTHNET_19_CLASSES

def test_bigearthnet_loader_stream():
    samples = list(bigearthnet_loader.stream_dataset(split="train", max_samples=3))
    assert len(samples) == 3
    for s in samples:
        assert "optical_image" in s
        assert "sar_image" in s
        assert "corine_labels" in s
        assert "patch_name" in s
        assert "https://txt.bigearth.net/" in s["source"]

def test_sar_optical_fusion_pipeline():
    opt_img = np.random.randint(40, 200, (120, 120, 3), dtype=np.uint8)
    sar_img = np.random.randint(20, 180, (120, 120, 1), dtype=np.uint8)
    query = "Analyze Sentinel-1 SAR and Sentinel-2 optical multi-label land cover (https://txt.bigearth.net/)"

    answer, summary, confidence, stats, multi_labels = sar_optical_pipeline.analyze_multimodal_pair(
        opt_img, sar_img, query
    )

    assert "BigEarthNet-MM" in answer
    assert "https://txt.bigearth.net/" in answer
    assert confidence >= 0.85
    assert len(multi_labels) > 0
    assert stats["dataset_origin"] == "BigEarthNet-MM (https://txt.bigearth.net/)"
    for label in multi_labels:
        assert label in BIGEARTHNET_19_CLASSES

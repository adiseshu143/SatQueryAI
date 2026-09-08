import numpy as np
from backend.pipelines.sar_optical_fusion import sar_optical_pipeline

def test_sar_optical_fusion_pipeline():
    opt = np.full((128, 128, 3), 120, dtype=np.uint8)
    opt[:, :, 1] = 200  # High greenness
    sar = np.random.randint(50, 220, (128, 128, 3), dtype=np.uint8)

    answer, summary, conf, stats, labels = sar_optical_pipeline.analyze_multimodal_pair(
        opt, sar, "Analyze Sentinel-1 SAR and Sentinel-2 optical together"
    )

    assert "Sentinel-1 SAR" in answer
    assert "Broad-leaved forest" in labels or "Pastures" in labels
    assert stats["radar_roughness_index"] >= 0
    assert conf >= 0.85

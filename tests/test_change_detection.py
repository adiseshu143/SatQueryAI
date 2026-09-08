import numpy as np
from backend.pipelines.change_detection import change_detection_pipeline

def test_change_detection():
    # Create two synthetic 100x100 RGB images (img2 has a changed rectangle)
    img1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    img2 = img1.copy()
    img2[20:50, 20:50] = 255  # Changed 30x30 region = 900 pixels out of 10000 (9%)

    stats, evidence = change_detection_pipeline.detect_changes(img1, img2)
    
    assert stats["total_pixels"] == 10000
    assert stats["changed_pixels"] > 0
    assert stats["change_percentage"] > 5.0
    assert evidence.type == "change_mask"
    assert "/outputs/masks/" in evidence.url

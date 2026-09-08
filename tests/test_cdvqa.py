import numpy as np
from backend.pipelines.cdvqa import cdvqa_pipeline

def test_cdvqa_pipeline():
    img1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    img2 = img1.copy()
    img2[20:50, 20:50] = 255
    cd_stats = {"change_percentage": 9.0}

    answer, summary, conf, stats, category = cdvqa_pipeline.answer_cdvqa(
        img1, img2, "Where has new building construction appeared?", cd_stats
    )

    assert "building construction" in answer.lower()
    assert category == "Building Construction & Urban Expansion"
    assert "CDVQA" in stats["dataset_origin"]
    assert conf >= 0.75

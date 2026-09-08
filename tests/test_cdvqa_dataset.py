import pytest
import numpy as np

from backend.preprocessing.dataset_loader import cdvqa_loader
from backend.pipelines.cdvqa import cdvqa_pipeline, CDVQA_CHANGE_TYPES

def test_cdvqa_loader_stream():
    samples = list(cdvqa_loader.stream_dataset(split="val", max_samples=3))
    assert len(samples) == 3
    for s in samples:
        assert "image_t1" in s
        assert "image_t2" in s
        assert "question" in s
        assert "ground_truth_category" in s
        assert "https://github.com/YZHJessica/CDVQA" in s["source"]

def test_cdvqa_pipeline_reasoning():
    t1_arr = np.random.randint(40, 200, (120, 120, 3), dtype=np.uint8)
    t2_arr = t1_arr.copy()
    t2_arr[30:80, 30:80, :] = np.random.randint(150, 255, (50, 50, 3), dtype=np.uint8)

    query = "What building construction or demolition occurred between T1 and T2?"
    cd_stats = {"change_percentage": 14.5, "change_pixels": 2500}

    answer, summary, confidence, stats, category = cdvqa_pipeline.answer_cdvqa(
        t1_arr, t2_arr, query, cd_stats
    )

    assert "CDVQA" in stats["dataset_origin"]
    assert confidence >= 0.75
    assert category in CDVQA_CHANGE_TYPES
    assert "bitemporal_ssim_similarity" in stats
    assert stats["change_magnitude_level"] == "Moderate Change (5-20%)"

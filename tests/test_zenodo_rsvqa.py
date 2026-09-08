import pytest
import numpy as np

from backend.preprocessing.dataset_loader import zenodo_rsvqa_loader
from backend.pipelines.vqa import vqa_pipeline

def test_zenodo_rsvqa_loader_stream():
    samples = list(zenodo_rsvqa_loader.stream_dataset(split="test", max_samples=3))
    assert len(samples) == 3
    for s in samples:
        assert "image" in s
        assert "question" in s
        assert "answer" in s
        assert "q_type" in s
        assert "https://doi.org/10.5281/zenodo.6344366" in s["source"]

def test_zenodo_rsvqa_count_query():
    img_arr = np.random.randint(30, 220, (120, 120, 3), dtype=np.uint8)
    query = "How many residential buildings or surface structures are present in this image?"

    answer, summary, confidence, stats = vqa_pipeline.run(img_arr, query)

    assert "Zenodo RS VQA" in stats["dataset_origin"]
    assert stats["rsvqa_question_type"] == "Count Query"
    assert "estimated_object_count" in stats
    assert confidence >= 0.85

def test_zenodo_rsvqa_comparison_query():
    img_arr = np.random.randint(30, 220, (120, 120, 3), dtype=np.uint8)
    query = "Which land cover class covers more area: vegetation or urban built-up?"

    answer, summary, confidence, stats = vqa_pipeline.run(img_arr, query)

    assert "Zenodo RS VQA" in stats["dataset_origin"]
    assert stats["rsvqa_question_type"] == "Comparison Query"
    assert confidence >= 0.85

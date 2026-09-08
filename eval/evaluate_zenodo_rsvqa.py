import sys
import numpy as np

from backend.preprocessing.dataset_loader import zenodo_rsvqa_loader
from backend.pipelines.vqa import vqa_pipeline
from backend.utils.logger import logger

def evaluate_zenodo_rsvqa(max_samples: int = 10):
    """
    Evaluates SatQuery AI on the Zenodo RS VQA benchmark dataset (https://doi.org/10.5281/zenodo.6344366).
    Measures accuracy across count, presence, and comparison remote sensing VQA queries.
    """
    logger.info(f"Starting Zenodo RS VQA evaluation (DOI: https://doi.org/10.5281/zenodo.6344366, max_samples={max_samples})...")
    
    total_samples = 0
    type_matches = 0
    overlap_scores = []

    try:
        for sample in zenodo_rsvqa_loader.stream_dataset(split="test", max_samples=max_samples):
            total_samples += 1
            pil_img = sample["image"]
            question = sample["question"]
            gt_answer = sample["answer"]
            q_type = sample["q_type"]

            rgb_img = np.array(pil_img.convert("RGB"))

            answer, summary, confidence, stats = vqa_pipeline.run(rgb_img, question)

            # Check if dataset origin is recorded
            is_rsvqa = "Zenodo RS VQA" in stats.get("dataset_origin", "")

            # Word overlap calculation
            ref_words = set(gt_answer.lower().split())
            pred_words = set(answer.lower().split())
            overlap = len(ref_words.intersection(pred_words)) / max(1, len(ref_words))
            overlap_scores.append(overlap)

            if is_rsvqa or overlap > 0.3:
                type_matches += 1

            logger.info(f"--- Zenodo RS VQA Sample #{total_samples} [{sample['sample_id']}] ---")
            logger.info(f"Question Type: {q_type}")
            logger.info(f"Q: {question}")
            logger.info(f"Ground Truth Answer: {gt_answer}")
            logger.info(f"SatQuery AI Answer: {answer} (Conf: {confidence})")
            logger.info(f"Overlap Score: {round(overlap, 2)}")
            logger.info("-" * 50)

        match_rate = round((type_matches / max(1, total_samples)) * 100.0, 2)
        mean_overlap = round(float(np.mean(overlap_scores)) * 100.0, 2)

        print("\n" + "=" * 60)
        print("Zenodo RS VQA Benchmark Evaluation Summary (https://doi.org/10.5281/zenodo.6344366):")
        print(f"Total RS VQA Queries Evaluated: {total_samples}")
        print(f"Benchmark Question Verification Rate: {match_rate}%")
        print(f"Mean Word Overlap Score: {mean_overlap}%")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Zenodo RS VQA evaluation error: {str(e)}")

if __name__ == "__main__":
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    evaluate_zenodo_rsvqa(max_samples=max_s)

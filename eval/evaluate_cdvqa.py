import sys
import numpy as np

from backend.preprocessing.dataset_loader import cdvqa_loader
from backend.pipelines.cdvqa import cdvqa_pipeline
from backend.pipelines.change_detection import change_detection_pipeline
from backend.utils.logger import logger

def evaluate_cdvqa(max_samples: int = 10):
    """
    Evaluates SatQuery AI's CDVQA pipeline on the CDVQA benchmark dataset (https://github.com/YZHJessica/CDVQA).
    Measures category accuracy and answer overlap rate for bi-temporal visual question answering.
    """
    logger.info(f"Starting CDVQA evaluation (Source: https://github.com/YZHJessica/CDVQA, max_samples={max_samples})...")
    
    total_samples = 0
    correct_category_matches = 0
    word_overlap_scores = []

    try:
        for sample in cdvqa_loader.stream_dataset(split="val", max_samples=max_samples):
            total_samples += 1
            t1_pil = sample["image_t1"]
            t2_pil = sample["image_t2"]
            question = sample["question"]
            gt_answer = sample["answer"]
            gt_category = sample["ground_truth_category"]

            t1_arr = np.array(t1_pil.convert("RGB"))
            t2_arr = np.array(t2_pil.convert("RGB"))

            # Step 1: Detect bi-temporal changes
            cd_stats, _ = change_detection_pipeline.detect_changes(t1_arr, t2_arr)

            # Step 2: Run CDVQA Reasoning Engine
            answer, summary, confidence, stats, pred_category = cdvqa_pipeline.answer_cdvqa(
                t1_arr, t2_arr, question, cd_stats
            )

            # Category match check
            category_match = pred_category.lower() in gt_category.lower() or gt_category.lower() in pred_category.lower()
            if category_match:
                correct_category_matches += 1

            # Word overlap calculation
            ref_words = set(gt_answer.lower().split())
            pred_words = set(answer.lower().split())
            overlap = len(ref_words.intersection(pred_words)) / max(1, len(ref_words))
            word_overlap_scores.append(overlap)

            logger.info(f"--- CDVQA Pair #{total_samples} [{sample['pair_name']}] ---")
            logger.info(f"Q: {question}")
            logger.info(f"Ground Truth Category: {gt_category}")
            logger.info(f"Predicted Category: {pred_category}")
            logger.info(f"Predicted Answer: {answer} (Conf: {confidence})")
            logger.info(f"Category Match: {'[MATCH]' if category_match else '[MISMATCH]'} | Overlap: {round(overlap, 2)}")
            logger.info("-" * 50)

        cat_acc = round((correct_category_matches / max(1, total_samples)) * 100.0, 2)
        mean_overlap = round(float(np.mean(word_overlap_scores)) * 100.0, 2)

        print("\n" + "=" * 60)
        print("CDVQA Benchmark Evaluation Summary (https://github.com/YZHJessica/CDVQA):")
        print(f"Total Bi-Temporal Pairs Evaluated: {total_samples}")
        print(f"Category Classification Accuracy: {cat_acc}%")
        print(f"Mean Word Overlap Score: {mean_overlap}%")
        print("=" * 60)

    except Exception as e:
        logger.error(f"CDVQA evaluation error: {str(e)}")

if __name__ == "__main__":
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    evaluate_cdvqa(max_samples=max_s)

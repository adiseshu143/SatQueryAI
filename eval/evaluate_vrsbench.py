import sys
import numpy as np
from PIL import Image

from backend.preprocessing.dataset_loader import vrsbench_loader
from backend.pipelines.vqa import vqa_pipeline
from backend.utils.logger import logger

def evaluate_vrsbench(max_samples: int = 10):
    """
    Evaluates SatQuery AI VQA pipeline on streaming VRSBench remote sensing dataset.
    """
    logger.info(f"Starting VRSBench evaluation (max_samples={max_samples})...")
    
    total_qa = 0
    correct_matches = 0

    try:
        sample_count = 0
        for sample in vrsbench_loader.stream_dataset(split="train"):
            pil_img = sample["image"]
            caption = sample["caption"]
            vqa_pairs = sample["vqa"]

            if pil_img is None:
                continue

            sample_count += 1

            # Convert PIL to RGB numpy array
            rgb_img = np.array(pil_img.convert("RGB"))

            logger.info(f"--- Processing VRSBench Sample #{sample_count} ---")
            logger.info(f"Caption: {str(caption)[:100]}...")

            # Evaluate each Q&A pair in the sample
            if isinstance(vqa_pairs, list):
                for qa in vqa_pairs:
                    if not isinstance(qa, dict):
                        continue

                    total_qa += 1
                    question = qa.get("question", qa.get("q", "What is visible?"))
                    reference_answer = qa.get("answer", qa.get("a", ""))

                    # Run SatQuery AI VQA Pipeline
                    predicted_answer, summary, confidence, stats = vqa_pipeline.run(rgb_img, question)

                    # Keyword overlap match metric
                    ref_words = set(str(reference_answer).lower().split())
                    pred_words = set(str(predicted_answer).lower().split())
                    overlap = len(ref_words.intersection(pred_words)) / max(1, len(ref_words))

                    if overlap > 0.3:
                        correct_matches += 1

                    logger.info(f"Q: {question}")
                    logger.info(f"Ground Truth: {reference_answer}")
                    logger.info(f"SatQuery AI Answer: {predicted_answer} (Conf: {confidence})")
                    logger.info(f"Word Overlap Score: {round(overlap, 2)}")
                    logger.info("-" * 40)

            if sample_count >= max_samples:
                break

        accuracy = round((correct_matches / max(1, total_qa)) * 100.0, 2)
        print("\n" + "=" * 50)
        print(f"VRSBench Evaluation Summary:")
        print(f"Total Samples Tested: {sample_count}")
        print(f"Total Q&A Pairs Evaluated: {total_qa}")
        print(f"Accuracy / Keyword Match Rate: {accuracy}%")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    evaluate_vrsbench(max_samples=max_s)

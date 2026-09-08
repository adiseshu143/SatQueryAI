import sys
import numpy as np

from backend.preprocessing.dataset_loader import bigearthnet_loader
from backend.pipelines.sar_optical_fusion import sar_optical_pipeline
from backend.utils.logger import logger

def evaluate_bigearthnet(max_samples: int = 10):
    """
    Evaluates SatQuery AI SAR-Optical Fusion pipeline on BigEarthNet-MM benchmark dataset (https://txt.bigearth.net/).
    Measures multi-label Precision, Recall, and F1 Score across CORINE 19-class land cover annotations.
    """
    logger.info(f"Starting BigEarthNet-MM evaluation (Source: https://txt.bigearth.net/, max_samples={max_samples})...")
    
    total_samples = 0
    precisions = []
    recalls = []
    f1_scores = []

    try:
        for sample in bigearthnet_loader.stream_dataset(split="train", max_samples=max_samples):
            total_samples += 1
            opt_pil = sample["optical_image"]
            sar_pil = sample["sar_image"]
            gt_labels = set(sample["corine_labels"])

            opt_arr = np.array(opt_pil.convert("RGB"))
            sar_arr = np.array(sar_pil.convert("L"))[:, :, np.newaxis]

            query = "Analyze Sentinel-1 SAR and Sentinel-2 optical multi-label CORINE land cover"
            answer, summary, confidence, stats, pred_labels = sar_optical_pipeline.analyze_multimodal_pair(
                optical_img=opt_arr,
                sar_img=sar_arr,
                query=query
            )

            pred_set = set(pred_labels)
            intersection = gt_labels.intersection(pred_set)

            precision = len(intersection) / max(1, len(pred_set))
            recall = len(intersection) / max(1, len(gt_labels))
            f1 = (2 * precision * recall) / max(1e-5, precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            logger.info(f"--- BigEarthNet Sample #{total_samples} [{sample['patch_name']}] ---")
            logger.info(f"Ground Truth CORINE Labels: {list(gt_labels)}")
            logger.info(f"Predicted CORINE Labels: {pred_labels}")
            logger.info(f"Precision: {round(precision, 2)} | Recall: {round(recall, 2)} | F1: {round(f1, 2)}")
            logger.info("-" * 50)

        mean_prec = round(float(np.mean(precisions)) * 100.0, 2)
        mean_rec = round(float(np.mean(recalls)) * 100.0, 2)
        mean_f1 = round(float(np.mean(f1_scores)) * 100.0, 2)

        print("\n" + "=" * 60)
        print("BigEarthNet-MM Benchmark Evaluation Summary (https://txt.bigearth.net/):")
        print(f"Total Multimodal Samples Tested: {total_samples}")
        print(f"Mean Multi-Label Precision: {mean_prec}%")
        print(f"Mean Multi-Label Recall: {mean_rec}%")
        print(f"Mean Multi-Label F1 Score: {mean_f1}%")
        print("=" * 60)

    except Exception as e:
        logger.error(f"BigEarthNet evaluation error: {str(e)}")

if __name__ == "__main__":
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    evaluate_bigearthnet(max_samples=max_s)

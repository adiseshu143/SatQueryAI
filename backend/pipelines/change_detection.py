import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

from backend.config import settings
from backend.schemas.evidence import VisualEvidence

class ChangeDetectionPipeline:
    def detect_changes(
        self, 
        img1_arr: np.ndarray, 
        img2_arr: np.ndarray,
        pixel_resolution_m: Optional[float] = None
    ) -> Tuple[Dict[str, Any], VisualEvidence]:
        """
        Runs change detection between registered image pair.
        Generates binary change mask and visual evidence heatmap overlay.
        Calculates change percentage and affected area in hectares if pixel resolution exists.
        """
        # Ensure identical spatial dimensions
        h, w = img1_arr.shape[:2]
        if img2_arr.shape[:2] != (h, w):
            img2_arr = cv2.resize(img2_arr, (w, h))

        # Convert to grayscale for differencing
        gray1 = cv2.cvtColor(img1_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Absolute differencing
        diff = cv2.absdiff(gray1, gray2)

        # Gaussian blur filtering to suppress sensor noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)

        # Otsu / Adaptive thresholding to generate binary mask
        norm_diff = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary_mask = cv2.threshold(norm_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological post-processing cleanup (Opening & Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate statistics
        total_pixels = h * w
        changed_pixels = int(np.count_nonzero(cleaned_mask))
        change_pct = round((changed_pixels / total_pixels) * 100.0, 2)

        stats = {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "change_percentage": change_pct
        }

        # Calculate real-world area in hectares if geospatial metadata exists
        if pixel_resolution_m and pixel_resolution_m > 0:
            pixel_area_m2 = pixel_resolution_m * pixel_resolution_m
            changed_area_m2 = changed_pixels * pixel_area_m2
            changed_area_hectares = round(changed_area_m2 / 10000.0, 2)
            stats["changed_area_hectares"] = changed_area_hectares

        # Generate visual evidence overlay (Red heatmap highlight on change areas)
        overlay = img2_arr.copy()
        overlay[cleaned_mask > 0] = [255, 50, 50]  # Highlight red
        blended = cv2.addWeighted(img2_arr, 0.6, overlay, 0.4, 0)

        filename = f"change_mask_{uuid.uuid4().hex[:8]}.png"
        output_path = settings.MASK_DIR / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        evidence = VisualEvidence(
            type="change_mask",
            url=f"/outputs/masks/{filename}",
            description=f"Change mask overlay highlighting {change_pct}% detected structural/terrain change (red regions)."
        )

        return stats, evidence

change_detection_pipeline = ChangeDetectionPipeline()

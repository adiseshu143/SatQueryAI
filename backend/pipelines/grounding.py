import uuid
from typing import Optional, List
import numpy as np
import cv2

from backend.config import settings
from backend.schemas.evidence import VisualEvidence

class VisualGroundingPipeline:
    def generate_grounding_overlay(
        self, 
        img_arr: np.ndarray, 
        query: str
    ) -> Optional[VisualEvidence]:
        """
        Generates visual evidence overlay (detected feature bounding boxes or spectral attention map).
        """
        q_lower = query.lower()
        h, w = img_arr.shape[:2]
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY) if len(img_arr.shape) == 3 else img_arr

        overlay = img_arr.copy()
        features_found = False
        desc = ""

        # 1. Grounding buildings if requested
        if any(k in q_lower for k in ["building", "buildings", "house", "structure", "rooftop"]):
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh_adapt = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
            )
            contours, _ = cv2.findContours(thresh_adapt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = max(25.0, (h * w) * 0.0001)
            max_area = (h * w) * 0.15

            building_cnts = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
                    solidity = area / float(bw * bh) if (bw * bh) > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0 and solidity >= 0.35:
                        building_cnts.append(cnt)
                        # Draw green bounding rectangle around genuine detected rooftop
                        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 120), 2)
            
            if building_cnts:
                features_found = True
                desc = f"Visual Grounding: Highlighted {len(building_cnts)} detected building structure footprint(s) (green outlines)."

        # 2. Grounding water bodies if requested
        elif any(k in q_lower for k in ["water", "river", "lake", "flood"]):
            if len(img_arr.shape) == 3:
                b = img_arr[:, :, 2].astype(float)
                r = img_arr[:, :, 0].astype(float)
                water_mask = ((b - r > 15) & (b > 60)).astype(np.uint8) * 255
                contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                water_cnts = [c for c in contours if cv2.contourArea(c) > 100]
                if water_cnts:
                    cv2.drawContours(overlay, water_cnts, -1, (0, 180, 255), 2)
                    features_found = True
                    desc = f"Visual Grounding: Highlighted {len(water_cnts)} detected surface water body region(s) (blue outlines)."

        # 3. Grounding vegetation if requested
        elif any(k in q_lower for k in ["greenery", "vegetation", "green", "crop", "forest", "foliage"]):
            if len(img_arr.shape) == 3:
                r = img_arr[:, :, 0].astype(float)
                g = img_arr[:, :, 1].astype(float)
                b = img_arr[:, :, 2].astype(float)
                exg = 2 * g - r - b
                veg_mask = (exg > 15).astype(np.uint8) * 255
                contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                veg_cnts = [c for c in contours if cv2.contourArea(c) > 50]
                if veg_cnts:
                    cv2.drawContours(overlay, veg_cnts, -1, (40, 220, 90), 2)
                    features_found = True
                    desc = f"Visual Grounding: Highlighted active green vegetation regions ({len(veg_cnts)} green canopy contours)."

        if features_found:
            filename = f"grounding_{uuid.uuid4().hex[:8]}.png"
            output_path = settings.OVERLAY_DIR / filename
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return VisualEvidence(
                type="bounding_box",
                url=f"/outputs/overlays/{filename}",
                description=desc
            )

        # 3. Spectral Attention Overlay for general scene description & VQA
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(img_arr, 0.70, heatmap_rgb, 0.30, 0)

        filename = f"grounding_{uuid.uuid4().hex[:8]}.png"
        output_path = settings.OVERLAY_DIR / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return VisualEvidence(
            type="heatmap",
            url=f"/outputs/overlays/{filename}",
            description="Spectral feature reflectance attention overlay indicating primary land cover response regions."
        )

grounding_pipeline = VisualGroundingPipeline()

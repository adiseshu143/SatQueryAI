import cv2
import numpy as np
from typing import Tuple, Dict, Any
from backend.utils.exceptions import RegistrationFailedError
from backend.utils.logger import logger
from backend.config import settings

class ImageRegistrationPipeline:
    def register_pair(self, img1_arr: np.ndarray, img2_arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aligns img2 to img1 using feature matching (ORB + RANSAC).
        Returns warped img2 and alignment statistics.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_arr, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_arr, cv2.COLOR_RGB2GRAY)

        # Detect features using ORB
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.warning("Insufficient feature keypoints detected for registration.")
            # Fallback to direct resize if registration fails gracefully
            resized_img2 = cv2.resize(img2_arr, (img1_arr.shape[1], img1_arr.shape[0]))
            return resized_img2, {"matches": 0, "inliers": 0, "registered": False, "warning": "Low feature count"}

        # BFMatcher with Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < settings.REGISTRATION_MIN_MATCHES:
            logger.warning(f"Matches count {len(matches)} below threshold {settings.REGISTRATION_MIN_MATCHES}")
            resized_img2 = cv2.resize(img2_arr, (img1_arr.shape[1], img1_arr.shape[0]))
            return resized_img2, {"matches": len(matches), "inliers": 0, "registered": False, "warning": "Insufficient matching points"}

        # Extract location of matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate Homography using RANSAC
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        if H is None:
            resized_img2 = cv2.resize(img2_arr, (img1_arr.shape[1], img1_arr.shape[0]))
            return resized_img2, {"matches": len(matches), "inliers": 0, "registered": False, "warning": "Homography estimation failed"}

        inliers = int(np.sum(mask)) if mask is not None else 0
        h, w = img1_arr.shape[:2]
        warped_img2 = cv2.warpPerspective(img2_arr, H, (w, h))

        stats = {
            "matches": len(matches),
            "inliers": inliers,
            "inlier_ratio": round(inliers / max(1, len(matches)), 2),
            "registered": True
        }
        logger.info(f"Image registration completed: {inliers} inliers out of {len(matches)} matches.")
        return warped_img2, stats

registration_pipeline = ImageRegistrationPipeline()

import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from backend.utils.exceptions import InvalidImageError
from backend.config import settings

MAX_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

def load_image_bytes(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Image.Image, Dict[str, Any]]:
    """
    Loads raw image bytes, validates dimensions, file size, corruption, and format, and returns:
    - numpy ndarray (RGB)
    - PIL Image instance
    - extracted metadata dictionary
    """
    if not file_bytes or len(file_bytes) == 0:
        raise InvalidImageError(f"Uploaded file '{filename}' is empty (0 bytes). Please upload a valid satellite image.")

    if len(file_bytes) > MAX_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_MB
        raise InvalidImageError(f"File size exceeds maximum limit of {max_mb} MB. Uploaded size: {round(len(file_bytes)/(1024*1024), 1)} MB.")

    ext = Path(filename).suffix.lower()
    allowed = settings.ALLOWED_FORMATS
    if ext not in allowed:
        raise InvalidImageError(f"Unsupported image format '{ext}'. Allowed formats: {', '.join(allowed)}.")

    try:
        # Verify file header and corruption
        bio = io.BytesIO(file_bytes)
        pil_verify = Image.open(bio)
        pil_verify.verify()

        # Re-open after verify() (verify invalidates Image object)
        bio.seek(0)
        pil_img = Image.open(bio)
        width, height = pil_img.size
        mode = pil_img.mode
        bands = len(pil_img.getbands())

        if width < 16 or height < 16:
            raise InvalidImageError(f"Image dimensions ({width}x{height}) are too small. Minimum supported resolution is 16x16 pixels.")

        is_geotiff = ext in [".tif", ".tiff"]
        geotiff_info = None

        if is_geotiff and hasattr(pil_img, "tag_v2"):
            # Inspect authentic GeoTIFF tags if present
            tags = pil_img.tag_v2
            geotiff_info = {
                "is_geotiff": True,
                "model_pixel_scale": str(tags.get(33550, "Not specified")),
                "tie_points": str(tags.get(33922, "Not specified"))
            }

        # Convert to RGB array for CV processing
        rgb_pil = pil_img.convert("RGB")
        np_arr = np.array(rgb_pil)

        basic_metadata = {
            "filename": filename,
            "width": width,
            "height": height,
            "channels": bands,
            "mode": mode,
            "file_format": ext,
            "file_size_kb": round(len(file_bytes) / 1024, 1),
            "is_geotiff": is_geotiff,
            "geotiff_details": geotiff_info or "Standard Raster Imagery"
        }

        return np_arr, rgb_pil, basic_metadata

    except InvalidImageError as iie:
        raise iie
    except Exception as e:
        raise InvalidImageError(f"Corrupted or invalid image file '{filename}': {str(e)}")

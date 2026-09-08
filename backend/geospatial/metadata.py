from typing import Dict, Any, Optional, Tuple
from backend.schemas.metadata import ImageMetadata

def extract_geospatial_metadata(basic_meta: Dict[str, Any], pil_img) -> ImageMetadata:
    """
    Extracts geospatial metadata (CRS, pixel resolution, bounding box) from image tags or GeoTIFF headers.
    Falls back gracefully to non-geospatial metadata if tags are absent.
    """
    crs: Optional[str] = None
    pixel_res: Optional[float] = None
    bbox: Optional[list[float]] = None
    has_geo = False

    # Check for TIFF / GeoTIFF tag metadata if available
    try:
        if hasattr(pil_img, "tag_v2"):
            tags = pil_img.tag_v2
            # GeoKeyDirectoryTag (34735), ModelPixelScaleTag (33550), ModelTiepointTag (33922)
            if 33550 in tags:
                scales = tags[33550]
                pixel_res = float(scales[0])  # Scale X in meters/degrees
                has_geo = True
            if 34735 in tags or 33922 in tags:
                has_geo = True
                crs = "EPSG:4326"  # Default WGS84 fallback if GeoTIFF keys present

            # Tiepoint tag (33922) -> Extract origin coordinates if available
            if 33922 in tags and 33550 in tags:
                tp = tags[33922]
                ps = tags[33550]
                if len(tp) >= 6 and len(ps) >= 2:
                    min_x = float(tp[3])
                    max_y = float(tp[4])
                    max_x = min_x + (basic_meta["width"] * float(ps[0]))
                    min_y = max_y - (basic_meta["height"] * float(ps[1]))
                    bbox = [min_x, min_y, max_x, max_y]
    except Exception:
        pass

    return ImageMetadata(
        filename=basic_meta["filename"],
        width=basic_meta["width"],
        height=basic_meta["height"],
        channels=basic_meta["channels"],
        file_format=basic_meta["file_format"],
        crs=crs,
        pixel_resolution_m=pixel_res,
        has_geospatial_metadata=has_geo,
        bounding_box=bbox
    )

def check_same_area_overlap(meta1: ImageMetadata, meta2: ImageMetadata) -> Tuple[bool, str]:
    """
    Checks whether Image A and Image B represent overlapping or compatible geographic coverage.
    Returns: (is_overlapping, status_message)
    """
    if meta1.bounding_box and meta2.bounding_box:
        # Extracted Bounding Box format: [min_x, min_y, max_x, max_y]
        b1 = meta1.bounding_box
        b2 = meta2.bounding_box

        # Check spatial intersection
        overlap_x = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        overlap_y = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        
        if overlap_x <= 0 or overlap_y <= 0:
            return False, "These images appear to represent different geographic areas. For Change Analysis, upload Before and After images covering the same area."
        
        return True, "Geospatial bounding boxes confirmed spatial coverage overlap."

    if meta1.has_geospatial_metadata and meta2.has_geospatial_metadata:
        return True, "GeoTIFF headers present in both images."

    return True, "Geographic metadata unavailable; spatial correspondence could not be fully verified."

from typing import Optional
from pydantic import BaseModel, Field

class ImageMetadata(BaseModel):
    filename: str
    width: int
    height: int
    channels: int
    file_format: str
    crs: Optional[str] = None
    pixel_resolution_m: Optional[float] = None
    has_geospatial_metadata: bool = False
    bounding_box: Optional[list[float]] = None

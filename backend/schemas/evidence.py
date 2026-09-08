from typing import Optional
from pydantic import BaseModel

class VisualEvidence(BaseModel):
    type: str  # e.g., "change_mask", "heatmap", "bounding_box"
    url: str
    description: str

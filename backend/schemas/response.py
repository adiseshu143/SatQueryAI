from typing import Any, Optional
from pydantic import BaseModel, Field
from backend.schemas.evidence import VisualEvidence

class ConfidenceDetail(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    label: str  # "low", "medium", "high"
    method: str  # "heuristic", "model_probability", etc.

class SatQueryResponse(BaseModel):
    success: bool = True
    query: str
    intent: str
    answer: str
    summary: str
    confidence: ConfidenceDetail
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[VisualEvidence] = Field(default_factory=list)
    execution_steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

from typing import Any, List, Optional
from backend.schemas.response import SatQueryResponse, ConfidenceDetail
from backend.schemas.evidence import VisualEvidence

def build_satquery_response(
    query: str,
    intent: str,
    answer: str,
    summary: str,
    confidence_score: float,
    confidence_label: str,
    confidence_method: str,
    execution_steps: List[str],
    statistics: Optional[dict] = None,
    metadata: Optional[dict] = None,
    evidence: Optional[List[VisualEvidence]] = None,
    warnings: Optional[List[str]] = None
) -> SatQueryResponse:
    """
    Standard response builder for SatQuery AI.
    Ensures consistent API schema across all pipelines and filters output text.
    """
    from backend.services.llm_gateway import llm_gateway

    clean_answer = llm_gateway.sanitize_output(answer, query=query)
    clean_summary = llm_gateway.sanitize_output(summary, query=query)

    return SatQueryResponse(
        success=True,
        query=query,
        intent=intent,
        answer=clean_answer,
        summary=clean_summary,
        confidence=ConfidenceDetail(
            score=round(confidence_score, 2),
            label=confidence_label,
            method=confidence_method
        ),
        statistics=statistics or {},
        metadata=metadata or {},
        evidence=evidence or [],
        execution_steps=execution_steps,
        warnings=warnings or []
    )

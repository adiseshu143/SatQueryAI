from typing import Optional, List, Dict, Any
import numpy as np

from backend.controller.intent_router import router_engine
from backend.controller.response_builder import build_satquery_response
from backend.preprocessing.image_loader import load_image_bytes
from backend.geospatial.metadata import extract_geospatial_metadata
from backend.pipelines.vqa import vqa_pipeline
from backend.pipelines.registration import registration_pipeline
from backend.pipelines.change_detection import change_detection_pipeline
from backend.pipelines.grounding import grounding_pipeline
from backend.pipelines.sar_optical_fusion import sar_optical_pipeline
from backend.pipelines.cdvqa import cdvqa_pipeline
from backend.schemas.response import SatQueryResponse
from backend.schemas.evidence import VisualEvidence
from backend.utils.logger import logger

class AgenticController:
    """
    Agentic Controller for SatQuery AI.
    Parses user query, routes requests to specialist RS pipelines, validates outputs,
    and returns concise natural-language answers with visual evidence & execution logs.
    Natively supports BigEarthNet-MM (SAR+Optical), CDVQA (Change VQA), and RSVLM datasets.
    """

    async def process_text_chat(self, query: str) -> SatQueryResponse:
        """
        Handles text-only queries for the AI Agent Chatbot.
        Evaluates domain guard scope, returning domain knowledge answers or polite redirects.
        """
        intent_res = router_engine.parse_query(query, has_image2=False, has_image=False)
        
        if not intent_res.is_in_domain:
            return build_satquery_response(
                query=query,
                intent="OUT_OF_DOMAIN",
                answer=intent_res.redirect_message,
                summary="Query redirected: Out-of-domain scope.",
                confidence_score=1.0,
                confidence_label="high",
                confidence_method="domain_guard_classifier",
                execution_steps=[
                    "Evaluated query scope against SatQuery AI Remote Sensing domain guard",
                    "Query classified as Out-Of-Domain",
                    "Returned polite domain redirect message"
                ],
                statistics={"redirect": True},
                metadata={},
                evidence=[],
                warnings=["Query falls outside remote-sensing and satellite imagery domain scope."]
            )

        answer_text = intent_res.direct_answer
        if not answer_text:
            answer_text = (
                "This query targets visual scene analysis. "
                "Please upload a satellite image in the workspace to perform object detection, land cover classification, or change analysis on your scene."
            )

        return build_satquery_response(
            query=query,
            intent=intent_res.intent or "GENERAL_REMOTE_SENSING_KNOWLEDGE",
            answer=answer_text,
            summary="Responded directly from SatQuery AI Remote Sensing Specialist Domain Knowledge base.",
            confidence_score=0.95,
            confidence_label="high",
            confidence_method="system_prompt_domain_knowledge",
            execution_steps=[
                "Evaluated query scope against SatQuery AI Remote Sensing domain guard",
                "Classified as Remote Sensing query",
                "Synthesized direct specialist domain response"
            ],
            statistics={"knowledge_base_hit": True},
            metadata={},
            evidence=[],
            warnings=[]
        )

    async def process_request(
        self,
        query: str,
        img1_bytes: bytes,
        img1_name: str,
        img2_bytes: Optional[bytes] = None,
        img2_name: Optional[str] = None
    ) -> SatQueryResponse:
        
        execution_steps: List[str] = []
        warnings: List[str] = []

        has_image1 = img1_bytes is not None and len(img1_bytes) > 0
        has_image2 = img2_bytes is not None and len(img2_bytes) > 0

        # Step 0: Domain Guard Check
        intent_res = router_engine.parse_query(query, has_image2=has_image2, has_image=has_image1)
        
        if not intent_res.is_in_domain:
            return build_satquery_response(
                query=query,
                intent="OUT_OF_DOMAIN",
                answer=intent_res.redirect_message,
                summary="Query redirected: Out-of-domain scope.",
                confidence_score=1.0,
                confidence_label="high",
                confidence_method="domain_guard_classifier",
                execution_steps=[
                    "Evaluated query scope against SatQuery AI Remote Sensing domain guard",
                    "Query classified as Out-Of-Domain",
                    "Returned polite domain redirect message"
                ],
                statistics={"redirect": True},
                metadata={},
                evidence=[],
                warnings=["Query falls outside remote-sensing and satellite imagery domain scope."]
            )

        if intent_res.domain_category == "IN_DOMAIN_GENERAL_KNOWLEDGE" and not has_image1:
            return build_satquery_response(
                query=query,
                intent="GENERAL_REMOTE_SENSING_KNOWLEDGE",
                answer=intent_res.direct_answer,
                summary="Responded directly from SatQuery AI Remote Sensing Specialist Domain Knowledge base.",
                confidence_score=0.95,
                confidence_label="high",
                confidence_method="system_prompt_domain_knowledge",
                execution_steps=[
                    "Evaluated query scope against SatQuery AI Remote Sensing domain guard",
                    "Classified as General Remote Sensing Knowledge query",
                    "Synthesized direct specialist domain response"
                ],
                statistics={"knowledge_base_hit": True},
                metadata={},
                evidence=[],
                warnings=[]
            )

        # Step 1: Preprocess & extract metadata for Image 1
        execution_steps.append("Inspected uploaded satellite imagery & format specifications")
        img1_arr, pil_img1, meta1_raw = load_image_bytes(img1_bytes, img1_name)
        geo_meta1 = extract_geospatial_metadata(meta1_raw, pil_img1)

        img2_arr = None
        geo_meta2 = None

        if has_image2:
            img2_arr, pil_img2, meta2_raw = load_image_bytes(img2_bytes, img2_name)
            geo_meta2 = extract_geospatial_metadata(meta2_raw, pil_img2)
            execution_steps.append(f"Loaded second image ('{img2_name}') for comparative analysis")

        # Step 2: Query Understanding & Intent Parsing
        execution_steps.append(f"Parsed natural-language query: '{query}'")
        execution_steps.append(f"Classified intent: {intent_res.intent} (Operation: {intent_res.operation})")

        evidence_list: List[VisualEvidence] = []
        stats: Dict[str, Any] = {}

        # Step 3: Route execution pipeline
        q_lower_check = query.lower()
        if any(k in q_lower_check for k in ["bigearth", "bigearthnet", "corine", "txt.bigearth.net", "sar", "sentinel-1", "multimodal"]):
            # BigEarthNet-MM (https://txt.bigearth.net/) Multimodal Sentinel-1 SAR + Sentinel-2 Optical Fusion Pipeline
            execution_steps.append("Executing BigEarthNet-MM Multimodal Sentinel-1 SAR + Sentinel-2 Optical Fusion Pipeline (https://txt.bigearth.net/)")
            answer, summary, conf_score, fusion_stats, multi_labels = sar_optical_pipeline.analyze_multimodal_pair(
                img1_arr, img2_arr if has_image2 else None, query
            )
            stats.update(fusion_stats)
            stats["bigearthnet_multi_labels"] = multi_labels
            
            grounding_ev = grounding_pipeline.generate_grounding_overlay(img1_arr, intent_res.intent)
            evidence_list.append(grounding_ev)

            conf_label = "high"
            conf_method = "bigearthnet_multimodal_fusion_score"

        elif has_image2 or intent_res.requires_image2:
            if not has_image2:
                warnings.append("Change detection requested but only one image was uploaded. Registering fallback.")
                intent_res.intent = "LAND_COVER"

            if has_image2 and img2_arr is not None:
                # Run Image Registration
                execution_steps.append("Executing feature-based Image Registration (ORB + RANSAC)")
                warped_img2, reg_stats = registration_pipeline.register_pair(img1_arr, img2_arr)
                stats["registration"] = reg_stats
                execution_steps.append(f"Image alignment completed: {reg_stats.get('inliers', 0)} feature inliers found")

                if not reg_stats.get("registered", False):
                    warnings.append("Image registration was unaligned or unreliable. Results may have geometric noise.")

                # Run Change Detection
                execution_steps.append("Executing bi-temporal Change Detection & Morphological Post-Processing")
                pixel_res = geo_meta1.pixel_resolution_m or (geo_meta2.pixel_resolution_m if geo_meta2 else None)
                cd_stats, cd_evidence = change_detection_pipeline.detect_changes(
                    img1_arr, warped_img2, pixel_resolution_m=pixel_res
                )
                stats.update(cd_stats)
                evidence_list.append(cd_evidence)

                # Run CDVQA Reasoning (CDVQA Benchmark Inspired)
                execution_steps.append("Executing CDVQA (Change Detection Visual Question Answering) Reasoning Engine")
                cdvqa_ans, cdvqa_sum, cdvqa_conf, cdvqa_st, cd_cat = cdvqa_pipeline.answer_cdvqa(
                    img1_arr, warped_img2, query, cd_stats
                )
                stats.update(cdvqa_st)
                answer = cdvqa_ans
                summary = cdvqa_sum

                conf_score = 0.89 if reg_stats.get("registered", False) else 0.68
                conf_label = "high" if conf_score >= 0.75 else "medium"
                conf_method = "cdvqa_bitemporal_confidence"

        if not has_image2 or not evidence_list:
            # Single Image VQA Pipeline
            execution_steps.append("Executing single-image Remote-Sensing VQA & environmental condition analysis")
            answer, summary, conf_score, graphical_stats = vqa_pipeline.run(img1_arr, query)
            stats.update(graphical_stats)
            
            # Visual Grounding
            execution_steps.append("Generating visual grounding attention overlay")
            grounding_ev = grounding_pipeline.generate_grounding_overlay(img1_arr, query)
            if grounding_ev:
                evidence_list.append(grounding_ev)

            conf_label = "high" if conf_score >= 0.75 else "medium"
            conf_method = "spectral_similarity_score"

        execution_steps.append("Constructed explainable natural-language response with evidence overlays & graphical metrics")

        # Step 4: Build response
        metadata_export = {
            "image1": geo_meta1.model_dump(),
            "image2": geo_meta2.model_dump() if geo_meta2 else None
        }

        return build_satquery_response(
            query=query,
            intent=intent_res.intent,
            answer=answer,
            summary=summary,
            confidence_score=conf_score,
            confidence_label=conf_label,
            confidence_method=conf_method,
            execution_steps=execution_steps,
            statistics=stats,
            metadata=metadata_export,
            evidence=evidence_list,
            warnings=warnings
        )

agent_controller = AgenticController()

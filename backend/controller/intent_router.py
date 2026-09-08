import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from backend.controller.domain_classifier import domain_classifier, DomainClassificationResult

class IntentResult(BaseModel):
    intent: str
    entities: List[str]
    operation: str
    requires_image2: bool
    requires_geospatial_data: bool
    is_in_domain: bool = True
    domain_category: str = "IN_DOMAIN_IMAGE_WORKFLOW"
    redirect_message: Optional[str] = None
    direct_answer: Optional[str] = None

class IntentRouter:
    """
    Parses natural language query intent, extracts domain entities, enforces domain-guarding,
    and determines whether single-image or bi-temporal comparison pipelines are required.
    """
    
    CHANGE_KEYWORDS = [
        "change", "changed", "difference", "compare", "before", "after", 
        "growth", "deforestation", "flood expansion", "new construction", "destroyed", "built"
    ]
    
    VEGETATION_KEYWORDS = ["forest", "vegetation", "crop", "agriculture", "ndvi", "tree", "plant", "greenery"]
    WATER_KEYWORDS = ["flood", "water", "river", "lake", "inundated", "submerged", "flooded"]
    URBAN_KEYWORDS = ["building", "urban", "city", "construction", "house", "road", "structure"]
    SAR_KEYWORDS = ["sar", "radar", "sentinel-1", "backscatter", "speckle"]
    
    # New disaster & weather condition keywords
    EARTHQUAKE_KEYWORDS = ["earthquake", "seismic", "fault", "displacement", "tremor", "landslide", "tectonic"]
    RAIN_KEYWORDS = ["rain", "rainy", "precipitation", "storm", "monsoon", "cloud", "moisture"]
    SNOW_KEYWORDS = ["snow", "ice", "glacier", "freezing", "snowpack", "frost", "cold"]

    def parse_query(self, query: str, has_image2: bool = False, has_image: bool = True) -> IntentResult:
        # Step 0: Domain Guard Classification
        domain_res = domain_classifier.classify(query, has_image=has_image)

        if not domain_res.is_in_domain:
            return IntentResult(
                intent="OUT_OF_DOMAIN",
                entities=[],
                operation="redirect",
                requires_image2=False,
                requires_geospatial_data=False,
                is_in_domain=False,
                domain_category="OUT_OF_DOMAIN",
                redirect_message=domain_res.redirect_message
            )

        if domain_res.category == "IN_DOMAIN_GENERAL_KNOWLEDGE":
            return IntentResult(
                intent="GENERAL_REMOTE_SENSING_KNOWLEDGE",
                entities=[],
                operation="explain_rs_knowledge",
                requires_image2=False,
                requires_geospatial_data=False,
                is_in_domain=True,
                domain_category="IN_DOMAIN_GENERAL_KNOWLEDGE",
                direct_answer=domain_res.direct_answer
            )

        q = query.lower()
        entities: List[str] = []
        requires_image2 = False
        requires_geo = "area" in q or "hectare" in q or "square km" in q or "km2" in q

        # Extract entities
        if any(k in q for k in self.VEGETATION_KEYWORDS):
            entities.append("vegetation")
        if any(k in q for k in self.WATER_KEYWORDS):
            entities.append("water")
        if any(k in q for k in self.URBAN_KEYWORDS):
            entities.append("urban")
        if any(k in q for k in self.SAR_KEYWORDS):
            entities.append("sar")
        if any(k in q for k in self.EARTHQUAKE_KEYWORDS):
            entities.append("earthquake")
        if any(k in q for k in self.RAIN_KEYWORDS):
            entities.append("rain")
        if any(k in q for k in self.SNOW_KEYWORDS):
            entities.append("snow")

        # Intent classification
        if any(k in q for k in self.EARTHQUAKE_KEYWORDS):
            intent = "EARTHQUAKE_RISK"
            operation = "analyze_seismic_risk"

        elif any(k in q for k in self.RAIN_KEYWORDS):
            intent = "RAIN_FLOOD_CONDITION"
            operation = "analyze_rain_condition"

        elif any(k in q for k in self.SNOW_KEYWORDS):
            intent = "SNOW_ICE_CONDITION"
            operation = "analyze_snow_cover"

        elif has_image2 or any(k in q for k in self.CHANGE_KEYWORDS):
            requires_image2 = True
            if "forest" in q or "deforestation" in q or "tree" in q:
                intent = "DEFORESTATION"
            elif "flood" in q or "water" in q:
                intent = "FLOOD_ANALYSIS"
            elif "building" in q or "construction" in q or "urban" in q:
                intent = "URBAN_GROWTH"
            elif any(k in q for k in self.SAR_KEYWORDS):
                intent = "MULTIMODAL_COMPARISON"
            else:
                intent = "CHANGE_DETECTION"
            operation = "compare"

        elif any(k in q for k in self.WATER_KEYWORDS):
            intent = "FLOOD_ANALYSIS"
            operation = "analyze_water"

        elif any(k in q for k in self.VEGETATION_KEYWORDS):
            intent = "VEGETATION_ANALYSIS"
            operation = "analyze_vegetation"

        elif "land cover" in q or "land use" in q or "type of area" in q or "what is shown" in q:
            intent = "LAND_COVER"
            operation = "classify"

        elif any(k in q for k in self.SAR_KEYWORDS):
            intent = "SAR_ANALYSIS"
            operation = "analyze_sar"

        elif "how many" in q or "count" in q:
            intent = "COUNTING"
            operation = "count"

        else:
            intent = "GENERAL_VQA"
            operation = "vqa"

        return IntentResult(
            intent=intent,
            entities=entities,
            operation=operation,
            requires_image2=requires_image2,
            requires_geospatial_data=requires_geo
        )

router_engine = IntentRouter()

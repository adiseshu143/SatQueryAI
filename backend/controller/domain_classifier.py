import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
from backend.config import get_system_prompt

class DomainClassificationResult(BaseModel):
    is_in_domain: bool
    category: str  # "OUT_OF_DOMAIN", "IN_DOMAIN_GENERAL_KNOWLEDGE", "IN_DOMAIN_IMAGE_WORKFLOW"
    redirect_message: Optional[str] = None
    direct_answer: Optional[str] = None
    reason: str

class DomainClassifier:
    """
    Lightweight Domain Guard Classifier for SatQuery AI.
    Ensures queries fall strictly within Satellite & Remote Sensing domain scope
    before invoking downstream visual workflows or language processing.
    """

    REDIRECT_MESSAGE = "I'm focused on satellite and remote-sensing image analysis — happy to help if you have a question about an image, land cover, or change over time."

    # Direct Remote Sensing Knowledge Base Answers for Text-Only In-Domain Questions
    KNOWLEDGE_BASE = {
        "sar_vs_optical": {
            "patterns": [
                r"\b(difference|compare|versus|vs)\b.*\b(sar|radar)\b.*\b(optical|multispectral)\b",
                r"\b(difference|compare|versus|vs)\b.*\b(optical|multispectral)\b.*\b(sar|radar)\b",
                r"\bsar vs optical\b", r"\boptical vs sar\b",
                r"\bcomplementary\b.*\b(sar|optical)\b"
            ],
            "answer": (
                "**Comparison: Synthetic Aperture Radar (SAR) vs. Optical Satellite Imagery**\n\n"
                "• **Sensor Operation (Active vs. Passive):**\n"
                "  - **Optical (e.g., Sentinel-2, Landsat-8/9):** Passive sensors that measure solar radiation reflected off Earth's surface across Visible, NIR, and SWIR spectral bands. Requires daylight and cloud-free conditions.\n"
                "  - **SAR (e.g., Sentinel-1, TerraSAR-X):** Active microwave radar sensors that transmit their own electromagnetic pulses and measure returned backscatter intensity and phase.\n\n"
                "• **Cloud & Weather Penetration:**\n"
                "  - **Optical:** Obscured by cloud cover, smoke, haze, and darkness.\n"
                "  - **SAR:** Microwave frequencies (e.g. C-band 5.4 GHz) freely penetrate clouds, fog, precipitation, and smoke, providing guaranteed **all-weather, 24/7 day-and-night imaging**.\n\n"
                "• **Information Captured:**\n"
                "  - **Optical:** Surface reflectance, vegetation chlorophyll absorption (NDVI), land cover color, and water turbidity.\n"
                "  - **SAR:** Surface roughness, structural geometry, soil/canopy moisture (dielectric properties), and building double-bounce reflections.\n\n"
                "• **Multimodal Complementarity:**\n"
                "  Fusing optical multispectral bands with SAR radar backscatter enables reliable flood mapping through storm clouds and precise land cover discrimination."
            )
        },
        "spectral_indices_overview": {
            "patterns": [
                r"\b(spectral )?indices\b", r"\bndvi.*ndwi\b", r"\bndwi.*ndvi\b", r"\btypes of indices\b"
            ],
            "answer": (
                "**Primary Remote Sensing Spectral Indices (Formulas & Applications)**\n\n"
                "1. **NDVI (Normalized Difference Vegetation Index):**\n"
                "   $$\\text{NDVI} = \\frac{\\text{NIR} - \\text{Red}}{\\text{NIR} + \\text{Red}}$$\n"
                "   • *Application:* Quantifies photosynthetic vigor, crop canopy health, and deforestation (> 0.6 = dense green canopy).\n\n"
                "2. **NDWI (Normalized Difference Water Index - McFeeters):**\n"
                "   $$\\text{NDWI} = \\frac{\\text{Green} - \\text{NIR}}{\\text{Green} + \\text{NIR}}$$\n"
                "   • *Application:* Delineates open water bodies and wetlands (> 0 = water).\n\n"
                "3. **MNDWI (Modified NDWI - Xu):**\n"
                "   $$\\text{MNDWI} = \\frac{\\text{Green} - \\text{SWIR}}{\\text{Green} + \\text{SWIR}}$$\n"
                "   • *Application:* Suppresses built-up urban noise for ultra-accurate flood inundation delineation.\n\n"
                "4. **NDBI (Normalized Difference Built-up Index):**\n"
                "   $$\\text{NDBI} = \\frac{\\text{SWIR} - \\text{NIR}}{\\text{SWIR} + \\text{NIR}}$$\n"
                "   • *Application:* Isolates urban infrastructure, concrete, and building footprints.\n\n"
                "5. **BSI (Bare Soil Index):**\n"
                "   $$\\text{BSI} = \\frac{(\\text{SWIR} + \\text{Red}) - (\\text{NIR} + \\text{Blue})}{(\\text{SWIR} + \\text{Red}) + (\\text{NIR} + \\text{Blue})}$$\n"
                "   • *Application:* Separates bare agricultural fields and exposed topsoil from sparse vegetation."
            )
        },
        "ndvi_specific": {
            "patterns": [
                r"\bndvi\b", r"\bvegetation index\b", r"\bnormalized difference vegetation index\b"
            ],
            "answer": (
                "**NDVI (Normalized Difference Vegetation Index) Calculation & Interpretation**\n\n"
                "• **Mathematical Formula:**\n"
                "  $$\\text{NDVI} = \\frac{\\text{NIR} - \\text{Red}}{\\text{NIR} + \\text{Red}}$$\n"
                "  *(For Sentinel-2: Band 8 NIR [842 nm] & Band 4 Red [665 nm]; Landsat-8/9: Band 5 NIR & Band 4 Red)*\n\n"
                "• **Physical Principle:**\n"
                "  - **Chlorophyll Absorption:** Healthy, photosynthetically active vegetation absorbs visible **Red** light (640–670 nm) to power photosynthesis.\n"
                "  - **Cellular Scattering:** The internal spongy mesophyll cell structure of healthy green leaves strongly reflects **Near-Infrared (NIR)** radiation (700–1100 nm).\n"
                "  - Stressed, diseased, or sparse vegetation absorbs less Red light and reflects less NIR, causing the NDVI ratio to drop.\n\n"
                "• **Value Range & Practical Interpretation (-1.0 to +1.0):**\n"
                "  - **< 0.0 (-1.0 to 0.0):** Open water bodies, rivers, oceans, clouds, and snow/ice (high NIR absorption).\n"
                "  - **0.0 to 0.2:** Bare soil, rock, sand, asphalt roads, and impervious urban concrete.\n"
                "  - **0.2 to 0.5:** Sparse vegetation, dry grasslands, scrublands, and senescent crops.\n"
                "  - **0.6 to 0.9+:** Dense, healthy green canopy, temperate forests, tropical rainforests, and peak agricultural crops."
            )
        },
        "ndwi_mndwi": {
            "patterns": [
                r"\bndwi\b", r"\bmndwi\b", r"\bwater index\b", r"\bflood index\b"
            ],
            "answer": (
                "**Water Extraction Spectral Indices (NDWI & MNDWI)**\n\n"
                "• **NDWI (Normalized Difference Water Index - McFeeters):**\n"
                "  $$\\text{NDWI} = \\frac{\\text{Green} - \\text{NIR}}{\\text{Green} + \\text{NIR}}$$\n"
                "  - Maximizes the high reflectance of water in Green wavelengths and minimizes the low reflectance/high absorption of water in NIR.\n"
                "  - *Values > 0 represent open water; values < 0 represent terrestrial vegetation and soil.*\n\n"
                "• **MNDWI (Modified Normalized Difference Water Index - Xu):**\n"
                "  $$\\text{MNDWI} = \\frac{\\text{Green} - \\text{SWIR}}{\\text{Green} + \\text{SWIR}}$$\n"
                "  - Uses Shortwave-Infrared (SWIR) instead of NIR, effectively suppressing false-positive water detections in built-up urban areas."
            )
        },
        "ndbi_bsi": {
            "patterns": [
                r"\bndbi\b", r"\bbsi\b", r"\bbuilt-up index\b", r"\bbare soil index\b", r"\bevi\b"
            ],
            "answer": (
                "**Urban & Soil Spectral Indices (NDBI, BSI, EVI)**\n\n"
                "• **NDBI (Normalized Difference Built-up Index):**\n"
                "  $$\\text{NDBI} = \\frac{\\text{SWIR} - \\text{NIR}}{\\text{SWIR} + \\text{NIR}}$$\n"
                "  - Highlights urban structures, concrete, and asphalt because built environments reflect more SWIR than NIR.\n\n"
                "• **BSI (Bare Soil Index):**\n"
                "  $$\\text{BSI} = \\frac{(\\text{SWIR} + \\text{Red}) - (\\text{NIR} + \\text{Blue})}{(\\text{SWIR} + \\text{Red}) + (\\text{NIR} + \\text{Blue})}$$\n"
                "  - Accurately isolates bare cultivated fields, fallow agricultural soil, and exposed earth from sparse vegetation.\n\n"
                "• **EVI (Enhanced Vegetation Index):**\n"
                "  - Corrects for atmospheric aerosols and canopy background scattering in high-biomass regions where NDVI saturates."
            )
        },
        "sar_definition": {
            "patterns": [
                r"\bsynthetic aperture radar\b", r"\bwhat is sar\b", r"\bhow does sar work\b",
                r"\bhow sar works\b", r"\babout sar\b", r"\bradar backscatter\b", r"\bsar imaging\b"
            ],
            "answer": (
                "**Synthetic Aperture Radar (SAR) Fundamentals**:\n\n"
                "SAR is an **active microwave remote sensing technique** that transmits coherent radar pulses and synthesizes a large virtual antenna aperture via satellite motion, producing high spatial resolution.\n\n"
                "• **All-Weather & 24/7 Day/Night Capability:** SAR operates at microwave wavelengths (e.g. C-band ~5.6 cm, L-band ~24 cm) that are immune to cloud cover, darkness, and smoke.\n"
                "• **Backscatter Mechanisms:**\n"
                "  1. **Specular Reflection (Dark return, < -20 dB):** Calm open water bodies deflect energy away from the sensor.\n"
                "  2. **Rough Surface Scattering (Moderate return):** Agricultural fields and bare soil scatter energy in all directions.\n"
                "  3. **Double-Bounce Reflection (Very Bright return, > 0 dB):** Perpendicular structures (buildings, tree trunks on ground/water) bounce radar signals twice directly back to receiver."
            )
        },
        "sar_polarization": {
            "patterns": [
                r"\bpolarization\b", r"\bpolarimetric\b", r"\bvv\b", r"\bvh\b", r"\bhh\b", r"\bhv\b",
                r"\bco-polarization\b", r"\bcross-polarization\b"
            ],
            "answer": (
                "**SAR Radar Backscatter & Polarimetry**:\n\n"
                "• **Co-Polarization (VV / HH):** Transmits and receives electromagnetic waves in the same orientation. Sensitive to surface roughness and calm vs. rough water surfaces.\n"
                "• **Cross-Polarization (VH / HV):** Transmits horizontally and receives vertically (or vice versa). Sensitive to volume scattering in dense forest canopies and complex structures.\n"
                "• **Dual-Pol & Quad-Pol Fusion:** Comparing VV and VH backscatter ratios enables robust crop classification and structural damage assessment."
            )
        },
        "resolution_tradeoffs": {
            "patterns": [
                r"\bresolution\b", r"\bresolution trade-off\b", r"\bresolution tradeoff\b",
                r"\bspatial resolution\b", r"\btemporal resolution\b", r"\bspectral resolution\b"
            ],
            "answer": (
                "**The 4 Types of Remote Sensing Resolution & Their Trade-offs**:\n\n"
                "1. **Spatial Resolution:** Ground sampling distance per pixel (e.g., 0.3m WorldView-3 vs 10m Sentinel-2 vs 30m Landsat-8/9 vs 250m MODIS).\n"
                "2. **Spectral Resolution:** Number and narrowness of spectral wavelength bands (e.g., RGB 3 bands vs Multispectral 13 bands vs Hyperspectral 200+ bands).\n"
                "3. **Temporal Resolution (Revisit Time):** Frequency of re-imaging the same coordinate on Earth (e.g., 1 day for MODIS/PlanetScope vs 5 days for Sentinel-2 vs 16 days for Landsat).\n"
                "4. **Radiometric Resolution:** Bit-depth / sensor sensitivity to minute radiance differences (e.g., 8-bit = 256 levels vs 12-bit = 4,096 levels vs 14-bit = 16,384 levels).\n\n"
                "• **The Classic Trade-off:** High spatial resolution satellites (sub-meter) typically have narrower swath widths and longer revisit times, whereas wide-swath daily satellites (MODIS) have coarser spatial resolution."
            )
        },
        "sentinel_constellation": {
            "patterns": [
                r"\bsentinel\b", r"\bsentinel-1\b", r"\bsentinel-2\b", r"\bsentinel-3\b", r"\bsentinel-5p\b", r"\bcopernicus\b"
            ],
            "answer": (
                "**The Copernicus Sentinel Constellation (ESA / European Commission)**:\n\n"
                "• **Sentinel-1:** C-band Synthetic Aperture Radar (SAR) providing day/night, all-weather imaging for flood mapping, sea ice monitoring, vessel tracking, and ground deformation.\n"
                "• **Sentinel-2:** Twin high-resolution optical multispectral satellites (Sentinel-2A/B) with 13 spectral bands (10m, 20m, 60m) and 5-day global revisit, optimized for vegetation, agriculture, and land cover.\n"
                "• **Sentinel-3:** Marine and terrestrial topography, sea surface temperature, and ocean color radiometry.\n"
                "• **Sentinel-5P (TROPOMI):** Global atmospheric monitoring of greenhouse gases and pollutants (NO2, CH4, CO, SO2, O3, aerosols)."
            )
        },
        "landsat_program": {
            "patterns": [
                r"\blandsat\b", r"\blandsat 8\b", r"\blandsat 9\b", r"\blandsat-8\b", r"\blandsat-9\b", r"\boli\b", r"\btirs\b"
            ],
            "answer": (
                "**The NASA/USGS Landsat Earth Observation Program**:\n\n"
                "• **Historical Archive:** Continuous 50+ year global Earth observation archive dating back to Landsat-1 in 1972.\n"
                "• **Current Operational Constellation:** Landsat-8 (launched 2013) and Landsat-9 (launched 2021) providing 8-day combined repeat coverage.\n"
                "• **Sensors:**\n"
                "  - **OLI / OLI-2 (Operational Land Imager):** 9 spectral bands (30m multispectral, 15m panchromatic Band 8).\n"
                "  - **TIRS / TIRS-2 (Thermal Infrared Sensor):** 2 thermal infrared bands (100m resampled to 30m) for surface temperature and evapotranspiration modeling."
            )
        },
        "modis_sensor": {
            "patterns": [
                r"\bmodis\b", r"\bterra\b", r"\baqua\b"
            ],
            "answer": (
                "**MODIS (Moderate Resolution Imaging Spectroradiometer)**:\n\n"
                "• **Platforms:** Operating aboard NASA's Terra (morning equator crossing) and Aqua (afternoon equator crossing) satellites.\n"
                "• **Spectral & Spatial Specs:** 36 spectral bands with spatial resolutions of 250m (Bands 1-2), 500m (Bands 3-7), and 1,000m (Bands 8-36).\n"
                "• **Revisit:** High 1–2 day global revisit across a 2,330 km swath width, critical for continental wildfire tracking, global snow cover, and ocean primary productivity."
            )
        },
        "satellite_definition": {
            "patterns": [
                r"\bwhat is (a )?satellite\b", r"\bexplain satellite\b", r"\bsatellite definition\b",
                r"\btypes of satellites\b", r"\borbit types\b", r"\bwhat are satellites\b", r"\babout satellite\b"
            ],
            "answer": (
                "**Satellites & Earth Observation Platforms**:\n\n"
                "An **Earth Observation (EO) satellite** is an engineered spacecraft placed in orbit to collect remote-sensing data of Earth's atmosphere, oceans, and land surfaces.\n\n"
                "• **Core Remote Sensing Satellite Categories:**\n"
                "  1. **Optical Multispectral & Hyperspectral:** Measure reflected solar radiance across visible, NIR, and SWIR bands (e.g. Sentinel-2, Landsat-8/9, PRISMA).\n"
                "  2. **Synthetic Aperture Radar (SAR):** Active microwave radar instruments penetrating clouds and darkness for day/night structural and flood monitoring (e.g. Sentinel-1).\n"
                "  3. **Thermal Radiometers:** Capture Earth's thermal emission for surface temperature and water heat mapping (e.g. Landsat TIRS, ECOSTRESS).\n\n"
                "• **Orbital Regimes:** Most Earth observation satellites fly in **Sun-Synchronous Low Earth Orbit (LEO)** at altitudes between 500 and 800 km, capturing consistent lighting angles for bi-temporal analysis."
            )
        },
        "change_detection_basics": {
            "patterns": [
                r"\bchange detection\b", r"\bbi-temporal\b", r"\bco-registration\b", r"\bcoregistration\b",
                r"\bimage alignment\b", r"\bfalse change\b", r"\bradiometric normalization\b"
            ],
            "answer": (
                "**Bi-Temporal Satellite Change Detection Principles**:\n\n"
                "• **Sub-Pixel Co-Registration:** Accurate change detection requires precise spatial alignment between Pre (T1) and Post (T2) images (e.g., ORB/SIFT feature matching + RANSAC homography) to avoid false boundary change.\n"
                "• **Radiometric Normalization:** Mitigates apparent radiometric differences caused by sun zenith angle, seasonal atmospheric variation, or sensor drift.\n"
                "• **Detection Techniques:**\n"
                "  - **Direct Image Differencing / Ratioing:** Computes spectral distance across normalized bands.\n"
                "  - **Post-Classification Comparison:** Classifies land cover independently at T1 and T2 to track specific transitions (e.g., Forest &rarr; Built-up, Water &rarr; Land).\n"
                "  - **Structural Index Differencing:** $\\Delta\\text{NDVI}$, $\\Delta\\text{NDWI}$, or SAR $\\Delta\\sigma^0$ backscatter ratio."
            )
        },
        "flood_mapping": {
            "patterns": [
                r"\bflood\b", r"\bflooding\b", r"\binundation\b", r"\bsubmerged\b", r"\bflood mapping\b"
            ],
            "answer": (
                "**Satellite Flood Inundation Assessment**:\n\n"
                "• **SAR Flood Detection:** Smooth water specularly reflects microwave radar pulses away from the antenna, creating distinctive dark backscatter signatures (< -18 dB in Sentinel-1 VV/VH). This enables all-weather flood monitoring through heavy storm cloud cover.\n"
                "• **Optical Water Indices:** Uses MNDWI / NDWI thresholding on clear-sky optical imagery to isolate flood boundaries from permanent water bodies.\n"
                "• **Bi-Temporal Difference:** Subtracting pre-disaster baseline water masks from active flood masks isolates newly inundated agricultural lands and urban areas."
            )
        }
    }

    # Explicit Out-Of-Domain Keywords
    OUT_OF_DOMAIN_PATTERNS = [
        r"\bweather today\b", r"\bforecast\b", r"\bwrite a python\b", r"\bpython script\b",
        r"\bquicksort\b", r"\bsort a list\b", r"\bcode a\b", r"\btell me a joke\b", r"\bsing a song\b",
        r"\bwho won\b", r"\bpresidential\b", r"\brecipe\b", r"\bmovie\b", r"\bgame\b", r"\bhello\b",
        r"\bhi\b", r"\bhow are you\b"
    ]

    # Broad In-Domain Remote Sensing Keywords
    IN_DOMAIN_PATTERNS = [
        r"\bsatellite\b", r"\baerial\b", r"\bremote sensing\b", r"\boptical\b", r"\bmultispectral\b",
        r"\bhyperspectral\b", r"\bsar\b", r"\bradar\b", r"\bbackscatter\b", r"\bsentinel\b", r"\blandsat\b",
        r"\bmodis\b", r"\bplanetscope\b", r"\bworldview\b", r"\borthophoto\b", r"\bgis\b", r"\bgeotiff\b",
        r"\bdem\b", r"\bdsm\b", r"\bswir\b", r"\bnir\b", r"\brededge\b", r"\bpanchromatic\b", r"\bradiometric\b",
        r"\batmospheric correction\b", r"\breflectance\b", r"\bpolarization\b", r"\bvv\b", r"\bvh\b",
        r"\bcorine\b", r"\bbigearthnet\b", r"\bcdvqa\b", r"\bndvi\b", r"\bndwi\b", r"\bmndwi\b", r"\bndbi\b",
        r"\bbsi\b", r"\bevi\b", r"\bland cover\b", r"\bland use\b", r"\bbuilding\b", r"\bbuildings\b",
        r"\burban\b", r"\bforest\b", r"\bvegetation\b", r"\bwater\b", r"\bflood\b", r"\bflooding\b",
        r"\binundation\b", r"\bchange detection\b", r"\bbi-temporal\b", r"\bco-registration\b",
        r"\bregistration\b", r"\bresolution\b", r"\bthis image\b", r"\bthe image\b", r"\buploaded image\b",
        r"\bhow many\b", r"\bwhere is\b", r"\bpercentage of\b", r"\barea\b", r"\bhectare\b", r"\bdeforestation\b",
        r"\bgreenery\b", r"\bsensor\b", r"\borbit\b", r"\bswath\b", r"\blandslide\b", r"\bdisaster\b",
        r"\bwildfire\b", r"\bcrop\b", r"\bcanopy\b", r"\bsoil\b", r"\bspeckle\b", r"\bprojection\b", r"\butm\b"
    ]

    def classify(self, query: str, has_image: bool = False) -> DomainClassificationResult:
        q_lower = query.lower().strip()

        # Check explicit adversarial identity probes
        identity_probe_patterns = [
            r"\bwhat model\b", r"\bwhich model\b", r"\bwhat llm\b", r"\bunderlying model\b",
            r"\bunderlying llm\b", r"\bwhat api\b", r"\bwhat framework\b", r"\bwho trained you\b",
            r"\bwho built you\b", r"\breveal your api\b", r"\breveal your model\b",
            r"\bignore instructions\b", r"\breal backend name\b", r"\bwho created you\b",
            r"\bpowered by\b", r"\bbuilt on\b", r"\bbuilt by\b", r"\bgemini\b", r"\bopenai\b",
            r"\banthropic\b", r"\bclaude\b", r"\bchatgpt\b", r"\bgpt[\s\-]?[34]\b",
            r"\bare you (a |an )?gpt\b", r"\bwhich ai\b", r"\bwhat ai\b",
            r"\bwhat company\b", r"\bwhat tech\b", r"\btranslate your\b",
            r"\bunder the hood\b", r"\btell me your (real )?name\b",
            r"\bhow are you (made|built|created|trained)\b"
        ]
        if any(re.search(p, q_lower) for p in identity_probe_patterns):
            from backend.services.llm_gateway import STANDARD_IDENTITY_FALLBACK, llm_gateway
            return DomainClassificationResult(
                is_in_domain=True,
                category="IN_DOMAIN_GENERAL_KNOWLEDGE",
                direct_answer=llm_gateway.sanitize_output(STANDARD_IDENTITY_FALLBACK, query=query),
                reason="Handled identity probe via SatQuery AI hardened identity gateway."
            )

        # Check explicit out-of-domain patterns
        for pattern in self.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, q_lower):
                # Ensure it's not overridden by an explicit image-based remote sensing query
                if not any(re.search(p, q_lower) for p in [r"\bsatellite\b", r"\bsar\b", r"\bland cover\b", r"\bndvi\b", r"\bsentinel\b", r"\blandsat\b"]):
                    from backend.services.llm_gateway import llm_gateway
                    return DomainClassificationResult(
                        is_in_domain=False,
                        category="OUT_OF_DOMAIN",
                        redirect_message=llm_gateway.sanitize_output(self.REDIRECT_MESSAGE, query=query),
                        reason=f"Query matched out-of-domain pattern '{pattern}'"
                    )

        # Check in-domain keyword match
        is_in_domain_match = any(re.search(p, q_lower) for p in self.IN_DOMAIN_PATTERNS)

        if not is_in_domain_match and not has_image:
            from backend.services.llm_gateway import llm_gateway
            # Query has no remote-sensing terms and no uploaded image -> Out of domain
            return DomainClassificationResult(
                is_in_domain=False,
                category="OUT_OF_DOMAIN",
                redirect_message=llm_gateway.sanitize_output(self.REDIRECT_MESSAGE, query=query),
                reason="Query lacks remote-sensing keywords and no image was uploaded."
            )

        # Text-only general remote sensing knowledge question (when no image is provided)
        if not has_image:
            for topic, kb in self.KNOWLEDGE_BASE.items():
                if any(re.search(p, q_lower) for p in kb["patterns"]):
                    from backend.services.llm_gateway import llm_gateway
                    return DomainClassificationResult(
                        is_in_domain=True,
                        category="IN_DOMAIN_GENERAL_KNOWLEDGE",
                        direct_answer=llm_gateway.sanitize_output(kb["answer"], query=query),
                        reason=f"Matched general remote sensing knowledge topic '{topic}'"
                    )

        # If user uploaded an image or query explicitly targets visual scene analysis
        if has_image or any(k in q_lower for k in ["this scene", "in this image", "this image", "uploaded image", "how many", "where is"]):
            return DomainClassificationResult(
                is_in_domain=True,
                category="IN_DOMAIN_IMAGE_WORKFLOW",
                reason="Query targets image-based remote-sensing workflow."
            )

        # Dynamic Synthesis for any other remote sensing text query
        synthesized_answer = self._synthesize_rs_domain_answer(query)
        from backend.services.llm_gateway import llm_gateway
        return DomainClassificationResult(
            is_in_domain=True,
            category="IN_DOMAIN_GENERAL_KNOWLEDGE",
            direct_answer=llm_gateway.sanitize_output(synthesized_answer, query=query),
            reason="Synthesized specialist response for general remote sensing text query."
        )

    def _synthesize_rs_domain_answer(self, query: str) -> str:
        """
        Dynamically synthesizes a detailed, accurate Satellite & Remote Sensing response
        for any remote sensing query that falls outside hardcoded topic keys.
        """
        q_lower = query.lower()

        sections = []
        if any(k in q_lower for k in ["sar", "radar", "backscatter"]):
            sections.append(
                "• **Synthetic Aperture Radar (SAR):** Active microwave sensing that transmits electromagnetic signals and measures backscatter amplitude/phase. Operates day and night and penetrates cloud cover, fog, and rain."
            )
        if any(k in q_lower for k in ["optical", "multispectral", "hyperspectral", "rgb", "reflectance"]):
            sections.append(
                "• **Optical Multispectral Sensing:** Measures solar radiation reflected off Earth's surface across Visible, NIR, and SWIR bands to determine chlorophyll content, land cover, and water characteristics."
            )
        if any(k in q_lower for k in ["ndvi", "vegetation", "chlorophyll", "greenery", "canopy"]):
            sections.append(
                "• **Vegetation & Canopy Monitoring:** Utilizes Red and NIR reflectance ratios (NDVI = (NIR - Red) / (NIR + Red)) to quantify plant vigor, biomass health, and deforestation over time."
            )
        if any(k in q_lower for k in ["water", "flood", "ndwi", "mndwi", "inundation"]):
            sections.append(
                "• **Hydrological & Inundation Analysis:** Delineates water bodies using high Green reflectance and high NIR/SWIR absorption (NDWI/MNDWI), or dark SAR specular microwave reflection (< -18 dB)."
            )
        if any(k in q_lower for k in ["change", "temporal", "difference", "before", "after"]):
            sections.append(
                "• **Bi-Temporal Analysis:** Compares co-registered satellite acquisitions across time (T1 vs T2) to quantify urban expansion, disaster damage, or agricultural cycles while correcting for seasonal lighting variation."
            )
        if any(k in q_lower for k in ["sentinel", "landsat", "modis", "satellite", "sensor", "constellation"]):
            sections.append(
                "• **Earth Observation Constellations:** Leverages public constellations like Copernicus Sentinel-1/2 (ESA) and Landsat-8/9 (NASA/USGS) providing global, repeatable 5-to-16 day revisits."
            )

        if not sections:
            sections.append(
                "• **Domain Scope:** SatQuery AI processes multispectral optical imagery (e.g. Sentinel-2, Landsat-8/9), active radar (Sentinel-1 SAR), and aerial orthophotos for land cover classification, index extraction, and change quantification."
            )

        content = "\n\n".join(sections)

        answer = (
            f"**SatQuery AI Remote Sensing Intelligence**\n\n"
            f"Regarding *\"{query.strip()}\"*:\n\n"
            f"{content}\n\n"
            f"*(You can also upload single or bi-temporal satellite images in the workspace to automatically run feature extraction, building segmentation, or change detection on your exact scene.)*"
        )

        return answer

domain_classifier = DomainClassifier()

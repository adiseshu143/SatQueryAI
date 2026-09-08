from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status

from backend.schemas.response import SatQueryResponse
from backend.controller.agent import agent_controller
from backend.services.ai_service import ai_service
from backend.utils.logger import logger

router = APIRouter()

@router.post("/analyze", response_model=SatQueryResponse)
async def analyze_imagery(
    query: str = Form(...),
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None)
):
    """
    Preserved legacy endpoint for full backwards compatibility.
    """
    logger.info(f"API Request /analyze -> query: '{query}', img1: {image1.filename}, img2: {image2.filename if image2 else 'None'}")
    
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty."
        )

    img1_bytes = await image1.read()
    if not img1_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded Image 1 file is empty."
        )

    img2_bytes = None
    img2_name = None
    if image2:
        img2_bytes = await image2.read()
        img2_name = image2.filename

    response = await agent_controller.process_request(
        query=query,
        img1_bytes=img1_bytes,
        img1_name=image1.filename,
        img2_bytes=img2_bytes,
        img2_name=img2_name
    )

    return response


@router.post("/analyze/single")
async def analyze_single(
    query: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Feature 1 Endpoint: Image Detection & VQA for a single image.
    """
    logger.info(f"API Request /analyze/single -> query: '{query}', image: {image.filename}")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    res = await ai_service.analyze_single_image(
        query=query,
        img_bytes=img_bytes,
        filename=image.filename
    )
    return res


@router.post("/analyze/change")
async def analyze_change(
    query: str = Form(...),
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    date_a: Optional[str] = Form(None),
    sensor_a: Optional[str] = Form(None),
    location_a: Optional[str] = Form(None),
    date_b: Optional[str] = Form(None),
    sensor_b: Optional[str] = Form(None),
    location_b: Optional[str] = Form(None)
):
    """
    Feature 2 Endpoint: Change Analysis for Image A (Before) and Image B (After).
    """
    logger.info(f"API Request /analyze/change -> query: '{query}', img_a: {image_a.filename}, img_b: {image_b.filename}, date_a: '{date_a}', date_b: '{date_b}'")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    img1_bytes = await image_a.read()
    img2_bytes = await image_b.read()

    if not img1_bytes or not img2_bytes:
        raise HTTPException(status_code=400, detail="Both Image A and Image B must be non-empty files.")

    res = await ai_service.analyze_change(
        query=query,
        img1_bytes=img1_bytes,
        img1_name=image_a.filename,
        img2_bytes=img2_bytes,
        img2_name=image_b.filename,
        date_a=date_a,
        sensor_a=sensor_a,
        location_a=location_a,
        date_b=date_b,
        sensor_b=sensor_b,
        location_b=location_b
    )
    return res


@router.post("/analyze/optical-sar")
async def analyze_optical_sar(
    query: str = Form(...),
    optical_image: UploadFile = File(...),
    sar_image: Optional[UploadFile] = File(None)
):
    """
    Feature 3 Endpoint: Joint Multimodal Analysis for Optical and SAR imagery.
    """
    logger.info(f"API Request /analyze/optical-sar -> query: '{query}', optical: {optical_image.filename}, sar: {sar_image.filename if sar_image else 'None'}")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    optical_bytes = await optical_image.read()
    if not optical_bytes:
        raise HTTPException(status_code=400, detail="Optical image file is empty.")

    sar_bytes = None
    sar_name = None
    if sar_image:
        sar_bytes = await sar_image.read()
        sar_name = sar_image.filename

    res = await ai_service.analyze_optical_sar(
        query=query,
        optical_bytes=optical_bytes,
        optical_name=optical_image.filename,
        sar_bytes=sar_bytes,
        sar_name=sar_name
    )
    return res


@router.post("/agent/route")
async def agent_route(
    query: str = Form(""),
    image_count: int = Form(1),
    mode_selected: Optional[str] = Form(None),
    has_sar: bool = Form(False),
    has_optical: bool = Form(False)
):
    """
    Feature 4 Endpoint: AI Agent Intelligent Workflow Selection.
    """
    logger.info(f"API Request /agent/route -> query: '{query}', count: {image_count}, mode: {mode_selected}")
    res = await ai_service.route_query(
        query=query,
        image_count=image_count,
        mode_selected=mode_selected,
        has_sar=has_sar,
        has_optical=has_optical
    )
    return res


@router.post("/agent/chat", response_model=SatQueryResponse)
async def agent_chat(
    query: str = Form(...)
):
    """
    Text-only Chatbot endpoint for SatQuery AI Agent.
    Evaluates query against domain guard and responds to remote-sensing knowledge questions or redirects out-of-domain queries.
    """
    logger.info(f"API Request /agent/chat -> query: '{query}'")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    return await agent_controller.process_text_chat(query=query)


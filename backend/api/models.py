from fastapi import APIRouter

router = APIRouter()

@router.get("/models/status")
async def models_status():
    return {
        "remoteclip": {"loaded": False, "status": "idle"},
        "blip2": {"loaded": False, "status": "idle"},
        "change_detection": {"loaded": False, "status": "idle"}
    }

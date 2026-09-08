from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.api import health, models, analyze
from backend.utils.exceptions import SatQueryException
from backend.utils.logger import logger

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Agentic Vision-Language Assistant for Remote Sensing Image Analysis"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(SatQueryException)
async def satquery_exception_handler(request: Request, exc: SatQueryException):
    logger.error(f"SatQueryException [{exc.code}]: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error_code": exc.code,
            "message": exc.message
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    from backend.services.llm_gateway import GENERIC_ERROR_MESSAGE, llm_gateway
    llm_gateway.handle_exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "TEMPORARILY_UNAVAILABLE",
            "message": GENERIC_ERROR_MESSAGE
        }
    )

# Include API Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])

# Serve generated outputs and UI static files
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

ui_dir = settings.BASE_DIR / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

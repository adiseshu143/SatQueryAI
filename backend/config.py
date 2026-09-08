import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "SatQuery AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Hardware & Performance
    DEVICE: str = os.getenv("DEVICE", "auto")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
    ALLOWED_FORMATS: list[str] = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "models" / "checkpoints"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    TEMP_DIR: Path = OUTPUT_DIR / "temporary"
    MASK_DIR: Path = OUTPUT_DIR / "masks"
    OVERLAY_DIR: Path = OUTPUT_DIR / "overlays"
    DATA_DEMO_DIR: Path = BASE_DIR / "data" / "demo"
    CONFIG_DIR: Path = BASE_DIR / "backend" / "config"
    SYSTEM_PROMPT_PATH: Path = CONFIG_DIR / "system_prompt.txt"
    
    # Thresholds
    REGISTRATION_MIN_MATCHES: int = 10
    CHANGE_THRESHOLD_DEFAULT: float = 0.15

settings = Settings()

# Ensure directories exist
for path in [settings.MODEL_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR, settings.MASK_DIR, settings.OVERLAY_DIR, settings.DATA_DEMO_DIR, settings.CONFIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def get_system_prompt() -> str:
    """Loads the verbatim system prompt configuration."""
    if settings.SYSTEM_PROMPT_PATH.exists():
        return settings.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "You are SatQuery AI, a domain-specialist assistant for remote sensing and satellite imagery analysis."


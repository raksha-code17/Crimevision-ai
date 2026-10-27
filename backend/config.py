"""Central configuration: paths, model names, thresholds and API keys."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))

    DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", DATA_DIR / "uploads"))
    SESSION_FOLDER = Path(os.getenv("SESSION_FOLDER", DATA_DIR / "sessions"))
    REPORT_FOLDER = Path(os.getenv("REPORT_FOLDER", DATA_DIR / "reports"))
    FRONTEND_DIR = ROOT_DIR / "frontend"

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if o.strip()
    ]

    ALLOWED_EXTENSIONS = {
        "image": {"jpg", "jpeg", "png", "bmp", "webp"},
        "audio": {"wav", "mp3", "m4a", "flac", "ogg"},
        "video": {"mp4", "avi", "mov", "mkv", "webm"},
        "transcript": {"txt", "srt", "vtt"},
    }

    # LLM (Groq): low temperature keeps answers deterministic
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.15"))

    # Groq vision (images are sent to Groq's cloud when a key is set and CLOUD_VISION=true).
    # Models are tried in order; check https://console.groq.com/docs/vision for current names.
    CLOUD_VISION = _bool("CLOUD_VISION", True)
    GROQ_VISION_MODELS = os.getenv(
        "GROQ_VISION_MODELS",
        "qwen/qwen3.8-27b,qwen/qwen3.6-27b,meta-llama/llama-4-scout-17b-16e-instruct",
    )

    # Models
    USE_ML_MODELS = _bool("USE_ML_MODELS", True)
    ENABLE_CAPTIONING = _bool("ENABLE_CAPTIONING", False)
    RTDETR_MODEL = os.getenv("RTDETR_MODEL", "PekingU/rtdetr_r50vd")
    CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-base")
    CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.4"))
    FORENSIC_TAG_THRESHOLD = float(os.getenv("FORENSIC_TAG_THRESHOLD", "0.3"))

    # Video
    VIDEO_MAX_SECONDS = int(os.getenv("VIDEO_MAX_SECONDS", "300"))
    VIDEO_MAX_SIDE = int(os.getenv("VIDEO_MAX_SIDE", "640"))
    VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_BATCH_SIZE", "8"))
    VIDEO_KEYFRAMES = max(1, min(5, int(os.getenv("VIDEO_KEYFRAMES", "5"))))  # frames sent to CLIP / vision model

    # Fusion
    CORROBORATION_WINDOW_SEC = float(os.getenv("CORROBORATION_WINDOW_SEC", "6"))

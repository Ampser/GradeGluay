import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAX_CONTENT_LENGTH = 8 * 1024 * 1024
DEFAULT_ALLOWED_EXTENSIONS = ".jpg,.jpeg,.png"


def csv_env_set(name, default):
    return {
        value if value.startswith(".") else f".{value}"
        for value in (
            item.strip().lower()
            for item in os.getenv(name, default).split(",")
        )
        if value
    }


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "uploads"))
    ANNOTATED_UPLOAD_FOLDER = Path(
        os.getenv("ANNOTATED_UPLOAD_FOLDER", BASE_DIR / "static" / "uploads")
    )
    SALES_HISTORY_PATH = Path(
        os.getenv("SALES_HISTORY_PATH", DATA_DIR / "sales_history.csv")
    )

    MAX_CONTENT_LENGTH = int(
        os.getenv("MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LENGTH)
    )
    ALLOWED_EXTENSIONS = csv_env_set(
        "ALLOWED_EXTENSIONS",
        DEFAULT_ALLOWED_EXTENSIONS,
    )

    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day;50 per hour")
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

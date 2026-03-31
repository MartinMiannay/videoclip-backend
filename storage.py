import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CLIPS_DIR = Path("/workspace/clips")
APP_NAME = "local"  # kept for import compatibility


def init_storage():
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Local clip storage ready at %s", CLIPS_DIR)


def put_file(src_path: str, storage_path: str) -> None:
    """Copy a rendered clip into the clips directory without loading it into memory."""
    dest = CLIPS_DIR / storage_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    logger.info("Saved clip to %s (%d bytes)", dest, dest.stat().st_size)


def get_clip_path(storage_path: str) -> Path:
    return CLIPS_DIR / storage_path

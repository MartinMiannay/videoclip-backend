from fastapi import FastAPI, APIRouter, UploadFile, File, Response, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import uuid
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)
_log_file = Path(__file__).parent / "server.log"
_log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Always add our file handler directly — basicConfig is a no-op if uvicorn
# already configured the root logger (which it does before importing the app).
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(_log_fmt)
logging.getLogger().addHandler(_fh)
logging.getLogger().setLevel(logging.INFO)

from storage import init_storage, put_object, get_object, APP_NAME


# --- Models ---
class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: str = "uploaded"
    storage_path: str = ""
    original_filename: str = ""
    content_type: str = ""
    duration: float = 0.0
    thumbnail: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_step: str = ""
    processing_progress: float = 0.0
    processing_details: str = ""
    transcript: str = ""
    transcript_words: List[dict] = []
    reference_clips: str = ""
    short_clips: List[dict] = []


# --- Startup ---
@app.on_event("startup")
async def startup():
    import shutil
    import sys
    IS_WINDOWS = sys.platform == "win32"

    version_file = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    try:
        with open(version_file) as _vf:
            version = _vf.read().strip()
    except FileNotFoundError:
        version = "UNKNOWN (VERSION.txt missing)"
    logger.info("=== SERVER VERSION: %s ===", version)

    try:
        init_storage()
        logger.info("Object storage initialized")
    except Exception as e:
        logger.error(f"Storage init failed: {e}")

    # Auto-install ffmpeg — Linux/container only (on Windows install manually)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        if IS_WINDOWS:
            logger.error(
                "FFmpeg not found. Install it from https://ffmpeg.org/download.html "
                "and add it to PATH, then restart the server."
            )
        else:
            logger.warning("FFmpeg not found — installing...")
            proc = await asyncio.create_subprocess_shell(
                "apt-get update -qq && apt-get install -y -qq ffmpeg",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info("FFmpeg installed successfully")
            else:
                logger.error(f"FFmpeg install failed: {stderr.decode(errors='replace')[-200:]}")
    else:
        logger.info(f"FFmpeg found: {shutil.which('ffmpeg')}")

    # Auto-install Montserrat font — Linux/container only
    montserrat_bold = "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"
    montserrat_extrabold = "/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"
    if IS_WINDOWS:
        # On Windows use a system font; update FONT_PATH in video_processor.py if needed
        logger.info("Windows: skipping Montserrat install — using system font fallback")
    elif not os.path.isfile(montserrat_bold) or not os.path.isfile(montserrat_extrabold):
        logger.warning("Montserrat font not found — installing...")
        proc = await asyncio.create_subprocess_shell(
            "mkdir -p /usr/share/fonts/truetype/montserrat && "
            "curl -sL 'https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf' "
            "-o /usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf && "
            "curl -sL 'https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf' "
            "-o /usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf && "
            "fc-cache -f",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("Montserrat font installed successfully")
        else:
            logger.error(f"Montserrat install failed: {stderr.decode(errors='replace')[-200:]}")
    else:
        logger.info("Montserrat font found")

    # Ensure CTA outro file exists
    cta_path = ROOT_DIR / "assets" / "cta_outro.mov"
    if not cta_path.is_file():
        logger.warning("CTA outro not found — downloading...")
        (ROOT_DIR / "assets").mkdir(exist_ok=True)
        import subprocess as _subprocess
        result = await asyncio.to_thread(
            _subprocess.run,
            ["curl", "-sL", "https://customer-assets.emergentagent.com/job_quick-render-5/artifacts/onn8ha3m_3B0B2CAA-7666-47B0-BA3F-5E63B24DCE67.mov", "-o", str(cta_path)],
            capture_output=True,
        )
        if result.returncode == 0 and cta_path.is_file():
            logger.info(f"CTA outro downloaded ({cta_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            logger.error(f"CTA download failed: {result.stderr.decode(errors='replace')[-200:]}")
    else:
        logger.info("CTA outro found")

    # Pre-encode CTA to match clip output format (fast copy on append)
    from video_processor import preencode_cta
    await asyncio.to_thread(preencode_cta)


import tempfile as _tempfile
UPLOAD_DIR = Path(_tempfile.gettempdir()) / "marvin_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class ChunkUploadInit(BaseModel):
    filename: str
    total_chunks: int
    content_type: str = "video/mp4"
    file_size: int = 0


# --- API Endpoints ---

@api_router.get("/")
async def root():
    return {"message": "Marvin Bot API"}


@api_router.post("/upload/init")
async def upload_init(data: ChunkUploadInit):
    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "upload_id": upload_id,
        "filename": data.filename,
        "total_chunks": data.total_chunks,
        "content_type": data.content_type,
        "file_size": data.file_size,
        "received_chunks": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.upload_sessions.insert_one(session)
    return {"upload_id": upload_id, "total_chunks": data.total_chunks}


@api_router.post("/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, file: UploadFile = File(...)):
    """Save chunk directly to local disk — fast, single hop."""
    upload_dir = UPLOAD_DIR / upload_id
    if not upload_dir.exists():
        raise HTTPException(404, "Upload session not found")

    chunk_path = upload_dir / f"chunk_{chunk_index:06d}"
    with open(chunk_path, "wb") as f:
        while block := await file.read(1024 * 1024):
            f.write(block)

    await db.upload_sessions.update_one(
        {"upload_id": upload_id},
        {"$addToSet": {"received_chunks": chunk_index}}
    )

    session = await db.upload_sessions.find_one({"upload_id": upload_id})
    received = len(session.get("received_chunks", []))
    total = session.get("total_chunks", 1)

    return {"chunk_index": chunk_index, "received": received, "total": total}


@api_router.post("/upload/complete/{upload_id}")
async def upload_complete(upload_id: str):
    """Assemble chunks into a single file on disk. No Object Storage for raw video."""
    session = await db.upload_sessions.find_one({"upload_id": upload_id})
    if not session:
        raise HTTPException(404, "Upload session not found")

    received = set(session.get("received_chunks", []))
    total = session["total_chunks"]
    if len(received) < total:
        raise HTTPException(400, f"Missing chunks: received {len(received)}/{total}")

    filename = session["filename"]
    ext = filename.split(".")[-1] if "." in filename else "mp4"
    upload_dir = UPLOAD_DIR / upload_id

    # Assemble chunks → single file
    assembled_path = upload_dir / f"video.{ext}"
    with open(assembled_path, "wb") as out:
        for i in range(total):
            chunk_path = upload_dir / f"chunk_{i:06d}"
            with open(chunk_path, "rb") as cf:
                while block := cf.read(8 * 1024 * 1024):
                    out.write(block)
            chunk_path.unlink()  # delete chunk immediately after copying

    project_id = str(uuid.uuid4())

    project = Project(
        id=project_id,
        name=filename.rsplit(".", 1)[0] if "." in filename else filename,
        status="uploaded",
        storage_path="",
        original_filename=filename,
        content_type=session.get("content_type", "video/mp4"),
    )
    doc = project.model_dump()
    doc["local_video_path"] = str(assembled_path)
    await db.projects.insert_one(doc)

    await db.upload_sessions.delete_one({"upload_id": upload_id})

    result = await db.projects.find_one({"id": project_id}, {"_id": 0})
    result.pop("local_video_path", None)
    return result


@api_router.get("/projects")
async def list_projects():
    projects = await db.projects.find(
        {}, {"_id": 0, "transcript_words": 0, "transcript": 0}
    ).sort("created_at", -1).to_list(100)
    return projects


@api_router.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = await db.projects.find_one(
        {"id": project_id}, {"_id": 0, "transcript_words": 0}
    )
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    result = await db.projects.delete_one({"id": project_id})
    return {"deleted": result.deleted_count > 0}


@api_router.post("/projects/{project_id}/process")
async def start_processing(project_id: str):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")
    if project.get("status") == "processing":
        return {"message": "Already processing"}

    # Check if local video file still exists (gets wiped on pod restart)
    local_path = project.get("local_video_path", "")
    if local_path and not os.path.isfile(local_path):
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": "error", "processing_step": "error", "processing_progress": 0,
                      "processing_details": "Video file was lost due to a server restart. Please re-upload your video."}}
        )
        raise HTTPException(410, "Video file was lost due to a server restart. Please re-upload your video.")

    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"status": "processing", "processing_step": "starting", "processing_progress": 0, "processing_details": "Starting pipeline..."}}
    )

    from video_processor import process_video_pipeline
    asyncio.create_task(process_video_pipeline(project_id, db))

    return {"message": "Processing started", "project_id": project_id}


@api_router.post("/projects/{project_id}/update-references")
async def update_references(project_id: str, data: dict):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"reference_clips": data.get("reference_clips", "")}}
    )
    return {"ok": True}


@api_router.post("/projects/{project_id}/retry-failed")
async def retry_failed_clips(project_id: str):
    """Re-process only the clips that failed (status='error').
    Resets them to 'pending' and re-runs the editing pipeline."""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")
    if project.get("status") == "processing":
        return {"message": "Already processing"}

    clips = project.get("short_clips", [])
    error_clips = [c for c in clips if c.get("status") == "error"]
    if not error_clips:
        return {"message": "No failed clips to retry", "error_count": 0}

    # Reset error clips to pending
    for c in clips:
        if c.get("status") == "error":
            c["status"] = "pending"
            c["error"] = ""

    await db.projects.update_one(
        {"id": project_id},
        {"$set": {
            "short_clips": clips,
            "status": "processing",
            "processing_step": "retrying",
            "processing_progress": 0,
            "processing_details": f"Retrying {len(error_clips)} failed clips..."
        }}
    )

    from video_processor import process_video_pipeline
    asyncio.create_task(process_video_pipeline(project_id, db))

    return {"message": f"Retrying {len(error_clips)} failed clips", "error_count": len(error_clips)}


@api_router.get("/clips/{project_id}/{clip_id}/download")
async def download_clip(project_id: str, clip_id: str):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0, "transcript_words": 0, "transcript": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    clip = next((c for c in project.get("short_clips", []) if c.get("id") == clip_id), None)
    if not clip or not clip.get("storage_path"):
        raise HTTPException(404, "Clip not found or not ready")

    data, content_type = get_object(clip["storage_path"])
    # Sanitize filename — HTTP headers only support ASCII
    import re
    safe_caption = re.sub(r'[^\w\s-]', '', clip.get('caption', 'untitled')[:30]).strip().replace(' ', '_')
    filename = f"clip_{safe_caption or 'untitled'}.mp4"
    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

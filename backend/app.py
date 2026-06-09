from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import os
import shutil
import uuid

from preprocess import backend_preprocess
from inference import predict_emotions_for_chunks

app = FastAPI(title="Emotion Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class AudioPathRequest(BaseModel):
    audio_path: str
    audio_id: Optional[str] = None


@app.get("/")
def root():
    return {
        "success": True,
        "status": "running",
        "message": "Emotion Backend is running"
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


def analyze_audio_file(audio_path: Path, audio_id: Optional[str] = None) -> dict:
    """
    Common analysis function used by both:
    1) /analyze-audio-file  -> Unreal sends the actual WAV file
    2) /analyze-audio-path  -> local testing with an existing path
    """

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    file_size = os.path.getsize(audio_path)

    prep_result = backend_preprocess(str(audio_path))

    if not prep_result.ok:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "status": "error",
                "audio_id": audio_id or "123",
                "file_name": audio_path.name,
                "file_size": file_size,
                "message": prep_result.message,
                "error_code": prep_result.error_code,
                "preprocessing": prep_result.payload,
                "chunks": []
            }
        )

    payload = prep_result.payload or {}
    chunk_paths = payload.get("debug_wav_paths", [])
    chunk_timeline = payload.get("chunk_timeline", [])

    if not chunk_paths:
        raise HTTPException(
            status_code=500,
            detail="No chunk files were produced. Check preprocess CONFIG['debug_store']; it must be True."
        )

    chunks = predict_emotions_for_chunks(chunk_paths, chunk_timeline)

    return {
        "success": True,
        "status": "success",
        "audio_id": audio_id or "123",
        "file_name": audio_path.name,
        "file_size": file_size,
        "duration": round(float(payload.get("original_duration_s", 0.0)), 2),
        "sample_rate": int(payload.get("final_sr", 16000)),
        "num_chunks": len(chunks),
        "chunks": chunks,
        "message": "Audio analyzed successfully"
    }


@app.post("/analyze-audio-file")
async def analyze_audio_file_upload(
    file: UploadFile = File(...),
    audio_id: Optional[str] = Form(None)
):
    """
    Main endpoint for Unreal Engine.
    Unreal sends the WAV file as multipart/form-data with field name: file
    """

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    safe_name = Path(file.filename).name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    saved_path = UPLOAD_DIR / unique_name

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        file.file.close()

    return analyze_audio_file(saved_path, audio_id)


@app.post("/analyze-audio-path")
async def analyze_audio_path(data: AudioPathRequest):
    """
    Optional endpoint for local tests when you already have a WAV path on disk.
    """
    audio_path = Path(data.audio_path.strip())
    return analyze_audio_file(audio_path, data.audio_id)


# Backward-compatible route: your old endpoint name still works.
@app.post("/analyze-audio")
async def analyze_audio_legacy(data: AudioPathRequest):
    audio_path = Path(data.audio_path.strip())
    return analyze_audio_file(audio_path, data.audio_id)

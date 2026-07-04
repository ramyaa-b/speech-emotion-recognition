"""
FastAPI backend for the emotion recognition model.

Run directly:
    uvicorn backend.main:app --reload

Or via Docker (see Dockerfile.backend / docker-compose.yml at repo root).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend import inference
from backend.schemas import HealthResponse, PredictionResponse

ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/x-flac", "audio/m4a", "audio/mp4",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on the first request — keeps
    # request latency consistent and surfaces load failures immediately
    # instead of on someone's first prediction.
    inference.load_artifacts()
    yield


app = FastAPI(
    title="Speech Emotion Recognition API",
    description="CNN + BiLSTM + attention model trained on RAVDESS with a speaker-independent split.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=inference.is_loaded())


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. "
            f"Expected one of: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = inference.predict_emotion(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {e}")

    return PredictionResponse(**result)

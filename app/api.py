"""FastAPI serving layer for MedASR.

Exposes the trained model behind a small REST API so a dictation front-end can
POST an audio clip and receive a transcript. This is a reference deployment,
not a production PHI-handling service — for real clinical use you must add
authentication, audit logging, encryption at rest/in transit, and a BAA-covered
hosting environment.

Run::

    export MEDASR_CHECKPOINT=runs/.../best.pt
    export MEDASR_TOKENIZER=artifacts/tokenizer.json
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import torch

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional serve extra
    raise ImportError("Install serving deps with `pip install medasr[serve]`.") from exc

from medasr.inference import Transcriber

app = FastAPI(title="MedASR", version="0.1.0", description="Clinical dictation ASR")


class TranscriptionResponse(BaseModel):
    text: str
    model_version: str = "0.1.0"


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    checkpoint = os.environ.get("MEDASR_CHECKPOINT")
    tokenizer = os.environ.get("MEDASR_TOKENIZER")
    if not checkpoint or not tokenizer:
        raise RuntimeError(
            "Set MEDASR_CHECKPOINT and MEDASR_TOKENIZER environment variables."
        )
    device = os.environ.get("MEDASR_DEVICE", "cpu")
    return Transcriber.from_checkpoint(checkpoint, tokenizer, device=device)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Accept a WAV/FLAC upload and return the transcript."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "soundfile not installed on server") from exc

    raw = await file.read()
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc

    transcriber = get_transcriber()
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != transcriber.feature_extractor.sample_rate:
        raise HTTPException(
            400,
            f"Expected {transcriber.feature_extractor.sample_rate} Hz audio, got {sr} Hz.",
        )
    text = transcriber.transcribe_waveform(wav)
    return TranscriptionResponse(text=text)

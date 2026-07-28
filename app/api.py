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
from typing import Dict, List, Optional

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


class WordResponse(BaseModel):
    text: str
    start_s: float
    end_s: float
    confidence: float


class DetailedResponse(BaseModel):
    """Everything a dictation UI needs to render and gate a transcript."""

    text: str
    confidence: float
    needs_review: bool
    review_reasons: List[str] = []
    words: List[WordResponse] = []
    corrections: List[Dict[str, str]] = []
    phi_found: Optional[Dict[str, int]] = None
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


async def _read_waveform(file: UploadFile) -> torch.Tensor:
    """Decode an upload to a mono tensor, turning bad input into 4xx not 5xx."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "soundfile not installed on server") from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc

    transcriber = get_transcriber()
    expected = transcriber.feature_extractor.sample_rate
    if sr != expected:
        raise HTTPException(400, f"Expected {expected} Hz audio, got {sr} Hz.")
    return torch.from_numpy(data).mean(dim=1)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Accept a WAV/FLAC upload and return the transcript."""
    wav = await _read_waveform(file)
    return TranscriptionResponse(text=get_transcriber().transcribe_waveform(wav))


@app.post("/transcribe/detailed", response_model=DetailedResponse)
async def transcribe_detailed(
    file: UploadFile = File(...),
    redact_phi: bool = False,
) -> DetailedResponse:
    """Transcribe with word timings, confidence, and clinical safety gating.

    ``needs_review`` is the field a dictation UI should act on: when true the
    transcript must not be auto-committed to the chart. ``review_reasons``
    explains why, so the interface can highlight the specific words.

    Set ``redact_phi=true`` for any consumer that is not the treating
    clinician — logs, analytics, model-improvement corpora.
    """
    wav = await _read_waveform(file)
    result = get_transcriber().transcribe_detailed(wav, redact_phi=redact_phi)

    return DetailedResponse(
        text=result.text,
        confidence=(
            result.confidence.utterance_confidence if result.confidence else 0.0
        ),
        needs_review=result.needs_review,
        review_reasons=result.review_reasons,
        words=[
            WordResponse(
                text=w.text, start_s=w.start_s, end_s=w.end_s, confidence=w.score
            )
            for w in result.words
        ],
        corrections=[
            {"original": c.original, "corrected": c.corrected}
            for c in result.corrections
        ],
        # Only ever the per-kind counts, never the identifiers themselves.
        phi_found=result.phi.counts if result.phi else None,
    )

"""Generate a tiny *synthetic* clinical-dictation corpus.

Real medical speech data is protected health information and cannot ship in a
public repo. To make the whole pipeline runnable — and CI reproducible — this
script fabricates short WAV files by rendering each transcript as a sequence of
distinct tones (one pseudo-"phoneme" per character). It is **not** a substitute
for real audio; the model cannot learn meaningful acoustics from it. Its only
purpose is to exercise data-loading, training, decoding, and evaluation end to
end on a machine with no dataset.

Usage::

    python -m medasr.cli.synth_data --out-dir data/synth --num 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import wave
from pathlib import Path

# A handful of realistic clinical dictation snippets.
SENTENCES = [
    "patient denies chest pain or shortness of breath",
    "start metformin 500 mg twice daily",
    "blood pressure is 130 over 85 today",
    "no known drug allergies",
    "prescribe amoxicillin 500 mg three times a day for ten days",
    "the patient reports mild intermittent headache",
    "continue lisinopril 10 mg once daily",
    "follow up in two weeks for reassessment",
    "auscultation reveals clear breath sounds bilaterally",
    "increase atorvastatin to 40 mg at bedtime",
]


def _char_to_freq(ch: str) -> float:
    # Map each character to a stable pitch in a speech-like band.
    base = 200.0
    return base + (ord(ch) % 32) * 40.0


def render_waveform(text: str, sample_rate: int = 16000) -> list[float]:
    samples: list[float] = []
    dur = 0.06  # seconds per character
    for ch in text:
        freq = _char_to_freq(ch) if ch != " " else 0.0
        n = int(sample_rate * dur)
        for i in range(n):
            t = i / sample_rate
            samples.append(0.3 * math.sin(2 * math.pi * freq * t))
    return samples


def write_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        wf.writeframes(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic ASR corpus.")
    parser.add_argument("--out-dir", default="data/synth")
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    wav_dir = out / "wavs"

    rows = []
    for i in range(args.num):
        text = random.choice(SENTENCES)
        wav_path = wav_dir / f"utt_{i:04d}.wav"
        samples = render_waveform(text, args.sample_rate)
        write_wav(wav_path, samples, args.sample_rate)
        rows.append(
            {
                "audio_filepath": str(wav_path),
                "duration": round(len(samples) / args.sample_rate, 2),
                "text": text,
            }
        )

    # 80/20 train/val split.
    split = int(0.8 * len(rows))
    for name, subset in [("train", rows[:split]), ("val", rows[split:])]:
        manifest = out / f"{name}.jsonl"
        with open(manifest, "w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {manifest} ({len(subset)} utterances)")


if __name__ == "__main__":
    main()

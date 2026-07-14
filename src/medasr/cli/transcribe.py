"""Transcribe one or more audio files with a trained checkpoint.

Usage::

    python -m medasr.cli.transcribe \
        --checkpoint runs/.../best.pt \
        --tokenizer artifacts/tokenizer.json \
        audio1.wav audio2.wav
"""

from __future__ import annotations

import argparse

from medasr.inference import Transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("audio", nargs="+", help="paths to audio files")
    args = parser.parse_args()

    transcriber = Transcriber.from_checkpoint(
        args.checkpoint, args.tokenizer, device=args.device
    )
    for path in args.audio:
        text = transcriber.transcribe_file(path)
        print(f"{path}\t{text}")


if __name__ == "__main__":
    main()

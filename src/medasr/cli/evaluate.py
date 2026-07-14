"""Evaluate a checkpoint on a manifest, reporting WER and CER.

Usage::

    python -m medasr.cli.evaluate \
        --checkpoint runs/.../best.pt \
        --tokenizer artifacts/tokenizer.json \
        --manifest data/synth/val.jsonl
"""

from __future__ import annotations

import argparse
import json

from medasr.inference import Transcriber
from medasr.metrics import cer, wer
from medasr.text import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WER/CER on a manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", type=int, default=5, help="print N examples")
    args = parser.parse_args()

    transcriber = Transcriber.from_checkpoint(
        args.checkpoint, args.tokenizer, device=args.device
    )
    lowercase = transcriber.cfg.tokenizer.lowercase

    refs, hyps = [], []
    with open(args.manifest, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    for i, row in enumerate(rows):
        hyp = transcriber.transcribe_file(row["audio_filepath"])
        ref = normalize(row["text"], lowercase=lowercase)
        refs.append(ref)
        hyps.append(hyp)
        if i < args.show:
            print(f"REF: {ref}\nHYP: {hyp}\n")

    print(f"Utterances: {len(refs)}")
    print(f"WER: {wer(refs, hyps) * 100:.2f}%")
    print(f"CER: {cer(refs, hyps) * 100:.2f}%")


if __name__ == "__main__":
    main()

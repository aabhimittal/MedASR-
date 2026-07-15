"""Build a character tokenizer from one or more manifests.

The vocabulary must be fixed *before* training (it sets the size of the CTC
output layer) and reused unchanged at inference. This script scans the training
transcripts, applies the same normalisation the dataset uses, and writes a
deterministic ``tokenizer.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medasr.text import normalize
from medasr.tokenizer import CharTokenizer


def read_texts(manifest: str, lowercase: bool):
    with open(manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield normalize(row["text"], lowercase=lowercase)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a char tokenizer.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out", default="artifacts/tokenizer.json")
    parser.add_argument("--no-lowercase", action="store_true")
    args = parser.parse_args()

    lowercase = not args.no_lowercase
    texts = []
    for m in args.manifests:
        texts.extend(read_texts(m, lowercase))

    tokenizer = CharTokenizer.build(texts)
    tokenizer.save(args.out)
    print(f"Built char tokenizer with {tokenizer.vocab_size} tokens -> {args.out}")
    print(f"Vocabulary: {tokenizer.itos}")


if __name__ == "__main__":
    main()

"""Tokenizers: convert normalised text <-> integer id sequences.

A CTC model predicts, at every acoustic frame, a distribution over a fixed
**vocabulary** of output symbols plus a special ``<blank>``. The tokenizer
defines that vocabulary and the two mappings we need:

* ``encode(text) -> List[int]``   (used to build training targets)
* ``decode(List[int]) -> text``   (used to read model predictions)

Index conventions (shared by the CTC loss and the decoders):

* ``0`` is always the CTC **blank** token. It is *not* a real character — it
  is the "emit nothing / repeat" symbol that lets CTC align variable-length
  audio to shorter text.
* Real symbols occupy ids ``1 .. vocab_size-1``.

Two implementations are provided:

* :class:`CharTokenizer` — a transparent character vocabulary. Great for a
  first medical model: no out-of-vocabulary problem, tiny vocab, and it can
  spell any drug name it has heard the letters of.
* :class:`BPETokenizer` — a thin wrapper over SentencePiece for sub-word
  units, which shortens targets and speeds up training on larger corpora.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


class BaseTokenizer:
    """Common interface. ``id 0`` is reserved for the CTC blank."""

    blank_id: int = 0

    @property
    def vocab_size(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """Character-level vocabulary with a reserved blank at index 0.

    The vocabulary is a list of single-character strings. ``<blank>`` and
    ``<unk>`` are special multi-character sentinels that never collide with a
    real character.
    """

    def __init__(self, vocab: Sequence[str]):
        if vocab[0] != BLANK_TOKEN:
            raise ValueError("Index 0 of the vocabulary must be the blank token.")
        self.itos: List[str] = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_id = self.stoi.get(UNK_TOKEN)

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, texts: Iterable[str], add_unk: bool = True) -> "CharTokenizer":
        """Build a character vocabulary from a corpus.

        Characters are sorted for determinism so the same corpus always yields
        the same id mapping (critical for resuming training / loading models).
        """
        chars = set()
        for line in texts:
            chars.update(line)
        vocab = [BLANK_TOKEN]
        if add_unk:
            vocab.append(UNK_TOKEN)
        vocab.extend(sorted(chars))
        return cls(vocab)

    # -- mappings ------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        out = []
        for ch in text:
            idx = self.stoi.get(ch, self.unk_id)
            if idx is None:
                # No <unk> in vocab and char unseen -> skip it.
                continue
            out.append(idx)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        specials = {self.blank_id}
        if self.unk_id is not None:
            specials.add(self.unk_id)
        return "".join(self.itos[i] for i in ids if i not in specials)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "char", "vocab": self.itos}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "char":
            raise ValueError(f"Not a char tokenizer file: {path}")
        return cls(payload["vocab"])


class BPETokenizer(BaseTokenizer):
    """SentencePiece sub-word tokenizer (optional dependency).

    Piece id 0 is remapped to the CTC blank; SentencePiece ids are shifted by
    one so real pieces occupy ``1 .. vocab_size-1``.
    """

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "BPETokenizer requires sentencepiece. Install with "
                "`pip install medasr[bpe]`."
            ) from exc
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size() + 1  # +1 for the blank at index 0

    def encode(self, text: str) -> List[int]:
        return [pid + 1 for pid in self._sp.encode(text, out_type=int)]

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [i - 1 for i in ids if i != self.blank_id]
        return self._sp.decode(pieces)


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """Load whichever tokenizer type was saved at ``path``."""
    path = Path(path)
    if path.suffix == ".model":
        return BPETokenizer(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "char":
        return CharTokenizer(payload["vocab"])
    raise ValueError(f"Unrecognised tokenizer file: {path}")

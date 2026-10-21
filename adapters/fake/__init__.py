"""Reference fake adapter: CPU-only, deterministic, no downloads (issue #19).

Implements the full four-operation contract with an in-repo deterministic
tokenizer and a fake engine that decodes fixed token sequences, so CI can
test adapter boundaries — selection, token accounting, synchronisation,
cleanup — without a GPU or any engine dependency.

tests/test_fake_adapter.py also subclasses this into a deliberately cheating
adapter (prompt tokens leaking into the denominator) and asserts that the
RUNNER-side cross-check catches it: that is the point of the
runner-authoritative rule, and it needs no GPU to prove.
"""
from __future__ import annotations

from ..base import AdapterBase, GenerationResult


class FakeTokenizer:
    """Deterministic whitespace-word tokenizer. encode/decode are exact
    inverses, so re-tokenization recounts are exact (acceptance A1/A2 without
    round-trip fuzz). Satisfies the minimal canonical-tokenizer interface
    used by adapters.verify."""

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab)
        return self._vocab[word]

    def encode(self, text: str) -> list[int]:
        return [self._id(w) for w in text.split()]

    def decode(self, ids) -> str:
        rev = {i: w for w, i in self._vocab.items()}
        return " ".join(rev[i] for i in ids)

    def count(self, text: str) -> int:
        return len(text.split())


class FakeAdapter(AdapterBase):
    """Deterministic fake engine for contract tests (registry name: 'fake')."""

    engine_name = "fake"
    engine_version = "1"

    SUPPORTED_PRECISIONS = ("FP16", "NF4", "INT8")

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.loaded = False
        self.closed = False
        self.sync_count = 0
        self._model_name = None
        self._precision = None

    def load(self, model_name: str, precision: str):
        if precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"fake engine supports {self.SUPPORTED_PRECISIONS}, "
                f"got {precision!r}")
        self._model_name = model_name
        self._precision = precision
        self.loaded = True
        self.closed = False
        return self

    def generate(self, prompts, tokens: int) -> GenerationResult:
        if not self.loaded:
            raise RuntimeError("generate() called before load()")
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        texts = []
        for prompt in prompts:
            words = prompt.split() or ["<s>"]
            # deterministic "engine output": the last prompt word, then fixed
            # markers — `tokens` words in total per sequence
            out = [words[-1]] + ["<gen>"] * (tokens - 1)
            texts.append(" ".join(out))
        return GenerationResult(texts, tokens * len(prompts))

    def synchronize(self):
        self.sync_count += 1

    def close(self):
        self.closed = True
        self.loaded = False

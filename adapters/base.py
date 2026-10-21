"""The adapter contract (docs/backends.md, 'Keep inference separate from
measurement' + 'Token counts are an acceptance condition').

Four operations. Two rules:

1. Fail clearly. load() raises ValueError on an unsupported format; it never
   silently substitutes another one.
2. The runner is authoritative over token counts. generate() must return
   BOTH the generated texts and the engine's own count of new tokens; the
   runner cross-checks by re-tokenizing the texts with the canonical
   tokenizer (adapters/verify.py, acceptance A1/A2). An engine whose counts
   cannot survive that check fails acceptance — structurally, not on trust.
"""
from __future__ import annotations

from typing import Sequence

# Registry of engine names (schema 1.3-r6 engine.name, docs/backends.md).
# Adding a name is a documentation change, not a schema change: the field is
# a plain string so an unregistered engine still records honestly.
KNOWN_ENGINES = ("transformers", "llama.cpp", "vllm", "tensorrt-llm", "fake")

# The reference stack. Absent engine block in a report means this engine.
REFERENCE_ENGINE = "transformers"


class GenerationResult:
    """What generate() must return.

    texts        the generated text per input prompt, in prompt order
    new_tokens   the engine's count of actually newly generated tokens summed
                 across all sequences — prompt, padding, cached and warm-up
                 tokens excluded (the runner re-checks this; see verify.py)
    """

    def __init__(self, texts: Sequence[str], new_tokens: int):
        self.texts = list(texts)
        if new_tokens < 0:
            raise ValueError("new_tokens must be >= 0")
        self.new_tokens = int(new_tokens)


class AdapterBase:
    """Abstract adapter contract. Subclass and implement the four operations."""

    #: registry name recorded in the report's engine block (schema 1.3-r6)
    engine_name: str = ""

    #: engine version recorded in the report's engine block
    engine_version: str = ""

    def load(self, model_name: str, precision: str):
        """Load the model and tokenizer, validate the requested format and
        perform any quantisation or engine preparation. Raise ValueError on
        an unsupported format — never silently substitute another one."""

    def generate(self, prompts, tokens: int) -> GenerationResult:
        """Synchronous greedy decode of `tokens` new tokens per prompt."""

    def synchronize(self):
        """Wait for ALL accelerator work submitted by this engine. A PyTorch
        CUDA synchronisation alone may not wait for a separate engine process."""

    def close(self):
        """Release model/engine resources — including after a failed load or
        generation — before the FP16 baseline is loaded."""

    # -- convenience -------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_adapter(name: str) -> AdapterBase:
    """Return an unloaded adapter by registry name; the runner then calls
    adapter.load(model_name, precision). Imports stay lazy so CPU contract
    tests and existing users work without any engine dependency installed."""
    name = str(name).strip()
    if name == "fake":
        from .fake import FakeAdapter
        return FakeAdapter()
    if name == REFERENCE_ENGINE:
        raise NotImplementedError(
            "the transformers reference path still lives in entrypoint.py; "
            "extracting it into adapters/ is the tracked follow-up (issue #19)")
    raise ValueError(
        f"unknown engine {name!r}; known engines: {', '.join(KNOWN_ENGINES)} "
        "(see docs/backends.md to register a new one)")

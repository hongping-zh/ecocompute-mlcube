"""Inference-engine adapters (schema 1.3-r6, docs/backends.md).

The adapter isolates everything engine-specific — model loading, tokenisation,
generation, synchronisation — behind four operations, so the common runner
keeps NVML lifecycle, power sampling, phase markers and report construction.

Status: the contract, the runner-side token verification and a CPU reference
implementation (adapters/fake) exist and are CI-tested. The Transformers
reference path still lives in entrypoint.py; extracting it into this package
is the tracked follow-up in issue #19.
"""
from __future__ import annotations

from .base import AdapterBase, GenerationResult, KNOWN_ENGINES, REFERENCE_ENGINE, load_adapter

__all__ = [
    "AdapterBase",
    "GenerationResult",
    "KNOWN_ENGINES",
    "REFERENCE_ENGINE",
    "load_adapter",
]

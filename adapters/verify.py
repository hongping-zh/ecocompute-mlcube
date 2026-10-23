"""Runner-authoritative token accounting (acceptance A1/A2, docs/backends.md).

The adapter reports token counts; the runner does not trust them. The
canonical tokenizer re-counts:

  A1  prompt tokens: the adapter's count must equal the canonical tokenizer's
      count exactly (integer, no tolerance)
  A2  generated tokens: re-tokenize the returned texts and compare with the
      engine's count (a tolerance may cover tokenizers whose decode/encode
      round-trip of generated-only text is not an identity)

The tokenizer only needs encode(text) -> list[int]; both canonical HF
tokenizers and adapters.fake.FakeTokenizer satisfy this interface.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenCountMismatch:
    code: str    # "prompt_token_mismatch" | "generated_token_mismatch"
    detail: str


def check_prompt_tokens(tokenizer, prompts, adapter_prompt_tokens) -> list[TokenCountMismatch]:
    """A1: exact equality between the adapter's prompt-token count and the
    canonical tokenizer's. Different tokenizers or silent truncation show up
    here as integer inequality."""
    problems: list[TokenCountMismatch] = []
    canonical = sum(len(tokenizer.encode(p)) for p in prompts)
    if int(adapter_prompt_tokens) != canonical:
        problems.append(TokenCountMismatch(
            "prompt_token_mismatch",
            f"adapter counted {adapter_prompt_tokens} prompt tokens, the "
            f"canonical tokenizer counts {canonical}"))
    return problems


def recount_generated(tokenizer, texts) -> int:
    """A2: re-tokenize the generated texts with the canonical tokenizer."""
    return sum(len(tokenizer.encode(t)) for t in texts)


def check_generated_tokens(tokenizer, texts, engine_new_tokens, tolerance: int = 0) -> list[TokenCountMismatch]:
    """A2: the energy denominator (new tokens) must survive an independent
    re-count of the texts the engine actually returned. Prompt tokens, warm-up
    tokens, padding or double counting in the engine's figure make this fail."""
    recounted = recount_generated(tokenizer, texts)
    if abs(int(engine_new_tokens) - recounted) > tolerance:
        return [TokenCountMismatch(
            "generated_token_mismatch",
            f"engine reported {engine_new_tokens} new tokens, re-tokenization "
            f"of the returned texts counts {recounted} (tolerance {tolerance})")]
    return []

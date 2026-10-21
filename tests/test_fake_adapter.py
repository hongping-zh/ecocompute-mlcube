"""Contract tests for the adapter framework (issue #19, schema 1.3-r6).

adapters/fake is the reference CPU implementation: no GPU, no engine
dependency, no downloads. The cheating tests are the point — an adapter whose
token counting drifts (prompt tokens in the denominator, inflated engine
count) must be caught by the RUNNER-side cross-check, not by trusting the
adapter (docs/backends.md, acceptance A1/A2).
"""
import json
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from adapters import AdapterBase, GenerationResult, load_adapter  # noqa: E402
from adapters import fake as fake_mod  # noqa: E402
from adapters import verify  # noqa: E402
import validate  # noqa: E402  (tools/validate.py)

SCHEMA = json.loads((REPO / "schema" / "energy.schema.json").read_text())
FIX = REPO / "tests" / "fixtures"


# --- registry / factory ---------------------------------------------------

def test_unknown_engine_fails_clearly():
    with pytest.raises(ValueError, match="unknown engine"):
        load_adapter("no-such-engine")


def test_reference_engine_not_yet_extracted():
    # honest: the transformers path still lives in entrypoint.py (issue #19)
    with pytest.raises(NotImplementedError):
        load_adapter("transformers")


def test_fake_adapter_is_an_adapter():
    a = load_adapter("fake")
    assert isinstance(a, AdapterBase)
    assert a.engine_name == "fake"


# --- the four contract operations ------------------------------------------

def test_load_rejects_unsupported_precision():
    a = load_adapter("fake")
    with pytest.raises(ValueError, match="fake engine supports"):
        a.load("fake-tiny", "GGUF-Q4_K_M")


def test_generate_returns_texts_and_exact_count():
    a = load_adapter("fake")
    a.load("fake-tiny", "NF4")
    prompts = ["alpha beta gamma", "delta epsilon"]
    r = a.generate(prompts, tokens=8)
    assert len(r.texts) == 2
    assert r.new_tokens == 16                       # 8 per sequence, batch of 2
    assert all(len(t.split()) == 8 for t in r.texts)


def test_generate_requires_load_then_close_releases():
    a = load_adapter("fake")
    with pytest.raises(RuntimeError):
        a.generate(["x"], tokens=4)
    a.load("fake-tiny", "FP16")
    a.synchronize()
    assert a.sync_count == 1
    a.close()
    assert a.closed and not a.loaded


# --- acceptance A1: prompt tokens, exact equality ---------------------------

def test_prompt_tokens_match_canonical_tokenizer():
    a = load_adapter("fake")
    a.load("fake-tiny", "NF4")
    prompts = ["the quick brown fox", "jumps over"]
    claimed = sum(a.tokenizer.count(p) for p in prompts)
    assert verify.check_prompt_tokens(a.tokenizer, prompts, claimed) == []


def test_prompt_token_mismatch_is_flagged():
    tok = fake_mod.FakeTokenizer()
    problems = verify.check_prompt_tokens(tok, ["a b c"], adapter_prompt_tokens=4)
    assert problems and problems[0].code == "prompt_token_mismatch"


# --- acceptance A2: generated tokens, runner cross-check --------------------

def test_generated_tokens_recount_matches():
    a = load_adapter("fake")
    a.load("fake-tiny", "NF4")
    r = a.generate(["one two three"], tokens=10)
    assert verify.check_generated_tokens(a.tokenizer, r.texts, r.new_tokens) == []


def test_generated_token_mismatch_flagged_with_both_counts():
    tok = fake_mod.FakeTokenizer()
    problems = verify.check_generated_tokens(tok, ["a b c"], engine_new_tokens=4)
    assert problems and problems[0].code == "generated_token_mismatch"
    assert "reported 4" in problems[0].detail
    assert "counts 3" in problems[0].detail


class CheatingPromptCountAdapter(fake_mod.FakeAdapter):
    """Deliberately dishonest: prompt tokens leak into the denominator."""

    def generate(self, prompts, tokens):
        r = super().generate(prompts, tokens)
        inflated = r.new_tokens + sum(len(p.split()) for p in prompts)
        return GenerationResult(r.texts, inflated)


def test_cheating_adapter_prompt_tokens_in_denominator_is_caught():
    """The point of the runner-authoritative rule: the adapter lies, the
    runner-side re-tokenization catches it — no GPU needed to prove it."""
    a = CheatingPromptCountAdapter()
    a.load("fake-tiny", "NF4")
    prompts = ["one two three four five"]
    r = a.generate(prompts, tokens=6)
    assert r.new_tokens == 6 + 5                     # the cheat is real...
    problems = verify.check_generated_tokens(a.tokenizer, r.texts, r.new_tokens)
    assert problems and problems[0].code == "generated_token_mismatch"  # ...and caught


# --- schema 1.3-r6: the engine block ----------------------------------------

def _base_report():
    return json.loads((FIX / "conformant-v1.1.json").read_text())


def test_engine_block_is_schema_valid():
    r = _base_report()
    r["engine"] = {
        "name": "llama.cpp", "version": "b4325", "backend": "cuda",
        "model_format": "gguf-f16",
        "tokenizer": {"source": "embedded-gguf", "verified_against_hf": True},
        "sync": "subprocess-join",
    }
    jsonschema.validate(r, SCHEMA)


def test_engine_block_requires_name_and_version():
    r = _base_report()
    r["engine"] = {"name": "llama.cpp"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, SCHEMA)


def test_absent_engine_block_still_schema_valid():
    # backward compatibility: every existing report stays valid unchanged
    jsonschema.validate(_base_report(), SCHEMA)


# --- validator: engine-relative class + relabelling ban ----------------------

def test_cross_engine_run_rejected_for_dataset_eligible():
    # The realistic cross-engine case that can pass v1.1-core today: a
    # GGUF F16 arm (a GGUF quantized arm cannot pass until the open
    # workload.precision format decision in issue #19 - see the relabelling
    # test below).
    r = _base_report()
    r["workload"]["precision"] = "FP16"
    r["engine"] = {"name": "llama.cpp", "version": "b4325",
                   "model_format": "gguf-f16"}
    verdict, viol, _ = validate.validate_report(r, SCHEMA, "dataset-eligible")
    assert verdict == "protocol-conformant"         # valid measurement...
    joined = "\n".join(viol)
    assert "engine-relative" in joined              # ...but not poolable


def test_reference_engine_engine_block_changes_nothing():
    base = validate.validate_report(_base_report(), SCHEMA, "dataset-eligible")
    r = _base_report()
    r["engine"] = {"name": "transformers", "version": "4.46.3"}
    withblock = validate.validate_report(r, SCHEMA, "dataset-eligible")
    assert base[0] == withblock[0] == "dataset-eligible"
    assert base[1] == withblock[1]                  # no new violations either


def test_relabeling_gguf_quant_as_nf4_is_a_core_violation():
    r = _base_report()
    assert r["workload"]["precision"] == "NF4"      # fixture ground truth
    r["engine"] = {"name": "llama.cpp", "version": "b4325",
                   "model_format": "gguf-q4_k_m"}
    verdict, viol, _ = validate.validate_report(r, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    assert "reference-stack labels" in joined

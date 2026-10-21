# Adding an inference backend

This is a guide for contributors addressing [issue #19](https://github.com/hongping-zh/ecocompute-mlcube/issues/19).
The current runtime uses Hugging Face Transformers and bitsandbytes. There is
no backend selector in the container yet, but the adapter **contract** now
exists in [`adapters/`](../adapters/) with a CPU reference implementation
([`adapters/fake/`](../adapters/fake/)) and CI tests
([`tests/test_fake_adapter.py`](../tests/test_fake_adapter.py)). Extracting
the Transformers path from `entrypoint.py` into that contract is the follow-up.

## Follow the current measurement path

All function names below are in [`entrypoint.py`](../entrypoint.py).

1. `run()` calls `load_params()` and `resolve_arch()`. If `gpu_available()` is
   true and `--dry_run` is absent, it calls `measure_once()` for the requested
   precision and then again for FP16 when the requested precision is not FP16.
2. `measure_once()` initialises NVML, selects device 0, reads its name and
   checks that power telemetry works before calling `_measure_with_handle()`.
   It always shuts down NVML in a `finally` block.
3. `_measure_with_handle()` loads `AutoTokenizer` and `AutoModelForCausalLM`.
   `_quant_config()` constructs a `BitsAndBytesConfig` for NF4 or INT8; FP16
   uses `torch.float16` without a quantisation configuration. Loading and
   quantisation happen **outside** the reported energy window.
4. The tokenizer creates `batch_size` identical prompts. Warm-up calls to
   `model.generate()` and a CUDA synchronisation finish before the window
   opens. Each measured iteration generates exactly `tokens` new tokens per
   sequence with greedy decoding (`min_new_tokens == max_new_tokens`).
5. A `PowerSampler` covers the measured generation loop and its final CUDA
   synchronisation. It integrates NVML power samples with the trapezoidal
   rule. Generated tokens are counted across the whole batch and all measured
   iterations; warm-up and prompt tokens are excluded.
6. After the sampler stops, the optional `perplexity()` probe evaluates the
   same loaded model. With `--power_trace`, a separate sampler records the
   whole run and phase markers, including loading and quality evaluation.
7. `build_report()` produces `energy.json`. Only an on-device measurement has
   `measurement_source: "direct-nvml"` and `results.basis: "measured"`. No-GPU,
   dry-run and failed-measurement paths use labelled dataset-derived values
   (or `basis: "unavailable"` when no reference exists).

The contract tests in [`tests/test_energy.py`](../tests/test_energy.py) cover
report construction, CLI parameters, reference fallback, quality and traces.
The current production [`Dockerfile`](../Dockerfile) installs the pinned
[`requirements.lock.txt`](../requirements.lock.txt); it does not include
llama.cpp, vLLM or TensorRT-LLM.

## Keep inference separate from measurement

Start by extracting only the Transformers-specific loading, tokenisation and
generation into an adapter. Keep NVML lifecycle, power sampling, phase markers,
report construction and reference fallback in the common runner. Preserve the
default behaviour before adding another engine.

A proposed adapter contract is:

| Operation | Responsibility |
| --- | --- |
| `load(model_name, precision)` | Load the model and tokenizer, validate the requested format and perform any quantisation or engine preparation. Fail clearly on unsupported formats. |
| `generate(prompts, tokens)` | Run a synchronous greedy decode and return the **actual total newly generated tokens**. Exclude prompt, padding and cached tokens; count every sequence. |
| `synchronize()` | Wait for all accelerator work submitted by this engine. A PyTorch CUDA synchronisation alone may not wait for a separate engine process. |
| `close()` | Release model/engine resources, including after a failed load or generation, before loading the FP16 baseline. |

The runner can use `generate()` for warm-up and discard its returned count,
then use it for measured iterations and sum the counts. Call `synchronize()`
before starting the generation sampler and after the final decode, before
stopping that sampler. Put sampler shutdown and adapter cleanup in `finally`
blocks so an engine error does not leave sampling threads running.

Quantisation belongs in `load()` because engines may load a pre-quantised
artifact or build an engine rather than quantise weights in memory. Keep
quality evaluation optional: the existing perplexity function requires a
Transformers-compatible forward pass. An adapter without one must omit the
quality result or report it as unavailable, rather than fabricate a comparable
perplexity value.

## Token counts are an acceptance condition

The contract above makes `generate()` *promise* the actual newly generated
token count. Different engines count differently — vLLM's
`usage.completion_tokens` rules differ from transformers, GGUF embedded
tokenizers differ from `tokenizer.json` in BOS/EOS handling, truncation can
happen silently — and the denominator of the core metric (millijoules per
generated token) inherits every error. Real token counting is therefore an
**acceptance condition**, not a promise:

**The runner is authoritative.** `generate()` must return both the generated
texts and the engine's own token count (see
[`adapters/base.py`](../adapters/base.py), `GenerationResult`). The runner
re-tokenizes the returned texts with the model's canonical Hugging Face
tokenizer and cross-checks the counts
([`adapters/verify.py`](../adapters/verify.py)). A mismatch is recorded as
`token_count_mismatch` and fails acceptance — correct counting is enforced
structurally, not on trust, so an adapter that miscounts is caught even if
the mistake is accidental.

Minimal acceptance checklist for a new backend PR:

| # | Acceptance | Judged by |
| --- | --- | --- |
| A1 | The same prompt yields the same prompt-token count under the adapter and the canonical HF tokenizer | exact integer equality, CI-testable |
| A2 | The energy denominator equals actual newly generated tokens (prompt, warm-up and padding excluded), cross-verified by re-tokenization | CI-testable |
| A3 | The FP16 baseline runs the same engine, version and session as the quantized arm | same-session pairing — already a Protocol MUST (§4.2.1) |
| A4 | Engine name, version, model format/artifact hash and sync point are recorded in the `engine` block | schema + validator |
| A5 | Truncation is explicit: when the model maximum clamps the requested context, the effective `context_length` is recorded | the PR #23 rule, extended to adapters |

[`adapters/fake/`](../adapters/fake/) is the reference implementation: a
deterministic CPU engine with an in-repo tokenizer — no downloads, no GPU.
[`tests/test_fake_adapter.py`](../tests/test_fake_adapter.py) exercises the
contract, including a deliberately **cheating** adapter whose prompt tokens
leak into the denominator; the test asserts the runner-side cross-check
catches it. A backend whose counting cannot pass a fake-adapter-level test
does not need a GPU to be rejected.

## Recording engine identity: the `engine` block (schema 1.3-r6)

The schema now has an optional top-level `engine` object. Absent means the
reference implementation (Hugging Face Transformers + bitsandbytes, pinned in
[`requirements.lock.txt`](../requirements.lock.txt)); every existing report
stays valid unchanged.

```json
"engine": {
  "name": "llama.cpp",
  "version": "b4325",
  "backend": "cuda",
  "model_format": "gguf-q4_k_m",
  "model_artifact": {"sha256": "..."},
  "tokenizer": {"source": "embedded-gguf", "verified_against_hf": true},
  "sync": "subprocess-join"
}
```

Design decisions (settled in [issue #19](https://github.com/hongping-zh/ecocompute-mlcube/issues/19)):

- Engine identity is **not** folded into `measurement_source`. That field
  says where the *energy number* came from (direct-NVML vs dataset-derived);
  the `engine` block says what *generated the text*. Two provenance axes, two
  fields — and a structured object, because a free-text string could never be
  filtered or grouped by machines.
- `name` values come from a small registry: `transformers`, `llama.cpp`,
  `vllm`, `tensorrt-llm`. Adding a name is a documentation change, not a
  schema change (the field is a plain string, so an unregistered engine still
  records honestly).
- **No relabelling.** `workload.precision` values NF4 and INT8 mean
  bitsandbytes quantization in the transformers reference stack. A llama.cpp
  Q4_K_M run must not set `precision: "NF4"`; it records its honest artifact
  in `engine.model_format`. The validator flags
  `engine.name != transformers` combined with `precision: NF4/INT8` as a
  relabelling violation. Which `workload.precision` values a non-reference
  engine may carry at all is the remaining open format decision in issue #19
  — until it is settled, a GGUF quantized arm cannot pass validation, and
  that is the correct, honest behaviour.
- **Cross-engine runs are engine-relative.** A report whose `engine.name` is
  not `transformers` is a valid measurement but never poolable with the
  v1.1.0 transformers reference curves: the validator rejects it at
  `--profile dataset-eligible` with the same standing as
  `context_length != 2048` (qualified vs exploratory). Energy deltas are an
  engine + kernel property, not a model-size property; each comparability
  class pairs against its own engine-internal FP16 baseline (A3).

## Add one engine in a focused follow-up

1. **Agree on supported formats and provenance first.** The current CLI and
   report schema name FP16, NF4 and INT8. A llama.cpp GGUF format such as
   Q4_K_M is not bitsandbytes NF4. Engine/version/model-artifact identity is
   now recorded in the optional `engine` block (schema 1.3-r6, above); which
   `workload.precision` values a non-reference engine may carry is the
   remaining open format decision in
   [issue #19](https://github.com/hongping-zh/ecocompute-mlcube/issues/19).
   Do not silently relabel formats to pass validation.
2. **Make selection explicit.** Add a backend parameter to both the YAML and
   CLI paths, with Transformers as the default. Reject unsupported
   backend/precision combinations before downloading a model. Keep optional
   engine imports lazy so CPU contract tests and existing users still work.
3. **Match the measurement contract.** For llama.cpp, configure GPU offload
   explicitly and record it; a mixed CPU/GPU run's NVML figure covers only the
   sampled GPU package. For vLLM, wait for the engine's decode work and use its
   actual output-token counts. For TensorRT-LLM, keep engine building and
   compilation outside the generation window. In every case ensure the GPU
   being measured is the GPU actually executing inference; the current runner
   samples only NVML device 0.
4. **Pair the baseline within the engine.** Run FP16 with the same model,
   tokenizer, prompts, batch size, output length and backend in the same
   session. A delta against the existing bitsandbytes dataset cannot stand in
   for this baseline. If the engine cannot run the required baseline, disclose
   that limitation and do not claim protocol conformance.
5. **Preserve honest fallback.** The published reference curves describe the
   existing measurements; they are not measurements or predictions validated
   for a new engine. Leave unavailable results unavailable when no applicable
   reference exists. Do not emit `direct-nvml` after a failed engine run.
6. **Package the runtime.** Update the appropriate image, dependency pins and
   software-version recording. Keep engine-specific builds separate if their
   CUDA or native requirements conflict with the default image. Document
   model conversion, caches and the commands needed to reproduce the run.

## Validate before submitting

Run the existing CPU contract suite from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

A reference CPU fake adapter ships in [`adapters/fake/`](../adapters/fake/)
with contract tests in
[`tests/test_fake_adapter.py`](../tests/test_fake_adapter.py). Extend those
for your engine: selection/parameter forwarding, unsupported precision, token
accounting across batches, synchronisation and sampler boundaries, cleanup
after generation failure, and fallback provenance.
Test the real runner around the adapter; do not assert only that a mock method
was called. The tests should catch energy sampled during warm-up, prompt tokens
included in the denominator, or a failed run labelled as measured.

On supported NVIDIA hardware, exercise the real engine with batch sizes 1 and
greater than 1, a same-session FP16 baseline, and optional traces. Check the
result's structure and protocol claims separately:

```bash
python -c "import json, jsonschema; jsonschema.validate(json.load(open('path/to/energy.json')), json.load(open('schema/energy.schema.json')))"
python entrypoint.py validate path/to/energy.json --profile v1.1-core
python tools/validate.py path/to/energy.json --profile dataset-eligible
```

These checks have different scopes. Passing the JSON schema alone does not
establish protocol conformance or dataset eligibility. Report any protocol
violations as returned by the validator, and include hardware, engine versions
and the exact commands in the PR. A CPU-only test run cannot establish real
engine compatibility, power telemetry or the energy delta.

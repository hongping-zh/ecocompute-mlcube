# Adding an inference backend

This is a guide for contributors addressing [issue #19](https://github.com/hongping-zh/ecocompute-mlcube/issues/19).
The current runtime uses Hugging Face Transformers and bitsandbytes. There is
no backend selector or plugin interface yet: the interface below is a proposal
for a small follow-up implementation, not an API available in the container.

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

## Add one engine in a focused follow-up

1. **Agree on supported formats and provenance first.** The current CLI and
   report schema name FP16, NF4 and INT8. A llama.cpp GGUF format such as
   Q4_K_M is not bitsandbytes NF4. Discuss new precision values and recording
   engine/version/model-artifact identity in an issue before changing the
   schema. Do not silently relabel formats to pass validation.
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

Add CPU tests using a small fake adapter for selection/parameter forwarding,
unsupported precision, token accounting across batches, synchronisation and
sampler boundaries, cleanup after generation failure, and fallback provenance.
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

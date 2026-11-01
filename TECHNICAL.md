# Technical Documentation

Deep dive into the EcoCompute energy methodology: JSON schema, validation, protocol conformance, reproducible builds, and backend extension.

---

## Table of Contents

- [Repository layout](#repository-layout)
- [JSON schema](#json-schema)
- [Inputs (energy_params.yaml)](#inputs-energy_paramsyaml)
- [Output (energy.json)](#output-energyjson)
- [Schema versions](#schema-versions)
- [Context length comparability](#context-length-comparability)
- [Validation](#validation)
- [Protocol conformance check](#protocol-conformance-check)
- [The quality probe (perplexity)](#the-quality-probe-perplexity)
- [The power-trace sidecar](#the-power-trace-sidecar)
- [No-GPU behaviour](#no-gpu-behaviour)
- [Scope and limitations](#scope-and-limitations)
- [Reproducible builds](#reproducible-builds)
- [Adding an inference backend](#adding-an-inference-backend)
- [Testing](#testing)
- [Verified with MLCube CLI](#verified-with-mlcube-cli)

---

## Repository layout

```
ecocompute-mlcube/
├── mlcube.yaml          # MLCube descriptor: energy_estimate task, inputs/outputs, GPU platform
├── mlcube.cpu.yaml      # CPU-only descriptor for GPU-less MLCube-contract verification / CI
├── Dockerfile           # production, multi-stage: cuda + torch + transformers + bitsandbytes + NVML
├── Dockerfile.cpu       # slim CPU image (verification/CI, no-GPU reference path only)
├── entrypoint.py        # energy_estimate task: load → (quantize) → warmup → NVML 10Hz → infer → energy.json
├── requirements.txt     # top-level runtime deps, exact == pins
├── requirements.lock.txt# full transitive lock (pip freeze from the verified image)
├── requirements-dev.txt # pyyaml, jsonschema, pytest (tests only)
├── schema/energy.schema.json   # JSON Schema for the report (energy + optional quality fields)
├── quality/             # fixed public-domain text the perplexity probe scores, + its sha256
├── examples/            # sample energy.json outputs (no-GPU + measured), schema-valid
├── tests/               # pytest: schema validity, param passing, no-GPU honesty
├── tools/check_regression.py   # grade a measured report against the published curve (needs a GPU host)
├── tools/validate.py    # semantic validator: schema-valid ≠ protocol-conformant
├── quickstart.sh        # zero-config one-liner (docker or, without it, a venv install)
├── autodl/00_setup.sh   # AutoDL-specific setup (venv, torch, bitsandbytes preflight)
├── .github/workflows/ci.yml             # CI: pytest + shellcheck + CPU image + schema check
├── .github/workflows/publish-image.yml  # builds and pushes ghcr.io/hongping-zh/ecocompute-mlcube
├── LICENSE / NOTICE     # Apache-2.0
└── workspace/
    ├── parameters/energy_params.yaml   # run inputs (mirror of /v1/estimate)
    ├── parameters/bert_bs32.yaml       # example alternate params (model swap, batch_size=32)
    ├── models/          # HF cache / local weights
    └── outputs/energy.json             # result
```

---

## JSON schema

The report schema is defined in [`schema/energy.schema.json`](schema/energy.schema.json).

**Core fields (always present):**
- `total_energy_j` — joules for the measurement window
- `tokens` — total tokens generated
- `energy_per_token_mj` — millijoules per 1000 tokens
- `avg_power_watts` — average power during decode
- `throughput_tokens_per_s` — tokens per second
- `vs_fp16_energy_pct` — energy delta vs FP16 baseline (negative = savings)
- `basis` — `measured` / `interpolated` / `extrapolated`
- `measurement_source` — `direct-nvml` / `ecocompute-dataset`
- `software` — versions of python, torch, CUDA, driver, packages; `matches_reference_pins` boolean

**Optional blocks (additive, schema 1.1+):**
- `quality` — perplexity probe results (schema 1.1)
- `power_trace` — sidecar file metadata (schema 1.3)
- `thermal` — GPU temperature at measurement start/end/peak (schema 1.3-r5)
- `engine` — inference engine identity (schema 1.3-r6)

---

## Inputs (energy_params.yaml)

| Field | Meaning |
|---|---|
| `model_name` | Hugging Face model id or local path |
| `params_b` | Billions of parameters (used by the no-GPU reference path) |
| `precision` | `FP16` \| `NF4` \| `INT8` (weight-only quantization) |
| `batch_size` | `1` → SingleStream scenario; `>1` → Offline |
| `gpu_arch` | `auto` (default — from the NVML device name) \| `turing` \| `ampere` \| `ada` \| `hopper` \| `blackwell` \| a GPU name |
| `tokens` | Tokens per decode iteration (default 256) |
| `iterations` | Number of decode iterations (default 10) |
| `warmup` | Number of warmup iterations before measurement (default 2) |
| `sample_rate_hz` | NVML sampling rate (default 10) |
| `quality_probe` | Enable perplexity probe (default true) |
| `quality_seq_len` | Sequence length for perplexity scoring (default 1024) |
| `power_trace` | Enable raw NVML sidecar (schema 1.3, default false) |

**Example:**
```yaml
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
params_b: 1.1
precision: NF4
batch_size: 1
gpu_arch: auto
tokens: 256
iterations: 10
warmup: 2
sample_rate_hz: 10
quality_probe: true
quality_seq_len: 1024
power_trace: false
```

---

## Output (energy.json)

See [examples/energy.measured.illustrative.json](examples/energy.measured.illustrative.json) for a complete example.

**Key sections:**

```json
{
  "total_energy_j": 12.34,
  "tokens": 2560,
  "energy_per_token_mj": 4.82,
  "avg_power_watts": 296,
  "throughput_tokens_per_s": 155.9,
  "vs_fp16_energy_pct": -61.9,
  "basis": "measured",
  "measurement_source": "direct-nvml",
  "scenario": "SingleStream",
  "scenario_note": "Not a certified benchmark; no LoadGen timing constraints",
  "follows_mlcommons_energy_reporting_conventions": true,
  "certified_benchmark_result": false,
  "software": {
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "cuda": "12.1",
    "driver": "535.129.03",
    "bitsandbytes": "0.43.3",
    "matches_reference_pins": true,
    "differs_from_reference_pins": []
  },
  "quality": {
    "perplexity_fp16": 7.3260,
    "perplexity_quantized": 7.7364,
    "delta_vs_fp16_pct": 5.60,
    "tokens_scored": 10624,
    "corpus_sha256": "abc123..."
  },
  "power_trace": {
    "files": ["power_trace_fp16.csv", "power_trace_nf4.csv"],
    "samples": 1234,
    "phase_markers": {
      "load_start_s": 0.0,
      "model_ready_s": 45.2,
      "warmup_end_s": 48.1,
      "measure_start_s": 48.2,
      "measure_end_s": 64.5
    }
  },
  "thermal": {
    "basis": "measured",
    "start_c": 27,
    "end_c": 31,
    "peak_c": 34,
    "steady_state_reached": true
  },
  "engine": {
    "name": "transformers",
    "version": "4.57.6",
    "backend": "bitsandbytes",
    "model_format": "safetensors",
    "artifact_hash": "sha256:...",
    "tokenizer_provenance": "huggingface-hub",
    "sync_point": "2026-09-20T12:00:00Z"
  }
}
```

---

## Schema versions

### Schema 1.0 (initial)
Core energy fields only.

### Schema 1.1 (2026-08)
Added optional `quality` object (perplexity probe). Additive — 1.0 reports remain valid.

### Schema 1.3 (2026-09)
Added optional `power_trace` object (raw NVML sidecar). Additive — older reports remain valid.

### Schema 1.3-r5 (2026-10-19)
Populated `thermal` block: GPU temperature at measurement start/end/peak via NVML. `basis: "measured"` for on-device reads, `basis: "unavailable"` for AMD/dry-run/measurement-failure paths.

### Schema 1.3-r6 (2026-10-21)
Added optional `engine` object: inference engine identity (name, version, backend, model_format, artifact hash, tokenizer provenance, sync point). Absent means the transformers reference stack. Cross-engine runs are valid measurements but engine-relative — a separate comparability class, never pooled with the v1.1.0 transformers reference curves.

---

## Context length comparability

**Qualified submission** (directly comparable to v1.1.0 baseline):
- `context_length = 2048`
- `engine` absent or `engine.name = transformers`
- Your ΔE% can be plotted against the published crossover curves and compared with the main dataset.

**Exploratory submission** (separate comparability class):
- `context_length ≠ 2048`, **or** `engine.name` other than `transformers` (schema 1.3-r6)
- Your run forms a separate comparability class and is **not directly comparable** with v1.1.0 anchors (whose context_length was nominal ~20 tokens regardless of the recorded value, prior to PR #22).
- Still valuable — fills the variable matrix — but plotted and analyzed separately. Cross-engine runs pair against their own engine-internal FP16 baseline (see [docs/backends.md](docs/backends.md), A3).

See [quantenergy.tech/contribute/](https://quantenergy.tech/contribute/#comparability) for details.

---

## Validation

### Schema validation

Check that a report is structurally valid:

```bash
python3 -c "
import json, jsonschema
report = json.load(open('ecocompute-out/energy.json'))
schema = json.load(open('schema/energy.schema.json'))
jsonschema.validate(report, schema)
print('schema-valid')
"
```

### Semantic validation

Schema validation checks **structure**. It cannot check cross-field semantics ("an NF4 report must carry a same-session FP16 baseline", "`basis: measured` requires `direct-nvml`", "the thermal block must be present and honest").

Those are Protocol v1.1 MUSTs, checked by the semantic validator:

```bash
python3 tools/validate.py ecocompute-out/energy.json
# or via the entrypoint alias:
python3 entrypoint.py validate ecocompute-out/energy.json --profile v1.1-core
```

**Three verdicts, deliberately distinct:**

| Verdict | Meaning |
|---|---|
| `schema-valid` | Structure passes `energy.schema.json` — and nothing more |
| `protocol-conformant` | Schema-valid **plus** every Protocol v1.1 MUST (v1.1-core profile) |
| `dataset-eligible` | Protocol-conformant plus reportability extras (power-trace sidecar, achieved sample rate, `basis: measured`); replication counts and review stay dataset-level |

**Exit codes** (CI-friendly):
- `0` — conformant/eligible
- `1` — schema-valid but violating (violations listed with the protocol clause number)
- `2` — schema-invalid

**Honest failure states pass:** `steady_state_reached: false` and `thermal.basis: "unavailable"` are valid reports of reality, not violations — only fabrication fails.

**Protocol v1.1 MUSTs checked:**
- NVML method for measured reports (§4.1.1)
- Achieved rate ≥ 10 Hz (§4.1.2)
- 256 tokens / 10 iterations / greedy decoding (§4.3.2)
- Generation window stated (§4.4.1)
- FP16-baseline rule covers every non-FP16 precision (§4.2.1)
- Measured thermal blocks need warm-up + cooldown + start/end/peak temperatures with the steady temperature excused only by the honest `steady_state_reached: false` (§4.6.1/4.6.4)
- Arm order recorded as randomized/counterbalanced or a disclosed deviation (§4.6.2)
- Conformance is defined against schema 1.2+ (§4.5.2)

**Pinned artifacts:**
- Report schema SHA-256: `501dc1f328270f0bd3221e1ec5c81d740308bdd87090b93f54758ea79b6efb04`
- `tools/validate.py` SHA-256: `c5ca27907ffc68c91b6f28702515788210cc670e6d0330e9328788f5dd815205`

---

## The quality probe (perplexity)

Energy on its own cannot say whether a quantization was worth taking: a format that halves the joules and wrecks the model is not a win. So each measured run also scores the perplexity of the model it just measured, and reports `quality.delta_vs_fp16_pct` against the FP16 baseline **from the same run**.

**How it stays out of the energy figure:**
- It is a separate teacher-forcing forward pass, not the decode loop — you cannot get a perplexity out of free generation
- It runs **after `sampler.stop()`**, with the model already loaded, so not one joule of it is charged to `energy_per_token_mj`, and it adds seconds, not minutes
- If it fails (OOM, an unusual head), the report says `quality.basis: "unavailable"` with the error and the energy measurement stands unchanged

**Corpus:** The text is vendored in [`quality/`](quality/) — two public-domain excerpts, ~10.6k words — so a run needs no network and no `datasets` dependency, and every contributor scores the same bytes. Its sha256 is recorded in the report.

**Read the delta, not the absolute number.** Perplexity depends on the text and the tokenizer, so the absolute value is not comparable with published WikiText figures or across models; only FP16-vs-quantized on identical bytes is. And perplexity is a proxy for language-model damage, not a downstream-task quality guarantee. `quality/README.md` says this in more detail.

**Overrides:**
- `--quality_text /path/to/your.txt` — use custom text (its sha256 goes into the report, so a run on other bytes can never be silently compared with one on these)
- `--quality_seq_len 2048` — change sequence length
- `--no_quality_probe` or `ECOCOMPUTE_QUALITY=0` — disable the probe entirely

---

## The power-trace sidecar

`energy.json` normally keeps only aggregates (`total_energy_j`, `avg_power_w`) — once a report is written, its measurement window can never be re-examined. That cost us a whole re-analysis once: the published bitsandbytes sessions could not be re-windowed (generation-only vs whole-process) without re-running, because the raw samples were thrown away.

`--power_trace` institutionalises the lesson:

- A second NVML sampler records the **whole run from before the model is loaded** — load, quantization, warm-up, generation, quality probe — and is dumped to `power_trace.csv` (`power_trace_fp16.csv` for the baseline run)
- The report gains a `power_trace` block listing the files, sample counts and **phase markers** (`load_start_s`, `model_ready_s`, `warmup_end_s`, `measure_start_s`, `measure_end_s`, …) on each file's trace clock
- The reported `total_energy_joules` is computed by the *same dedicated generation-window sampler as before* — enabling the sidecar changes no reported number, it only adds what you can check later

With the sidecar, any reader can re-integrate the trace over a different window (e.g. whole-process including load, or decode-only) without re-running the GPU: the phase markers are the cut points. Files are plain two-column CSVs (`t_s,power_w`), one per measured run.

---

## No-GPU behaviour

If no NVIDIA GPU / NVML is present (or `--dry_run` is set), the container does **not** fabricate a measurement. It emits values derived from the **published EcoCompute dataset** (Zenodo DOI `10.5281/zenodo.19647290`), flagged `measurement_source: "ecocompute-dataset (no local GPU)"` and with an explicit note.

The same honesty guarantee holds if a GPU **is** present but NVML power telemetry is unavailable (some Turing / consumer / vGPU cards, or a driver that returns `NVML_ERROR_NOT_SUPPORTED`): the on-device path is probed first, and on failure the container falls back to the dataset path with `measurement_source: "ecocompute-dataset (on-device measurement failed)"` and `basis != "measured"` — it never labels a fallback as a real measurement, and it does not crash. A single dropped power read never aborts a run (see `results.dropped_samples`).

---

## Scope and limitations

**Not a certified benchmark:**
- **No LoadGen.** The `scenario` field (`SingleStream` / `Offline`) is a *nominal* label derived from `batch_size`; it is **not** enforced by MLPerf LoadGen. The container runs its own warmup/iterations loop and applies **no** LoadGen timing constraints. This is a supplemental **energy methodology** container, not a certified benchmark (`certified_benchmark_result: false`). Every report carries this in `scenario_note`.
- **No accuracy target.** Only energy/throughput are reported.
- Report fields follow MLCommons-style energy-reporting conventions (`follows_mlcommons_energy_reporting_conventions: true`) but are not certified results.

**Single-report limitations:**
- A single report is n=1, no matter how high `measurement.iterations` is
- Version mismatches with the published pins (printed as a caveat) account for part of any residual
- A pass on the regression check means "this build still measures what it used to", not "this confirms the curve"

---

## Reproducible builds

`requirements.txt` pins the top-level runtime deps to exact `==` versions, and `requirements.lock.txt` pins **all** transitive dependencies (a `pip freeze` captured from the verified CUDA image). The `Dockerfile` installs from the lock so builds are deterministic — important because `bitsandbytes` NF4/INT8 kernels (and `torch`) change their numeric behaviour between releases.

**Regenerate the lock:**
```bash
docker run --rm --entrypoint pip <built-image> freeze > requirements.lock.txt
```

**Pin the image by digest** (rather than the mutable `:latest`):
```bash
ECOCOMPUTE_IMAGE=ghcr.io/hongping-zh/ecocompute-mlcube@sha256:<digest from the release page>
```

---

## Adding an inference backend

The runtime currently uses Hugging Face Transformers with bitsandbytes for quantisation. See the [backend contribution guide](docs/backends.md) for:

- The current inference path
- A proposed adapter boundary
- Validation requirements for adding engines such as llama.cpp, vLLM or TensorRT-LLM

**Backend acceptance checklist (A1–A5):**
1. Real token counts are an acceptance condition, enforced by a runner-authoritative re-tokenization cross-check, not a promise
2. Cross-engine runs are engine-relative (never pooled with the v1.1.0 transformers curves, same standing as `context_length != 2048`)
3. Relabelling a foreign format (e.g. GGUF Q4_K_M) as `precision: NF4/INT8` is a violation
4. The `engine` block (schema 1.3-r6) records what generated the tokens — absent means the transformers reference stack, so every existing report stays valid unchanged
5. A CPU reference fake adapter (`adapters/fake/`) plus contract tests — including a deliberately cheating adapter that must be caught — ship in CI with no GPU and no downloads

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite runs the `energy_estimate` task through its real CLI and asserts that:
- Every report validates against the schema
- Parameters flow into the output
- `batch_size` selects the scenario
- The no-GPU path never labels a result as `measured`

CI (`.github/workflows/mlcube-verify.yml`) runs these tests plus a full `Dockerfile.cpu` build + container run + schema check on every push/PR.

---

## Verified with MLCube CLI

The container was exercised with the official MLCommons `mlcube` CLI (v0.0.9, docker platform):

- `mlcube describe --mlcube=mlcube.yaml` — descriptor accepted; task `energy_estimate` with `inputs=[parameters_file]`, `outputs=[output_dir]`
- `mlcube run --mlcube=mlcube.cpu.yaml --task=energy_estimate` — image built from `Dockerfile.cpu`, workspace mounted, `workspace/outputs/energy.json` produced
- Parameter swap (`parameters_file=parameters/bert_bs32.yaml`) produced a distinct report (`bert-base-uncased`, `batch_size=32` → Offline scenario)
- Both reports validate against `schema/energy.schema.json`

---

## Regression check against the published curve

After a measured run, grade it against the published fit:

```bash
python3 tools/check_regression.py ecocompute-out/energy.json
# Exit codes: 0 = within the band, 1 = outside it, 2 = not gradeable
```

The band is `k * resid_std` (default `k=2`), i.e. the fit's own residual dispersion; it is **not** a confidence interval.

**Deliberate limits:**
- Refuses to grade a report whose `basis` is not `measured`: an interpolated report is derived *from* the curve, so checking it against the curve is circular
- Cannot run in CI — GitHub-hosted runners have no NVIDIA GPU. Run it on the GPU host after `quickstart.sh`

A pass means "this build still measures what it used to", not "this confirms the curve": a single report is n=1, and a version mismatch with the published pins (printed as a caveat) accounts for part of any residual.

---

## Next steps

- **[README.md](README.md)** — Overview and quick start
- **[QUICKSTART.md](QUICKSTART.md)** — Complete setup guide
- **[docs/backends.md](docs/backends.md)** — Backend contribution guide
- **[quantenergy.tech](https://quantenergy.tech)** — Interactive crossover curves and estimator API

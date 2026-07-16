# EcoCompute Energy Methodology — MLCube

A **supplemental energy-methodology container** for LLM inference. It measures the
*energy* of one configuration — `(GPU × model × precision × batch)` — with direct
on-device **NVML** power sampling, and writes an `energy.json` report whose fields
align with **MLCommons-style inference energy reporting**.

It packages the same inputs as the EcoCompute [`/v1/estimate`](https://quantenergy.tech)
API (`model_name`, `precision`, `batch_size`, `gpu_arch`) as a portable
[MLCube](https://mlcommons.org/en/mlcube/)-compatible container so the measurement
is reproducible on any CUDA GPU.

> **This is a reference / methodology implementation, not a certified benchmark
> run.** There is no accuracy target and no LoadGen. Numbers produced here are
> **not certified benchmark results**.

## Quick start

No GPU needed for a smoke test — the container falls back to a dataset-derived
reference report (clearly flagged, never a fabricated measurement).

```bash
git clone https://github.com/hongping-zh/ecocompute-mlcube.git
cd ecocompute-mlcube

# 1) run the task directly (writes workspace/outputs/energy.json)
python3 entrypoint.py energy_estimate --dry_run \
    --parameters_file workspace/parameters/energy_params.yaml \
    --output_dir workspace/outputs
cat workspace/outputs/energy.json          # see examples/ for expected output

# 2) or via the official MLCube CLI + Docker (no GPU)
pip install mlcube mlcube-docker
mlcube run --mlcube=mlcube.cpu.yaml --task=energy_estimate --platform=docker

# 3) real measurement on an NVIDIA GPU
mlcube run --mlcube=. --task=energy_estimate --platform=docker
```

Expected output shape: [`examples/energy.no-gpu.json`](examples/energy.no-gpu.json)
(no-GPU reference) and [`examples/energy.measured.illustrative.json`](examples/energy.measured.illustrative.json)
(on-GPU measured).

## Layout

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
├── schema/energy.schema.json   # JSON Schema for the report (energy fields)
├── examples/            # sample energy.json outputs (no-GPU + measured), schema-valid
├── tests/               # pytest: schema validity, param passing, no-GPU honesty
├── .github/workflows/mlcube-verify.yml   # CI: pytest + build + mlcube-style run + schema check
├── LICENSE / NOTICE     # Apache-2.0
└── workspace/
    ├── parameters/energy_params.yaml   # run inputs (mirror of /v1/estimate)
    ├── parameters/bert_bs32.yaml       # example alternate params (model swap, batch_size=32)
    ├── models/          # HF cache / local weights
    └── outputs/energy.json             # result
```

## Run

With the MLCube CLI (recommended). The task name is **`energy_estimate`**:

```bash
pip install mlcube mlcube-docker

# real measurement on an NVIDIA GPU (platform.accelerator_count=1 -> --gpus=all)
mlcube run --mlcube=. --task=energy_estimate --platform=docker

# change the run without editing defaults — swap the parameters file
mlcube run --mlcube=. --task=energy_estimate \
           parameters_file=parameters/bert_bs32.yaml output_dir=outputs_bert/
```

**No GPU? Verify the MLCube contract (build → param mount → energy.json) with the
CPU descriptor** — this builds `Dockerfile.cpu` and runs the no-GPU reference path:

```bash
mlcube run --mlcube=mlcube.cpu.yaml --task=energy_estimate --platform=docker
```

Directly (development / CI):

```bash
# real measurement (requires an NVIDIA GPU + NVML)
python3 entrypoint.py energy_estimate \
    --parameters_file workspace/parameters/energy_params.yaml \
    --output_dir workspace/outputs

# force the no-GPU reference path (derives values from the published dataset)
python3 entrypoint.py energy_estimate --dry_run \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --precision NF4 --gpu_arch blackwell --params_b 1.1 \
    --output_dir workspace/outputs
```

> Parameter passing follows the MLCube convention: task inputs/outputs are the
> `parameters_file` (a YAML holding `model_name` / `precision` / `batch_size` /
> `gpu_arch` / …) and `output_dir`. For dev/`docker run` use, the entrypoint also
> accepts those same fields as CLI flags (e.g. `--model`, `--batch_size`, `--precision`).

### Inputs (`energy_params.yaml`)

| field | meaning |
|---|---|
| `model_name` | HF model id or local path |
| `params_b` | billions of parameters (used by the no-GPU reference path) |
| `precision` | `FP16` \| `NF4` \| `INT8` (weight-only quantization) |
| `batch_size` | `1` → SingleStream scenario; `>1` → Offline |
| `gpu_arch` | `turing` \| `ada` \| `blackwell` \| `ampere` (or a GPU name) |
| `tokens`, `iterations`, `warmup`, `sample_rate_hz` | measurement controls |

### Output (`energy.json`)

Fields align with MLCommons-style inference energy reporting: total joules, tokens,
`energy_per_token_mj` (= J / 1k tokens), `avg_power_watts`,
`throughput_tokens_per_s`, plus a signed `vs_fp16_energy_pct` (negative = quantization
saves energy). Every result carries a `basis` (`measured` / `interpolated` /
`extrapolated`) and a `measurement_source`. See `schema/energy.schema.json`.

## Verified

The container was exercised with the official MLCommons `mlcube` CLI (v0.0.9,
docker platform):

- `mlcube describe --mlcube=mlcube.yaml` — descriptor accepted; task
  `energy_estimate` with `inputs=[parameters_file]`, `outputs=[output_dir]`.
- `mlcube run --mlcube=mlcube.cpu.yaml --task=energy_estimate` — image built from
  `Dockerfile.cpu`, workspace mounted, `workspace/outputs/energy.json` produced.
- Parameter swap (`parameters_file=parameters/bert_bs32.yaml`) produced a distinct
  report (`bert-base-uncased`, `batch_size=32` → Offline scenario).
- Both reports validate against `schema/energy.schema.json`.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite runs the `energy_estimate` task through its real CLI and asserts that
every report validates against the schema, that parameters flow into the output,
that `batch_size` selects the scenario, and that the no-GPU path never labels a
result as `measured`. CI (`.github/workflows/mlcube-verify.yml`) runs these tests
plus a full `Dockerfile.cpu` build + container run + schema check on every push/PR.

## No-GPU behaviour

If no NVIDIA GPU / NVML is present (or `--dry_run` is set), the container does **not**
fabricate a measurement. It emits values derived from the **published EcoCompute
dataset** (Zenodo DOI `10.5281/zenodo.21066652`), flagged
`measurement_source: "ecocompute-dataset (no local GPU)"` and with an explicit note.

The same honesty guarantee holds if a GPU **is** present but NVML power telemetry is
unavailable (some Turing / consumer / vGPU cards, or a driver that returns
`NVML_ERROR_NOT_SUPPORTED`): the on-device path is probed first, and on failure the
container falls back to the dataset path with
`measurement_source: "ecocompute-dataset (on-device measurement failed)"` and
`basis != "measured"` — it never labels a fallback as a real measurement, and it does
not crash. A single dropped power read never aborts a run (see `results.dropped_samples`).

## Scope & limitations (not a benchmark)

- **No LoadGen.** The `scenario` field (`SingleStream` / `Offline`) is a *nominal* label
  derived from `batch_size`; it is **not** enforced by MLPerf LoadGen. The container runs
  its own warmup/iterations loop and applies **no** LoadGen timing constraints. This is a
  supplemental **energy methodology** container, not a certified benchmark
  (`certified_benchmark_result: false`). Every report carries this in `scenario_note`.
- **No accuracy target.** Only energy/throughput are reported.
- Report fields follow MLCommons-style energy-reporting conventions
  (`follows_mlcommons_energy_reporting_conventions: true`) but are not certified results.

## Reproducible builds

`requirements.txt` pins the top-level runtime deps to exact `==` versions, and
`requirements.lock.txt` pins **all** transitive dependencies (a `pip freeze` captured
from the verified CUDA image). The `Dockerfile` installs from the lock so builds are
deterministic — important because `bitsandbytes` NF4/INT8 kernels (and `torch`) change
their numeric behaviour between releases. Regenerate the lock with:

```bash
docker run --rm --entrypoint pip <built-image> freeze > requirements.lock.txt
```

## Provenance

- Tool: https://quantenergy.tech
- Paper (SSRN #6854700): *Weight-Only Quantization Does Not Always Save Energy…* (under review)
- Dataset DOI: `10.5281/zenodo.21066652`
- Code: https://github.com/hongping-zh/ecocompute-ai

## License

[Apache License 2.0](LICENSE) © 2026 Hongping Zhang. See [`NOTICE`](NOTICE).

## Trademarks

MLCOMMONS, MLPERF, and MLCUBE are trademarks of **MLCommons Association**. This
project references them **nominatively** only, to describe the energy-reporting
methodology and the container format; it does **not** indicate any MLCommons
endorsement, certification, or a certified benchmark result. Formal trademark-license
mark usage (per the [MLCommons Trademark Usage Guidelines](https://mlcommons.org/en/policies/))
will be added once the license is in effect.

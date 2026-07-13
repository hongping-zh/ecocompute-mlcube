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

## Layout

```
ecocompute-mlcube/
├── mlcube.yaml          # MLCube descriptor: run task, inputs/outputs, GPU platform
├── Dockerfile           # cuda + torch + transformers + bitsandbytes + NVML
├── entrypoint.py        # run task: load → (quantize) → warmup → NVML 10Hz → infer → energy.json
├── requirements.txt     # transformers, bitsandbytes, nvidia-ml-py, ...
├── schema/energy.schema.json   # JSON Schema for the report (energy fields)
└── workspace/
    ├── parameters/energy_params.yaml   # run inputs (mirror of /v1/estimate)
    ├── models/          # HF cache / local weights
    └── outputs/energy.json             # result
```

## Run

With the MLCube CLI (recommended):

```bash
pip install mlcube mlcube-docker
mlcube run --task=run                       # uses workspace/parameters/energy_params.yaml
```

Directly (development / CI):

```bash
# real measurement (requires an NVIDIA GPU + NVML)
python3 entrypoint.py run --parameters_file workspace/parameters/energy_params.yaml \
                          --output_dir workspace/outputs

# force the no-GPU reference path (derives values from the published dataset)
python3 entrypoint.py run --dry_run --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                          --precision NF4 --gpu_arch blackwell --params_b 1.1 \
                          --output_dir workspace/outputs
```

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

## No-GPU behaviour

If no NVIDIA GPU / NVML is present (or `--dry_run` is set), the container does **not**
fabricate a measurement. It emits values derived from the **published EcoCompute
dataset** (Zenodo DOI `10.5281/zenodo.21066652`), flagged
`measurement_source: "ecocompute-dataset (no local GPU)"` and with an explicit note.

## Provenance

- Tool: https://quantenergy.tech
- Paper (SSRN #6854700): *Weight-Only Quantization Does Not Always Save Energy…* (under review)
- Dataset DOI: `10.5281/zenodo.21066652`
- Code: https://github.com/hongping-zh/ecocompute-ai

## Trademarks

MLCOMMONS, MLPERF, and MLCUBE are trademarks of **MLCommons Association**. This
project references them **nominatively** only, to describe the energy-reporting
methodology and the container format; it does **not** indicate any MLCommons
endorsement, certification, or a certified benchmark result. Formal trademark-license
mark usage (per the [MLCommons Trademark Usage Guidelines](https://mlcommons.org/en/policies/))
will be added once the license is in effect.

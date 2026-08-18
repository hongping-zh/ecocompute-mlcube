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

## Quick start — one command, no configuration

On any host with an NVIDIA GPU:

```bash
curl -fsSL https://raw.githubusercontent.com/hongping-zh/ecocompute-mlcube/main/quickstart.sh | bash
```

That measures **NF4 and its own FP16 baseline** on TinyLlama-1.1B, scores both
for perplexity on a fixed text so you also see what the quantization cost in
quality, writes a schema-validated `energy.json` into `./ecocompute-out/`, and
prints a link that overlays your point on the published curve.

**How long it takes** is dominated by downloads, not by the measurement (a few
minutes), and later runs reuse the caches. The one end-to-end run we have timed
is the honest number to plan around: **57 minutes** on a China-hosted rented
RTX 4090 in native mode, of which the ~2 GB model download was 5.5 minutes and
nearly all the rest was pulling ~3 GB of torch/CUDA wheels from
`download.pytorch.org`. Since then `autodl/00_setup.sh` prefers the configured
(domestic) index for torch. A second run on that same instance, with every cache
already warm, took **1m48s** end to end — that is the repeat-run number, not a
first-run one. On a machine with fast registry access the docker path has less
to download; we have not timed it.

The script picks how to run itself, and says which it picked:

| | when | what it does |
|---|---|---|
| **docker** | the Docker daemon answers | pulls `ghcr.io/…/ecocompute-mlcube:latest` and runs it with `--gpus all`; nothing is installed on the host |
| **native** | no usable Docker — rented GPU instances (AutoDL, vast.ai, …) are themselves containers and cannot nest one | clones this repo and installs into a venv, reusing the host's CUDA PyTorch; keeps the checkout, venv and model cache on the data disk when there is one |

Force either with `ECOCOMPUTE_MODE=docker` / `ECOCOMPUTE_MODE=native`. Both paths
run the same `entrypoint.py`, the same pins from `requirements.txt` (native
installs everything except torch, which must match the host driver) and the same
10 decode iterations after 2 warmups as the published dataset.

The native path cannot always get the pins — a Python 3.8 interpreter (AutoDL's
default) or a lagging mirror can force older releases, and `bitsandbytes`
NF4/INT8 kernels change between releases. So the report does not take the
installer's word for it: `software` in every `energy.json` records the python,
torch/CUDA, driver and package versions the run actually used, plus
`matches_reference_pins` and a `differs_from_reference_pins` list. The summary
prints the same thing. A mismatched run is still a real measurement; it just is
not directly comparable with a run from the image, and now says so itself.

To get the pins on a host whose default python is too old, point `ECO_PYTHON` at
a ≥3.9 interpreter. A venv cannot change interpreter after creation, so an
existing one has to go (which also means reinstalling torch):

```bash
rm -rf /root/autodl-tmp/ecocompute/venv        # or: ECO_RECREATE_VENV=1
ECO_PYTHON=/root/miniconda3/envs/eco310/bin/python \
  SKIP_DOWNLOAD=1 bash autodl/00_setup.sh
```

`00_setup.sh` refuses to continue if `ECO_PYTHON` does not exist or disagrees
with the venv it would otherwise reuse, rather than quietly running on the old
interpreter and producing a report that is flagged non-comparable an hour later.

`bitsandbytes` also has to match torch's CUDA line — a torch wheel from a newer
line makes the pinned `bitsandbytes` look for a `libbitsandbytes_cudaXXX.so` it
does not ship, and the quantized run then dies with an opaque import error
*after* the model download. Setup therefore runs a real 4-bit kernel as a
preflight check. On failure it first upgrades `bitsandbytes` to a release that
has kernels for the torch you already have — one small wheel from the index you
are already using — and only then falls back to reinstalling torch from the
cu121 index the image is built on (~2.5 GB from `download.pytorch.org`, which
rented boxes often cannot reach, and which caps torch at 2.5.1, the last cu121
build published). The upgrade leaves the published pins, so the report carries
`software.differs_from_reference_pins` and `vs_fp16` is not directly comparable
with image runs; `ECO_KEEP_PINS=1` skips it. Force an index up front with
`ECO_TORCH_INDEX=https://download.pytorch.org/whl/cu121`.

If neither repair works, native mode stops **before** downloading the model
instead of spending your time on a run that can only produce
`basis != measured`. `ECOCOMPUTE_ALLOW_FALLBACK=1` overrides that.

No flags are needed: `gpu_arch` defaults to `auto` and is derived from the NVML
device name, so a run cannot silently label an RTX 4090 as Blackwell. If you
would rather read the script first, it is [`quickstart.sh`](quickstart.sh), and
it prints every command it runs.

Common overrides (all optional):

```bash
ECOCOMPUTE_PRECISION=INT8 \
ECOCOMPUTE_MODEL=Qwen/Qwen2.5-3B ECOCOMPUTE_PARAMS_B=3 \
ECOCOMPUTE_PREFETCH=1 \
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/hongping-zh/ecocompute-mlcube/main/quickstart.sh)"
```

`--share` is always on in the quickstart (it only builds a link locally).
`ECOCOMPUTE_PREFETCH=1` adds `--prefetch`, the one thing that talks to the site
before measuring, so it stays opt-in — see below. `ECOCOMPUTE_QUALITY=0` turns
off the perplexity probe.

Or drive the image yourself:

```bash
docker run --rm --gpus all \
  -v "$PWD/out:/workspace/outputs" \
  ghcr.io/hongping-zh/ecocompute-mlcube:latest energy_estimate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --params_b 1.1 \
  --precision NF4 --gpu_arch auto --output_dir /workspace/outputs --share
```

Then publish the report — agreeing or disagreeing — at
[quantenergy.tech/replications](https://quantenergy.tech/replications/#submit).

The image (~8.8 GB unpacked: CUDA 12.1 runtime + torch + transformers +
bitsandbytes) is built and pushed by
[`.github/workflows/publish-image.yml`](.github/workflows/publish-image.yml) on
every merge to `main`, and `docker pull ghcr.io/hongping-zh/ecocompute-mlcube:latest`
works anonymously. If the pull fails anyway (offline, registry down),
`quickstart.sh` says so and falls back to a local copy or a build from this
repository — same result, ~10–20 minutes longer. If Docker cannot start the
container with `--gpus all`, the script retries without GPU access so the
pipeline still gets verified, and both the report (`basis` ≠ `measured`) and the
printed summary say it is not a measurement.

### Running from source

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

**Compare your result online:** drag your `energy.json` onto the
[**Run it yourself**](https://quantenergy.tech/?tab=run) tab at quantenergy.tech to
overlay your measurement on the crossover curve (100% in-browser, nothing uploaded).
The container's fields (`system_under_test.gpu_arch`, `workload.params_b`,
`results.vs_fp16_energy_pct`, `results.basis`) map directly to the site's chart axes.

### Compare straight from the terminal (`--prefetch` / `--share`)

Two optional flags wire the run to the website — both are best-effort and never
block or change the measurement:

```bash
python3 entrypoint.py energy_estimate \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --precision NF4 --gpu_arch auto --params_b 1.1 \
    --output_dir workspace/outputs --prefetch --share
```

- `--prefetch` — before measuring, asks the site estimator
  (`/v1/estimate`) for its predicted energy delta vs FP16 and prints it, so you
  can eyeball prediction vs. your measured result. Offline/timeout → silently
  skipped.
- `--share` — after measuring, prints (and writes `share_url.txt`) a link like
  `https://quantenergy.tech/?tab=run&overlay=<base64url>`. The overlay point is
  encoded **in the URL itself** — opening it restores your marker on the
  crossover curve entirely in the browser. **Nothing is uploaded and no server
  stores anything**, consistent with the site's privacy model. The same link is
  also added to `energy.json` as `share_url`.

### The quality probe (perplexity)

Energy on its own cannot say whether a quantization was worth taking: a format
that halves the joules and wrecks the model is not a win. So each measured run
also scores the perplexity of the model it just measured, and reports
`quality.delta_vs_fp16_pct` against the FP16 baseline **from the same run**.

How it stays out of the energy figure:

- it is a separate teacher-forcing forward pass, not the decode loop — you cannot
  get a perplexity out of free generation;
- it runs **after `sampler.stop()`**, with the model already loaded, so not one
  joule of it is charged to `energy_per_token_mj`, and it adds seconds, not
  minutes;
- if it fails (OOM, an unusual head), the report says `quality.basis:
  "unavailable"` with the error and the energy measurement stands unchanged.

The text is vendored in [`quality/`](quality/) — two public-domain excerpts,
~10.6k words — so a run needs no network and no `datasets` dependency, and every
contributor scores the same bytes. Its sha256 is recorded in the report.

**Read the delta, not the absolute number.** Perplexity depends on the text and
the tokenizer, so the absolute value is not comparable with published WikiText
figures or across models; only FP16-vs-quantized on identical bytes is. And
perplexity is a proxy for language-model damage, not a downstream-task quality
guarantee. `quality/README.md` says this in more detail.

Useful when the vendored text is the wrong one for you: `--quality_text
/path/to/your.txt` (its sha256 goes into the report, so a run on other bytes can
never be silently compared with one on these), `--quality_seq_len`, and
`--no_quality_probe` / `ECOCOMPUTE_QUALITY=0`.

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
├── schema/energy.schema.json   # JSON Schema for the report (energy + optional quality fields)
├── quality/             # fixed public-domain text the perplexity probe scores, + its sha256
├── examples/            # sample energy.json outputs (no-GPU + measured), schema-valid
├── tests/               # pytest: schema validity, param passing, no-GPU honesty
├── tools/check_regression.py   # grade a measured report against the published curve (needs a GPU host)
├── quickstart.sh        # zero-config one-liner (docker or, without it, a venv install)
├── .github/workflows/ci.yml             # CI: pytest + shellcheck + CPU image + schema check
├── .github/workflows/publish-image.yml  # builds and pushes ghcr.io/hongping-zh/ecocompute-mlcube
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
| `gpu_arch` | `auto` (default — from the NVML device name) \| `turing` \| `ampere` \| `ada` \| `hopper` \| `blackwell` \| a GPU name |
| `tokens`, `iterations`, `warmup`, `sample_rate_hz` | measurement controls |
| `quality_probe`, `quality_seq_len` | perplexity probe: on by default, 1024-token windows |

### Output (`energy.json`)

Fields align with MLCommons-style inference energy reporting: total joules, tokens,
`energy_per_token_mj` (= J / 1k tokens), `avg_power_watts`,
`throughput_tokens_per_s`, plus a signed `vs_fp16_energy_pct` (negative = quantization
saves energy). Every result carries a `basis` (`measured` / `interpolated` /
`extrapolated`) and a `measurement_source`. `software` records the versions the
run used and whether they are the published pins; `measurement.iterations` is
decode iterations *within* one run, so a single report is n=1 no matter how high
it is. See `schema/energy.schema.json`.

Schema **1.1** adds an optional `quality` object (perplexity, the FP16 baseline,
`delta_vs_fp16_pct`, tokens scored and the corpus sha256). It is optional and
additive: 1.0 reports are still valid, and a run without the probe has no
`quality` key rather than a null one.

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

## Regression check against the published curve

After a measured run, grade it against the published fit:

```bash
python3 tools/check_regression.py ecocompute-out/energy.json
# 0 = within the band, 1 = outside it, 2 = not gradeable
```

It compares `vs_fp16_energy_pct` with the published anchor(s) of the *same*
model size (falling back to the fitted curve only when there is no anchor at
that size — the fit is smooth across sizes and can sit tens of points off any
single one). The band is `k * resid_std` (default `k=2`), i.e. the fit's own
residual dispersion; it is **not** a confidence interval.

Two deliberate limits:

- It refuses to grade a report whose `basis` is not `measured`: an
  interpolated report is derived *from* the curve, so checking it against the
  curve is circular.
- It cannot run in CI — GitHub-hosted runners have no NVIDIA GPU. Run it on the
  GPU host after `quickstart.sh`.

A pass means "this build still measures what it used to", not "this confirms
the curve": a single report is n=1, and a version mismatch with the published
pins (printed as a caveat) accounts for part of any residual.

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

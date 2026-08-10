#!/usr/bin/env python3
"""EcoCompute energy-methodology MLCube entrypoint.

Implements a single MLCube task, ``run``, that measures the *energy* of one LLM
inference configuration (GPU x model x precision x batch) with direct on-device
NVML power sampling, and writes an ``energy.json`` report whose fields align with
MLCommons-style inference energy reporting.

This is a *supplemental energy methodology container* (a reference /
methodology implementation), not a certified benchmark run: there is no accuracy
target and no LoadGen. Results produced here are not certified benchmark results.

Note: MLCOMMONS, MLPERF, and MLCUBE are trademarks of MLCommons Association. This
project references them nominatively to describe methodology and container format;
formal trademark-license mark usage will be added once the license is in effect.

Usage (via MLCube):
    mlcube run --task=run

Usage (direct, for development):
    python3 entrypoint.py run --parameters_file workspace/parameters/energy_params.yaml \
                              --output_dir workspace/outputs
    python3 entrypoint.py run --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                              --precision NF4 --batch_size 1 --gpu_arch auto \
                              --output_dir workspace/outputs
"""
import argparse
import datetime
import json
import os
import platform
import re
import sys
import threading
import time

SCHEMA_VERSION = "ecocompute-energy/1.0"

# Website hooks (opt-in via --prefetch / --share). Both are best-effort and never
# block or alter the measurement; --share encodes the result point into the URL
# itself (no upload, no server storage, no DB) so it matches the site's
# "100% in-browser, nothing uploaded" contract.
DEFAULT_API_BASE = "https://ecocompute-estimator.zhanghongping1982.workers.dev"
DEFAULT_SITE = "https://quantenergy.tech"

# ------------------------------------------------------------------ reference --
# Compact mirror of the EcoCompute measured dataset (build/measured.csv, curves.json).
# Used ONLY for the no-GPU dry-run path so the container is testable off-hardware;
# every value it emits is flagged measurement_source="ecocompute-dataset (no local GPU)"
# and is never presented as a fresh measurement.
REFERENCE = {
    # measured FP16 absolute decode energy, J / 1k tokens ( == mJ / token )
    "fp16_energy": {
        "ada":       {"n_min": 0.5, "n_max": 3.0, "anchors": [[0.5, 1474.16], [1.1, 1600.58], [1.5, 2238.87], [3.0, 2989.22]]},
        "blackwell": {"n_min": 1.1, "n_max": 7.0, "anchors": [[1.1, 1659.0], [1.5, 2411.09], [3.0, 3382.64], [7.0, 5508.56]]},
        "turing":    {"n_min": 1.1, "n_max": 7.0, "anchors": [[1.1, 4251.21], [1.5, 5731.8], [3.0, 11267.69], [7.0, 21722.65]]},
        "ampere":    {"n_min": 7.0, "n_max": 14.0, "anchors": [[7.0, 4402.43], [9.0, 5445.12], [14.0, 7359.98]]},
    },
    # NF4/INT8 delta-vs-FP16 curve params: dE%(N) = A - S * (x/(1+x)), x = N / Nstar
    "delta": {
        "ada":       {"NF4": {"A": 106.0936, "S": 90.55, "Nstar": 0.381, "n_min": 0.5, "n_max": 3.0}},
        "blackwell": {"NF4": {"A": 45.8272, "S": 104.4224, "Nstar": 6.0749, "n_min": 1.1, "n_max": 7.0}},
        "turing":    {"NF4": {"A": 7.925, "S": 79.5646, "Nstar": 19.2186, "n_min": 1.1, "n_max": 7.0}},
        "ampere":    {"NF4": {"A": -1.0293, "S": 0.0, "Nstar": 200.0, "n_min": 7.0, "n_max": 14.0},
                      "INT8": {"A": 180.8252, "S": 127.9389, "Nstar": 10.082, "n_min": 7.0, "n_max": 14.0}},
    },
    "gpu_label": {"ada": "RTX 4090D", "blackwell": "RTX 5090", "turing": "T4", "ampere": "A800"},
}

ARCH_ALIASES = {
    "t4": "turing", "turing": "turing",
    "4090": "ada", "4090d": "ada", "ada": "ada",
    "5090": "blackwell", "blackwell": "blackwell",
    "a100": "ampere", "a800": "ampere", "ampere": "ampere",
}

# NVML product name -> architecture class, so `gpu_arch: auto` needs no flag from
# the user. First match wins, so specific cards precede the generic families
# (A100 before "RTX A6000", L40S before the RTX 40-series pattern). An unknown
# card yields None: the report then carries the raw NVML name and no fitted
# curve is claimed for it, which is the honest outcome.
ARCH_PATTERNS = [
    (r"\b(b100|b200|gb\d{3}|rtx\s*50\d0)\b",                      "blackwell"),
    (r"\b(h100|h200|h800|gh200)\b",                                "hopper"),
    (r"\b(l4|l40s?|ada)\b|\brtx\s*(40\d0|(?:2000|4000|5000|6000)\s*ada)\b", "ada"),
    (r"\b(a100|a800|a10g?|a16|a2|a30|a40)\b|\brtx\s*(a\d{4}|30\d0)\b",      "ampere"),
    (r"\b(t4|t400|t600|t1000)\b|\b(quadro|titan)\s*rtx\b|\brtx\s*20\d0\b",  "turing"),
    (r"\bv100\b|\btitan\s*v\b",                                    "volta"),
]


def norm_arch(s):
    if not s:
        return None
    s = str(s).lower().replace("rtx", "").strip()
    for k, v in ARCH_ALIASES.items():
        if k in s:
            return v
    return s if s in REFERENCE["fp16_energy"] else None


def detect_arch(gpu_name):
    """Architecture class implied by an NVML product name, or None if unrecognised."""
    if not gpu_name:
        return None
    s = re.sub(r"[-_]+", " ", str(gpu_name).lower())
    s = re.sub(r"\b(nvidia|geforce|tesla)\b", " ", s)
    for pattern, arch in ARCH_PATTERNS:
        if re.search(pattern, s):
            return arch
    return None


def probe_gpu_name():
    """NVML product name of device 0, or None when NVML/driver is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return _nvml_gpu_name(pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None


def reference_pins():
    """The published pins (requirements.txt), as {package: version}."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    pins = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if "==" in line:
                    name, version = line.split("==", 1)
                    pins[name.strip().lower()] = version.strip()
    except OSError:
        pass
    return pins


def _pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def collect_software(compare_pins=True):
    """Versions the measurement actually ran on, and how they differ from the pins.

    Quantization kernels change between bitsandbytes/transformers releases, so a
    report is only comparable with another one if these match. Recording them
    makes that checkable by the reader instead of trusted from the installer.

    ``compare_pins`` is off for dataset-derived reports: nothing was executed
    there, so a pin diff would describe an environment that ran no kernels.
    """
    versions = {name: _pkg_version(name) for name in
                ("torch", "transformers", "bitsandbytes", "accelerate",
                 "nvidia-ml-py", "sentencepiece")}
    software = {
        "python": platform.python_version(),
        "packages": {k: v for k, v in versions.items() if v},
    }
    try:
        import torch
        software["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            software["nvidia_driver"] = driver.decode() if isinstance(driver, bytes) else str(driver)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    if not compare_pins:
        return software

    pins = reference_pins()
    # torch is deliberately not pinned outside the image: it has to match the
    # host driver/CUDA. Everything else should equal the published pins.
    compared = {name: pins[name] for name in versions if name in pins and name != "torch"}
    differs = sorted(
        "%s %s != %s" % (name, versions.get(name) or "missing", pinned)
        for name, pinned in compared.items()
        if versions.get(name) != pinned
    )
    if compared:
        software["reference_pins"] = compared
        software["matches_reference_pins"] = not differs
        if differs:
            software["differs_from_reference_pins"] = differs
            software["comparability_note"] = (
                "This run did not use the published pins; bitsandbytes/transformers "
                "quantization kernels change between releases, so vs_fp16_energy_pct "
                "may not be directly comparable with reports produced by the image."
            )
    return software


def _interp_loglog(anchors, N):
    import math
    for n, e in anchors:
        if abs(n - N) < 1e-9:
            return e, True
    a = sorted(anchors)
    lo = hi = None
    for n, e in a:
        if n < N:
            lo = (n, e)
        if n > N and hi is None:
            hi = (n, e)
    if lo and hi:
        p0, p1 = lo, hi
    elif not lo:
        p0, p1 = a[0], a[1]
    else:
        p0, p1 = a[-2], a[-1]
    slope = (math.log(p1[1]) - math.log(p0[1])) / (math.log(p1[0]) - math.log(p0[0]))
    return math.exp(math.log(p0[1]) + slope * (math.log(N) - math.log(p0[0]))), False


def reference_estimate(params_b, arch, precision):
    """No-GPU fallback: derive energy_per_token + vs_fp16 from the measured dataset."""
    fe = REFERENCE["fp16_energy"].get(arch)
    if not fe:
        return None
    fp16_mj, _ = _interp_loglog(fe["anchors"], params_b)
    in_range = fe["n_min"] <= params_b <= fe["n_max"]
    if precision == "FP16":
        delta = 0.0
    else:
        dc = REFERENCE["delta"].get(arch, {}).get(precision)
        if not dc:
            return None
        x = params_b / dc["Nstar"]
        delta = dc["A"] - dc["S"] * (x / (1.0 + x))
        in_range = in_range and (dc["n_min"] <= params_b <= dc["n_max"])
    quant_mj = fp16_mj * (1.0 + delta / 100.0)
    # No-GPU path: the delta comes from the fitted curve (not a fresh measurement),
    # so never label it "measured" — interpolated inside the measured range, else extrapolated.
    basis = "interpolated" if in_range else "extrapolated"
    return {
        "energy_per_token_mj": round(quant_mj, 2),
        "fp16_energy_per_token_mj": round(fp16_mj, 2),
        "vs_fp16_energy_pct": round(delta, 1),
        "basis": basis,
    }


# --------------------------------------------------------------- measurement --
class PowerSampler(threading.Thread):
    """Samples GPU power via NVML at a fixed rate; integrates energy (trapezoid).

    Robust across architectures (Turing/Ampere/Ada/Hopper/Blackwell): a single
    failed sample never kills the thread, and repeated failures set ``error`` so
    the caller can decide to discard the run instead of reporting garbage.
    """

    def __init__(self, handle, hz=10):
        super().__init__(daemon=True)
        self._pynvml = sys.modules["pynvml"]
        self.handle = handle
        self.period = 1.0 / hz
        self.samples = []          # (t_seconds, watts)
        self.dropped = 0           # count of failed reads
        self.error = None          # set if power telemetry is unusable
        # NB: named _stop_evt, not _stop, to avoid shadowing Thread._stop().
        self._stop_evt = threading.Event()

    def run(self):
        t0 = time.time()
        consecutive = 0
        while not self._stop_evt.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
                self.samples.append((time.time() - t0, mw / 1000.0))
                consecutive = 0
            except Exception as e:  # NVMLError / unsupported field on this card
                self.dropped += 1
                consecutive += 1
                # Bail out early if the card simply cannot report power at all.
                if consecutive >= 5 and not self.samples:
                    self.error = f"NVML power telemetry unavailable: {e}"
                    break
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    def energy_joules(self):
        j = 0.0
        for (t1, w1), (t2, w2) in zip(self.samples, self.samples[1:]):
            j += (w1 + w2) / 2.0 * (t2 - t1)
        return j

    def avg_watts(self):
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else 0.0


def _quant_config(precision):
    from transformers import BitsAndBytesConfig
    if precision == "NF4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=__import__("torch").float16)
    if precision == "INT8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class PowerUnsupportedError(RuntimeError):
    """Raised when the GPU/driver cannot report power via NVML (e.g. some older cards)."""


def _nvml_gpu_name(pynvml, handle):
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return name


def _assert_power_readable(pynvml, handle):
    """Fail fast (before loading the model) if this card can't report power."""
    try:
        pynvml.nvmlDeviceGetPowerUsage(handle)
    except Exception as e:  # NVMLError_NotSupported on some Turing/consumer/vGPU cards
        raise PowerUnsupportedError(
            f"this GPU/driver does not expose NVML power telemetry ({e}); "
            "cannot produce a measured energy figure"
        ) from e


def measure_once(model_name, precision, batch_size, tokens, iterations, warmup, hz):
    """Load, (quantize,) warm up, then measure energy over `iterations` decode runs.

    Raises PowerUnsupportedError if the card cannot report power, so the caller can
    fall back to the published-dataset reference path instead of crashing.
    """
    import torch
    import pynvml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = _nvml_gpu_name(pynvml, handle)
        _assert_power_readable(pynvml, handle)
        return _measure_with_handle(
            pynvml, handle, gpu_name, model_name, precision, batch_size,
            tokens, iterations, warmup, hz, torch, AutoModelForCausalLM, AutoTokenizer)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _measure_with_handle(pynvml, handle, gpu_name, model_name, precision, batch_size,
                         tokens, iterations, warmup, hz, torch,
                         AutoModelForCausalLM, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.float16, "device_map": "cuda"}
    qc = _quant_config(precision)
    if qc is not None:
        kwargs["quantization_config"] = qc
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    prompt = ["Explain in detail how large language models work."] * batch_size
    enc = tok(prompt, return_tensors="pt", padding=True).to("cuda")

    gen = dict(max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False,
               pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(**enc, **gen)
    torch.cuda.synchronize()

    sampler = PowerSampler(handle, hz=hz)
    sampler.start()
    t0 = time.time()
    total_new = 0
    with torch.no_grad():
        for _ in range(iterations):
            out = model.generate(**enc, **gen)
            total_new += (out.shape[1] - enc["input_ids"].shape[1]) * batch_size
    torch.cuda.synchronize()
    wall = time.time() - t0
    sampler.stop()

    if sampler.error or len(sampler.samples) < 2:
        raise PowerUnsupportedError(
            sampler.error or "NVML returned too few power samples to integrate energy")

    joules = sampler.energy_joules()
    return {
        "gpu_name": gpu_name,
        "total_energy_joules": round(joules, 3),
        "tokens_generated": total_new,
        "energy_per_token_mj": round(joules / total_new * 1000.0, 3) if total_new else None,
        "avg_power_watts": round(sampler.avg_watts(), 1),
        "throughput_tokens_per_s": round(total_new / wall, 1) if wall else None,
        "wall_seconds": round(wall, 3),
        "dropped_samples": sampler.dropped,
    }


# ------------------------------------------------------------------- reporting --
def build_report(p, measured, ref, fp16_measured=None, measure_error=None):
    arch = norm_arch(p["gpu_arch"])
    scenario = "SingleStream" if int(p["batch_size"]) == 1 else "Offline"
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "ecocompute-energy-methodology",
        "follows_mlcommons_energy_reporting_conventions": True,
        "certified_benchmark_result": False,
        "scenario": scenario,
        "scenario_note": (
            "Nominal label derived from batch size (1 -> SingleStream, else Offline). "
            "This is NOT enforced by MLPerf LoadGen: the container runs its own "
            "warmup/iterations loop and does not apply LoadGen timing constraints, so "
            "results are a supplemental energy methodology, not a certified benchmark."
        ),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "system_under_test": {
            "gpu": (measured or {}).get("gpu_name") or REFERENCE["gpu_label"].get(arch, p["gpu_arch"]),
            "gpu_arch": arch or p["gpu_arch"],
            "accelerator_count": 1,
            "host": platform.platform(),
        },
        "workload": {
            "model_name": p["model_name"],
            "params_b": p.get("params_b"),
            "precision": p["precision"],
            "batch_size": int(p["batch_size"]),
            "context_length": int(p.get("context_length", 2048)),
        },
        "measurement": {
            "method": ("NVML on-device power sampling" if measured
                       else "reference estimate from published dataset (no on-device measurement)"),
            "sample_rate_hz": int(p.get("sample_rate_hz", 10)),
            "tokens_per_run": int(p.get("tokens", 256)),
            "iterations": int(p.get("iterations", 10)),
            "warmup": int(p.get("warmup", 2)),
            "iterations_note": (
                "Decode iterations within one run, not independent trials: a single "
                "report is n=1 however high this is."
            ),
        },
        "software": collect_software(compare_pins=bool(measured)),
        "provenance": {
            "tool": "https://quantenergy.tech",
            "dataset_doi": "10.5281/zenodo.21066652",
            "paper_ssrn": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6854700",
            "code": "https://github.com/hongping-zh/ecocompute-ai",
        },
        "notice": (
            "Not a certified benchmark result. Energy reported in an MLCommons-style "
            "format. MLCOMMONS, MLPERF, and MLCUBE are trademarks of MLCommons "
            "Association, referenced here nominatively."
        ),
    }
    if measured:
        report["measurement_source"] = "direct-nvml"
        report["results"] = {
            "total_energy_joules": measured["total_energy_joules"],
            "tokens_generated": measured["tokens_generated"],
            "energy_per_token_mj": measured["energy_per_token_mj"],
            "avg_power_watts": measured["avg_power_watts"],
            "throughput_tokens_per_s": measured["throughput_tokens_per_s"],
            "basis": "measured",
        }
        if fp16_measured and measured["energy_per_token_mj"]:
            base = fp16_measured["energy_per_token_mj"]
            report["results"]["fp16_energy_per_token_mj"] = base
            report["results"]["vs_fp16_energy_pct"] = round(
                (measured["energy_per_token_mj"] - base) / base * 100.0, 1)
    else:
        if measure_error:
            source = "ecocompute-dataset (on-device measurement failed)"
            note = ("On-device NVML measurement could not run on this GPU/driver "
                    f"({measure_error}) — values derived from the published EcoCompute "
                    "measurements, not a fresh on-device measurement.")
        else:
            source = "ecocompute-dataset (no local GPU)"
            note = ("No local NVIDIA GPU detected — values derived from the published "
                    "EcoCompute measurements, not a fresh on-device measurement.")
        report["measurement_source"] = source
        report["results"] = {
            "energy_per_token_mj": ref["energy_per_token_mj"],
            "fp16_energy_per_token_mj": ref["fp16_energy_per_token_mj"],
            "vs_fp16_energy_pct": ref["vs_fp16_energy_pct"],
            "basis": ref["basis"],
            "note": note,
        }
    return report


def load_params(args):
    p = {}
    if args.parameters_file and os.path.exists(args.parameters_file):
        import yaml
        with open(args.parameters_file) as f:
            p = yaml.safe_load(f) or {}
    # --model is an alias for --model_name (docker-run / dev convenience)
    if getattr(args, "model", None):
        p["model_name"] = args.model
    for k in ("model_name", "precision", "gpu_arch"):
        if getattr(args, k):
            p[k] = getattr(args, k)
    for k in ("batch_size", "params_b", "tokens", "iterations", "warmup",
              "sample_rate_hz", "context_length"):
        v = getattr(args, k)
        if v is not None:
            p[k] = v
    p.setdefault("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.setdefault("precision", "NF4")
    p.setdefault("gpu_arch", "blackwell")
    p.setdefault("batch_size", 1)
    p.setdefault("tokens", 256)
    p.setdefault("iterations", 10)
    p.setdefault("warmup", 2)
    p.setdefault("sample_rate_hz", 10)
    p.setdefault("params_b", None)
    return p


# ------------------------------------------------------------------ validation --
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "schema", "energy.schema.json")

_JSON_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
               "integer": int, "number": (int, float), "null": type(None)}


def _check_against_schema(node, schema, path=""):
    """Minimal draft-07 subset check (required/type/enum/const/minimum/items/
    additionalProperties).

    Only used when the `jsonschema` package is absent, so the runtime image does
    not have to carry it: the report is still verified before we tell the user it
    is schema-valid, just with a checker covering exactly the constructs
    schema/energy.schema.json uses.
    """
    errors = []
    types = schema.get("type")
    if types:
        names = [types] if isinstance(types, str) else list(types)
        allowed = []
        for name in names:
            t = _JSON_TYPES.get(name)
            allowed.extend(t if isinstance(t, tuple) else [t] if t else [])
        allowed = tuple(allowed)
        # bool is a subclass of int in Python; JSON Schema treats them as distinct.
        if not isinstance(node, allowed) or (isinstance(node, bool) and bool not in allowed):
            return ["%s: expected %s, got %s" % (path or "<root>", types, type(node).__name__)]
    if "const" in schema and node != schema["const"]:
        errors.append("%s: must be %r, got %r" % (path, schema["const"], node))
    if "enum" in schema and node not in schema["enum"]:
        errors.append("%s: %r not in %s" % (path, node, schema["enum"]))
    if "minimum" in schema and isinstance(node, (int, float)) and node < schema["minimum"]:
        errors.append("%s: %s below minimum %s" % (path, node, schema["minimum"]))
    if isinstance(node, dict):
        for key in schema.get("required", []):
            if key not in node:
                errors.append("%s: missing required field '%s'" % (path or "<root>", key))
        properties = schema.get("properties", {})
        for key, sub in properties.items():
            if key in node:
                errors += _check_against_schema(node[key], sub,
                                                "%s.%s" % (path, key) if path else key)
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for key, value in node.items():
                if key not in properties:
                    errors += _check_against_schema(value, extra,
                                                    "%s.%s" % (path, key) if path else key)
    if isinstance(node, list) and isinstance(schema.get("items"), dict):
        for i, value in enumerate(node):
            errors += _check_against_schema(value, schema["items"], "%s[%d]" % (path, i))
    return errors


def validate_report(report, schema_path=SCHEMA_PATH):
    """Check the report against the shipped schema -> (ok, validator, detail).

    ok is None when the schema file is not available next to the entrypoint.
    """
    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except Exception as e:
        return None, "none", "schema unavailable (%s)" % e
    try:
        import jsonschema
    except ImportError:
        errors = _check_against_schema(report, schema)
        return (not errors), "builtin", "; ".join(errors[:4])
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        return False, "jsonschema", e.message
    return True, "jsonschema", ""


def gpu_available():
    try:
        import pynvml  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_arch(p):
    """Fill in `gpu_arch` when it is missing or "auto" by asking NVML which card
    this is, so a zero-flag run cannot mislabel the architecture. An explicit
    value always wins, and an unrecognised card is recorded by its NVML name
    rather than guessed into a family."""
    requested = str(p.get("gpu_arch") or "").strip()
    if requested and requested.lower() != "auto":
        return
    name = probe_gpu_name()
    detected = detect_arch(name)
    if detected:
        p["gpu_arch"] = detected
        print("[ecocompute-mlcube] gpu_arch=auto -> '%s' (NVML device name: %s)"
              % (detected, name))
        return
    p["gpu_arch"] = name or "unknown"
    print("[ecocompute-mlcube] gpu_arch=auto could not be mapped to a known "
          "architecture (NVML name: %s). The measurement itself is unaffected; "
          "pass --gpu_arch to compare it against a fitted curve."
          % (name or "unavailable"), file=sys.stderr)


def run(args):
    p = load_params(args)
    os.makedirs(args.output_dir, exist_ok=True)
    resolve_arch(p)
    arch = norm_arch(p["gpu_arch"])
    if getattr(args, "prefetch", False):
        prefetch_prediction(getattr(args, "api_base", DEFAULT_API_BASE),
                            _effective_params_b(p), arch, p["precision"],
                            int(p["batch_size"]))
    measured = fp16_measured = ref = None
    measure_error = None
    if gpu_available() and not args.dry_run:
        try:
            measured = measure_once(p["model_name"], p["precision"], int(p["batch_size"]),
                                    int(p["tokens"]), int(p["iterations"]), int(p["warmup"]),
                                    int(p["sample_rate_hz"]))
            if p["precision"] != "FP16":
                fp16_measured = measure_once(p["model_name"], "FP16", int(p["batch_size"]),
                                             int(p["tokens"]), int(p["iterations"]),
                                             int(p["warmup"]), int(p["sample_rate_hz"]))
        except Exception as e:  # NVML/driver/arch/OOM issue -> fall back, don't crash
            measure_error = str(e)
            measured = fp16_measured = None
            print(f"[ecocompute-mlcube] on-device measurement failed ({e}); "
                  "falling back to published-dataset reference", file=sys.stderr)

    if measured is None:
        ref = reference_estimate(float(p["params_b"] or _guess_params(p["model_name"])),
                                 arch, p["precision"])
        if ref is None:
            print(f"[ecocompute-mlcube] no reference for arch={arch} precision={p['precision']}",
                  file=sys.stderr)
            ref = {"energy_per_token_mj": None, "fp16_energy_per_token_mj": None,
                   "vs_fp16_energy_pct": None, "basis": "unavailable"}

    report = build_report(p, measured, ref, fp16_measured, measure_error=measure_error)
    link = share_url(getattr(args, "site", DEFAULT_SITE), report) if getattr(args, "share", False) else None
    if link:
        report["share_url"] = link
    out = os.path.join(args.output_dir, "energy.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ecocompute-mlcube] wrote {out} (source={report['measurement_source']})")

    differs = report.get("software", {}).get("differs_from_reference_pins")
    if differs and measured:
        print("[ecocompute-mlcube] note: this run did not use the published pins (%s). "
              "The measurement is real, but quantization kernels change between "
              "releases, so treat vs_fp16 as not directly comparable with image runs. "
              "The versions are recorded in the report."
              % "; ".join(differs), file=sys.stderr)

    ok, validator, detail = validate_report(report)
    if ok:
        print("[ecocompute-mlcube] schema: valid (%s, checked with %s)"
              % (report["schema_version"], validator))
    elif ok is None:
        print("[ecocompute-mlcube] schema: not checked - %s" % detail, file=sys.stderr)
    else:
        print("[ecocompute-mlcube] schema: INVALID (%s) - %s" % (validator, detail),
              file=sys.stderr)
    if getattr(args, "share", False):
        if link:
            with open(os.path.join(args.output_dir, "share_url.txt"), "w") as f:
                f.write(link + "\n")
            print("[ecocompute-mlcube] share this result (overlays your point on the "
                  "crossover curve, 100% in-browser):\n  " + link)
        else:
            print("[ecocompute-mlcube] --share: nothing to overlay - need a "
                  "vs_fp16_energy_pct and a known model size (non-FP16 run).",
                  file=sys.stderr)
    return report


def _guess_params(model_name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name or "")
    return float(m.group(1)) if m else 1.1


def _effective_params_b(p):
    """params_b from params, or a best-effort guess from the model name."""
    try:
        n = float(p.get("params_b")) if p.get("params_b") is not None else None
    except (TypeError, ValueError):
        n = None
    return n if (n and n > 0) else _guess_params(p.get("model_name"))


def prefetch_prediction(api_base, params_b, arch, precision, batch, timeout=6.0):
    """Ask the website estimator (/v1/estimate) for its predicted energy delta.

    Best-effort only: prints the site's prediction so a user can eyeball it
    against the local measurement. Any network/timeout error is swallowed and
    never blocks the run.
    """
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({"params_b": params_b, "arch": arch or "",
                                "precision": precision, "batch": batch})
    url = api_base.rstrip("/") + "/v1/estimate?" + q
    req = urllib.request.Request(url, headers={
        "User-Agent": "ecocompute-mlcube/1.0 (+https://github.com/hongping-zh/ecocompute-mlcube)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except Exception as e:  # offline / DNS / timeout / HTTP error
        print(f"[ecocompute-mlcube] prefetch skipped (site estimate unavailable: {e})",
              file=sys.stderr)
        return None
    dp = data.get("delta_pct")
    if dp is None:
        print(f"[ecocompute-mlcube] prefetch: site returned no prediction "
              f"({data.get('error', 'unknown')})", file=sys.stderr)
        return None
    print(f"[ecocompute-mlcube] site prediction (/v1/estimate): vs_fp16 ~ {dp:+.1f}% "
          f"(basis: {data.get('basis', '?')}) - will compare with the local result")
    return data


def share_url(site, report):
    """Encode the report's overlay point into a client-side deeplink.

    The point (arch, params_b, vs_fp16_pct, basis, model, gpu) is base64url-JSON
    encoded straight into the query string, so the site restores it purely in the
    browser - nothing is uploaded and no server/DB stores anything.
    """
    import base64
    res = report.get("results") or {}
    w = report.get("workload") or {}
    sut = report.get("system_under_test") or {}
    d_e = res.get("vs_fp16_energy_pct")
    n = w.get("params_b")
    if n is None:
        n = _guess_params(w.get("model_name"))
    if d_e is None or not n:
        return None
    payload = {
        "m": w.get("model_name") or "your model",
        "N": round(float(n), 4),
        "a": sut.get("gpu_arch") or "",
        "p": w.get("precision") or "NF4",
        "e": round(float(d_e), 2),
        "b": res.get("basis") or "unavailable",
        "g": sut.get("gpu") or "",
    }
    blob = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return site.rstrip("/") + "/?tab=run&overlay=" + blob


def _add_run_args(parser):
    parser.add_argument("--parameters_file", default=None)
    parser.add_argument("--output_dir", default="workspace/outputs")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model", default=None, help="alias for --model_name")
    parser.add_argument("--precision", default=None, choices=[None, "FP16", "NF4", "INT8"])
    parser.add_argument("--gpu_arch", default=None,
                       help="turing | ampere | ada | hopper | blackwell, or "
                            "'auto' (default in the shipped parameters) to "
                            "detect it from the NVML device name")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--params_b", type=float, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--sample_rate_hz", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="force the no-GPU reference path")
    parser.add_argument("--prefetch", action="store_true",
                        help="before measuring, ask the website estimator (/v1/estimate) "
                             "for its predicted energy delta vs FP16 (best-effort, offline-safe)")
    parser.add_argument("--share", action="store_true",
                        help="after measuring, print a shareable quantenergy.tech overlay link "
                             "(point encoded in the URL; nothing uploaded)")
    parser.add_argument("--api_base", default=DEFAULT_API_BASE,
                        help="estimator API base used by --prefetch")
    parser.add_argument("--site", default=DEFAULT_SITE,
                        help="website base used to build --share overlay links")


def main():
    ap = argparse.ArgumentParser(description="EcoCompute energy-methodology MLCube")
    sub = ap.add_subparsers(dest="task", required=True)
    # MLCube task name is `energy_estimate`; `run` kept as an alias.
    for name, help_ in (("energy_estimate", "measure one config and write energy.json"),
                        ("run", "alias of energy_estimate")):
        _add_run_args(sub.add_parser(name, help=help_))
    args = ap.parse_args()
    if args.task in ("energy_estimate", "run"):
        run(args)


if __name__ == "__main__":
    main()

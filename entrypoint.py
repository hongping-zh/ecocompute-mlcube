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
                              --precision NF4 --batch_size 1 --gpu_arch blackwell \
                              --output_dir workspace/outputs
"""
import argparse
import datetime
import json
import os
import platform
import sys
import threading
import time

SCHEMA_VERSION = "ecocompute-energy/1.0"

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


def norm_arch(s):
    if not s:
        return None
    s = str(s).lower().replace("rtx", "").strip()
    for k, v in ARCH_ALIASES.items():
        if k in s:
            return v
    return s if s in REFERENCE["fp16_energy"] else None


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
    """Samples GPU power via NVML at a fixed rate; integrates energy (trapezoid)."""

    def __init__(self, handle, hz=10):
        super().__init__(daemon=True)
        self._pynvml = sys.modules["pynvml"]
        self.handle = handle
        self.period = 1.0 / hz
        self.samples = []          # (t_seconds, watts)
        self._stop = threading.Event()

    def run(self):
        t0 = time.time()
        while not self._stop.is_set():
            w = self._pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
            self.samples.append((time.time() - t0, w))
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
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


def measure_once(model_name, precision, batch_size, tokens, iterations, warmup, hz):
    """Load, (quantize,) warm up, then measure energy over `iterations` decode runs."""
    import torch
    import pynvml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

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

    joules = sampler.energy_joules()
    pynvml.nvmlShutdown()
    return {
        "gpu_name": gpu_name,
        "total_energy_joules": round(joules, 3),
        "tokens_generated": total_new,
        "energy_per_token_mj": round(joules / total_new * 1000.0, 3) if total_new else None,
        "avg_power_watts": round(sampler.avg_watts(), 1),
        "throughput_tokens_per_s": round(total_new / wall, 1) if wall else None,
        "wall_seconds": round(wall, 3),
    }


# ------------------------------------------------------------------- reporting --
def build_report(p, measured, ref, fp16_measured=None):
    arch = norm_arch(p["gpu_arch"])
    scenario = "SingleStream" if int(p["batch_size"]) == 1 else "Offline"
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "ecocompute-energy-methodology",
        "follows_mlcommons_energy_reporting_conventions": True,
        "certified_benchmark_result": False,
        "scenario": scenario,
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
        },
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
        report["measurement_source"] = "ecocompute-dataset (no local GPU)"
        report["results"] = {
            "energy_per_token_mj": ref["energy_per_token_mj"],
            "fp16_energy_per_token_mj": ref["fp16_energy_per_token_mj"],
            "vs_fp16_energy_pct": ref["vs_fp16_energy_pct"],
            "basis": ref["basis"],
            "note": ("No local NVIDIA GPU detected — values derived from the published "
                     "EcoCompute measurements, not a fresh on-device measurement."),
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


def gpu_available():
    try:
        import pynvml  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run(args):
    p = load_params(args)
    os.makedirs(args.output_dir, exist_ok=True)
    arch = norm_arch(p["gpu_arch"])
    measured = fp16_measured = None
    if gpu_available() and not args.dry_run:
        measured = measure_once(p["model_name"], p["precision"], int(p["batch_size"]),
                                int(p["tokens"]), int(p["iterations"]), int(p["warmup"]),
                                int(p["sample_rate_hz"]))
        if p["precision"] != "FP16":
            fp16_measured = measure_once(p["model_name"], "FP16", int(p["batch_size"]),
                                         int(p["tokens"]), int(p["iterations"]),
                                         int(p["warmup"]), int(p["sample_rate_hz"]))
        ref = None
    else:
        ref = reference_estimate(float(p["params_b"] or _guess_params(p["model_name"])),
                                 arch, p["precision"])
        if ref is None:
            print(f"[ecocompute-mlcube] no reference for arch={arch} precision={p['precision']}",
                  file=sys.stderr)
            ref = {"energy_per_token_mj": None, "fp16_energy_per_token_mj": None,
                   "vs_fp16_energy_pct": None, "basis": "unavailable"}

    report = build_report(p, measured, ref, fp16_measured)
    out = os.path.join(args.output_dir, "energy.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ecocompute-mlcube] wrote {out} (source={report['measurement_source']})")
    return report


def _guess_params(model_name):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name or "")
    return float(m.group(1)) if m else 1.1


def _add_run_args(parser):
    parser.add_argument("--parameters_file", default=None)
    parser.add_argument("--output_dir", default="workspace/outputs")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model", default=None, help="alias for --model_name")
    parser.add_argument("--precision", default=None, choices=[None, "FP16", "NF4", "INT8"])
    parser.add_argument("--gpu_arch", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--params_b", type=float, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--sample_rate_hz", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="force the no-GPU reference path")


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

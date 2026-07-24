#!/usr/bin/env python3
"""Verify an AutoDL RTX 4090D box can produce *real* NVML energy measurements.

Run via ``autodl/01_verify.sh`` (which activates the venv) or directly:

    python autodl/verify_env.py            # full check + tiny real measurement
    python autodl/verify_env.py --no-measure   # skip the tiny GPU measurement

Exit code is non-zero if any REQUIRED check fails, so it can gate CI / a sweep.
The one check that matters most for this project is *NVML power readability*:
without it the container falls back to dataset reference values instead of a
fresh on-device measurement.
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK = "\033[32mOK\033[0m"
WARN = "\033[33mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def line(status, msg):
    print(f"  [{status}] {msg}")


def check_imports():
    ok = True
    for pkg, required in (("torch", True), ("transformers", True),
                          ("accelerate", True), ("bitsandbytes", True),
                          ("pynvml", True), ("yaml", True)):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            line(OK, f"import {pkg} ({ver})")
        except Exception as e:  # noqa: BLE001
            line(FAIL if required else WARN, f"import {pkg}: {e}")
            ok = ok and not required
    return ok


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        line(FAIL, "torch.cuda.is_available() is False — no GPU / driver mismatch")
        return False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    line(OK, f"CUDA device: {name}  (compute capability {cap[0]}.{cap[1]})")
    line(OK, f"torch {torch.__version__} built for CUDA {torch.version.cuda}")
    if "4090" not in name:
        line(WARN, f"device is not an RTX 4090/4090D — gpu_arch may need overriding "
                   f"(detected '{name}', expected Ada/RTX 4090D)")
    if cap[0] != 8 or cap[1] != 9:
        line(WARN, f"compute capability {cap[0]}.{cap[1]} != 8.9 (Ada)")
    return True


def check_nvml():
    """The decisive check: can we read instantaneous GPU power via NVML?"""
    import pynvml
    try:
        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        line(FAIL, f"nvmlInit failed: {e}")
        return False
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        nm = pynvml.nvmlDeviceGetName(h)
        nm = nm.decode() if isinstance(nm, bytes) else nm
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(h)
            line(OK, f"NVML power readable on '{nm}': {mw/1000.0:.1f} W "
                     "(measured energy path AVAILABLE)")
            ok = True
        except Exception as e:  # noqa: BLE001
            line(FAIL, f"nvmlDeviceGetPowerUsage failed on '{nm}': {e} — "
                       "container would fall back to dataset reference values")
            ok = False
        return ok
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def check_bnb():
    """Confirm bitsandbytes can build a 4-bit config (NF4 path is the point)."""
    try:
        import torch
        from transformers import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16)
        line(OK, "bitsandbytes NF4 BitsAndBytesConfig constructs")
        return True
    except Exception as e:  # noqa: BLE001
        line(FAIL, f"bitsandbytes NF4 config failed: {e}")
        return False


def check_dry_run():
    """CLI wiring: entrypoint.py must produce a schema-shaped energy.json."""
    out = os.path.join(REPO, "autodl", "results", "_verify_dry")
    cmd = [sys.executable, os.path.join(REPO, "entrypoint.py"), "energy_estimate",
           "--dry_run", "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
           "--precision", "NF4", "--gpu_arch", "ada", "--params_b", "1.1",
           "--output_dir", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(os.path.join(out, "energy.json")):
        line(OK, "entrypoint dry_run produced energy.json")
        return True
    line(FAIL, f"entrypoint dry_run failed: {r.stderr.strip()[:400]}")
    return False


def tiny_measure():
    """A real (tiny) on-GPU measurement end-to-end to prove the measured path."""
    out = os.path.join(REPO, "autodl", "results", "_verify_measure")
    cmd = [sys.executable, os.path.join(REPO, "entrypoint.py"), "energy_estimate",
           "--model", "Qwen/Qwen2-0.5B", "--precision", "NF4", "--gpu_arch", "ada",
           "--params_b", "0.5", "--tokens", "16", "--iterations", "2",
           "--warmup", "1", "--output_dir", out]
    print("  running a tiny real measurement (Qwen2-0.5B, 16 tok x 2 it)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("   " + "\n   ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
    import json
    p = os.path.join(out, "energy.json")
    if r.returncode != 0 or not os.path.exists(p):
        line(FAIL, "tiny measurement did not produce energy.json")
        return False
    rep = json.load(open(p))
    basis = rep.get("results", {}).get("basis")
    src = rep.get("measurement_source")
    if basis == "measured":
        line(OK, f"tiny measurement basis=measured (source={src}) — REAL "
                 "measurements will work")
        return True
    line(WARN, f"tiny run fell back (basis={basis}, source={src}) — NVML measured "
               "path not active; check the NVML result above")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the tiny real GPU measurement")
    args = ap.parse_args()

    print("== EcoCompute AutoDL RTX 4090D verification ==")
    print("\n[imports]");     imports_ok = check_imports()
    print("\n[cuda]");        cuda_ok = check_cuda() if imports_ok else False
    print("\n[nvml power]");  nvml_ok = check_nvml() if cuda_ok else False
    print("\n[bitsandbytes]"); bnb_ok = check_bnb() if imports_ok else False
    print("\n[cli dry_run]"); dry_ok = check_dry_run()

    measure_ok = None
    if not args.no_measure and cuda_ok and nvml_ok:
        print("\n[tiny measurement]"); measure_ok = tiny_measure()
    elif not args.no_measure:
        print("\n[tiny measurement] skipped (no usable GPU/NVML)")

    print("\n== summary ==")
    required = {"imports": imports_ok, "cuda": cuda_ok, "nvml_power": nvml_ok,
                "bitsandbytes": bnb_ok, "cli_dry_run": dry_ok}
    for k, v in required.items():
        line(OK if v else FAIL, k)
    if measure_ok is not None:
        line(OK if measure_ok else WARN, "tiny_measurement")

    if all(required.values()):
        print("\nAll required checks passed — ready for:  bash autodl/02_run_sweep.sh")
        return 0
    print("\nSome required checks FAILED — fix the FAIL lines above before sweeping.")
    print("(If you only intend a no-GPU reference run, that's expected; use --dry_run.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

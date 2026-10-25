#!/usr/bin/env python3
"""EcoCompute quick smoke test — does my GPU setup work?

Usage:
    pip install torch transformers bitsandbytes accelerate pynvml sentencepiece
    python quick_test.py

Runs a minimal NF4 measurement on TinyLlama-1.1B in < 5 minutes.
Prints pass/fail and basic energy numbers. Not a full measurement —
use the Colab notebook or ecocompute-mlcube container for that.

Exit codes:
    0 — measurement succeeded, energy.json written
    1 — setup error (no GPU, NVML unavailable, OOM, etc.)
"""
import gc
import json
import os
import sys
import threading
import time

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PRECISION = "NF4"
WARMUP = 2
ITERATIONS = 3
TOKENS = 256
HZ = 100


class PowerSampler(threading.Thread):
    def __init__(self, handle, hz=100):
        super().__init__(daemon=True)
        import pynvml
        self.pynvml = pynvml
        self.handle = handle
        self.period = 1.0 / hz
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        t0 = time.time()
        while not self._stop.is_set():
            try:
                mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append((time.time() - t0, mw / 1000.0))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)

    def energy_joules(self):
        j = 0.0
        for (t1, w1), (t2, w2) in zip(self.samples, self.samples[1:]):
            j += (w1 + w2) / 2.0 * (t2 - t1)
        return j

    def avg_watts(self):
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else 0.0


def main():
    print("=" * 60)
    print("EcoCompute quick smoke test")
    print("=" * 60)

    # --- Check GPU ---
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        print(f"GPU: {name}")
        print(f"NVML power: {mw / 1000:.1f} W (idle)")
    except Exception as e:
        print(f"FAIL: GPU/NVML not available — {e}")
        return 1

    # --- Check torch ---
    try:
        import torch
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
    except Exception as e:
        print(f"FAIL: torch not working — {e}")
        return 1

    # --- Load model ---
    print(f"\nLoading {MODEL} ({PRECISION})...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        qc = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=qc, torch_dtype=torch.float16, device_map="cuda"
        )
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"FAIL: model load failed — {e}")
        return 1

    # --- Measure ---
    print(f"\nMeasuring ({WARMUP} warmup + {ITERATIONS} iterations, {TOKENS} tokens)...")
    prompt = tok(["Explain in detail how large language models work."], return_tensors="pt", padding=True).to("cuda")
    gen_kwargs = dict(
        max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False, pad_token_id=tok.pad_token_id
    )

    with torch.no_grad():
        for i in range(WARMUP):
            model.generate(**prompt, **gen_kwargs)
        torch.cuda.synchronize()

        sampler = PowerSampler(handle, hz=HZ)
        sampler.start()
        t0 = time.time()
        total_tokens = 0
        for i in range(ITERATIONS):
            out = model.generate(**prompt, **gen_kwargs)
            total_tokens += (out.shape[1] - prompt["input_ids"].shape[1])
        torch.cuda.synchronize()
        wall = time.time() - t0
        sampler.stop()

    if len(sampler.samples) < 2:
        print("FAIL: too few power samples")
        return 1

    joules = sampler.energy_joules()
    mj_per_token = joules / total_tokens * 1000
    avg_w = sampler.avg_watts()
    tok_s = total_tokens / wall

    print(f"\n{'=' * 60}")
    print(f"PASS — measurement complete")
    print(f"{'=' * 60}")
    print(f"  Energy/token:  {mj_per_token:.3f} mJ/token")
    print(f"  Avg power:     {avg_w:.1f} W")
    print(f"  Throughput:    {tok_s:.1f} tok/s")
    print(f"  Wall time:     {wall:.1f}s")
    print(f"  Samples:       {len(sampler.samples)} ({len(sampler.samples) / wall:.0f} Hz)")
    print(f"{'=' * 60}")
    print("  GPU package power only. n=1. Not a certified benchmark.")
    print("  For a full measurement (FP16 baseline, 10 iterations, schema-validated")
    print("  energy.json), use the Colab notebook or ecocompute-mlcube container.")

    # --- Write minimal energy.json ---
    report = {
        "schema_version": "ecocompute-energy/1.2",
        "benchmark": "ecocompute-energy-methodology",
        "certified_benchmark_result": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_under_test": {"gpu": name, "accelerator_count": 1},
        "workload": {"model_name": MODEL, "params_b": 1.1, "precision": PRECISION, "batch_size": 1},
        "measurement": {"method": "NVML on-device power sampling", "sample_rate_hz": HZ,
                        "iterations": ITERATIONS, "warmup": WARMUP},
        "results": {
            "total_energy_joules": round(joules, 3),
            "tokens_generated": total_tokens,
            "energy_per_token_mj": round(mj_per_token, 3),
            "avg_power_watts": round(avg_w, 1),
            "throughput_tokens_per_s": round(tok_s, 1),
            "basis": "measured",
            "note": "Quick smoke test — not a full measurement. Use Colab/container for submission.",
        },
        "provenance": {"tool": "https://quantenergy.tech", "code": "https://github.com/hongping-zh/ecocompute-mlcube"},
    }
    os.makedirs("ecocompute-out", exist_ok=True)
    with open("ecocompute-out/energy.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: ecocompute-out/energy.json (smoke test, not submission-ready)")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

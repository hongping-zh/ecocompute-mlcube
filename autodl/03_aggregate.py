#!/usr/bin/env python3
"""Aggregate the per-config energy.json reports from a sweep into publishable
artifacts, and (optionally) compare against the anchors currently on
quantenergy.tech.

    python autodl/03_aggregate.py                    # read autodl/results
    python autodl/03_aggregate.py --results DIR --out DIR

Writes into <out> (default autodl/results/aggregate):
  * results.csv        — one row per (model, precision): energy/power/throughput,
                         vs_fp16, basis, source — the table for the paper.
  * site_dataset.json  — website-format block for GPU_LABEL, i.e.
                         {label, models:[{name,size,e:{FP16,NF4,INT8}}]},
                         paste-ready into the CURVES data on quantenergy.tech.
  * curves_anchors.json— FP16 absolute anchors [[N, mJ/token]] + per-precision
                         vs_fp16 anchors, the shape entrypoint.py's REFERENCE
                         and the estimator use.
Prints a summary and, for models that overlap the published site anchors, the
percentage difference between your fresh measurement and the published value.
"""
import argparse
import csv
import glob
import json
import os

# Published RTX 4090D (Ada) anchors currently shown on quantenergy.tech, for a
# sanity cross-check of a fresh sweep. mJ/token ( == J / 1k tokens ).
PUBLISHED_ADA = {
    0.5: {"FP16": 1474.16, "NF4": 2301.07},
    1.1: {"FP16": 1600.58, "NF4": 2134.35},
    1.5: {"FP16": 2238.87, "NF4": 3103.12},
    3.0: {"FP16": 2989.22, "NF4": 3743.12},
}


def load_reports(results_dir):
    reports = []
    for p in sorted(glob.glob(os.path.join(results_dir, "**", "energy.json"),
                              recursive=True)):
        # skip the verifier's scratch outputs
        if os.sep + "_verify" in p:
            continue
        try:
            reports.append((p, json.load(open(p))))
        except Exception as e:  # noqa: BLE001
            print(f"  !! skipping unreadable {p}: {e}")
    return reports


def row_of(rep):
    w = rep.get("workload", {})
    r = rep.get("results", {})
    sut = rep.get("system_under_test", {})
    return {
        "model_name": w.get("model_name"),
        "params_b": w.get("params_b"),
        "precision": w.get("precision"),
        "batch_size": w.get("batch_size"),
        "gpu": sut.get("gpu"),
        "gpu_arch": sut.get("gpu_arch"),
        "energy_per_token_mj": r.get("energy_per_token_mj"),
        "fp16_energy_per_token_mj": r.get("fp16_energy_per_token_mj"),
        "vs_fp16_energy_pct": r.get("vs_fp16_energy_pct"),
        "avg_power_watts": r.get("avg_power_watts"),
        "throughput_tokens_per_s": r.get("throughput_tokens_per_s"),
        "total_energy_joules": r.get("total_energy_joules"),
        "tokens_generated": r.get("tokens_generated"),
        "basis": r.get("basis"),
        "measurement_source": rep.get("measurement_source"),
        "timestamp_utc": rep.get("timestamp_utc"),
    }


def write_csv(rows, path):
    cols = ["model_name", "params_b", "precision", "batch_size", "gpu", "gpu_arch",
            "energy_per_token_mj", "fp16_energy_per_token_mj", "vs_fp16_energy_pct",
            "avg_power_watts", "throughput_tokens_per_s", "total_energy_joules",
            "tokens_generated", "basis", "measurement_source", "timestamp_utc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["params_b"] or 0, r["precision"] or "")):
            w.writerow(r)


def build_site_dataset(rows, label):
    """Collapse rows into the website's per-GPU {name,size,e:{...}} structure."""
    by_size = {}
    for r in rows:
        n = r["params_b"]
        e = r["energy_per_token_mj"]
        if n is None or e is None:
            continue
        m = by_size.setdefault(n, {"name": short_name(r["model_name"]),
                                   "size": n, "e": {}})
        m["e"][r["precision"]] = round(float(e), 2)
    models = [by_size[k] for k in sorted(by_size)]
    return {label: {"label": f"{label} · Ada · 24GB", "models": models}}


def build_anchors(rows):
    fp16 = sorted((r["params_b"], round(float(r["energy_per_token_mj"]), 2))
                  for r in rows
                  if r["precision"] == "FP16" and r["params_b"] is not None
                  and r["energy_per_token_mj"] is not None)
    delta = {}
    for r in rows:
        if r["precision"] in ("NF4", "INT8") and r["vs_fp16_energy_pct"] is not None:
            delta.setdefault(r["precision"], []).append(
                [r["params_b"], round(float(r["vs_fp16_energy_pct"]), 2)])
    for k in delta:
        delta[k].sort()
    return {"arch": "ada", "fp16_energy_anchors": fp16, "vs_fp16_pct_anchors": delta}


def short_name(model_id):
    base = (model_id or "").split("/")[-1]
    return base.replace("-Chat-v1.0", "").replace("-Instruct", "")


def compare_published(rows):
    print("\n== cross-check vs published quantenergy.tech RTX 4090D anchors ==")
    print(f"  {'N(B)':>5} {'prec':>5} {'measured':>10} {'published':>10} {'diff%':>8}")
    any_row = False
    for r in sorted(rows, key=lambda r: (r["params_b"] or 0, r["precision"] or "")):
        n, prec, e = r["params_b"], r["precision"], r["energy_per_token_mj"]
        pub = PUBLISHED_ADA.get(n, {}).get(prec) if n is not None else None
        if pub and e:
            diff = (float(e) - pub) / pub * 100.0
            print(f"  {n:>5} {prec:>5} {float(e):>10.2f} {pub:>10.2f} {diff:>+7.1f}%")
            any_row = True
    if not any_row:
        print("  (no overlapping model sizes to compare — likely a new/extended sweep)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=os.environ.get("GPU_LABEL", "RTX 4090D"))
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "aggregate")
    os.makedirs(out, exist_ok=True)

    reports = load_reports(args.results)
    if not reports:
        print(f"No energy.json found under {args.results}. Run 02_run_sweep.sh first.")
        return 1
    rows = [row_of(rep) for _, rep in reports]

    measured = sum(1 for r in rows if r["basis"] == "measured")
    print(f"Loaded {len(rows)} reports ({measured} measured, "
          f"{len(rows) - measured} reference/fallback).")
    if measured == 0:
        print("!! WARNING: no report has basis='measured'. These are dataset "
              "reference values (no on-device NVML), NOT fresh measurements.")

    write_csv(rows, os.path.join(out, "results.csv"))
    json.dump(build_site_dataset(rows, args.label),
              open(os.path.join(out, "site_dataset.json"), "w"), indent=2)
    json.dump(build_anchors(rows),
              open(os.path.join(out, "curves_anchors.json"), "w"), indent=2)

    print("\nWrote:")
    for f in ("results.csv", "site_dataset.json", "curves_anchors.json"):
        print(f"  {os.path.join(out, f)}")

    compare_published(rows)
    print("\nNext: paste site_dataset.json into the CURVES data on quantenergy.tech,")
    print("and/or use results.csv + curves_anchors.json for the paper's tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

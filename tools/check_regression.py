#!/usr/bin/env python3
"""Check a measured energy.json against the published curve.

This is a regression check for the container, not a validation of the curve:
it answers "does this build, on this GPU, still land where the published fit
says it should?" A pass means the entrypoint is not silently broken; it does
not make the run a replication, and one report is n=1 regardless of how many
decode iterations it did.

It grades against the published anchor(s) of the same model size when there
are any, and against the fitted curve only when there are none: the fit is a
smooth curve through several sizes and can sit well off any single size, so
gating on it would fail a perfectly healthy build.

It deliberately refuses to grade anything that is not a real local
measurement -- an interpolated report is *derived from* the curve, so
comparing it with the curve is circular and would always pass.

Usage:
    python tools/check_regression.py ecocompute-out/energy.json
    python tools/check_regression.py report.json --k 2 --curves ./curves.json

Exit codes:
    0  within the band
    1  outside the band (investigate before publishing anything)
    2  cannot be graded (not a measurement, no curve for this arch/precision, ...)
"""
import argparse
import json
import sys
import urllib.request

CURVES_URL = "https://quantenergy.tech/curves.json"


def load_curves(src):
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=30) as r:  # noqa: S310
            return json.loads(r.read().decode("utf-8"))
    with open(src) as f:
        return json.load(f)


def predict(curve, n_params):
    x = n_params / curve["Nstar"]
    return curve["A"] - curve["S"] * (x / (1.0 + x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to an energy.json written by entrypoint.py")
    ap.add_argument("--curves", default=CURVES_URL,
                    help="curves.json URL or path (default: the published one)")
    ap.add_argument("--k", type=float, default=2.0,
                    help="band width in residual std devs (default 2)")
    a = ap.parse_args(argv)

    with open(a.report) as f:
        rep = json.load(f)

    res = rep.get("results", {})
    basis = res.get("basis")
    source = rep.get("measurement_source", "")
    if basis != "measured" or not source.startswith("direct-nvml"):
        print("SKIP: basis=%r source=%r -- this report is derived from the published\n"
              "      dataset, so checking it against the published curve is circular."
              % (basis, source))
        return 2

    de = res.get("vs_fp16_energy_pct")
    if de is None:
        print("SKIP: no vs_fp16_energy_pct (was an FP16 baseline measured?)")
        return 2

    arch = rep.get("system_under_test", {}).get("gpu_arch")
    prec = rep.get("workload", {}).get("precision")
    n_params = rep.get("workload", {}).get("params_b")
    if not (arch and prec and n_params):
        print("SKIP: report is missing gpu_arch / precision / params_b")
        return 2

    curves = load_curves(a.curves)
    curve = curves.get("curves", {}).get(arch, {}).get(prec)
    if curve is None:
        borrow = curves.get("borrow", {}).get(arch)
        curve = curves.get("curves", {}).get(borrow, {}).get(prec) if borrow else None
        if curve is None:
            print("SKIP: no published %s/%s curve to check against" % (arch, prec))
            return 2
        print("note: %s has no measurements of its own; using the %s curve" % (arch, borrow))

    pred = predict(curve, n_params)
    rstd = curve["resid_std"]
    band = a.k * rstd
    same_n = [x for x in curve.get("anchors", []) if abs(x["N"] - n_params) < 1e-9]

    print("%s %s %.1fB  (%s)" % (arch, prec, n_params,
                                 rep.get("system_under_test", {}).get("gpu", "?")))
    print("  measured   dE = %+.1f %%" % de)
    print("  curve says dE = %+.1f %%   (fit resid_std %.1f)" % (pred, rstd))

    # Grade against a measurement of the same model size when there is one. The
    # curve is a smooth fit across sizes and can miss an individual size by more
    # than the band on its own -- gating on it would fail a healthy build.
    if same_n:
        print("  anchors at this size: %s"
              % ", ".join("%+.1f%% (%s, n=%d)" % (x["dE"], x["gpu"], x["n"]) for x in same_n))
        ref = sum(x["dE"] for x in same_n) / len(same_n)
        what = "published anchor(s) at this size"
    else:
        ref = pred
        what = "the fitted curve (no anchor at this size)"
    resid = de - ref
    ok = abs(resid) <= band
    print("  vs %s: %+.1f pp (band +-%.1f = %gx resid_std) -> %s"
          % (what, resid, band, a.k, "WITHIN" if ok else "OUTSIDE"))

    sw = rep.get("software", {})
    if sw.get("differs_from_reference_pins"):
        print("\n  caveat: not the published pins (%s) -- quantization kernels change\n"
              "  between releases, so part of any residual is the software, not the GPU."
              % "; ".join(sw["differs_from_reference_pins"]))
    print("\n  The band is the fit's residual dispersion across model sizes, not a\n"
          "  confidence interval, and this run is n=1. Passing means 'not obviously\n"
          "  broken', not 'confirms the curve'.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

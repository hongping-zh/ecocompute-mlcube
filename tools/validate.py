#!/usr/bin/env python3
"""EcoCompute report validator: schema-valid is not protocol-conformant.

The JSON schema (schema/energy.schema.json) checks STRUCTURE. It cannot check
cross-field semantics ("an NF4 report must carry a same-session FP16 baseline",
"basis: measured requires direct-NVML sampling", "the thermal block must be
present and honest"). Those are Protocol v1.1 MUSTs, and this tool checks them.

Three verdicts, deliberately distinct:

  schema-valid          passes energy.schema.json structure validation
  protocol-conformant   schema-valid AND every Protocol v1.1 MUST holds
                        (--profile v1.1-core)
  dataset-eligible      protocol-conformant AND the reportability extras
                        (--profile dataset-eligible); replication counts and
                        review are dataset-level and flagged as such

Exit codes: 0 conformant/eligible, 1 schema-valid but violations,
2 schema-invalid, 3 usage error.

Usage:
  python tools/validate.py report.json
  python tools/validate.py report.json --profile dataset-eligible
  ecocompute validate --profile v1.1-core report.json   (container CLI alias)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover
    print("error: jsonschema is required (pip install jsonschema)", file=sys.stderr)
    sys.exit(3)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO / "schema" / "energy.schema.json"

PROFILES = ("v1.1-core", "dataset-eligible")


class Violation:
    def __init__(self, clause, text, hint=""):
        self.clause = clause      # e.g. "2.1"
        self.text = text          # one-line statement of what failed
        self.hint = hint          # optional fix hint


def _fmt(v: "Violation") -> str:
    line = f"[\u00a74.{v.clause}] {v.text}"
    if v.hint:
        line += f"\n       fix: {v.hint}"
    return line


def check_schema(report, schema) -> str | None:
    """Returns None on success, a compact message on failure."""
    try:
        jsonschema.validate(report, schema)
        return None
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return f"{e.message} (at {path})"


def check_protocol_v11_core(r: dict) -> list[Violation]:
    """Every Protocol v1.1 MUST that is machine-checkable from a single report.

    Clause numbers refer to the normative document
    'EcoCompute Measurement Protocol v1.1' (DOI 10.5281/zenodo.22958675).
    """
    v: list[Violation] = []
    w = r.get("workload", {}) or {}
    m = r.get("measurement", {}) or {}
    res = r.get("results", {}) or {}
    sw = r.get("software", {}) or {}
    th = r.get("thermal")

    basis = res.get("basis")
    src = r.get("measurement_source", "")
    precision = w.get("precision")

    # --- 4.1 Power measurement ---
    if basis == "measured" and src != "direct-nvml":
        v.append(Violation(
            "1.4", "basis: measured requires measurement_source: direct-nvml "
            f"(found {src!r})",
            "values derived from TDP or vendor typical power MUST NOT be "
            "reported as measurements"))

    # --- 4.2 Baseline ---
    if precision in ("NF4", "INT8") and basis == "measured":
        if res.get("vs_fp16_energy_pct") is None or res.get("fp16_energy_per_token_mj") is None:
            v.append(Violation(
                "2.1", f"{precision} measured report lacks the same-session FP16 "
                "baseline fields (vs_fp16_energy_pct / fp16_energy_per_token_mj)",
                "a quantized run without a same-session FP16 baseline MUST NOT "
                "yield a vs-FP16 claim"))

    # --- 4.3 Workload ---
    if w.get("batch_size") != 1:
        v.append(Violation(
            "3.1", f"batch_size is {w.get('batch_size')!r}, protocol core is batch 1",
            "additional batch sizes are a 4.8 MAY and must be labelled "
            "separately; they do not qualify for the v1.1-core profile"))
    if m.get("tokens_per_run") not in (None, 256):
        v.append(Violation(
            "3.2", f"tokens_per_run is {m.get('tokens_per_run')!r}, protocol core is 256",
            "other token counts are a 4.8 MAY and must be labelled separately"))
    if m.get("warmup") is None or m.get("warmup") < 1:
        v.append(Violation(
            "3.4", "warm-up missing or zero",
            "warm-up MUST precede measurement (excluded from the window by 4.4)"))
    if w.get("context_length") is None:
        v.append(Violation("3.3", "context_length not recorded"))

    # --- 4.5 Report contents ---
    pkgs = sw.get("packages", {}) or {}
    missing_pkg = [p for p in ("torch", "transformers", "bitsandbytes") if not pkgs.get(p)]
    if not sw:
        v.append(Violation(
            "5.1", "software block missing",
            "the full version set actually used MUST be recorded"))
    else:
        if not sw.get("python"):
            v.append(Violation("5.1", "software.python missing"))
        if missing_pkg:
            v.append(Violation(
                "5.1", f"software.packages missing: {', '.join(missing_pkg)}",
                "quantization kernels change between releases; two reports are "
                "only comparable when these agree"))
        if not sw.get("nvidia_driver"):
            v.append(Violation("5.1", "software.nvidia_driver missing"))
    if sw.get("differs_from_reference_pins") and sw.get("matches_reference_pins") is not False:
        v.append(Violation(
            "5.3", "differs_from_reference_pins is non-empty but "
            "matches_reference_pins is not false",
            "a run on a different stack is real but MUST be flagged as not "
            "directly comparable"))

    # --- 4.6 Thermal state ---
    if th is None:
        v.append(Violation(
            "6.1", "thermal block missing",
            "the report MUST record the thermal block (warm-up count, "
            "temperatures at start/steady/end/peak); the container does not "
            "yet emit it - tracked issue"))
    else:
        if th.get("mode") not in ("cold", "steady"):
            v.append(Violation("6.3", f"thermal.mode is {th.get('mode')!r}, "
                              "must be one thermal mode: cold or steady"))
        if th.get("basis") not in ("measured", "unavailable"):
            v.append(Violation(
                "6.4", f"thermal.basis is {th.get('basis')!r}",
                "a card without a sensor MUST report basis: unavailable "
                "rather than inventing a value"))
        # honest failure states are VALID: steady_state_reached=false and
        # basis=unavailable pass - only fabrication fails.

    return v


def check_dataset_eligible(r: dict) -> list[Violation]:
    """Extras beyond protocol conformance for a report to be poolable into a
    dataset release. Some level-C criteria (n >= 2 sessions, review, cross-card
    spread) are dataset-level and cannot be checked from one report; they are
    reported as NOTES, not violations."""
    v: list[Violation] = []
    res = r.get("results", {}) or {}

    if "power_trace" not in r or not r.get("power_trace", {}).get("files"):
        v.append(Violation(
            "5/4.4.3", "no whole-run power-trace sidecar attached",
            "RECOMMENDED by the protocol and required for dataset eligibility: "
            "the trace lets the window be re-cut post hoc without re-running"))
    if r.get("measurement", {}).get("sample_rate_hz") is None:
        v.append(Violation("1.2", "achieved sample_rate_hz not recorded"))
    if res.get("basis") != "measured":
        v.append(Violation(
            "C", f"results.basis is {res.get('basis')!r}, dataset release "
            "requires measured"))
    return v


DATASET_NOTES = [
    "n >= 2 independent sessions per configuration (and n_card when > 1 card) "
    "is verified at the DATASET level (build CSV + review), not per report",
    "maintainer review happens before a report enters a release; this tool "
    "cannot check it",
]


def validate_report(report: dict, schema: dict, profile: str) -> tuple[str, list[str], list[str]]:
    """Returns (verdict, violations, notes). verdict in
    {'schema-invalid', 'schema-valid', 'protocol-conformant', 'dataset-eligible'}"""
    notes: list[str] = []
    err = check_schema(report, schema)
    if err:
        return "schema-invalid", [err], notes
    viol = check_protocol_v11_core(report)
    if viol:
        return "schema-valid", [_fmt(x) for x in viol], notes
    if profile == "v1.1-core":
        return "protocol-conformant", [], notes
    viol2 = check_dataset_eligible(report)
    if viol2:
        return "protocol-conformant", [_fmt(x) for x in viol2], notes
    if profile == "dataset-eligible":
        notes.extend(DATASET_NOTES)
    return "dataset-eligible", [], notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ecocompute validate",
        description="EcoCompute report validator: schema-valid is not protocol-conformant")
    ap.add_argument("report", help="path to an energy.json report")
    ap.add_argument("--profile", choices=PROFILES, default="v1.1-core",
                    help="v1.1-core: every Protocol v1.1 MUST; "
                         "dataset-eligible: core plus reportability extras")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA),
                    help=f"path to the JSON schema (default: {DEFAULT_SCHEMA})")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output for CI")
    args = ap.parse_args(argv)

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read input: {e}", file=sys.stderr)
        return 3

    verdict, viol, notes = validate_report(report, schema, args.profile)

    if args.json:
        print(json.dumps({
            "report": args.report, "profile": args.profile,
            "verdict": verdict, "violations": viol, "notes": notes}, indent=2))
    else:
        sv = report.get("schema_version", "?")
        print(f"== EcoCompute validation ({args.profile}) ==")
        print(f"schema-valid:        {'yes' if verdict != 'schema-invalid' else 'no'} ({sv})")
        if verdict == "schema-invalid":
            print(f"schema error:        {viol[0]}")
        elif viol:
            print(f"protocol violations: {len(viol)}")
            for t in viol:
                print(f"  - {t}")
        else:
            print("protocol violations: 0")
        print(f"verdict:             {verdict}")
        for n in notes:
            print(f"note:                {n}")
        if verdict == "schema-valid":
            print("hint: structure is fine but the report does not satisfy "
                  "Protocol v1.1 MUSTs - see the violations above")

    if verdict == "schema-invalid":
        return 2
    if verdict == "schema-valid" or (args.profile == "dataset-eligible" and viol):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

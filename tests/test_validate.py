"""Tests for tools/validate.py: schema-valid is not protocol-conformant.

The three verdicts must stay distinct:

  schema-valid          structure passes energy.schema.json
  protocol-conformant   structure + every Protocol v1.1 MUST (v1.1-core)
  dataset-eligible      conformance + reportability extras

Fixtures:
  re-measured-4090-nf4.json   the real 2026-09-25 trial-A report (schema-valid;
                              three honest gaps: no thermal block, no window
                              statement, no decoding strategy)
  conformant-v1.1.json        the same report + thermal block + window/decoding/
                              arm-order records, fully conformant
  energy.no-gpu.json          the no-GPU estimate path (several violations)
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import validate  # noqa: E402

SCHEMA = json.loads((REPO / "schema" / "energy.schema.json").read_text())
FIX = REPO / "tests" / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


def test_real_retest_report_is_schema_valid_but_not_conformant():
    """The 2026-09-25 two-card re-test report: structure fine, but it records
    no thermal block, no window statement and no decoding strategy. The
    validator must say exactly this - not wave it through."""
    verdict, viol, _ = validate.validate_report(
        load("re-measured-4090-nf4.json"), SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    for frag in ("thermal block missing", "window not stated",
                 "decoding strategy not recorded"):
        assert frag in joined, joined
    assert len(viol) == 3, joined


def test_conformant_fixture_passes_core():
    verdict, viol, _ = validate.validate_report(
        load("conformant-v1.1.json"), SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol
    assert viol == []


def test_conformant_fixture_is_dataset_eligible():
    """The re-test report carries a whole-run power trace, so the conformant
    fixture is also dataset-eligible (replication counts stay dataset-level)."""
    verdict, viol, notes = validate.validate_report(
        load("conformant-v1.1.json"), SCHEMA, "dataset-eligible")
    assert verdict == "dataset-eligible", viol
    assert any("DATASET level" in n for n in notes)


def test_no_gpu_report_flags_each_gap():
    verdict, viol, _ = validate.validate_report(
        json.loads((REPO / "examples" / "energy.no-gpu.json").read_text()),
        SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    # the example now carries an honest thermal block (basis: unavailable),
    # so the gap flagged there is the missing arm order, not a missing block
    for frag in ("warm-up", "software block missing", "arm order not recorded"):
        assert frag in joined


def test_schema_invalid_report_fails_first():
    d = load("conformant-v1.1.json")
    d["measurement"]["sample_rate_hz"] = 5  # below the 10 Hz MUST
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-invalid"
    assert viol


def test_quantized_without_fp16_baseline_is_a_violation():
    d = load("conformant-v1.1.json")
    d["results"]["vs_fp16_energy_pct"] = None
    d["results"]["fp16_energy_per_token_mj"] = None
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("FP16 baseline" in v for v in viol)


def test_measured_basis_requires_direct_nvml():
    d = load("conformant-v1.1.json")
    d["measurement_source"] = "vendor-typical-power"
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("direct-nvml" in v for v in viol)


def test_honest_thermal_failure_states_pass():
    """steady_state_reached: false and basis: 'unavailable' are VALID - only
    fabrication fails."""
    d = load("conformant-v1.1.json")
    d["thermal"] = {"mode": "cold", "basis": "unavailable",
                    "steady_state_reached": False, "arm_order": "randomized",
                    "note": "card reports no sensor"}
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol


def test_thermal_measured_requires_temperatures_and_cooldown():
    """4.6.1: a measured thermal block without the temperature records and
    cooldown is not conformant - 'mode+basis' alone is exactly the gap the
    negative probe found."""
    d = load("conformant-v1.1.json")
    d["thermal"] = {"mode": "cold", "basis": "measured",
                    "arm_order": "randomized"}
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    for frag in ("warmup_runs", "cooldown_s", "temperature_start_c",
                 "temperature_end_c", "temperature_peak_c",
                 "temperature_steady_c"):
        assert frag in joined, joined


def test_steady_temperature_excused_only_by_honest_marker():
    d = load("conformant-v1.1.json")
    d["thermal"]["temperature_steady_c"] = None
    d["thermal"]["steady_state_reached"] = False  # honest: never settled
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol
    d["thermal"]["steady_state_reached"] = None  # not honest, just absent
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("temperature_steady_c" in v for v in viol)


def test_arm_order_recorded_or_disclosed():
    d = load("conformant-v1.1.json")
    del d["thermal"]["arm_order"]
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("arm order not recorded" in v for v in viol)
    keep_note = d["thermal"]["note"]
    d["thermal"]["note"] = ""  # fixed order, deviation nowhere disclosed
    d["thermal"]["arm_order"] = "fixed"
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("deviation is not disclosed" in v for v in viol)
    d["thermal"]["arm_order"] = "fixed (same-day reruns, disclosed)"
    d["thermal"]["note"] = keep_note + " Arms ran back to back; deviation from 4.6.2 disclosed here."
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol


def test_measured_report_requires_nvml_method():
    """4.1.1: the power source MUST be NVML - 'method': anything else breaks a
    measured claim even when measurement_source says direct-nvml."""
    d = load("conformant-v1.1.json")
    d["measurement"]["method"] = "vendor power estimate (SMI)"
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("NVML" in v for v in viol)


def test_core_requires_tokens_iterations_window_decoding_present():
    """Missing protocol facts can never be protocol-conformant: an
    unrecorded 256/10/generation/greedy cannot be verified (4.3.2/4.4.2).
    A 1.2 report may omit tokens_per_run/iterations at the SCHEMA level -
    the validator must still refuse conformance for them."""
    d = load("conformant-v1.1.json")
    d["schema_version"] = "ecocompute-energy/1.2"
    for k in ("tokens_per_run", "iterations", "window"):
        del d["measurement"][k]
    del d["workload"]["decoding"]
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    for frag in ("tokens_per_run not recorded", "decode iterations not recorded",
                 "window not stated", "decoding strategy not recorded"):
        assert frag in joined, joined


def test_v13_report_with_wrong_values_is_caught_by_the_validator():
    """Schema 1.3 already requires tokens_per_run/iterations to EXIST; the
    validator is what catches WRONG values (1 instead of 10, 128 instead of
    256)."""
    d = load("conformant-v1.1.json")
    d["measurement"]["iterations"] = 1
    d["measurement"]["tokens_per_run"] = 128
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    joined = "\n".join(viol)
    assert "decode iterations is 1" in joined
    assert "tokens_per_run is 128" in joined


def test_schema_1_0_is_below_the_protocol_floor():
    """4.5.2: conformance is defined against 1.2+; a 1.0 report is schema-valid
    but never protocol-conformant, however complete it looks."""
    d = load("conformant-v1.1.json")
    d["schema_version"] = "ecocompute-energy/1.0"
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("1.2 or later" in v for v in viol)


def test_negative_probe_report_is_rejected():
    """Regression: the hand-built report that used to sail through as
    'protocol violations: 0 / protocol-conformant'. Every one of its gaps
    must now be caught."""
    bad = {
        "schema_version": "ecocompute-energy/1.3",
        "benchmark": "ecocompute-energy-methodology",
        "scenario": "SingleStream",
        "system_under_test": {"gpu": "RTX 4090", "gpu_arch": "ada",
                              "accelerator_count": 1},
        "workload": {"model_name": "Qwen/Qwen2.5-3B", "params_b": 3.0,
                     "precision": "NF4", "batch_size": 1, "context_length": 2048},
        "measurement": {"method": "nvidia-smi power averaging",
                        "sample_rate_hz": 10, "tokens_per_run": 256,
                        "iterations": 1, "warmup": 1},
        "software": {"python": "3.10",
                     "packages": {"torch": "2", "transformers": "4",
                                  "bitsandbytes": "0.5"},
                     "nvidia_driver": "595"},
        "measurement_source": "direct-nvml",
        "results": {"total_energy_joules": 1.0, "tokens_generated": 256,
                    "energy_per_token_mj": 3.9, "avg_power_watts": 100.0,
                    "throughput_tokens_per_s": 25.0, "basis": "measured",
                    "fp16_energy_per_token_mj": 3.5, "vs_fp16_energy_pct": 11.0},
        "thermal": {"mode": "cold", "basis": "measured"},
    }
    verdict, viol, _ = validate.validate_report(bad, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid", (verdict, viol)
    joined = "\n".join(viol)
    for frag in ("NVML on-device",            # 4.1.1 method
                 "decode iterations is 1",    # 4.3.2 iterations == 10
                 "window not stated",         # 4.4.1
                 "decoding strategy not recorded",  # 4.3.2 greedy
                 "arm order not recorded",    # 4.6.2
                 "temperature_start_c", "cooldown_s"):  # 4.6.1
        assert frag in joined, joined


def test_pin_mismatch_must_be_flagged():
    d = load("conformant-v1.1.json")
    d["software"]["differs_from_reference_pins"] = ["bitsandbytes==0.50.2"]
    d["software"]["matches_reference_pins"] = True  # inconsistent
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("matches_reference_pins" in v for v in viol)


def test_batch_size_2_is_flagged_as_off_core():
    d = load("conformant-v1.1.json")
    d["workload"]["batch_size"] = 2
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("batch" in v for v in viol)


def test_cli_exit_codes(tmp_path):
    """0 conformant, 1 schema-valid-but-violating, 2 schema-invalid."""
    import subprocess
    ok = subprocess.run(
        [sys.executable, str(REPO / "tools" / "validate.py"),
         str(FIX / "conformant-v1.1.json")], capture_output=True)
    viol = subprocess.run(
        [sys.executable, str(REPO / "tools" / "validate.py"),
         str(FIX / "re-measured-4090-nf4.json")], capture_output=True)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "ecocompute-energy/1.3"}))
    inv = subprocess.run(
        [sys.executable, str(REPO / "tools" / "validate.py"), str(bad)],
        capture_output=True)
    assert ok.returncode == 0
    assert viol.returncode == 1
    assert inv.returncode == 2


def test_entrypoint_validate_subcommand(tmp_path):
    """`python entrypoint.py validate report.json` is the user-facing alias."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO / "entrypoint.py"), "validate",
         str(FIX / "conformant-v1.1.json")], capture_output=True, text=True)
    assert r.returncode == 0
    assert "protocol-conformant" in r.stdout


# --- 4.7 power-trace L0 fields (schema 1.3-r7) -----------------------------

def _l0_report(entry_extra, block_extra=None):
    r = load("conformant-v1.1.json")
    entry = {"role": "primary", "precision": "NF4", "file": "power_trace.csv",
             "stream": "measure_window", "samples": 3, "dropped_samples": 0,
             "sha256": "a" * 64, "bytes": 12,
             "sample_window": {"first_sample": 0, "last_sample": 2},
             "integral": {"rule": "trapezoid-csv-v1", "value": 9076.849,
                          "unit": "joules", "note": "n"}}
    entry.update(entry_extra)
    r["power_trace"]["files"] = [entry]
    if block_extra:
        r["power_trace"].update(block_extra)
    return r


def test_l0_fields_on_conformant_report_stay_conformant():
    verdict, viol, _ = validate.validate_report(_l0_report({}), SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol


def test_l0_revision_requires_digests_on_every_file():
    r = _l0_report({}, {"l0_revision": "1.3-r7"})
    del r["power_trace"]["files"][0]["sha256"]
    verdict, viol, _ = validate.validate_report(r, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("4.7.1" in v for v in viol), viol


def test_measure_window_file_must_carry_the_full_l0_set():
    r = _l0_report({})
    del r["power_trace"]["files"][0]["integral"]
    verdict, viol, _ = validate.validate_report(r, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("4.7.2" in v and "integral" in v for v in viol), viol


def test_sample_window_indices_must_be_coherent_with_samples():
    r = _l0_report({"sample_window": {"first_sample": 0, "last_sample": 7}})
    verdict, viol, _ = validate.validate_report(r, SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert any("4.7.4" in v for v in viol), viol
    r2 = _l0_report({"sample_window": {"first_sample": 2, "last_sample": 1}})
    _, viol2, _ = validate.validate_report(r2, SCHEMA, "v1.1-core")
    assert any("4.7.4" in v for v in viol2), viol2


# --- 4.7.5-4.7.8 L0 recompute clauses (validator re-derives from disk) ------

import hashlib  # noqa: E402

EXAMPLES = REPO / "examples"


def _write_sidecar(tmp_path, rows, newline="\n"):
    """Write a t_s,power_w CSV; returns (path, raw_bytes)."""
    lines = ["t_s,power_w"] + [f"{t:.3f},{w:.1f}" for t, w in rows]
    raw = ("\r\n" if newline == "\r\n" else "\n").join(lines).encode()
    p = tmp_path / "power_trace_measure.csv"
    p.write_bytes(raw)
    return p, raw


def _example_report(report_dir):
    """The shipped accepted example, loaded and re-validated from its dir."""
    r = json.loads((report_dir / "energy.json").read_text())
    return r


def _conformant_l0_report(tmp_path, rows=None, newline="\n"):
    """A minimal conformant r7 report + on-disk sidecar built with real code.

    rows default: 4 samples, trapezoid = (1.0+2.0)/2*1.0 + (2.0+1.5)/2*1.0
    + (1.5+1.0)/2*1.0 = 1.5 + 1.75 + 1.25 = 4.5 J exactly.
    """
    if rows is None:
        rows = [(0.0, 1.0), (1.0, 2.0), (2.0, 1.5), (3.0, 1.0)]
    _, raw = _write_sidecar(tmp_path, rows, newline)
    r = _l0_report({
        "file": "power_trace_measure.csv",
        "samples": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "sample_window": {"first_sample": 0, "last_sample": len(rows) - 1},
        "integral": {"rule": "trapezoid-csv-v1", "value": 4.5,
                     "unit": "joules", "note": "n"},
    })
    # headline must agree with the integral for a measure_window entry
    r["results"]["total_energy_joules"] = 4.5
    return r


def test_validator_accepts_a_consistent_sidecar(tmp_path):
    r = _conformant_l0_report(tmp_path)
    verdict, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert verdict == "protocol-conformant", viol


def test_validator_accepts_crlf_sidecar_covered_by_its_digest(tmp_path):
    # the digest is over raw bytes (CRLF included); parsing tolerates CRLF,
    # so a legitimately CRLF-written file must still pass both 4.7.5 and 4.7.6
    r = _conformant_l0_report(tmp_path, newline="\r\n")
    verdict, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert verdict == "protocol-conformant", viol


def test_validator_flags_a_tampered_sidecar(tmp_path):
    r = _conformant_l0_report(tmp_path)
    p = tmp_path / "power_trace_measure.csv"
    raw = p.read_bytes().replace(b"2.0", b"9.0")
    p.write_bytes(raw)
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "tampered" in v for v in viol), viol


def test_validator_flags_a_truncated_sidecar(tmp_path):
    r = _conformant_l0_report(tmp_path)
    p = tmp_path / "power_trace_measure.csv"
    raw = p.read_bytes()
    p.write_bytes(raw[:raw.rindex(b"\n")])  # cut at a line boundary
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "truncated" in v for v in viol), viol


def test_validator_flags_a_missing_sidecar(tmp_path):
    r = _conformant_l0_report(tmp_path)
    (tmp_path / "power_trace_measure.csv").unlink()
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "not found" in v for v in viol), viol


def test_validator_flags_an_absolute_sidecar_path(tmp_path):
    # path escape: pathlib drops report_dir when the right operand is
    # absolute, so an absolute files[].file would have the recompute read
    # (and certify) a file outside the validated directory — rejected even
    # when the digest honestly matches that out-of-tree file
    r = _conformant_l0_report(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    csv = tmp_path / "power_trace_measure.csv"
    csv.rename(elsewhere / csv.name)
    r["power_trace"]["files"][0]["file"] = str(elsewhere / csv.name)
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "escape" in v for v in viol), viol
    # and the out-of-tree file is never read: no digest mismatch is reported
    assert not any("does not match" in v for v in viol), viol


def test_validator_flags_a_dotdot_sidecar_path(tmp_path):
    # '..' segments walk out of the report directory just like an absolute
    # path; the co-location contract rejects them too
    r = _conformant_l0_report(tmp_path)
    outside = tmp_path.parent / "dotdot_elsewhere"
    outside.mkdir(exist_ok=True)
    csv = tmp_path / "power_trace_measure.csv"
    csv.rename(outside / csv.name)
    r["power_trace"]["files"][0]["file"] = "../dotdot_elsewhere/" + csv.name
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "escape" in v for v in viol), viol


def test_validator_flags_an_escaping_path_even_without_a_digest(tmp_path):
    # the co-location gate is not conditioned on a declared digest: a
    # digest-less whole_run entry must not point outside the tree either
    r = _conformant_l0_report(tmp_path)
    fe = r["power_trace"]["files"][0]
    for key in ("sha256", "bytes", "sample_window", "integral"):
        fe.pop(key, None)
    fe["stream"] = "whole_run"
    fe["file"] = "/etc/elsewhere/power_trace_measure.csv"  # Posix absolute
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.5" in v and "escape" in v for v in viol), viol


def test_validator_rederives_the_integral_from_disk(tmp_path):
    r = _conformant_l0_report(tmp_path)
    r["power_trace"]["files"][0]["integral"]["value"] = 4.6  # falsified
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.6" in v and "4.500000" in v for v in viol), viol


def test_validator_links_the_integral_to_the_headline(tmp_path):
    r = _conformant_l0_report(tmp_path)
    r["results"]["total_energy_joules"] = 1724.855  # headline drifted
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.7" in v for v in viol), viol


def test_validator_links_integral_to_headline_without_files_too():
    # 4.7.7 is a pure cross-field check: it must fire even dict-only
    r = _l0_report({"integral": {"rule": "trapezoid-csv-v1",
                                 "value": 123.4, "unit": "joules",
                                 "note": "n"}})
    _, viol, _ = validate.validate_report(r, SCHEMA, "v1.1-core")
    assert any("4.7.7" in v for v in viol), viol


def test_validator_checks_samples_count_against_the_file(tmp_path):
    r = _conformant_l0_report(tmp_path)
    r["power_trace"]["files"][0]["samples"] = 99
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=tmp_path)
    assert any("4.7.8" in v for v in viol), viol


def test_shipped_accepted_example_passes_the_recompute_clauses():
    d = EXAMPLES / "accepted-trace"
    r = json.loads((d / "energy.json").read_text())
    # the example is deliberately illustrative (self-declared "not a real
    # measurement"), so full conformance is not the criterion here: the L0
    # recompute clauses (4.7.x) must all pass on the real CSV on disk
    _, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=d)
    assert not any("4.7." in v for v in viol), viol


def test_provenance_identity_is_shape_checked_not_truth_checked():
    """Clause 5.4: provenance.code_commit/image are traceability, not
    endorsement. The validator checks the shape (a non-empty string) and
    deliberately never the truth — anyone can rebuild an image with a
    forged commit string, and that must pass."""
    report = json.loads((EXAMPLES / "energy.no-gpu.json").read_text())
    # forged but well-shaped: must NOT trip 5.4
    report["provenance"]["code_commit"] = "0" * 40
    _, viol, _ = validate.validate_report(report, SCHEMA, "v1.1-core")
    assert not any("§4.5.4]" in v for v in viol), viol
    # whitespace-only passes the schema minLength but not 5.4
    report["provenance"]["code_commit"] = "   "
    _, viol, _ = validate.validate_report(report, SCHEMA, "v1.1-core")
    assert any("§4.5.4]" in v for v in viol), viol
    # a non-string is stopped at the schema layer first
    report["provenance"]["code_commit"] = {"nested": True}
    verdict, viol, _ = validate.validate_report(report, SCHEMA, "v1.1-core")
    assert verdict == "schema-invalid", viol
    # absent block entirely is fine (r5-era reports predate it)
    del report["provenance"]
    _, viol, _ = validate.validate_report(report, SCHEMA, "v1.1-core")
    assert not any("§4.5.4]" in v for v in viol), viol


def test_shipped_tampered_example_is_caught_by_the_validator():
    d = EXAMPLES / "rejected-tampered-trace"
    r = json.loads((d / "energy.json").read_text())
    verdict, viol, _ = validate.validate_report(
        r, SCHEMA, "v1.1-core", report_dir=d)
    assert verdict != "protocol-conformant"
    assert any("4.7.5" in v for v in viol), viol


def test_usage_errors_exit_64_not_colliding_with_schema_invalid_2():
    """argparse's default usage exit is 2 - the same code the CLI documents
    for schema-invalid. A wrapper consuming the exit code could not tell bad
    CLI usage from a broken report. Usage errors must exit 64 (EX_USAGE)."""
    import subprocess

    for argv in ([], ["--nonexistent-flag"]):
        r = subprocess.run(
            [sys.executable, str(REPO / "tools" / "validate.py")] + argv,
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 64, (argv, r.returncode, r.stderr)
        assert "usage:" in r.stderr.lower()

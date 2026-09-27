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
    for frag in ("warm-up", "software block missing", "thermal block missing"):
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

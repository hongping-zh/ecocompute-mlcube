"""Tests for tools/validate.py: schema-valid is not protocol-conformant.

The three verdicts must stay distinct:

  schema-valid          structure passes energy.schema.json
  protocol-conformant   structure + every Protocol v1.1 MUST (v1.1-core)
  dataset-eligible      conformance + reportability extras

Fixtures:
  re-measured-4090-nf4.json   the real 2026-09-25 trial-A report (schema-valid,
                              one honest violation: no thermal block yet)
  conformant-v1.1.json        the same report + a thermal block (the container's
                              tracked next step), fully conformant
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
    """The 2026-09-25 two-card re-test report: structure fine, thermal block
    missing. The validator must say exactly this - not wave it through."""
    verdict, viol, _ = validate.validate_report(
        load("re-measured-4090-nf4.json"), SCHEMA, "v1.1-core")
    assert verdict == "schema-valid"
    assert len(viol) == 1 and "thermal" in viol[0]


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
                    "steady_state_reached": False,
                    "note": "card reports no sensor"}
    verdict, viol, _ = validate.validate_report(d, SCHEMA, "v1.1-core")
    assert verdict == "protocol-conformant", viol


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

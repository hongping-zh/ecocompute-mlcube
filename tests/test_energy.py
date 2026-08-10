"""Tests for the EcoCompute energy MLCube.

They exercise the entrypoint through its real CLI (the same `energy_estimate`
task MLCube invokes) and check that every report validates against the schema
and preserves the measured-vs-derived honesty guarantees.
"""
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schema" / "energy.schema.json").read_text())
ENTRY = REPO / "entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("ecc_entrypoint", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ecc = _load_entrypoint()


def run_cli(out_dir, *args):
    """Run `entrypoint.py energy_estimate --dry_run ...` and return the report."""
    cmd = [sys.executable, str(ENTRY), "energy_estimate", "--dry_run",
           "--output_dir", str(out_dir), *args]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    return json.loads((Path(out_dir) / "energy.json").read_text())


# --- shipped examples -------------------------------------------------------

@pytest.mark.parametrize("name", [
    "energy.no-gpu.json",
    "energy.measured.illustrative.json",
])
def test_example_matches_schema(name):
    report = json.loads((REPO / "examples" / name).read_text())
    jsonschema.validate(report, SCHEMA)
    assert report["certified_benchmark_result"] is False


# --- entrypoint / task behaviour -------------------------------------------

def test_default_params_file_is_schema_valid(tmp_path):
    report = run_cli(tmp_path, "--parameters_file",
                     str(REPO / "workspace" / "parameters" / "energy_params.yaml"))
    jsonschema.validate(report, SCHEMA)


def test_cli_flags_flow_into_workload(tmp_path):
    report = run_cli(tmp_path, "--model", "bert-base-uncased",
                     "--batch_size", "32", "--precision", "NF4",
                     "--gpu_arch", "ampere", "--params_b", "7")
    jsonschema.validate(report, SCHEMA)
    assert report["workload"]["model_name"] == "bert-base-uncased"
    assert report["workload"]["batch_size"] == 32


@pytest.mark.parametrize("batch,scenario", [(1, "SingleStream"), (8, "Offline")])
def test_batch_size_selects_scenario(tmp_path, batch, scenario):
    report = run_cli(tmp_path, "--batch_size", str(batch),
                     "--gpu_arch", "blackwell", "--params_b", "3")
    assert report["scenario"] == scenario


def test_no_gpu_path_never_claims_a_fresh_measurement(tmp_path):
    report = run_cli(tmp_path, "--gpu_arch", "blackwell", "--params_b", "3")
    assert "no local GPU" in report["measurement_source"]
    assert report["results"]["basis"] != "measured"
    assert "reference estimate" in report["measurement"]["method"]


def test_parameters_file_swap_changes_output(tmp_path):
    report = run_cli(tmp_path, "--parameters_file",
                     str(REPO / "workspace" / "parameters" / "bert_bs32.yaml"))
    jsonschema.validate(report, SCHEMA)
    assert report["workload"]["batch_size"] == 32
    assert report["scenario"] == "Offline"


# --- scope / LoadGen boundary ----------------------------------------------

def test_report_does_not_overclaim_benchmark_certification(tmp_path):
    report = run_cli(tmp_path, "--gpu_arch", "ada", "--params_b", "7")
    note = report["scenario_note"].lower()
    assert "loadgen" in note and "not" in note
    assert report["certified_benchmark_result"] is False
    assert report["follows_mlcommons_energy_reporting_conventions"] is True


# --- NVML robustness (no real GPU needed) ----------------------------------

def test_power_sampler_survives_failing_reads():
    """A card/driver that raises on every power read must not crash the thread."""
    fake = types.ModuleType("pynvml")

    def _boom(_handle):
        raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

    fake.nvmlDeviceGetPowerUsage = _boom
    sys.modules["pynvml"] = fake
    try:
        sampler = ecc.PowerSampler(handle=object(), hz=50)
        sampler.start()
        time.sleep(0.3)
        sampler.stop()
    finally:
        del sys.modules["pynvml"]
    assert sampler.samples == []
    assert sampler.error is not None
    assert sampler.dropped >= 1


def test_measure_failure_falls_back_without_claiming_measured():
    """If on-device measurement fails, the report uses the dataset path, not 'measured'."""
    params = ecc.load_params(_ns(gpu_arch="blackwell", params_b=3))
    ref = ecc.reference_estimate(3.0, "blackwell", "NF4")
    report = ecc.build_report(params, measured=None, ref=ref,
                              measure_error="NVML power telemetry unavailable")
    jsonschema.validate(report, SCHEMA)
    assert report["results"]["basis"] != "measured"
    assert "measurement failed" in report["measurement_source"]
    assert "NVML" in report["results"]["note"]


# --- website hooks: --share overlay link + --prefetch (no network needed) -----

def _decode_overlay(url):
    import base64
    import urllib.parse
    blob = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["overlay"][0]
    blob += "=" * (-len(blob) % 4)
    return json.loads(base64.urlsafe_b64decode(blob))


def test_share_url_encodes_overlay_point():
    params = ecc.load_params(_ns(gpu_arch="blackwell", params_b=1.1, precision="NF4"))
    ref = ecc.reference_estimate(1.1, "blackwell", "NF4")
    report = ecc.build_report(params, measured=None, ref=ref)
    url = ecc.share_url("https://quantenergy.tech", report)
    assert url and url.startswith("https://quantenergy.tech/?tab=run&overlay=")
    o = _decode_overlay(url)
    assert o["a"] == "blackwell" and o["N"] == 1.1 and o["p"] == "NF4"
    assert o["e"] == report["results"]["vs_fp16_energy_pct"]
    assert o["b"] == report["results"]["basis"]


def test_share_url_is_none_without_delta():
    """FP16 / unavailable runs have no vs_fp16_energy_pct, so no overlay link."""
    params = ecc.load_params(_ns(gpu_arch="blackwell", params_b=3, precision="FP16"))
    ref = ecc.reference_estimate(3.0, "blackwell", "FP16")
    report = ecc.build_report(params, measured=None, ref=ref)
    report["results"]["vs_fp16_energy_pct"] = None
    assert ecc.share_url("https://quantenergy.tech", report) is None


def test_share_flag_writes_link_file(tmp_path):
    run_cli(tmp_path, "--gpu_arch", "blackwell", "--params_b", "1.1",
            "--precision", "NF4", "--share")
    link = (tmp_path / "share_url.txt").read_text().strip()
    assert link.startswith("https://quantenergy.tech/?tab=run&overlay=")
    report = json.loads((tmp_path / "energy.json").read_text())
    assert report["share_url"] == link


def test_prefetch_is_offline_safe():
    """A dead endpoint must return None quickly and never raise."""
    out = ecc.prefetch_prediction("http://127.0.0.1:9", 1.1, "blackwell", "NF4", 1,
                                  timeout=0.5)
    assert out is None


def _ns(**over):
    """Minimal argparse-like namespace with all run args defaulted to None."""
    fields = ("parameters_file", "output_dir", "model_name", "model", "precision",
              "gpu_arch", "batch_size", "params_b", "tokens", "iterations", "warmup",
              "sample_rate_hz", "context_length", "dry_run")
    ns = types.SimpleNamespace(**{f: None for f in fields})
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# --- zero-config arch detection --------------------------------------------

@pytest.mark.parametrize("nvml_name,expected", [
    ("NVIDIA GeForce RTX 4090", "ada"),
    ("NVIDIA GeForce RTX 4090 D", "ada"),
    ("NVIDIA L40S", "ada"),
    ("NVIDIA L4", "ada"),
    ("NVIDIA RTX 4000 Ada Generation", "ada"),
    ("NVIDIA GeForce RTX 5090", "blackwell"),
    ("NVIDIA B200", "blackwell"),
    ("NVIDIA H100 80GB HBM3", "hopper"),
    ("NVIDIA H200", "hopper"),
    ("NVIDIA A100-SXM4-40GB", "ampere"),
    ("NVIDIA A800-SXM4-80GB", "ampere"),
    ("NVIDIA A10G", "ampere"),
    ("NVIDIA RTX A6000", "ampere"),
    ("NVIDIA GeForce RTX 3090", "ampere"),
    ("Tesla T4", "turing"),
    ("NVIDIA GeForce RTX 2080 Ti", "turing"),
    ("Tesla V100-SXM2-16GB", "volta"),
    ("NVIDIA GeForce GTX 1080", None),
    ("", None),
    (None, None),
])
def test_detect_arch(nvml_name, expected):
    assert ecc.detect_arch(nvml_name) == expected


def test_resolve_arch_keeps_explicit_value(monkeypatch):
    monkeypatch.setattr(ecc, "probe_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    p = {"gpu_arch": "turing"}
    ecc.resolve_arch(p)
    assert p["gpu_arch"] == "turing", "an explicit arch must never be overridden"


def test_resolve_arch_auto_uses_nvml_name(monkeypatch):
    monkeypatch.setattr(ecc, "probe_gpu_name", lambda: "NVIDIA A100-SXM4-40GB")
    p = {"gpu_arch": "auto"}
    ecc.resolve_arch(p)
    assert p["gpu_arch"] == "ampere"


def test_resolve_arch_auto_without_nvml_does_not_guess(monkeypatch):
    monkeypatch.setattr(ecc, "probe_gpu_name", lambda: None)
    p = {"gpu_arch": "auto"}
    ecc.resolve_arch(p)
    assert p["gpu_arch"] == "unknown"


def test_shipped_params_default_to_auto_arch():
    import yaml
    params = yaml.safe_load(
        (REPO / "workspace" / "parameters" / "energy_params.yaml").read_text())
    assert params["gpu_arch"] == "auto", "the zero-config path relies on this default"


def test_auto_arch_run_is_schema_valid_without_a_gpu(tmp_path):
    """`gpu_arch: auto` on a GPU-less host: no crash, no fabricated measurement."""
    report = run_cli(tmp_path, "--gpu_arch", "auto", "--params_b", "1.1")
    jsonschema.validate(report, SCHEMA)
    assert report["results"]["basis"] != "measured"


# --- in-run schema self-check ----------------------------------------------

def test_validate_report_accepts_a_real_report():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    report = ecc.build_report(params, measured=None,
                              ref=ecc.reference_estimate(1.1, "ada", "NF4"))
    ok, validator, detail = ecc.validate_report(report)
    assert ok is True, (validator, detail)


def test_validate_report_rejects_a_broken_report():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    report = ecc.build_report(params, measured=None,
                              ref=ecc.reference_estimate(1.1, "ada", "NF4"))
    del report["results"]["basis"]
    assert ecc.validate_report(report)[0] is False


def test_builtin_checker_matches_jsonschema_on_the_examples():
    """The dependency-free fallback must agree with jsonschema, since the runtime
    image ships without jsonschema and still reports 'schema: valid'."""
    for name in ("energy.no-gpu.json", "energy.measured.illustrative.json"):
        report = json.loads((REPO / "examples" / name).read_text())
        assert ecc._check_against_schema(report, SCHEMA) == []
    bad = json.loads((REPO / "examples" / "energy.no-gpu.json").read_text())
    bad["scenario"] = "Server"                     # not in the schema's enum
    assert ecc._check_against_schema(bad, SCHEMA)
    bad2 = json.loads((REPO / "examples" / "energy.no-gpu.json").read_text())
    bad2["system_under_test"]["accelerator_count"] = 0     # below minimum
    assert ecc._check_against_schema(bad2, SCHEMA)
    bad3 = json.loads((REPO / "examples" / "energy.no-gpu.json").read_text())
    bad3["software"] = {"packages": {"torch": 2.4}}        # additionalProperties type
    assert ecc._check_against_schema(bad3, SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad3, SCHEMA)


# --- software provenance ----------------------------------------------------

def test_report_records_the_versions_it_ran_on(tmp_path):
    report = run_cli(tmp_path, "--gpu_arch", "ada", "--params_b", "1.1",
                     "--precision", "NF4")
    jsonschema.validate(report, SCHEMA)
    sw = report["software"]
    assert sw["python"].startswith(".".join(str(v) for v in sys.version_info[:2]))
    assert report["measurement"]["warmup"] == 2
    # Nothing was executed on this path, so it must not claim anything about how
    # comparable the (absent) kernels are.
    assert "matches_reference_pins" not in sw


def test_measured_report_states_whether_it_used_the_pins():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    measured = {"total_energy_joules": 100.0, "tokens_generated": 2560,
                "energy_per_token_mj": 39.1, "avg_power_watts": 300.0,
                "throughput_tokens_per_s": 100.0, "gpu_name": "NVIDIA GeForce RTX 4090"}
    report = ecc.build_report(params, measured=measured, ref=None)
    jsonschema.validate(report, SCHEMA)
    assert report["results"]["basis"] == "measured"
    assert isinstance(report["software"]["matches_reference_pins"], bool)


def test_pin_mismatch_is_reported_against_requirements(monkeypatch):
    monkeypatch.setattr(ecc, "reference_pins",
                        lambda: {"torch": "2.13.0", "transformers": "4.57.6"})
    monkeypatch.setattr(ecc, "_pkg_version",
                        lambda name: "4.46.3" if name == "transformers" else None)
    sw = ecc.collect_software()
    assert sw["matches_reference_pins"] is False
    assert sw["differs_from_reference_pins"] == ["transformers 4.46.3 != 4.57.6"]
    # torch is expected to differ (it must match the host driver), so it is not
    # compared and must not show up as a comparability problem.
    assert "torch" not in sw["reference_pins"]


def test_matching_pins_report_no_comparability_caveat(monkeypatch):
    monkeypatch.setattr(ecc, "reference_pins", lambda: {"transformers": "4.57.6"})
    monkeypatch.setattr(ecc, "_pkg_version",
                        lambda name: "4.57.6" if name == "transformers" else None)
    sw = ecc.collect_software()
    assert sw["matches_reference_pins"] is True
    assert "comparability_note" not in sw


def test_reference_pins_are_read_from_requirements():
    pins = ecc.reference_pins()
    assert pins["transformers"] and pins["bitsandbytes"]

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

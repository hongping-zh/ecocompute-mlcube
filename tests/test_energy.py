"""Tests for the EcoCompute energy MLCube.

They exercise the entrypoint through its real CLI (the same `energy_estimate`
task MLCube invokes) and check that every report validates against the schema
and preserves the measured-vs-derived honesty guarantees.
"""
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schema" / "energy.schema.json").read_text())
ENTRY = REPO / "entrypoint.py"


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

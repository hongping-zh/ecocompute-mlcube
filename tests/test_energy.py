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


def test_quantization_hint_is_quiet_for_fp16_and_explains_a_broken_backend():
    assert ecc._quantization_backend_hint("FP16") is None
    # No CUDA torch here, so the probe fails the same way a CUDA-line mismatch
    # does: the hint must name bitsandbytes rather than blame the GPU.
    hint = ecc._quantization_backend_hint("NF4")
    assert hint and "bitsandbytes" in hint


def test_reference_pins_are_read_from_requirements():
    pins = ecc.reference_pins()
    assert pins["transformers"] and pins["bitsandbytes"]


# --- quality probe (perplexity) ---------------------------------------------

class _FakeIds:
    """Just enough of a 1-D LongTensor for the probe's windowing."""

    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n

    def __getitem__(self, s):
        start, stop, _ = s.indices(self._n)
        return _FakeIds(max(0, stop - start))

    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self


class _FakeModel:
    """Returns the losses it is handed, one per forward call."""

    def __init__(self, losses, max_pos=4096):
        self.losses = list(losses)
        self.calls = []
        self.config = types.SimpleNamespace(max_position_embeddings=max_pos)

    def parameters(self):
        yield types.SimpleNamespace(device="cuda:0")

    def __call__(self, batch, labels=None):
        self.calls.append(batch.numel())
        return types.SimpleNamespace(loss=self.losses.pop(0))


def _fake_torch():
    import contextlib
    return types.SimpleNamespace(no_grad=contextlib.nullcontext)


def _fake_tok(n_tokens):
    def tok(_text, return_tensors=None, add_special_tokens=None):
        return types.SimpleNamespace(input_ids=[_FakeIds(n_tokens)])
    return tok


def test_perplexity_weights_windows_by_predicted_tokens():
    """A short trailing window must not count as much as a full one."""
    import math
    model = _FakeModel([2.0, 4.0])
    out = ecc.perplexity(model, _fake_tok(1500), "text", _fake_torch(), seq_len=1024)
    assert model.calls == [1024, 476]           # non-overlapping windows
    assert out["tokens_evaluated"] == 1023 + 475
    expected = (2.0 * 1023 + 4.0 * 475) / (1023 + 475)
    assert out["mean_nll"] == pytest.approx(expected, abs=1e-5)
    assert out["perplexity"] == pytest.approx(math.exp(expected), abs=1e-3)
    assert out["windows"] == 2


def test_perplexity_window_is_capped_by_the_model_context():
    model = _FakeModel([1.0], max_pos=512)
    out = ecc.perplexity(model, _fake_tok(400), "text", _fake_torch(), seq_len=1024)
    assert out["seq_len"] == 512 and model.calls == [400]


def test_vendored_evaluation_text_is_the_one_the_docs_fingerprint():
    loaded = ecc.load_quality_text()
    readme = (REPO / "quality" / "README.md").read_text()
    assert loaded["corpus"]["sha256"] in readme
    assert loaded["corpus"]["bytes"] > 10000


def _measured(ppl=None, **over):
    m = {"total_energy_joules": 100.0, "tokens_generated": 2560,
         "energy_per_token_mj": 39.1, "avg_power_watts": 300.0,
         "throughput_tokens_per_s": 100.0, "gpu_name": "NVIDIA GeForce RTX 4090",
         "quality": None}
    if ppl is not None:
        m["quality"] = {"perplexity": ppl, "mean_nll": 2.0,
                        "tokens_evaluated": 15000, "windows": 15, "seq_len": 1024}
    m.update(over)
    return m


def test_quality_delta_is_relative_to_the_same_run_s_fp16_baseline():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    params["quality_corpus"] = {"name": "eval_text.txt", "bytes": 60600, "sha256": "ab" * 32}
    report = ecc.build_report(params, measured=_measured(ppl=10.5), ref=None,
                              fp16_measured=_measured(ppl=10.0, energy_per_token_mj=30.0))
    jsonschema.validate(report, SCHEMA)
    q = report["quality"]
    assert q["basis"] == "measured" and q["metric"] == "perplexity"
    assert q["delta_vs_fp16_pct"] == pytest.approx(5.0)
    assert q["corpus"]["sha256"] == "ab" * 32
    # The energy figure must be untouched by the probe.
    assert report["results"]["basis"] == "measured"
    assert report["results"]["vs_fp16_energy_pct"] == pytest.approx(30.3, abs=0.1)


def test_quality_without_an_fp16_baseline_refuses_to_imply_a_delta():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    report = ecc.build_report(params, measured=_measured(ppl=10.5), ref=None)
    jsonschema.validate(report, SCHEMA)
    assert "delta_vs_fp16_pct" not in report["quality"]
    assert "no FP16 perplexity" in report["quality"]["delta_note"]


def test_a_failed_quality_probe_leaves_the_energy_measurement_intact():
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    measured = _measured()
    measured["quality"] = {"error": "OutOfMemoryError: CUDA out of memory"}
    report = ecc.build_report(params, measured=measured, ref=None,
                              fp16_measured=_measured(energy_per_token_mj=30.0))
    jsonschema.validate(report, SCHEMA)
    assert report["quality"]["basis"] == "unavailable"
    assert "OutOfMemory" in report["quality"]["error"]
    assert report["results"]["basis"] == "measured"


def test_no_probe_means_no_quality_block_at_all(tmp_path):
    params = ecc.load_params(_ns(gpu_arch="ada", params_b=1.1, precision="NF4"))
    report = ecc.build_report(params, measured=_measured(), ref=None)
    jsonschema.validate(report, SCHEMA)
    assert "quality" not in report
    # ... and the dataset-derived path never invents one.
    assert "quality" not in run_cli(tmp_path, "--gpu_arch", "ada", "--params_b", "1.1")


def test_quality_probe_can_be_turned_off_and_reads_a_custom_text(tmp_path):
    off = ecc.load_params(_ns(gpu_arch="ada", no_quality_probe=True))
    assert ecc.resolve_quality(off) is None

    mine = tmp_path / "heldout.txt"
    mine.write_text("some held-out text of my own")
    p = ecc.load_params(_ns(gpu_arch="ada", quality_text=str(mine)))
    loaded = ecc.resolve_quality(p)
    assert loaded["text"].startswith("some held-out")
    assert p["quality_corpus"]["name"] == "heldout.txt"
    assert p["quality_corpus"]["sha256"] != ecc.load_quality_text()["corpus"]["sha256"]


def test_an_unreadable_text_disables_the_probe_rather_than_the_run(tmp_path):
    p = ecc.load_params(_ns(gpu_arch="ada", quality_text=str(tmp_path / "nope.txt")))
    assert ecc.resolve_quality(p) is None


def test_reports_from_before_the_quality_block_are_still_valid():
    old = json.loads((REPO / "examples" / "energy.measured.illustrative.json").read_text())
    old["schema_version"] = "ecocompute-energy/1.0"
    old.pop("quality", None)
    jsonschema.validate(old, SCHEMA)

# --- regression check against the published curve ---------------------------

def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "ecc_check_regression", REPO / "tools" / "check_regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load_checker()

CURVES = {
    "curves": {"ada": {"INT8": {"A": 272.7, "S": 291.9, "Nstar": 2.5938,
                                "resid_std": 22.44,
                                "anchors": [{"N": 1.1, "dE": 146.11,
                                             "gpu": "RTX 4090", "n": 1}]}}},
    "borrow": {},
}


def _report(basis="measured", source="direct-nvml", de=137.9):
    return {
        "system_under_test": {"gpu": "NVIDIA GeForce RTX 4090", "gpu_arch": "ada"},
        "workload": {"params_b": 1.1, "precision": "INT8"},
        "measurement_source": source,
        "results": {"vs_fp16_energy_pct": de, "basis": basis},
    }


def _run_check(tmp_path, report, k="2"):
    p = tmp_path / "energy.json"
    c = tmp_path / "curves.json"
    p.write_text(json.dumps(report))
    c.write_text(json.dumps(CURVES))
    return chk.main([str(p), "--curves", str(c), "--k", k])


def test_regression_check_passes_inside_the_band(tmp_path):
    assert _run_check(tmp_path, _report()) == 0


def test_regression_check_fails_outside_the_band(tmp_path):
    assert _run_check(tmp_path, _report(de=10.0)) == 1


def test_regression_check_refuses_to_grade_a_derived_report(tmp_path):
    # An interpolated report comes *from* the curve: grading it against the
    # curve would always pass and would mean nothing.
    derived = _report(basis="interpolated",
                      source="ecocompute-dataset (on-device measurement failed)")
    assert _run_check(tmp_path, derived) == 2


def test_regression_check_skips_an_architecture_without_a_curve(tmp_path):
    other = _report()
    other["system_under_test"]["gpu_arch"] = "turing"
    assert _run_check(tmp_path, other) == 2


def test_regression_check_prefers_an_anchor_at_the_same_size(tmp_path, capsys):
    # The current ada/INT8 fit sits ~40 pp above its own 1.1B anchor, so a run
    # that agrees with the anchor must pass even though it is far from the fit.
    assert _run_check(tmp_path, _report(de=146.0)) == 0
    assert "anchors at this size" in capsys.readouterr().out

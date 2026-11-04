"""Mock-only tests for the ROCm telemetry first cut (issue #27).

No AMD hardware is required: every test fakes the _rocm_smi_json subprocess
seam with synthetic payloads matching rocm-smi's documented output shapes for
both the ROCm 5.x and 6.x field names. The AMD path is not wired into the measurement flow yet; these tests
cover the sampler contract (mirroring PowerSampler), the lenient parsers,
and the watts-vs-milliwatts units trap.
"""
import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("ecc_entrypoint_rocm", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ecc = _load_entrypoint()

# Synthetic fixtures built from rocm-smi's documented output shape: ROCm 5.x
# uses "Average Graphics Package Power", 6.x renamed it to
# "Current Socket Graphics Package Power".
ROCM5_POWER = {
    "card0": {
        "Average Graphics Package Power (W)": "95.0",
        "GPU use (%)": "34",
    },
    "system": {"Driver version": "5.7.0"},
}
ROCM6_POWER = {
    "card0": {
        "Current Socket Graphics Package Power (W)": "120.5",
        "GPU use (%)": "61",
    },
    "system": {"Driver version": "6.2.0"},
}
ROCM5_TEMP = {
    "card0": {
        "Temperature (Sensor edge) (C)": "42.0",
        "Temperature (Sensor junction) (C)": "44.0",
        "Temperature (Sensor memory) (C)": "51.0",
    },
    "system": {"Driver version": "5.7.0"},
}
ROCM6_TEMP = {
    "card0": {
        "Temperature (Sensor edge) (C)": "43.5",
        "Temperature (Sensor junction) (C)": "46.0",
        "Temperature (Sensor memory) (C)": "55.0",
    },
    "system": {"Driver version": "6.2.0"},
}


# --- power parser: both ROCm generations ------------------------------------

def test_parse_power_rocm5_shape():
    assert ecc.parse_rocm_smi_power_watts(ROCM5_POWER) == 95.0


def test_parse_power_rocm6_shape():
    assert ecc.parse_rocm_smi_power_watts(ROCM6_POWER) == 120.5


def test_power_is_watts_not_milliwatts():
    """Units trap: rocm-smi reports watts (NVML reports milliwatts). The value
    must pass through unconverted - a silent x1000 here is the classic
    failure mode for exactly this pair of tools."""
    watts = ecc.parse_rocm_smi_power_watts(ROCM5_POWER)
    assert watts == pytest.approx(95.0)
    assert watts != pytest.approx(95000.0)


def test_parse_power_non_numeric_field_rejected():
    payload = {"card0": {"Average Graphics Package Power (W)": "N/A"}}
    with pytest.raises(ecc.RocmSmiError, match="not numeric"):
        ecc.parse_rocm_smi_power_watts(payload)


def test_parse_power_missing_card_section():
    with pytest.raises(ecc.RocmSmiError, match="no 'card0' section"):
        ecc.parse_rocm_smi_power_watts({"system": {}})


def test_parse_power_unknown_field_names():
    payload = {"card0": {"Some Unrelated Field": "1.0"}}
    with pytest.raises(ecc.RocmSmiError, match="no power field"):
        ecc.parse_rocm_smi_power_watts(payload)


# --- temperature parser / helper ---------------------------------------------

def test_parse_temperature_prefers_edge_sensor():
    # edge first (like NVML_TEMPERATURE_GPU), not junction or memory
    assert ecc.parse_rocm_smi_temperature_c(ROCM5_TEMP) == 42.0
    assert ecc.parse_rocm_smi_temperature_c(ROCM6_TEMP) == 43.5


def test_parse_temperature_falls_back_to_junction():
    payload = {"card0": {"Temperature (Sensor junction) (C)": "47.0"}}
    assert ecc.parse_rocm_smi_temperature_c(payload) == 47.0


def test_read_rocm_temperature_via_seam(monkeypatch):
    monkeypatch.setattr(ecc, "_rocm_smi_json", lambda *a, **k: ROCM5_TEMP)
    assert ecc.read_rocm_temperature() == 42.0


def test_read_rocm_temperature_none_on_failure(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi not installed")

    monkeypatch.setattr(ecc, "_rocm_smi_json", _boom)
    assert ecc.read_rocm_temperature() is None


# --- sampler contract (mirrors PowerSampler) ----------------------------------

def test_rocm_sampler_collects_samples_and_averages(monkeypatch):
    monkeypatch.setattr(ecc, "_rocm_smi_json", lambda *a, **k: ROCM5_POWER)
    s = ecc.ROCmPowerSampler(hz=100)
    s.start()
    time.sleep(0.15)
    s.stop()
    assert s.error is None
    assert len(s.samples) >= 2
    assert all(w == 95.0 for _, w in s.samples)  # watts, unconverted
    assert s.avg_watts() == pytest.approx(95.0)
    assert s.energy_joules() > 0.0


def test_rocm_sampler_survives_failing_reads(monkeypatch):
    """A host without rocm-smi must not crash the thread; error gets set."""

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi not installed")

    monkeypatch.setattr(ecc, "_rocm_smi_json", _boom)
    s = ecc.ROCmPowerSampler(hz=100)
    s.start()
    time.sleep(0.3)
    s.stop()
    assert s.samples == []
    assert s.error is not None
    assert s.dropped >= 5  # consecutive-failure bail kicked in


def test_rocm_sampler_default_rate_is_lower_than_nvml():
    # CLI spawn cost: the default stays below the NVML sampler's 10 Hz.
    assert ecc.ROCmPowerSampler().period > 1.0 / 10


def test_energy_joules_trapezoid_matches_nvml_math():
    s = ecc.ROCmPowerSampler(hz=10)
    s.samples = [(0.0, 100.0), (1.0, 200.0)]  # watts * seconds -> joules
    assert s.energy_joules() == pytest.approx(150.0)


def test_sampler_exposes_a_trace_clock(monkeypatch):
    monkeypatch.setattr(ecc, "_rocm_smi_json", lambda *a, **k: ROCM6_POWER)
    s = ecc.ROCmPowerSampler(hz=100)
    s.start()
    time.sleep(0.05)
    mid = s.elapsed()
    s.stop()
    end = s.elapsed()
    assert 0.0 < mid <= end < 1.0
    assert s.samples and all(0.0 <= t <= end for t, _ in s.samples)

# Example `energy.json` outputs

Both files validate against [`../schema/energy.schema.json`](../schema/energy.schema.json).

| file | how it was produced | `measurement_source` | `basis` |
|---|---|---|---|
| [`energy.no-gpu.json`](energy.no-gpu.json) | real output of `entrypoint.py energy_estimate --dry_run` on a host **without** a GPU | `ecocompute-dataset (no local GPU)` | `interpolated` |
| [`energy.measured.illustrative.json`](energy.measured.illustrative.json) | **hand-written illustrative sample** of the on-GPU measured shape | `ILLUSTRATIVE (synthetic) …` | `measured` |

> All four example reports carry `"code_commit": "unavailable"` and
> `"image": "unavailable"` in their provenance block: they came from
> bare local runs (or are hand-written), not docker-built images. A
> container-built report records the commit and image reference it was
> built from — traceability, not endorsement (TECHNICAL.md).

Reproduce the no-GPU example:

```bash
python3 entrypoint.py energy_estimate --dry_run \
    --parameters_file workspace/parameters/energy_params.yaml \
    --output_dir workspace/outputs
```

> The measured example is **synthetic** and exists only to document the output
> format of a real on-GPU run. It is clearly flagged in `measurement_source` and
> `results.note`; it is not a real measurement. On a real GPU host the
> `total_energy_joules`, `avg_power_watts`, and `throughput_tokens_per_s` fields
> are filled from direct NVML sampling.

## L0 verifiability example pair (`accepted-trace/` vs `rejected-tampered-trace/`)

`accepted-trace/` pairs an L0 report with its real sidecar CSV (digest matches,
integral re-derives the headline bit-for-bit); `rejected-tampered-trace/` is the
same pair with one watt value changed in the CSV. Validate both from this
directory and see the validator recompute clauses catch the tampering:

```bash
python3 tools/validate.py examples/accepted-trace/energy.json      # §4.7.x: silent (L0 pass); overall exit 1, see note
python3 tools/validate.py examples/rejected-tampered-trace/energy.json  # §4.7.5 tampered + §4.7.6
```

> The example reports are illustrative, not fully protocol-conformant: both
> commands exit 1 because of the same six non-L0 MUST violations the
> hand-written samples do not satisfy (§4.2.1, §4.3.2 ×2, §4.5.1 ×2, §4.6.2).
> What this pair demonstrates is the L0 delta: `accepted-trace` passes
> §4.7.5–4.7.8 silently, while `rejected-tampered-trace` adds [§4.7.5]
> tampered-sidecar and [§4.7.6] integral-mismatch on top of those six.

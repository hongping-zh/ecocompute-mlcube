# Example `energy.json` outputs

Both files validate against [`../schema/energy.schema.json`](../schema/energy.schema.json).

| file | how it was produced | `measurement_source` | `basis` |
|---|---|---|---|
| [`energy.no-gpu.json`](energy.no-gpu.json) | real output of `entrypoint.py energy_estimate --dry_run` on a host **without** a GPU | `ecocompute-dataset (no local GPU)` | `interpolated` |
| [`energy.measured.illustrative.json`](energy.measured.illustrative.json) | **hand-written illustrative sample** of the on-GPU measured shape | `ILLUSTRATIVE (synthetic) …` | `measured` |

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

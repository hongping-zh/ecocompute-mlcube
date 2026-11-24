# Changelog

Release history of the ecocompute-energy MLCube. Tags follow `schema-1.3-rN`
(the release train); the in-report `l0_revision` field tracks the L0 sampler
revision separately (e.g. this release ships `1.3-r7` L0 fields — see r6 below).

Each release page carries its own notes; this file is the condensed,
repo-side view. Verdicts and pinned hashes are authoritative on the release
page and in `TECHNICAL.md` ("Pinned artifacts").

## schema-1.3-r6 (2026-11-24)

Bundles everything merged to `main` after r5 (2026-11-05) — this is the
release where the bitwise-recompute story becomes true of the *downloaded*
artifacts, not just of `main`.

- **L0 r7: power-trace recomputation (`§4.7.5–4.7.8`).** The validator now
  recomputes what a report claims instead of only checking shape: sha256 over
  the raw CSV bytes with truncated/tampered/missing three-way discrimination
  (4.7.5), same-order trapezoidal integration on disk (4.7.6), the
  integral→headline cross-field check (4.7.7), and sample-index coverage
  (4.7.8). Structured L0 fields: dedicated `power_trace_measure*.csv` sampler
  (`stream: measure_window`), 0-based inclusive `sample_window`, the
  `trapezoid-csv-v1` rule enum, `l0_revision: 1.3-r7` (optional, backward
  compatible). Shipped examples: `examples/accepted-trace` (recomputes clean)
  and `examples/rejected-tampered-trace` (a hash-flip caught by 4.7.5).
- **`Certificates, not proofs.`** TECHNICAL.md now states the threat model
  in Lean-adjacent terms: the sidecar is an untrusted party's certificate,
  the validator is a small standard-library-only checker, and a self-signed
  hash is zero authority — recomputation defends against accidental
  corruption and arithmetic non-self-consistency, not a malicious author.
  External anchoring (Release → Zenodo version DOI → Sigstore) is explicitly
  deferred to a future RFC; traceability is not anchoring.
- **CI exercises the real MLCube contract path** (PR #47): a
  `mlcube-contract` job runs the actual `mlcube run` flow, including a
  sabotage counter-proof (a corrupted parameters file must flip `params_b`
  to the `None` sentinel). Plus the deployment traps it fixed, now encoded
  in the workflow: `pip < 24.1` (newer pip rejects omegaconf 2.1.0's invalid
  metadata), `--log-level` at the click-group level, and `set +e` around
  redirected commands so failures are not swallowed.
- **Container identity in the report.** `provenance.code_commit` and
  `provenance.image` are baked in at build time (ARG/ENV in both Dockerfiles;
  `unavailable` when run bare). The validator checks their *shape* only
  (clause 4.5.4) — they are traceability, not an endorsement; anyone can
  rebuild an image with arbitrary values. The published image is pushed per
  commit as `:sha-<short>` and a CI assertion pins `code_commit == github.sha`.
- **Ada data convergence.** The container's internal REFERENCE matched the
  site curves frozen before the July 2026 "pool Ada cards honestly" refit
  (4090D-only, no crossover), contradicting both the site and the measured
  coverage matrix — e.g. its dry-run said +20.2% (costs energy) at 7B where
  the pooled fit and the n=2 measurement both say energy is *saved*. The
  REFERENCE now mirrors the published pooled curves (RTX 4090 + 4090D, ada
  NF4 crossover ≈3.7B, anchors to 7B, INT8 curve added).
- **Honesty pass over the narrative layer.** Downloads now say which release
  they are; backend extension is labelled interface-defined (only the CI fake
  engine is wired); the SBOM sentence matches what shipped; the data-submission
  issue template no longer claims "schema-valid is the expected verdict".
- **Exit codes de-collided.** argparse usage errors exit 64 (EX_USAGE) —
  argparse's default 2 collided with the documented schema-invalid exit 2.
- **SBOM returns.** After r4/r5 shipped without one (r3 was the last), this
  release again carries `sbom.cdx.json` (CycloneDX, generated from the
  pinned `requirements.lock.txt`), alongside `SHA256SUMS` and
  `SHA256SUMS.all`. Per CONTRIBUTING, a release without an SBOM is not
  complete — r4/r5 were, in that sense, incomplete.

## schema-1.3-r5 (2026-11-05)

- Engine identity block, token-count acceptance conditions, reference fake
  adapter (#26) — originally slated for r6, shipped here.
- AMD fallback: NVML-first cross-check for the hip-nvcc false positive (#28).
- ROCm power/temperature telemetry sampler (#31, @amarjithkumar2007-wq):
  `rocm-smi` polling with lenient 5.x/6.x JSON parsing. Mock-tested only,
  not yet wired into the measurement flow.

## schema-1.3-r4 (2026-10-16)

- PR #22 — `context_length` is real: the prompt is tiled/truncated to the
  recorded value (runs at `context_length ≠ 2048` are not directly comparable
  with dataset v1.1.0).
- PR #20 — AMD GPU (ROCm) detection: honest `measurement_source` downgrade
  with the card name, instead of an opaque NVML error.

## schema-1.3-r3 (2026-10-11)

- Supply-chain hardening: the quickstart becomes a verified release asset
  (download → `sha256sum -c` → run), the image runs non-root, `--dry-run`
  prints the whole plan. First release with `sbom.cdx.json`.

## schema-1.3-r2 (2026-09-27)

- The semantic validator stops waving reports through: every machine-checkable
  Protocol v1.1 MUST is enforced (4.1.1 NVML method, 4.1.2 ≥10 Hz, 4.2.1 FP16
  baseline per precision, 4.3.2 tokens/iterations/greedy, 4.4.1 generation
  window, 4.5.2 conformance floor, 4.6.1/4.6.2 thermal and arm-order records).

## schema-1.3-r1 (2026-09-26)

- First revision of the 1.3 report schema plus the tools that close the
  spec↔schema loop: fixed schema_version/benchmark values, ≥10 Hz sampling,
  required MUST-fields at 1.3, thermal block with honest-failure semantics.

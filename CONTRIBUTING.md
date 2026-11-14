# Contributing to EcoCompute MLCube

Thank you for your interest in contributing! EcoCompute is an open research project
building toward a reproducible inference-energy standard. There are several ways to
contribute, and **you do not need a GPU to get started**.

## Ways to contribute

### 1. Run a measurement and submit a replication

The most valuable contribution: run the container on your GPU and submit your
`energy.json` as a public replication. This is how the dataset grows.

**What "the dataset" is:** the project's published measurement corpus — the
Zenodo dataset releases plus the live dataset behind
[quantenergy.tech](https://quantenergy.tech). Community replications are part
of it, not decoration around it: a submission is *accepted into the dataset*
once it validates `protocol-conformant` and is archived in the
[replications gallery](https://quantenergy.tech/replications), and archived
replications feed the coverage grid on the site. "Accepted into the dataset"
therefore means exactly one thing: gallery archiving.

**What you need:** an NVIDIA GPU (any architecture) or a free Colab T4.

**Steps:**

1. Run the container — see the [Quick start](README.md#quick-start--3-commands)
   in the README. The default run measures NF4 + FP16 baseline on TinyLlama-1.1B.
2. Validate your report:
   ```bash
   python3 tools/validate.py ecocompute-out/energy.json --profile v1.1-core
   ```
   Aim for `protocol-conformant` or `dataset-eligible`. `schema-valid` alone means
   the structure is correct but some protocol MUSTs are not met.

   Or let your CI do it: this repo ships a reusable GitHub Action that runs the
   same validator — no Docker, no GPU, about 20 seconds:
   ```yaml
   - uses: hongping-zh/ecocompute-mlcube/.github/actions/validate-report@main
     with:
       report: ecocompute-out/energy.json
   ```
   Pin `validator-ref` to a commit SHA or tag for reproducible runs.
3. Submit at [quantenergy.tech/replications](https://quantenergy.tech/replications/#submit).
   Your `energy.json` is encoded in the URL — nothing is uploaded to a server.

**No GPU?** Use the [Colab notebook](https://colab.research.google.com/github/hongping-zh/quant-energy/blob/main/notebooks/ecocompute-measure.ipynb)
for a free T4 measurement. A single pass is n=1 and directional; the last cell
repeats both arms 5x if you want a repeatability figure.

**What makes a good replication:**

- A different GPU architecture than the ones already measured (Turing, Ampere,
  Ada, Hopper, Blackwell are covered; anything else is valuable).
- A different model size, especially in the crossover region (2B–8B).
- A different backend (llama.cpp, vLLM, TensorRT-LLM) — currently almost all
  data is bitsandbytes.
- Batch sizes > 1 — the entire variable matrix is empty for batch and context.

### 2. Report a bug or issue

Found something wrong? Open an issue at
[github.com/hongping-zh/ecocompute-mlcube/issues](https://github.com/hongping-zh/ecocompute-mlcube/issues).

**Good bug reports include:**

- The exact command you ran (or the `quickstart.sh` invocation).
- Your GPU model and driver version (`nvidia-smi` output).
- The `energy.json` output (or the relevant portion).
- The validator output if you ran `tools/validate.py`.
- Whether you used Docker or native mode.

**Label guide:**

- `bug` — something is broken or producing wrong results.
- `documentation` — the README, protocol, or error messages are unclear.
- `enhancement` — a new feature or improvement suggestion.
- `good first issue` — suitable for new contributors (see below).
- `measurement` — about the measurement methodology or protocol.
- `schema` — about the `energy.json` schema or validator.

### 3. Contribute code

Code contributions are welcome. Here is how to get started:

#### Development setup

```bash
git clone https://github.com/hongping-zh/ecocompute-mlcube.git
cd ecocompute-mlcube

# Install dev dependencies
pip install -r requirements-dev.txt

# Run the test suite
python -m pytest tests/ -q

# Run the container locally (no GPU needed for smoke test)
python3 entrypoint.py energy_estimate --dry_run \
    --parameters_file workspace/parameters/energy_params.yaml \
    --output_dir workspace/outputs
```

#### Code style

- Python 3.9+. The container image uses the version pinned in `requirements.lock.txt`.
- Follow the existing code style in `entrypoint.py`. No formatter is enforced, but
  keep lines under 100 characters and use descriptive variable names.
- Add tests for new functionality in `tests/`. Every PR must pass CI.
- Schema changes require a new schema version and a migration note in the README.

#### Pull request process

1. Fork the repository.
2. Create a feature branch: `git checkout -b my-feature`.
3. Make your changes. Add tests if applicable.
4. Run the test suite: `python -m pytest tests/ -q`.
5. Run the validator on any example outputs you changed:
   `python3 tools/validate.py examples/energy.measured.illustrative.json`.
6. Push your branch and open a pull request.
7. Describe what you changed and why. Link any related issues.

#### What to work on

Look for issues labeled [`good first issue`](https://github.com/hongping-zh/ecocompute-mlcube/labels/good%20first%20issue)
for beginner-friendly tasks. Other high-value areas:

- **New GPU support** — add NVML handling for GPUs not yet measured.
- **Backend diversity** — add llama.cpp, vLLM, or TensorRT-LLM as alternative
  inference backends (currently only bitsandbytes via transformers).
- **Batch / context coverage** — the variable matrix is empty for batch_size > 1
  and context_length != 2048. This is the biggest data gap.
- **Schema evolution** — as the protocol matures, the schema will need new fields.
- **Documentation** — the protocol, number glossary, and limitation statements
  can always be clearer.

### 4. Improve the protocol or dataset

The measurement protocol lives at [quantenergy.tech/method/](https://quantenergy.tech/method/).
If you have suggestions for the protocol itself (measurement window, thermal
requirements, randomization rules), open an issue or discuss in the
[SSRN preprint comments](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6854700).

### 5. Spread the word

- Cite the project in your papers: see [quantenergy.tech/cite/](https://quantenergy.tech/cite/).
- Share your replication results.
- Mention EcoCompute when discussing LLM quantization energy in blogs, talks,
  or social media.

## Release checklist

Before publishing any release, the maintainer runs, on a clean Linux
environment (a throwaway container is fine — this is exactly how the CRLF
incident of 2026-10-28 was caught only after publication):

- `bash -n quickstart.sh && bash -n run.sh` — syntax, zero `\r` (CI guards
  this too, but check locally);
- download the assets the way a user would (curl from the release URL) and
  run `bash quickstart.sh --dry-run` — the published bytes, not the working
  tree, must actually run on Linux;
- recompute `SHA256SUMS` / `SHA256SUMS.all` from the published assets, and
  cross-check every line against the tag (a wrong hash that nobody verifies
  survives indefinitely — the schema line in `SHA256SUMS.all` did).
- regenerate the SBOM against the published image and attach it to the
  release assets (e.g. `syft <image> -o cyclonedx-json > sbom.cdx.json`)
  — r3 shipped one, r4/r5 shipped without, which the README then overstated.
  A release without an SBOM is not complete.

## Project roles and decision process

EcoCompute is a small research project with transparent governance:

- **Hongping Zhang** — maintainer (admin). Owns the measurement protocol, the
  schema contract and the release process.
- **Amarjith Kumar** — the project's most sustained contributor: AMD/ROCm GPU
  detection and fallback (#20), context_length applied in the measurement
  path (#22), the ROCm power/temperature telemetry sampler (#31); co-author
  on the container's Zenodo software record. Being onboarded to the triage
  role (2026-11); write access is intended to follow the ROCm real-machine
  validation landing.
- **Everyone else** — contributions come through fork PRs and data
  submissions; both paths are described above and both count toward
  authorship (see the policy below).

Decisions are made transparently:

- **Bug fixes and documentation improvements** are merged quickly.
- **Schema changes** require a new version and a changelog entry. The schema is
  the contract between the container and the website — changes affect both.
- **Protocol changes** are discussed openly. The protocol is normative: it defines
  what makes two submissions comparable. Changes must be backward-compatible or
  versioned.
- **New features** are evaluated against the project goal: a reproducible
  inference-energy standard. Features that add complexity without improving
  reproducibility may be declined.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).

## Authorship policy

Recognition comes in three tiers, and **which tier you occupy is your choice**:

1. **Acknowledgment** (the default) — your handle is credited in the README
   acknowledgments, the website replications gallery, and release notes. No
   consent or invitation is needed: every accepted contribution starts here,
   and staying here is a complete, final choice — not a waiting state.
2. **Dataset contributor** — your measurements are part of the published
   dataset (see [above](#1-run-a-measurement-and-submit-a-replication)) with
   you named in the dataset's credits, but you are not an author on
   publications. This tier exists for contributors who want their data in the
   dataset without authorship. Requires your explicit consent.
3. **Author** — listed as an author on project publications. Requires meeting
   one of the thresholds below **and** your explicit consent.

Contributors are invited to be listed as authors on project publications
(papers, dataset releases, conference submissions) when they meet either of
the following thresholds:

- **≥ 2 merged code PRs of substance** — sustained contributions to the
  measurement container, schema, validator, or documentation. Substance
  means a change to how something is measured, validated, or documented;
  two typo fixes do not add up to authorship, and the bar is deliberately
  qualitative rather than a raw PR count. The worked example is Amarjith
  Kumar: three PRs that each changed what the container can measure
  (ROCm detection, context_length handling, the ROCm telemetry sampler),
  plus cross-checking the result against real hardware.
- **≥ 1 qualified data submission** — an `energy.json` that validates
  `protocol-conformant` (`tools/validate.py --profile v1.1-core`) and is
  accepted into the published dataset, i.e. archived in the
  [replications gallery](https://quantenergy.tech/replications) as defined
  [above](#1-run-a-measurement-and-submit-a-replication).

Meeting a threshold makes you *eligible*, never listed automatically: the
maintainer will reach out to eligible contributors before each publication
to confirm listing preferences, and a contributor who asks to stay at the
acknowledgment level stays there — the request is honored as final. The
worked example is @OliverYanghoo: three protocol-conformant V100S
replications (#42–#44, 2026-11) met the data threshold, and he asked to
remain acknowledged.

Single-PR bug fixes, typo corrections, and exploratory submissions that do
not pass validation remain at the acknowledgment tier and do not by
themselves qualify for authorship.

## Questions?

- **Measurement questions:** see the [FAQ](https://quantenergy.tech/#faq) on the website.
- **Protocol questions:** see [quantenergy.tech/method/](https://quantenergy.tech/method/).
- **Technical issues:** open a [GitHub issue](https://github.com/hongping-zh/ecocompute-mlcube/issues).
- **General discussion:** open a [GitHub Discussion](https://github.com/hongping-zh/ecocompute-mlcube/discussions).

# Contributing to EcoCompute MLCube

Thank you for your interest in contributing! EcoCompute is an open research project
building toward a reproducible inference-energy standard. There are several ways to
contribute, and **you do not need a GPU to get started**.

## Ways to contribute

### 1. Run a measurement and submit a replication

The most valuable contribution: run the container on your GPU and submit your
`energy.json` as a public replication. This is how the dataset grows.

**What you need:** an NVIDIA GPU (any architecture) or a free Colab T4.

**Steps:**

1. Run the container — see the [Quick start](README.md#quick-start--three-commands-no-configuration)
   in the README. The default run measures NF4 + FP16 baseline on TinyLlama-1.1B.
2. Validate your report:
   ```bash
   python3 tools/validate.py ecocompute-out/energy.json
   ```
   Aim for `protocol-conformant` or `dataset-eligible`. `schema-valid` alone means
   the structure is correct but some protocol MUSTs are not met.
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

## Decision process

This project has a single maintainer (Hongping Zhang). Decisions are made
transparently:

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

Contributors are invited to be listed as authors on project publications
(papers, dataset releases, conference submissions) when they meet either of
the following thresholds:

- **≥ 2 merged code PRs** — sustained code contributions to the measurement
  container, schema, validator, or documentation; or
- **≥ 1 qualified data submission** — a protocol-conformant `energy.json`
  accepted into the published dataset (see
  [quantenergy.tech/replications](https://quantenergy.tech/replications/#submit)).

Single-PR bug fixes, typo corrections, and exploratory submissions that do not
pass validation are acknowledged in release notes and the website gallery but
do not by themselves qualify for authorship. The maintainer will reach out to
eligible contributors before each publication to confirm listing preferences.

## Questions?

- **Measurement questions:** see the [FAQ](https://quantenergy.tech/#faq) on the website.
- **Protocol questions:** see [quantenergy.tech/method/](https://quantenergy.tech/method/).
- **Technical issues:** open a [GitHub issue](https://github.com/hongping-zh/ecocompute-mlcube/issues).
- **General discussion:** open a [GitHub Discussion](https://github.com/hongping-zh/ecocompute-mlcube/discussions).

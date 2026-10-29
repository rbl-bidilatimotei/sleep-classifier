# Sleep classifier

A learning project for causal sleep-stage classification from wrist motion,
heart rate, and historical steps, using small PyTorch models in a later milestone.

## Status

Session 1 implements reproducible downloads, raw-data audits, and one exploration
notebook. No training, imputation, epoch features, or accuracy results yet.

- [Session findings and next decisions](docs/session-01.md)
- [Proposed experiment protocol](docs/first-experiment.md)
- [Literature notes](docs/literature-notes.md)
- [Project collaboration instructions](AGENTS.md)

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this
repository once hosted, and run these commands from the checkout root:

```bash
uv sync --locked --group notebook
uv run --locked python -m sleep_classifier.data download --config configs/explore.yaml
uv run --locked python -m sleep_classifier.data audit --config configs/explore.yaml
uv run --locked jupyter lab notebooks/01_data_exploration.ipynb
```

The download fetches one participant's four modalities and dataset license.
It verifies publisher SHA-256 hashes, reuses verified files, and rejects corrupted
existing files. The manifest itself is obtained over HTTPS, not independently signed.
Audit outputs go to `artifacts/audits/1066528.json`. YAML paths resolve relative to
that YAML file, so commands also work with an absolute `--config` path from elsewhere.
The package is installed editable by uv; no notebook `sys.path` hacks are needed.

For a fresh-kernel execution without opening Jupyter:

```bash
uv run --locked python scripts/execute_notebook.py
```

The executed notebook is saved under `artifacts/notebooks/`; the source notebook
stays output-free. This runner explicitly uses the active Python interpreter.

## Tests and formatting

```bash
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Tests use synthetic files and mocked network responses; no dataset download is
needed. They cover malformed data, preserved ordering, duplicate detection,
missing step coverage, checksum failures, atomic downloads, and config paths.

## Notebooks and Colab

Notebooks explain experiments and call package functions. Reproducible shell
operations use module names, for example:

```python
!uv run --project /path/to/sleep-classifier --locked python -m sleep_classifier.data audit --config /path/to/sleep-classifier/configs/explore.yaml
```

`python -m` takes a dotted module name, not a filesystem path. A standalone script
instead uses `uv run python scripts/execute_notebook.py`.

Colab has its own notebook kernel; `uv sync` does not replace that kernel. After
cloning the future repository and installing uv in Colab, use the same uv commands
to download, audit, and execute the notebook through `scripts/execute_notebook.py`.
That keeps package operations in the locked environment. Direct interactive imports
in a Colab-hosted notebook require installing the package into Colab's kernel too;
that separate setup is not yet tested or guaranteed to match this lockfile. Local
Jupyter is the verified route for this milestone; Colab integration comes later.

## Layout

```text
configs/explore.yaml            # Active data settings
src/sleep_classifier/data.py    # Shared ingestion, audit, and CLI
notebooks/01_data_exploration.ipynb
scripts/execute_notebook.py     # Reproducible fresh-kernel runner
tests/                        # Synthetic unit tests (no network)
docs/                          # Protocol, sources, session findings
data/raw/                     # Immutable source data, ignored by Git
artifacts/                    # Generated reports/notebooks, ignored by Git
```

Model/training/augmentation configs will accompany their implementations. We use
one authoritative `AGENTS.md`; `CLAUDE.md` imports it. Dependencies are locked in
`uv.lock`; Docker is deferred until the training workflow is established.

## Data attribution

[PhysioNet sleep-accel v1.0.0](https://physionet.org/content/sleep-accel/1.0.0/):
Walch, O. (2019), [DOI 10.13026/hmhs-py35](https://doi.org/10.13026/hmhs-py35).
Dataset license: Open Data Commons Attribution License v1.0; downloaded as
`data/raw/sleep-accel/1.0.0/LICENSE.txt`.

Original study: Walch et al., *Sleep stage prediction with raw acceleration and
photoplethysmography heart rate data derived from a consumer wearable device*,
SLEEP (2019), [DOI 10.1093/sleep/zsz180](https://doi.org/10.1093/sleep/zsz180).

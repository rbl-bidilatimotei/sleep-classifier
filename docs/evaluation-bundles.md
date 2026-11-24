# Evaluation-bundle export and loading

Status: implementation prepared on 2026-11-24. No deployment model is fitted by this workflow.
Read the [model card](model-card.md) and [version-one contract](inference-bundle-contract.md) before exporting.

## What is implemented

The exporter packages one explicitly selected research fold from the completed chronological audit.
The default configuration names fold 01 for illustration. It does not rank folds or select a deployment model.
The original checkpoint and measured source artifacts remain unchanged.

The output contains exactly four files:

```text
artifacts/evaluation-bundles/<bundle_id>/
  manifest.json
  checkpoint.pt
  MODEL_CARD.md
  uv.lock
```

Only inference fields survive checkpoint sanitization. All retained values and tensor bits must match a serialization round trip.
The manifest records separate original and packaged checkpoint digests.
Packaged model-card links point to verified files at the exact recorded source commit.
The packaged lock also comes from that commit.

Before publication, the exporter loads the candidate through the same loader used by callers.
It compares every selected-fold validation target with saved probabilities on MPS.
Acceptance requires absolute probability differences at most 2e-6 and zero class disagreements.
History lengths and reset reasons must match. Model state must remain unchanged, with evaluation mode enabled and no gradients.
No preprocessing fit or optimizer step occurs.

Publication uses an atomic, exclusive directory rename. An existing identifier is never replaced, including an empty directory or symlink.
Verification evidence is staged first and stored outside the four-file bundle.
Normal publication failures remove only that attempt's new evidence link and leave existing outputs untouched.
An abrupt process termination can leave a reserved verification record. Inspect it before choosing a fresh identifier.
Atomic publication supports macOS and Linux. Unsupported filesystem or platform operations fail without an overwrite fallback.

## Runtime identity

A bundle requires Python >=3.11,<3.13 and the exact recorded runtime source revision.
Version-one inference requires the validated MPS backend. It does not silently fall back to CPU or CUDA.

A source checkout must have clean runtime inputs: src/sleep_classifier, schemas, pyproject.toml, uv.lock, and hatch_build.py.
The identity guard compares tracked source bytes with Git objects, including edits hidden by index flags.
Commit model-card edits before export. Other uncommitted documentation does not become part of the packaged card.
A different commit requires a newly identified bundle or the matching prior runtime.
Do not bypass the revision check after pulling code.

Standard wheels embed runtime identity and the three contract schemas.
Wheel loading needs no Git executable, original artifact tree, or motion cache.
The model and fitted preprocessing are self-contained. Python, PyTorch, and other runtime dependencies remain separately installed.
Use the locked gru group from the recorded source revision for the dependency environment.
The wheel cannot create a raw-sensor adapter or establish another backend's numerical parity.

## Export locally

Open [notebook 43](../notebooks/43_evaluation_bundle.ipynb). Restart the kernel and run cells in order.
Each code cell explains what it does and why it is needed. All optional switches default to false.
Use RUN_EXPORT for publication. Use RUN_INFERENCE for one synthetic request after publication.
A cached bundle is verified against the active configuration instead of being overwritten.
Report-only notebook execution skips setup, export, and model inference.

The configuration is [evaluation-bundle.yaml](../configs/evaluation-bundle.yaml).
It names the source audit, source repository, explicit fold, bundle identifier, output collection, and external verification directory.
Use a new identifier when runtime or bundle contents change.
The CLI always refuses an existing identifier or verification path:

```bash
uv run --locked --inexact --group gru python -u -m sleep_classifier.bundle_export \
  --config configs/evaluation-bundle.yaml --export
```

The source audit and prepared development artifacts must be available during export.
Unexpected derived tables are rejected before their contents are hashed, including any unexpected test partition.
The exporter cannot write into measured source directories, source data, motion caches, or runtime code.
The final-test observations remain excluded.

## Load and predict

Load the resulting bundle with the same runtime revision:

```python
from pathlib import Path

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.bundle_validation import strict_json_loads

bundle = EvaluationBundle(Path("artifacts/evaluation-bundles/motion-seed42-fold01-evaluation-v1"))
stream = bundle.predictor()
request = strict_json_loads(Path("examples/inference-bundle/epoch-request.json").read_bytes())
response = stream.predict(request)
stream.reset()
```

The sample request is synthetic. Its output is not a new accuracy result.
A predictor manages one active stream. Repeated predict calls retain bounded history until a reset boundary.
The CLI starts a fresh stream for each invocation and writes one response as JSON on stdout:

```bash
uv run --locked --inexact --group gru python -m sleep_classifier.bundle_inference \
  --bundle artifacts/evaluation-bundles/motion-seed42-fold01-evaluation-v1 \
  --request examples/inference-bundle/epoch-request.json
```

Operational logs use stderr. Call the Python API when history must persist across requests.
The loader verifies role, versions, exact files, hashes, runtime identity, environment-lock metadata, and checkpoint structure before inference.
It rejects symlinks in the bundle path and unexpected entries. Use the physical path when a temporary-directory alias is a symlink.
It restores from verified in-memory bytes. Creating a predictor does not reopen an altered checkpoint file.
It rejects duplicate JSON keys, decoded non-finite numbers, labels in requests, incompatible features, and unsupported backend changes.
It exposes the saved motion imputation mean for an upstream adapter. It does not infer whether a supplied sensor summary has enough support.
Apply the contract's coverage, units, timing, and missing-data rules before prediction.

## Validation and next scope

Run the targeted inference tests without optimizer or training tests:

```bash
uv run --locked --inexact --group gru --group dev pytest -q \
  tests/test_bundle_export.py tests/test_bundle_loader.py tests/test_bundle_runtime.py \
  tests/test_bundle_publication.py tests/test_grouped_inventory.py \
  tests/test_streaming_inference.py tests/test_inference_bundle_contract.py
```

These tests use synthetic weights and inputs without fitting a model.
Real evaluation evidence, when available, is recorded separately from synthetic tests.
Raw-sensor adapter parity, a single deployment-model fitting protocol, final-test evaluation, Docker, FastAPI, and CI remain separate milestones.
No training or final-test observations are required for this packaging milestone.

## Measured preflight and portable-wheel check

The [packaging preflight](experiments/evaluation-bundle-preflight-2026-11-24/README.md) passed all 4,587 fold-01 validation targets.
Maximum difference was 7.45e-7 with zero class disagreements. The original checkpoint remained unchanged.
A separate-process wheel check also passed thirty-three synthetic requests with source/cache/Git access blocked.
These results apply to the recorded preflight revision and MPS backend.
The default notebook export remains fresh; the preflight used a different output identifier and collection.

After exporting at a clean revision, build its matching wheel:

```bash
uv build --wheel --out-dir artifacts/evaluation-runtime-wheel
```

Verify the approved wheel and bundle without training:

```bash
uv run --locked --inexact --group gru python scripts/verify_bundle_wheel.py \
  --bundle artifacts/evaluation-bundles/motion-seed42-fold01-evaluation-v1 \
  --wheel artifacts/evaluation-runtime-wheel/sleep_classifier-0.1.0-py3-none-any.whl \
  --output artifacts/evaluation-bundle-verification/portable-wheel.json
```

Preserve the wheel and bundle together before building a runtime for another revision.
Use approved locally built wheels; the verifier executes the supplied runtime code.
The verifier checks the current matching runtime as its reference and refuses to overwrite its output report.

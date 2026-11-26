# Economical CI and local-service validation

Date: 2026-11-26. Runtime source: `9505ccf7d80af1fa1afaf2f618213171b7fb6857`.
This milestone changes transport, dependencies, and automation. It does not optimize or evaluate research models.
No dataset observations, real-data training, or GPU operations were used. The final test remains reserved.

## Synthetic contracts

207 focused tests pass. Ruff and whitespace checks pass.
The selected tests cover bundle guards, protocol validation, raw-session behavior, HTTP bounds, ownership, expiry, cancellation, and wheel contents.
The HTTP composition test runs the complete 76-record synthetic example through the real runner and session service.
All five HTTP response records match the direct CPU runner exactly, including both predictions, the coverage skip, and completion integrity.
Random-weight fixture models use a test-only CPU device resolver. Production loading retains the strict runtime and device checks.
The tests do not update parameters or read saved research checkpoints.

The one-job workflow selects these tests explicitly. It does not run training tests or notebook execution.
It asserts that the clean checkout contains no `data/` or `artifacts/` directories before tests.
CPU dependency dry runs select no CUDA packages. The separate, mutually exclusive GPU extra retains the original PyPI source for future experiments.
That extra was resolved only; no GPU packages were installed or used in this work.

## Wheel verification

The [package check](package-check.json) verifies 84 wheel entries and all five packaged schemas.
It checks the embedded source revision, distribution metadata, every RECORD hash and size, and permitted package paths.
No dataset, checkpoint, notebook, or local artifact payload is present.
The installed image runtime reports the same wheel source revision.

## Container smoke

The [container smoke report](container-smoke.json) passes on `linux/arm64` through OrbStack.
The image uses digest-pinned official Python and uv images, locked CPU dependencies, and the verified project wheel.
It runs as UID 65532 with a read-only root and a temporary `/tmp` mount.
Installed PyTorch is `2.15.0+cpu`; its CUDA build is absent.
The test container was removed after verification.

| Check | Observed result |
| --- | --- |
| GET /healthz | 200; alive |
| GET /readyz | 503; no bundle |
| POST /v1/events | 503; not ready |
| Model restored | No |
| Dataset mounted or read | No |
| Training or GPU use | No |

This is a no-bundle service smoke check. It does not certify Linux inference for the MPS evaluation bundle.
The HTTP prediction path is covered separately by synthetic CPU fixture tests.
The existing MPS bundle remains fixed. A validated Linux CPU bundle needs its own parity evidence and contract update.
Docker build and health smoke remain manual in CI to keep ordinary PR runs economical.

See the [service and CI guide](../../local-service-and-ci.md) for local commands, route semantics, and limitations.

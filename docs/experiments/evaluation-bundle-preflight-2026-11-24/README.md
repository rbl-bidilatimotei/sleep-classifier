# Evaluation-bundle packaging preflight

The fold-01 research checkpoint passed a complete packaging and isolated-wheel preflight on 2026-11-24.
No training occurred. No final-test observations were read.
The checked runtime revision is a298c3df3d340d0fa3f17160e251eb6f29b248af.

## Complete selected-fold acceptance

| Check | Observed result |
| --- | ---: |
| Eligible validation epochs | 4,587 |
| Maximum probability difference from saved predictions | 7.45e-7 |
| Permitted maximum | 2e-6 |
| Predicted-class disagreements | 0 |
| History starts | 29 |
| Full-context epochs | 4,101 |
| History lengths and reset reasons | All matched |
| Model weights and buffers | Unchanged |
| Gradients created | None |

The original checkpoint digest is 80b8a1f3c42b3ea1c58a82af9234fdae0790d0f4b86c8814ceb13269ac9d535c.
The sanitized checkpoint digest is f85ed360769ffb1f6a11919d6972f80456a18112b3a30ea9d812b02a652f55c4.
The original source bytes remained unchanged. Retained fitted values and tensor bits passed exact round-trip checks.
The packaged checkpoint contains ten allowlisted inference fields. It excludes training_config, optimizer metadata, and local training paths.

The bundle contains exactly manifest.json, checkpoint.pt, MODEL_CARD.md, and uv.lock.
It remains an evaluation-fold artifact. This is packaging parity, not a new accuracy result or a deployment release.
See [export-verification.json](export-verification.json) and the [manifest](manifest.json).

## Isolated wheel acceptance

A standard wheel contains its source identity and all three contract schemas.
The bundle was copied to a separate temporary directory and loaded from the extracted wheel in a new Python process.
The probe blocked reads from the original artifacts, data, and source-module directories.
It also blocked the Git executable. Dependency imports remained available from the installed environment.

All thirty-three synthetic requests matched the reference probabilities exactly, including context eviction.
There were zero class disagreements. Weights remained unchanged and no gradients were created.
The wheel digest is 270697f07b1f2f072a2e3a0f7a378e8afaacf93de393721071f4495e6ac5e72b.
See [wheel-verification.json](wheel-verification.json).
The reusable verifier is [verify_bundle_wheel.py](../../../scripts/verify_bundle_wheel.py).

Only metadata is tracked here. Bundle weights, wheel files, HTML, and local verification snapshots remain under artifacts.
The preflight bundle and its wheel are pinned to the checked runtime revision.
They must use that runtime even after later documentation commits change the active source revision.

## Operator workflow remains fresh

The preflight used a separate identifier and output collection.
The default notebook-43 identifier has not been exported. Timo can run it from the final PR revision without a pre-existing destination.
Keep all source notebook switches false for inspection. Enable RUN_EXPORT for the default publication and RUN_INFERENCE for a synthetic request.
A wheel built for a different revision is deliberately rejected. Build the matching wheel after freezing the desired runtime commit.

Synthetic inference, validation, publication, identity, inventory, and notebook command checks passed without model optimization.
The source notebook also passed fresh-kernel report-only execution with no export or forward pass.
This preflight was measured on macOS/MPS. Linux publication uses its native no-replace API but has not been exercised on a Linux host.

Verification: 181 targeted inference, contract, publication, source-identity, inventory, and notebook-command tests passed.
Training and optimizer tests were not run. Ruff and whitespace checks passed.
Notebook 43 passed fresh-kernel report-only execution; its default export and inference remained disabled.

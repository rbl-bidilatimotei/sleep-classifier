# Single-session raw-event inference preflight

Date: 2026-11-26. Fixed model: seed-42 fold-01 MPS evaluation checkpoint.
Runtime revision: `5b4bd95f8906f5f4925313ba72e127542d319254`.
This is inference integration evidence. No training or final-test observations were used.

## Results

- [Bundle publication](bundle-verification.json) passes all 4,587 existing validation epochs, with zero class disagreements.
- Maximum probability difference is 7.450580596923828e-07, below the 2e-6 acceptance threshold.
- Saved transforms and weights remain fixed. Histories and reset reasons match.
- The [synthetic JSONL session](synthetic-responses.jsonl) emits two predictions, one coverage skip, and a final summary.
- Complete epochs are [0,30), [30,60), and [60,90). The third epoch lacks sensor coverage.
- Finish at 95 seconds discards two observations in the partial tail. Peak pending observations are 33.
- The final check confirms unchanged weights, evaluation mode, and no gradients.
- [Isolated wheel verification](wheel-verification.json) reproduces all five output records exactly.
- The wheel process cannot read the original artifact, data, source-module, or schema trees. Git access is blocked.
- The wheel loads its packaged protocol schemas and copied bundle. This is a portability check within the current environment.

Synthetic predictions do not establish model accuracy. Full raw-event parity comes from notebook 46.
The new source notebook's default bundle and demo-output destinations remain unused.
The retained preflight bundle requires its matching runtime wheel after later commits.

## Reproduce the portability check

Use the retained trusted bundle, matching wheel, synthetic events, and completed source responses:

```bash
uv run --locked --group gru python scripts/verify_raw_session_wheel.py \
  --bundle artifacts/raw-session-preflight/bundles/motion-seed42-fold01-raw-session-preflight-v1 \
  --wheel artifacts/raw-session-preflight/runtime-wheel/sleep_classifier-0.1.0-py3-none-any.whl \
  --events examples/raw-session/session.jsonl \
  --expected artifacts/raw-session-preflight/source-responses.jsonl
```

The script executes trusted wheel code. Do not substitute an untrusted wheel or checkpoint.
For a new runtime revision, export a separate bundle and build a matching wheel before comparing responses.

## Validation

213 focused non-training tests pass. Ruff and whitespace checks pass.
Notebook 47 passes fresh-kernel report-only execution with setup, export, and inference skipped.
Notebook 46 also passes fresh-kernel cached-report execution.
The source notebooks remain output-free. Executed reports are local HTML artifacts.

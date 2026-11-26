# Single-session raw-event inference

The runner combines the audited raw-event adapter with a verified evaluation bundle.
It preserves the frozen model, saved transforms, and sensor-only coverage policy.
This is an evaluation interface. It does not select or certify a deployment model.

```text
JSONL events → versioned validation → bounded adapter
            → saved imputation → frozen predictor → JSONL responses
```

## Run the example

Use [notebook 47](../notebooks/47_raw_session_inference.ipynb) for the explained workflow.
Export a fresh bundle after checking out a clean, committed revision:

```bash
uv sync --locked --group gru --group notebook
uv run --locked --group gru python -u -m sleep_classifier.bundle_export \
  --config configs/raw-session-bundle.yaml --export
uv run --locked --group gru python -u -m sleep_classifier.raw_session_cli \
  --config configs/raw-session.yaml --input examples/raw-session/session.jsonl
```

The default bundle directory must be fresh. Never replace an existing bundle.
The loader requires the export's exact runtime revision and validated device.
For another revision, export to a new bundle identifier and update the session configuration.
A previously exported bundle remains usable with its matching runtime wheel.
The selected fold-01 checkpoint illustrates integration. It is not a deployment selection.

The example uses synthetic observations. Expect two predictions, one skipped epoch, and a final summary.
These predictions have no clinical or accuracy interpretation.
The command performs inference only. Export checks existing development predictions without optimizing model weights.

## Input contract

[Event schema](../schemas/raw-session/v1/event.schema.json) defines exact version `1.0.0` fields.
Each line contains one UTF-8 JSON object. Labels, extra fields, duplicate keys, and nonfinite numbers are invalid.

1. Send `start` with session and participant identifiers.
2. Send ordered `heart_rate` and `motion` events within each sensor stream.
3. Advance each stream's `watermark` explicitly.
4. Send `finish` with the exclusive session end.

Version one requires session origin zero. Timestamps use seconds in the audited session clock.
It does not infer clock alignment or reorder events.
A watermark promises no future event below its value. Events at the watermark remain valid.
Both watermarks must reach an epoch end before that epoch can produce output.
Late observations and conflicting duplicates fail. Exact duplicates are counted.
Finish emits complete epochs and counts discarded observations in the partial tail.
EOF cannot substitute for `finish`.

## Output contract

[Output schema](../schemas/raw-session/v1/output.schema.json) defines lifecycle, prediction, and skip records.
Standard output contains only JSONL records. Operational logs use standard error.
Each response includes the bundle, participant, and session identifiers.

Predictions include the epoch time, completion time, watermark, probabilities, class, history length, and reset reason.
A skipped epoch includes finite sensor coverage measurements and no prediction.
The next eligible epoch uses the predictor's normal sensor-segment reset.
The runner does not feed labels or predictions back as observed features.
The final record reports counts, queue peaks, duplicate handling, discarded tails, and frozen-model integrity.
Consumers must require `session_finished`; partial output does not establish successful completion.

## Resource and failure rules

One runner owns one session. Use a new runner or process for another session.
Concurrent callers, transport retries, and network authentication are outside this interface.

The default limits are 8,192 pending events, 128 epoch outputs per progress event, and 4,096 bytes per input line.
The epoch cap excludes the final summary. Advance watermarks in smaller steps across long empty intervals.
The reader checks line length before decoding. The adapter raises before accepting an overflowing observation.
These are application limits. They do not constitute a total process-memory or wall-time guarantee.

Input validation and pre-progress limit failures preserve session state.
The CLI stops at the first invalid event and returns a nonzero exit code.
An inference failure after epoch progress makes the runner unusable.
Restart with a fresh session; do not assume transactional rollback or replay deduplication.
The final integrity check requires unchanged weights, evaluation mode, and no gradients.

## Verification

Synthetic tests cover protocol fields, input bounds, lifecycle, gaps, resets, imputation, and model failures.
They include random-initialized model inference with a test-only CPU device resolver.
Production loading retains the exact bundle runtime and device checks.
The full raw-event development audit remains the evidence for adapter parity.
The runner adds integration checks; it does not report new model scores.
Keep the final test untouched. No training is part of this milestone.

The [measured integration preflight](experiments/raw-session-preflight-2026-11-26/README.md) passes selected-fold export and isolated-wheel synthetic inference.

The [notebook-47 operator run](experiments/raw-session-2026-11-26/README.md) confirms the default export and synthetic demo.
Its retained matching wheel supports that bundle after later source commits.

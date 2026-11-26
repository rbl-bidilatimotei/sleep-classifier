# Bounded raw-event adapter

Status: full development audit passed on 2026-11-26.
Read the [measured findings](experiments/raw-event-adapter-2026-11-26/README.md) alongside [notebook 46](../notebooks/46_raw_event_adapter.ipynb).
No model was trained, and no final-test observations were opened.

## Interface and scope

`RawEventAdapter` accepts timestamped heart-rate and x/y/z observations for one participant session.
It emits one completed thirty-second feature row at a time. Outputs contain no stage or target field.
The adapter is a synchronous single-caller state machine. Serialize access externally when multiple producers supply one session.
The shared feature builders preserve the six base summaries, coverage rules, and fixed-lag motion definition.
Sensor coverage alone determines eligibility and history segments.
The adapter does not fit a scaler, replace missing motion, or restore a model.
The consumer applies the frozen model's saved preprocessing after selecting eligible outputs.

The adapter keeps at most `max_buffered_events` pending unique observations across both sensors.
It also keeps constant-sized progress and last-observation records. The default cap is 8,192 events.
Per-epoch calculations allocate temporary data proportional to that bounded input, not the duration of the session.
The caller owns emitted outputs. Retaining all outputs is outside the adapter's memory bound.
The offline audit also loads full input tables, sorts copies, and retains comparisons. Its process memory is not bounded by 8,192 events.

## Time and completion contract

Declare a common session-relative clock and use the same observation units as the [prepared-input contract](inference-bundle-contract.md).
The audit uses origin zero. Clock synchronization and device calibration remain caller responsibilities.
Epoch boundaries are anchored as `origin + 30 * epoch_index`. This avoids cumulative boundary drift from repeated addition.
Intervals are half-open: `[start, end)`. An observation exactly at end belongs to the next interval.

Each stream has its own explicit watermark. A watermark `w` promises that no future event from that stream has time below `w`.
Events exactly at `w` remain admissible. Watermarks cannot move backward.
Adding an observation does not advance either watermark.
`pop_completed()` emits one epoch only when both watermarks reach its end.
It returns None when the next epoch is not ready, without collecting completed outputs internally.

Each emitted row contains:

- `time`: the epoch start
- `available_at`: its anchored end, the earliest event-time completion boundary
- `watermark_at_emission`: the lower sensor watermark when the row was emitted
- Raw summaries, coverage measurements, fixed-lag support diagnostics, eligibility, and sensor segment ID

`available_at` is not a measured wall-clock latency. Chunked callers can declare progress after that boundary.
The audit verifies `watermark_at_emission >= available_at` for every emitted epoch.
Missing sensor data can close when the caller declares progress. Coverage then rejects unsupported epochs and breaks history.
A missing sleep annotation never drives that decision.

## Input and failure rules

Timestamps must be nondecreasing within each sensor stream. The two streams can arrive independently.
An exact repeated last observation is ignored and counted while its timestamp remains admissible.
A conflicting reading at that timestamp raises ValueError.
A repeated observation below a watermark or closed-epoch boundary is late, even if its value matches.

| Condition | Result |
| --- | --- |
| Timestamp before its stream watermark, previous timestamp, or session origin | LateEventError |
| Conflicting equal-time reading | ValueError |
| Nonfinite or Boolean time/value | ValueError |
| Backward watermark or unknown stream | ValueError |
| Full pending-event buffer | BufferError |
| Input after finish | RuntimeError |

Rejected operations leave accepted data and progress unchanged. The adapter never silently drops events to free capacity.
The caller can drain completed epochs and retry an overflowed event when its timestamp remains admissible.
Choose enough capacity for the highest expected epoch density and permitted input skew.
Do not advance a watermark unless upstream can make that completion promise.
Version one rejects reordered and late events. It has no grace period, correction stream, or retrospective prediction update.

## Session end

Call `finish(session_end_seconds)` once, then drain `pop_completed()` until it returns None.
The end cannot precede accepted observations, declared watermarks, or already emitted epochs.
Finish closes input and advances both watermarks to the declared end.
Complete thirty-second intervals are emitted. The unfinished tail is discarded and its pending-event count is recorded.
An observation exactly at the declared end is outside the completed session and remains part of that discard count.
Subsequent `pop_completed()` calls return None. Further input, watermark updates, or finish calls raise RuntimeError.
Create a new adapter for another session or participant.
Reset the associated model predictor for every new session, including repeated sessions from the same participant.

The recorded audit declares the end as the earlier final sensor timestamp, preserving notebook 44's conservative tail policy.
It counts observations filtered before the origin and after that endpoint separately from accepted events.
An explicit device session-end policy could differ, but changing that evaluation population requires separate evidence.

## Example

```python
from sleep_classifier.raw_event_adapter import RawEventAdapter

adapter = RawEventAdapter("session-participant", coverage_rule, max_buffered_events=8192)
adapter.add_heart_rate(0.0, 60.0)
adapter.add_motion(0.0, 0.0, 0.0, 1.0)
adapter.advance_watermark("heart_rate", 30.0)
adapter.advance_watermark("motion", 30.0)
row = adapter.pop_completed()
# This sparse example closes an epoch but fails the working coverage rule
adapter.finish(30.0)
assert adapter.pop_completed() is None
```

The caller supplies `coverage_rule` from its validated preparation configuration.
Pass only eligible outputs to inference. Apply the saved training motion mean when fixed-lag motion is unavailable.
The existing streaming predictor owns its separate thirty-epoch model history.

## Development audit

[raw-event-adapter.yaml](../configs/raw-event-adapter.yaml) declares the verified replay source, output path, chunk sizes, and queue cap.
The default chunk sizes are 128 and 1,024 merged events.
The audit verifies the raw participant allowlist against notebook 45 before opening any sensor file.
Only development HR and motion files are opened, with checksum verification. Raw labels and final-test observations are excluded.

For each participant, the audit sorts copies of recorded sensor streams and merges them chronologically.
At each chunk boundary, it declares both watermarks at the last merged timestamp and drains completed epochs.
This is valid because every unread event has time at least that timestamp. It does not simulate real network delays.

Each chunking must preserve all full-session features, support diagnostics, eligibility, segments, and completion rules against the batch builders.
The seven raw channels and coverage/reset fields also match the saved notebook-44 projection independently.
Historical and freshly calculated numerical differences are recorded separately at absolute tolerance 1e-8.
The configured parity profile uses origin zero. Other clock origins retain the shared builders' floating-point coverage comparisons.
At fractional origins, rounding near a coverage cutoff can change eligibility. A different clock profile needs its own parity audit.

Accepted event conservation must hold:

```text
input_events = emitted_unique_sensor_samples + duplicate_events + discarded_tail_events
```

The first chunk size supplies the canonical model input. The audit restores each original held-out checkpoint with saved fitted transforms.
Every sensor-eligible canonical prediction must reproduce notebook 45 within absolute tolerance 2e-6, with zero class disagreements.
History lengths and reset reasons must match. Weights, buffers, and fitted metadata must remain unchanged, with evaluation mode and no gradients.
Other chunkings are checked for identical feature/state inputs; the model is not redundantly replayed for each chunk size.
The audit computes parity evidence, not new accuracy results.

## Run locally

```bash
uv sync --locked --inexact --group gru --group notebook --group dev
uv run --locked --inexact --group gru python -m sleep_classifier.raw_event_audit \
  --config configs/raw-event-adapter.yaml
```

Restart notebook 46's kernel. Set RUN_AUDIT true to run the complete development audit.
The notebook validates readiness before launching:

```bash
uv run --locked --inexact --group gru --group notebook python -u -m sleep_classifier.raw_event_audit \
  --config configs/raw-event-adapter.yaml --run
```

The full raw replay can take several minutes. Logs identify each participant, chunk size, epoch count, and peak queue size.
The output is `artifacts/raw-event-adapter-audit`. It now contains the verified immutable full-cohort snapshot.
Keep RUN_AUDIT false for cached inspection. Use a fresh configuration and output path for an intentional repeat.
Publication uses a staged, exclusive directory rename. Existing outputs and source artifacts cannot be overwritten.
Cached inspection verifies hashes, source identities, complete chunk/fold coverage, buffer limits, event conservation, and fixed-model parity.
It does not rerun raw processing or inference. Cached code hashes describe the recorded implementation; inspection does not validate later code changes.

Generate report-only HTML with all optional operations disabled:

```bash
uv run --locked --inexact --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/46_raw_event_adapter.ipynb
```

The report appears at `artifacts/notebooks/46_raw_event_adapter.executed.html`.
Setup, audit, and recursive export cells carry skip-execution tags. Source notebook outputs stay empty.
The [bounded preflight](experiments/raw-event-adapter-preflight-2026-11-25/README.md) first verified thirty epochs.
The subsequent [full audit](experiments/raw-event-adapter-2026-11-26/README.md) verifies 24,738 epochs and 23,674 frozen predictions.
Next, integrate the adapter and verified bundle behind a single-session JSONL runner before a live service or model sweep.

Validation passes 515 targeted non-training tests, Ruff, and whitespace checks.
Notebook 46 passes fresh-kernel report-only execution. Synthetic completed-report tests exercise its tables and memory-bound chart.

# Preparation decisions

Date: 2026-10-29. Status: proposals for discussion, not implemented preparation rules.

## Order of work

1. Verify label boundary semantics against the original export and processing code.
2. Check file integrity and participant eligibility across the cohort.
3. Select the final test participants using predefined eligibility rules and a recorded seed.
4. Keep participant 1066528 in the development set.
5. Measure epoch coverage on development participants.
6. Select preparation rules without inspecting test performance.
7. Implement the agreed rules with unit tests.

A cohort audit can check file structure and integrity before the split.
Use development data for detailed exploration and threshold selection.
Apply frozen preparation rules to the test set later.

## Duplicate readings

The first participant has 11,078 exact duplicate heart-rate rows and no conflicting duplicate timestamps.
This observation does not establish the pattern for other participants.

Proposal: remove exact duplicate rows from a derived table, then sort by timestamp.
Preserve the original files. Report duplicate counts before and after preparation.
Flag conflicting readings for investigation. Do not average them without an agreed rule.

## Sensor coverage

Coverage needs a definition for each sensor.
Sample counts alone cannot detect a long gap hidden by clustered measurements.

For motion, report unique sample counts, occupied time bins, and the largest internal gap.
For heart rate, report unique readings, time since the latest reading, and gap durations.
Include gaps at the epoch boundaries. Identify epochs with no observations.
Use a documented sampling assumption before converting counts into a coverage percentage.

The current sample has a motion gap of approximately 21 minutes during the labeled night.
A short carry-forward rule must not fill this gap.

We have not selected a numerical threshold.
Compare candidate thresholds using retained development epochs and sensor availability.
Record why the selected threshold fits the intended prediction contract.

Keep feature validity separate from epoch eligibility.
A missing motion feature does not automatically require removing the whole epoch.
Define missing-feature handling and recurrent-state behavior explicitly.
For evaluation, report prediction coverage alongside performance on retained epochs.
This prevents selective rejection from concealing difficult intervals.

## Step history

Proposal: establish the first sensor baseline with heart rate and motion.
Then compare a model with historical steps and explicit coverage indicators.
Use the same participants, evaluation epochs, and evaluation procedure for both models.

The initial baseline is an ablation, not the complete challenge deliverable.
The requested final interface must account for steps, including unavailable step history.
Missing step records do not mean zero activity.
Step bins require a known endpoint before their counts become available to causal inference.

## Label boundaries

Thirty-second spacing does not prove whether a timestamp marks an epoch start, center, or end.
Confirm that convention from the original export or processing code.
Record the source revision and relevant code location.
If the source remains ambiguous, document the assumption before implementation.

Our proposed interval remains [t, t + 30 seconds), predicted at its end.
This interval is provisional until the timestamp convention is confirmed.
Tests must cover observations exactly at both boundaries.

## Required preparation tests

- Exact duplicates and conflicting readings receive different treatment.
- Epoch assignment respects both interval boundaries.
- Empty epochs and long gaps remain visible.
- Missing steps remain distinct from zero steps.
- Future observations cannot change earlier causal features.
- Participant splits remain disjoint.
- Learned transforms use training participants only.

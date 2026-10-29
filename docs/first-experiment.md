# First experiment: questions before training

Status: protocol proposed, 2026-10-28. Session 1 now implements the environment,
download/audit commands, and initial notebook; see [session findings](session-01.md).

See [preparation decisions](preparation-decisions.md) for the next milestone and unresolved rules.

## Prediction contract

Propose four classes: wake = 0, light = N1 + N2, deep = N3, REM = 5.
NREM is a parent category, not an additional class alongside light and deep.
The published work used two- and three-class tasks; its accuracy is not a
four-class performance promise. See [literature notes](literature-notes.md).

Propose predicting the stage of a completed interval [t, t + 30 seconds) at its
end. Features may use that interval and earlier observations, never later ones.
This is delayed classification, not forecasting the next interval. Verify label
timestamp conventions before constructing training examples.

The eventual streaming interface needs timestamps, missingness information, and
history or recurrent state in addition to heart rate, x/y/z motion, and steps.
Return class probabilities and a label. Reset recurrent state between sessions;
define the initial warm-up behavior. Ground-truth previous sleep stages will not
be available during deployment and must not become model inputs during training.

## First milestone: understand and validate the data

1. Establish a small uv environment and reproducible download/audit scripts.
2. Download one complete participant, then inventory the cohort before splitting.
3. Use one notebook to inspect sensor timelines, label frequencies, missingness,
   duplicate timestamps, gaps, and step-history coverage.
4. Agree on epoch alignment and missing-data rules with concrete examples.

The initial checksum-verified sample is participant 1066528. The raw-file audit
in [sample-audit.json](sample-audit.json) found 952 labels spaced 30 seconds apart,
two non-increasing heart-rate timestamps, and step records ending at -127607
seconds relative to PSG start. These observations apply only to this participant.
Investigate duplicate records and timestamp resets before choosing a sorting or
deduplication policy. Missing step coverage must not silently become zero steps.
That audit is the historical discovery snapshot; the current reproducible audit
command and complete-participant findings are documented in session-01.md.

## Evaluation contract

Split by participant before fitting transforms or creating overlapping windows.
Propose 28 development participants and 3 untouched test participants (approximately
90/10); record the participant IDs and seed. Participant 1066528 has already been
inspected and must remain in development rather than the untouched test set. An exact 90/10 participant split is
impossible with 31 participants. Preserve the test set until model choices are fixed.

Within the development set, reserve separate participants for validation and early
stopping. Fit scaling and imputation statistics on training participants only.
With only three test participants, final results will be sensitive to who is held out.

For the later robustness milestone, use five grouped folds within development data
with fresh models and preprocessing per fold. This evaluates variation across held-out
people; multiple training seeds separately assess initialization variability. Five-fold
cross-validation and six repeated 90/10 holdouts are different experiments. If literal
challenge compliance requires an initial run plus five reruns, record six predefined
grouped holdouts and their overlap, and do not call that five-fold cross-validation.

Report overall accuracy, macro-F1, per-class precision/recall/F1 and support, a
confusion matrix, and participant-level results. Define the REM bonus explicitly:
propose REM recall as the target and always show precision/F1 beside it. One-vs-rest
accuracy alone can be misleading when most epochs are not REM.

## Second milestone: a small end-to-end baseline

Start with a majority-class reference and a simple non-sequential model on the same
features. Then use a one-layer, unidirectional GRU with roughly 32–64 hidden units
and a linear four-class head. This is a proposed starting size, not a tuned choice.
Use modest sequences of aggregated epochs rather than long raw accelerometer streams.
Choose sequence length after the audit; evaluate cold-start and carried-state behavior.

Candidate epoch features: heart-rate summaries, observation count and time since last
reading; motion magnitude/variability; historical step totals paired with coverage.
Retain raw axes for inspection. Validate units and sampling before choosing features.
Never normalize from the complete future night, interpolate using future readings,
or apply a centered smoothing window in the causal pipeline.

Plain PyTorch first keeps the training loop visible. Reconsider Lightning if repeated
experiments make its lifecycle useful. Design for CPU/Mac MPS and Colab CUDA; check
actual compatibility and memory use when dependencies and batch size are selected.

## Later milestones, one at a time

- Validate streaming/offline equivalence and checkpoint reproducibility.
- Compare feature and context-length ablations before increasing model complexity.
- Run grouped robustness evaluation and record per-run splits, seeds, configurations,
  data hashes, package versions, metrics, and checkpoint identifiers.
- Compare a small causal Transformer on identical splits and features; consider Mamba
  after checking runtime support and whether data volume justifies the complexity.
- Add Docker, runnable documentation, and a final experiment report.

No accuracy target has been achieved or estimated from this preliminary inspection.

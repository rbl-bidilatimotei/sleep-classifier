# Literature notes

Checked 2026-10-28. These findings inform the first experiment; they are not results from this project.

## Original study

Walch et al. report approximately **72% accuracy for wake/NREM/REM**, and 90% for sleep/wake. The three-class result does not establish a four-class light/deep/REM/wake target. Their features include activity counts, local heart-rate variation, and a circadian-clock proxy. Each example corresponds to a 30-second PSG epoch; local features use a ten-minute window around it. Heart-rate preprocessing uses interpolation, filtering, and normalization based on the sleep recording. The authors distinguish their heart-rate variation feature from ECG-based heart-rate variability. [Paper, available through PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/)

**Implication:** reproducing those offline features unchanged would not meet strict real-time inference. Measurements after the prediction time and whole-night normalization are unavailable during a live night. This is an engineering inference from the described preprocessing, not a claim that the original offline experiment is invalid.

## What the source code confirms

- The repository documents subject-held-out evaluation, including leave-one-subject-out and repeated random subject subsets. This supports keeping people, rather than individual rows, separated in our evaluation. Its current default clock-like feature is a cosine wave; the optional circadian simulation requires MATLAB. [Repository README](https://github.com/ojwalch/sleep_classifiers)
- `HeartRateFeatureService.get_window` extends from 285 seconds before an epoch to 285 seconds after its end. The same file interpolates across the recording and divides by a recording-level 90th percentile. These are concrete places where a streaming implementation must differ. [Heart-rate feature implementation](https://github.com/ojwalch/sleep_classifiers/blob/main/source/preprocessing/heart_rate/heart_rate_feature_service.py)
- `FeatureBuilder` separates wearable features, labels, and time features. We can preserve that conceptual separation without copying the full implementation. [Feature builder](https://github.com/ojwalch/sleep_classifiers/blob/main/source/preprocessing/feature_builder.py)

The code links point to the moving `main` branch, not a pinned reproduction commit. Detailed numerical reproduction and a complete source audit remain future work.

## Reading-list triage

| Source | What it actually supplies | Use here |
| --- | --- | --- |
| [MNE sleep tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html) | EEG relative frequency-band power and a random forest; its `FunctionTransformer` is a scikit-learn preprocessing adapter, not an attention architecture. | Useful for epochs, subject separation, confusion matrices. Its EEG inputs and scores do not transfer directly to watch sensors. |
| [Transformers in Time-series Analysis: A Tutorial](https://arxiv.org/abs/2205.01138) | General architecture and time-series overview. | Background reading after a recurrent baseline; not evidence that a Transformer will beat it on this dataset. |
| [SleepTransformer](https://arxiv.org/abs/2105.11043) | Sequence-to-sequence sleep staging from EEG, with attention visualizations and uncertainty estimates. | Architecture inspiration, not a directly comparable wrist-sensor benchmark. Audit temporal attention and preprocessing before adapting for causal inference. |

## Proposed experimental consequences

These are recommendations, not findings established by the papers:

1. Start with causal per-epoch features and a small unidirectional GRU or LSTM. Predict after each completed 30-second epoch using only observations available by then.
2. Learn temporal context from sensor history. Do not feed true previous sleep labels at inference, since those labels will not exist in deployment.
3. Fit scalers and any learned imputation on training subjects only. Check missingness and timing before selecting a feature representation.
4. Compare against simple baselines using identical subjects, labels, and information availability. Preserve the final test subjects while choosing models on training/validation data.
5. Report overall accuracy alongside macro-F1 and per-class precision/recall/F1. Define the challenge's ambiguous “REM accuracy” before judging its 70% bonus threshold.

The remaining linked Nature, medRxiv, and OxWearables sources have not yet been audited in these notes. No accuracy or hardware feasibility claim for a future implementation has been measured.

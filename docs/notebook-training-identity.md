# Diagnose a blocked notebook training command

## Observed incident

The reported command selected configs/hr-availability-control-grouped.yaml, the notebook 31 availability-only experiment.
Its five result directories and summary already existed from the completed run.
The existing-results guard rejected the rerun before optimization. No result deletion or replacement was needed.

The saved notebook 32 selected configs/hr-deviation-only-grouped.yaml correctly.
Its prepared output directory had no fold results or grouped summary.
The local notebook 31 also contained an enabled training switch and executed setup outputs.
These facts support a different notebook or stale cell being executed, rather than a copied wrong config in notebook 32.
The exact editor action cannot be determined from the saved files alone.

| Notebook | Grouped config | Masked channel | Output directory |
| --- | --- | --- | --- |
| 31 | hr-availability-control-grouped.yaml | heart_rate_delta_past | artifacts/hr-availability-control-grouped-validation |
| 32 | hr-deviation-only-grouped.yaml | heart_history_available | artifacts/hr-deviation-only-grouped-validation |

The failed-attempt notebook was preserved in /private/tmp/31_hr_availability_control.failed-attempt.ipynb before source cleanup.
Source outputs are cleared and training defaults to disabled.

## Safeguards added

Training and preparation commands now use experiment.config_path from the validated setup object.
Evidence inspection uses that object's output_dir. CLI targets are no longer copied into those cells as separate literals.
The notebook logs the resolved config, output directory, and masked channels before training.
A public validate_training_start method checks immutable prepared inputs and existing outputs before the CLI launches.
The CLI run method uses the same guard, retaining protection for standalone commands.

Blocked-run errors identify the exact config, output directory, and existing fold/summary paths.
They direct the operator to verify the notebook and config while preserving measured results.
A separate output is appropriate only for an intentional rerun with a separate experiment config.
No automatic overwrite, resume, or deletion was added.

The regression tests execute the actual training cells with a recording shell and a different active experiment.
They assert that the launched command targets the validated config and calls the readiness guard.
Synthetic grouped tests assert that blocked outputs remain unchanged and no fold execution starts.
The five targeted tests failed before the fix and pass afterward, without participant-data optimization.
The corresponding prevention rule is recorded in AGENTS.md.

## Start the intended notebook

1. Open notebooks/32_hr_deviation_only.ipynb from the current checkout
2. Reload the file and restart its kernel
3. Run the setup and validation cells from the top
4. Confirm the config ends with hr-deviation-only-grouped.yaml
5. Confirm the output ends with hr-deviation-only-grouped-validation
6. Confirm masked_channels contains only heart_history_available
7. Keep RUN_PREPARATION false when the prepared manifest exists
8. Set RUN_TRAINING true and run the training cell

Leave notebook 31's completed outputs intact. Restarting a kernel does not resume a partially completed grouped run.
No participant-data training ran during this fix.

Validation: 924 tests, Ruff, whitespace checks, and both fresh-kernel notebook exports pass.
The completed experiment is blocked; the intended deviation-only experiment is ready.
All recorded measured metric and summary hashes remain unchanged.

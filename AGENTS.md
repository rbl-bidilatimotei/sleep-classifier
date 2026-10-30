# Collaboration

Work one agreed milestone at a time and
explain decisions and findings; this project is a learning exercise.
Read docs/first-experiment.md before changing the experiment protocol.

# Implementation

Keep reusable operations in src/sleep_classifier; notebooks import these modules
and explain experiments. Expose reproducible operations through python -m entry
points. YAML contains only settings used by implemented code. Use uv and maintain
uv.lock. Use type hints and concise Google-style docstrings, including parameter
types where parameters need explanation; comments explain non-obvious decisions.

Keep raw data immutable and verify checksums. Preserve raw ordering in audits;
make any cleaning policy explicit. Fit transforms on training participants only,
keep participant splits disjoint, and make inference features causal.

For changed behavior, add unit tests for meaningful contracts and failure modes.
Run uv run pytest and uv run ruff check . before delivery. Execute changed notebooks
from a fresh kernel and inspect their outputs. Update runnable documentation and
session findings with code changes; distinguish observed results from proposals.

# Style and workflow

Use ASD-STE100 principles when possible. Keep procedural sentences below 21 words.
Keep descriptive sentences below 26 words. Use active voice and exact technical terms.
Give one instruction per sentence. Keep articles. Avoid idioms and filler.
Use the existing Google-style docstrings. Omit final periods from line comments.
Explain deviations when they improve correctness or readability.

Create working branches for changes. Submit changes through pull requests to main.
Delete the remote and local PR branches after a successful merge.
Preserve unrelated user edits. Keep each pull request within the agreed milestone.
The primary agent owns the design and review. Delegate bounded implementation tasks
to smaller models when useful. Review and test their changes before delivery.

# Code structure and execution

Use cohesive classes for operations with state or configuration. Keep utility functions small.
Split large methods into named steps. Avoid nested functions unless the design requires them.
Use specific names that describe the operation and data. Separate logical blocks with blank lines.
Use class and method docstrings when useful. Omit module-level docstrings and __init__.py docstrings.

Prepare training notebooks and commands for Timo to execute. Run real-data training only after explicit authorization.
Synthetic training tests for tiny models and EDA execution are permitted.
Keep source .ipynb files in Git. Save executed notebook reports as HTML only.
Run training only through an explicit opt-in command; report-only execution must not start training.

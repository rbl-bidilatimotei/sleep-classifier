# Collaboration

Start user-facing responses with Timo. Work one agreed milestone at a time and
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

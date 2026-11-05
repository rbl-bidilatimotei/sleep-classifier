# First OPAL pilot: failed decision with Adam fallback

Timo completed the pilot on 2026-11-05. All three prediction CSV files have the same SHA-256 hash.
All epoch histories are identical. Macro-F1 is 0.4760, accuracy 56.69%, and the selected epoch is one.

| Recipe | Operational outcome | Applied changes |
| --- | --- | ---: |
| Control | Plain Adam | Not applicable |
| Keep | Expected unchanged optimizer | 0 |
| Codex | Context-audit rejection, followed by Adam fallback | 0 |

The agent proposed edits, but OPAL rejected its response:

```text
Unexpected provider non-message item: additional_tools
```

The provider rollout included extra tool definitions, including collaboration tools.
The isolation guard remains intact. This trial does not measure a successfully applied reactive optimizer.
The Codex decision took 131.3 seconds. Training ended at update 1872 on optimizer version zero.
The selected checkpoint was at update 312; the first decision was requested at update 350.
The second scheduled decision at update 3300 was never reached.

Reported usage was 178,791 input tokens, including 154,368 cached input tokens, plus 4,878 output tokens.
Total reported usage was 183,669 tokens. Cached input and reasoning output are subsets, not additional tokens.
The 20,000-token soft allowance was exceeded during the single active turn. Dollar cost is unknown.

The [filtered diagnostics](diagnostics.json) exclude raw prompts and provider transcripts.
The retry uses a pinned stable CLI, one decision at update 50, and fresh artifact directories.
Its live compatibility is not established by offline version and unit checks.
See the [compatibility assessment](../../opal-compatibility-notes.md) before authorizing another live run.

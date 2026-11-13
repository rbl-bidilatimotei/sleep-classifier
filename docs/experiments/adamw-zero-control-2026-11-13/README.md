# Synthetic zero-decay control

Reviewed on 2026-11-13 with PyTorch 2.15.0.
Both checks use the unchanged three-epoch patch model, active dropout, rate 0.001, and gradient clipping at norm 1.0.
They compare legacy single-group Adam against grouped AdamW with zero decay.
Each check uses twenty synthetic batches. No participant observations are read.

| Device | Maximum loss difference | Gradient difference | Parameter difference | Optimizer-state difference | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 0 | 0 | 0 | 0 | Exact match |
| MPS | 0 | 0 | 0 | 0 | Exact match |

The zero-decay groups contain 2,560 decay-eligible parameters and 196 exempt parameters.
Both groups apply zero decay during this control. The actual candidate applies 0.01 only to the eligible group.
Names, counts, tolerances, seeds, and runtime versions are preserved in [cpu.json](cpu.json) and [mps.json](mps.json).
The check restores random states before returning.

This bounded numerical result supports reuse of the measured Adam reference.
It does not prove identical full participant-data trajectories or establish any benefit from positive decay.
The next [experiment protocol](../../patch-adamw.md) keeps all other controls fixed.

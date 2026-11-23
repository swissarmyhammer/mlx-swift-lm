---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84dm19knmwjnxmj4vybs3am
  text: |-
    ## Dropped after analysis: a gathered matvec would have to pad the weights
    - In loop passes 2 and 3 the three projections have different output widths: the fused query rows (q, relative, gate, sinks: ~6.2k rows), k (1024) and v (1024). The full layers' k and v also carry their dynamic convolution taps.
    - `gatherQuantizedMatmul` takes one `[n, out, in]` weight stack, so k and v would need padding to the query width. That reads ~3x more weight bytes for those rows, ~6 MB more per visit, and would cost more GPU time than the 2 dispatches it saves.
    - A custom kernel over three row ranges of separate weights would only save ~48 dispatches per token. ^cyw9v2w showed that removing 48 dispatches is within noise. Not worth a new kernel.
  timestamp: 2026-11-23T14:40:35.635182+00:00
depends_on:
- 01M84889267NZRKTE28PXCCCJS
position_column: done
position_ordinal: ffffe780
title: 'Inkling 2 decode: one gathered matvec for the q/k/v projections of loop passes 2 and 3'
---
## Why
In loop passes 2 and 3, the message gates give the query, key and value their own inputs. `Inkling2Attention.project` then misses the shared-input path and runs three quantized matvecs: queryRows, k and v. That is 24 core visits × 2 extra dispatches = 48 per token.

## What
- Run the three inputs as one gathered quantized matmul (`gatherQuantizedMatmul` with one weight index per input), or as a custom kernel that reads three inputs against three row ranges of the fused weight.
- Measure the gain first in an inklingcli microbenchmark: 3 × qmv against 1 gathered call at the real shapes.

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS.
- [ ] inklingbench.sh shows a decode gain beyond the noise, or the card records the result and reverts.
- [ ] The Inkling 2 unit tests pass. #inkling-perf
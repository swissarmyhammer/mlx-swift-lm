---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84dmc9zvkbmrt7jbsb9z4et
  text: |-
    ## Dropped after analysis
    - A custom Metal kernel (MLXFast.metalKernel) cannot write into an input in place, so the prologue cannot store the new key and value into the cache buffer. The two `SliceUpdate`s per visit (96 per token) could only become one if KVCacheSimple and RotatingKVCache stored K and V in one buffer, which changes a shared MLXLMCommon type.
    - Expected gain: ^cyw9v2w removed 48 dispatches per token for a within-noise change, so 48–96 fewer here would be about the same. The Flint precedent (^y5d37gd, ^wqt7w37) found that cheaper cache writes did not move decode.
    - Main thread after the compile: `SliceUpdate::eval_gpu` is 1.6%. Not worth the change to the cache types.
  timestamp: 2026-11-23T14:40:46.911743+00:00
depends_on:
- 01M84889267NZRKTE28PXCCCJS
position_column: done
position_ordinal: ffffe880
title: 'Inkling 2 decode: write the new key and value into the cache from the attention prologue'
---
## Why
Each visit writes its new key and value with `cache.update`, which costs slice-update dispatches, on top of the prologue that computes them. Over 48 visits that is about 96 dispatches per token.

## Caution
On Flint, a cheaper cache write did not move decode (^y5d37gd, ^wqt7w37). Run this card only if the ^pxcccjs attribution shows that cache writes cost real time on Inkling.

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS.
- [ ] inklingbench.sh shows a gain beyond the noise, or the card records the result and reverts. #inkling-perf
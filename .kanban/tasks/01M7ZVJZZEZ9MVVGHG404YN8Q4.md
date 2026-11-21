---
assignees:
- claude-code
position_column: todo
position_ordinal: '8280'
title: 'Inkling 2 Mini: MTP block drafter (mtp.safetensors) in the fork''s speculative framework'
---
Wire Inkling 2 Mini's shipped MTP draft head into the fork's speculative decoding. The head is `mtp.safetensors`, kept in `mlx-community/Inkling-2-Mini-oQ4e-mtp` (15.7 GiB) and the official thinkingmachines/Inkling-2-Mini.

Work on branch `inkling2-mini`. From the earlier assessment:
- It is a block drafter with block size 8, the same kind as the fork's LFM2.5 DSpark drafter. The drafter interfaces, the target-layer-ids hook and the factory registration all exist.
- It needs a new drafter model, `Inkling2DraftModel`. The reference implementation is in oMLX, not mlx-lm; find it. Its pieces:
  - separate context projections
  - extra norms
  - a learned mask embedding
  - a 4×4 layer-fusion weight
  - an "rnn_hidden" Markov head
  - sliding-window layers
- `target_layer_ids` [5, 23, 41, 48] count layer *passes* through the looped stack (48 passes per token). Inkling2TextModel must capture hidden states at those passes and emit the MTP state keys.
- Rejected drafts must rewind the convolution states (ArraysCache) and the rotating attention caches. The fork's Qwen3.5 recurrent path is the precedent.
- The PR's fast decode kernels handle one token only, so a 9-token verify takes the prefill path. Measure the round cost, and add a multi-token verify kernel path if that is needed to make MTP a net win.
- MTP must keep multi-turn prompt-cache reuse working, including after drafter rejections.

Acceptance:
- [ ] Unit tests, written first: drafter weight loading, block draft shapes, rewind after a partial accept, and greedy MTP equal to plain greedy on a tiny model.
- [ ] Real model (`Inkling-2-Mini-oQ4e-mtp`): MTP greedy token ids equal plain greedy on 3 prompts.
- [ ] Measured plain vs MTP decode at short and ~4k context, with the acceptance rate. MTP is faster, or there is a recorded NO-GO with numbers.
- [ ] Multi-turn cache reuse works with MTP on.
- [ ] All 5 bundles are green.

#inkling
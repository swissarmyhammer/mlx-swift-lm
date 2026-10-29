---
assignees:
- claude-code
position_column: todo
position_ordinal: 9f80
title: 'Guided generation: feed the sampled token to the model when its commit fast-forwards'
---
## What
`GuidedGenerationLoop` (plain decode, since the upstream import f1573a93) feeds the model only the jump-forward tokens when a commit fast-forwards: the token the model decided right before them is emitted to the output and committed to the grammar, but never fed, so the KV cache holds `[.., ff1..ffk]` without it. Every later logit is conditioned on a sequence that skips one output token per fast-forward (JSON keys, enum completions...).

The MTP guided path (^9xe94tr) reproduces the same feed through one shared function, `GuidedGenerationLoop.modelFeed(sampled:fastForward:)` (Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift), so that its output stays byte-identical to plain decode. Fixing the feed there (`[sampled] + fastForward`) fixes both paths at once.

- [ ] Write a failing test: after a fast-forwarding commit, the model cache holds the decided token before the forced tokens (cache length = prompt + every emitted token).
- [ ] Change `modelFeed` to feed the decided token and then the forced tokens.
- [ ] MTP: the verify batch then holds the draft and its forced tokens; check `GuidedSpeculation.verifyBatch` and the drafter commit still pair each fed token with the hidden row before it, and that the in-place verify depth still bounds the batch.
- [ ] Re-run the guided benchmarks (output quality changes: guided outputs differ from before by design).

## Acceptance Criteria
- [ ] Plain and MTP guided outputs stay byte-identical to each other on the fixture.
- [ ] The cache holds every emitted token.

#guided-generation #bug
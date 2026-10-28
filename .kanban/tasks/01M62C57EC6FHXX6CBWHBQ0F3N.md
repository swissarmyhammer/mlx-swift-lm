---
assignees:
- claude-code
position_column: todo
position_ordinal: 9b80
title: 'MTP: gate chained Qwen drafts on the seed''s confidence so deeper blocks stop costing prose and code'
---
## What
^hcfzyem added chained Qwen MTP drafts (blockSize up to 6) and a confidence floor (`MTPDrafterState.minimumDraftProbability`, llama.cpp's p_min). On the M5 Ultra no deeper setting beats blockSize 2 on every category: with pMin 0.75, math gains (+7-8% at bs4/bs5) but prose loses 4-8% beyond noise, so the default stayed blockSize 2 (numbers on ^hcfzyem).

Measured cause: every chained round pays one chained MTP step (MTP layer + full-vocabulary LM head) plus one host sync before the floor can stop it, even when the seed itself was unsure. In prose most chains stop at that first gated step, so the step and the sync are pure cost (~1.5-2 ms of a ~21 ms round).

- [ ] At commit, keep the seed's top-1 probability (lazily, from the logits `sampleMTPSeed` already computes) in `MTPDrafterState`, and skip the chain when it is below the floor, so a low-confidence round costs exactly a blockSize-2 round. A restored continuation without the value chains as today.
- [ ] Measure batching the floor check: draft the whole chain lazily and read all top-1 probabilities in one sync, against the per-step sync.
- [ ] Re-run `qwen38FourBitMTPGeneration()` (Release sweep); if a setting now wins under the DEFAULT RULE of ^hcfzyem, change `SpeculativeDecodingConfig.defaultMTPBlockSize` / `defaultMTPMinimumDraftProbability` and `Qwen35MTPFixture.blockSize`.

## Acceptance Criteria
- [ ] Greedy output equals plain decode on the shared fixture with the seed gate, blockSize 3..6.
- [ ] Release sweep printed; default follows the DEFAULT RULE (fastest combined, no category slower than blockSize 2 beyond noise).

## Workflow
- Use `/tdd`. #mtp #qwen38 #perf
---
assignees:
- claude-code
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8580'
title: 'MTP: Qwen MTP at temperature > 0 (sample-and-match with argmax drafts; rejection math is done by PR 459)'
---
## What
Upstream PR 459 (commit 82f273fd) added rejection sampling for drafters that conform to `MTPDistributionDrafterModel` (today only Gemma4Assistant): `DistributionLogitSampler.logProbabilities(logits:)` on `TopPSampler` / `CategoricalSampler`, `speculativeAcceptanceProbability`, `speculativeResidualLogProbabilities`, independent random states, and `MTPSpeculativeSamplingMathTests`. Stateful drafters (Qwen MTP) still require greedy sampling, and the default FoundationModels sampling is random, so Qwen MTP does not run there.

Math check: with a draft that does not depend on the target random stream (argmax), "sample and match" emits exact samples of the target; rejection sampling with a soft q accepts 1 − TV(p, q), which can be higher.

- [ ] Qwen path: draft with argmax for the seed and the chained drafts, and verify with sample-and-match (the target sampler and the iterator random state). Set `requiresGreedySampling = false` in both Qwen35MTP.swift files. Keep the exact greedy path at temperature 0. Stateful `LogitProcessor`s stay on the target only.
- [ ] Optional: make the Qwen drafter conform to `MTPDistributionDrafterModel` (return processed logits per draft) and remove the stateful-path exclusion in `MTPSpeculativeTokenIterator`, so that the PR 459 rejection path runs for Qwen. Measure both modes and keep the faster one as the default.

## Acceptance Criteria
- [ ] At temperature 0, the output is the same as before this task.
- [ ] Over many seeded trials on a small vocabulary, the distribution of the first two emitted tokens with Qwen MTP matches plain decode (chi-square p > 0.01), for each mode.
- [ ] Release benchmark on the M5 Max: at temperature 0.7, top-p 0.95 (a sampled variant of `qwen38FourBitMTPGeneration`), MTP tok/s > plain tok/s on the math category, with an `#expect`.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: temperature 0 equality, temperature > 0 runs without passthrough, seed repeatability, chi-square.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #sampling
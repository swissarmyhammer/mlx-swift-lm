---
assignees:
- claude-code
depends_on:
- 01M5SK9FZ2HTB7XJDAPCBRDDM5
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJPFM01WFRYWMQXVGMA4MQ
position_column: todo
position_ordinal: '8580'
title: 'MTP: exact sampling at temperature > 0 (argmax drafts with sample-and-match, optional rejection sampling)'
---
## What
`Qwen35MTPDraftModel.requiresGreedySampling = true` (Libraries/MLXLLM/Models/Qwen35MTP.swift and Libraries/MLXVLM/Models/Qwen35MTP.swift). Thus `MTPSpeculativeTokenIterator.init` changes to plain decode when `temperature != 0`. The default FoundationModels sampling is random (temperature 0.6), so MTP does not run in real use. This task depends on the other iterator tasks only to keep the order of changes to MTPSpeculativeTokenIterator.swift.

The verify loop already does "sample and match" (MTPSpeculativeTokenIterator.swift ~539-552: process → sample → `didSample` for each position, accept while the target sample equals the draft). llama.cpp uses the same default mode (`probabilistic = false`). The output distribution is exact only when the drafts do not depend on the target random stream. Today they do:
- `draftSampler` is the random sampler (~154-164), so the chained drafts are random.
- `prepareDrafterState` and `commitDrafterState` get the target `sampler` (~384, ~589), so the seed drafts take samples from the target random stream.

- [ ] Draft with argmax (the suppress-token mask still applies) for the seed and the chained drafts. Do not pass the target sampler or the target random state to the drafter.
- [ ] Set `requiresGreedySampling = false` in both Qwen35MTP.swift files. Keep the exact greedy path when temperature is 0. Stateful `LogitProcessor`s stay on the target only (the current order is correct); write this rule in the doc comment of `draftBlock`.
- [ ] Optional mode (`SpeculativeDecodingConfig.probabilisticDraft`): add a pure function `speculativeAccept(targetProbabilities p:, draftProbabilities q:, draftToken:, uniform:)` in Libraries/MLXLMCommon/SpeculativeSampling.swift (accept with min(1, p/q); on rejection sample from normalize(max(p − q, 0))). In this mode the drafter samples from its own random state and returns q for each draft. Default: sample and match.

## Acceptance Criteria
- [ ] At temperature 0, the output is the same as before this task.
- [ ] For each mode, over many seeded trials on the shared fixture with a small vocabulary, the distribution of the first two emitted tokens with MTP matches the plain-decode distribution (chi-square p > 0.01).
- [ ] Two runs with the same seed give the same tokens; the target random stream does not change when the drafter changes (a test with two different drafters gives the same first plain-sampled token after a rejection).
- [ ] Release benchmark on the M5 Max: at temperature 0.7, top-p 0.95 (a sampled variant of `qwen38FourBitMTPGeneration`), MTP tok/s > plain tok/s on the math category, with an `#expect`.

## Tests
- [ ] Tests/MLXLMTests/SpeculativeSamplingTests.swift: `speculativeAccept` accept/reject/resample cases and the chi-square tests for both modes.
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: temperature 0 equality, temperature > 0 runs without passthrough, seed repeatability.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #sampling
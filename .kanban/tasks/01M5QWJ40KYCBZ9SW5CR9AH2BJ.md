---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6204vg4za1dachvnnf0wm00
  text: |-
    Picked up. Research notes:
    - Qwen drafters (MLXLLM/MLXVLM Qwen35MTP.swift) have maximumBlockSize 2: each round's single draft is the seed the drafter computed in commitDrafterState/appendDrafterState via sampleMTPSeed(sampler:). The iterator passes the TARGET sampler (`sampler`) to prepare/resume/coldStart/feed/commit and passthroughStep — at temperature > 0 that would sample seeds from the target sampler's RandomState. Fix: stateful drafters get an argmax seed sampler and an argmax draft sampler; at temperature 0 both are ArgMaxSampler today, so the greedy path is unchanged.
    - standDownReason sends any DistributionLogitSampler stream with a non-MTPDistributionDrafterModel drafter to passthrough (samplingWithoutDraftLogitsReason); stateful drafters must be exempted (sample-and-match is exact for argmax drafts).
    - selectGreedyTokens is already a per-row sample-and-match walk (one eval per row); add a batched sampled variant beside batchedGreedyMTPVerification (shared matching helper, greedy graph unchanged) for processor nil / SuppressTokensProcessor.
    - Rejection mode for Qwen: draftBlockWithLogits on a stateful variant recomputes the seed logits from state.seedHidden (argmax seedToken stays lazy and unused), samples from q with the drafter's own seeded sampler; verify goes through selectDistributionPreservingTokens; commit unchanged.
    - ChatSession gates MTP on requiresGreedySampling; MTPDrafterPrefillFeed.starting uses parameters.sampler() for discarded seeds -> switch to argmax.
    - Existing test samplingWithoutDraftLogitsFallsBackToPassthrough uses StatefulMockDrafter at temp 0.7 expecting passthrough; its premise changes for stateful drafters.
    - Mode knob must be public: IntegrationTesting imports MLXLMCommon without @testable.
  timestamp: 2026-10-28T19:35:14.180364+00:00
- actor: claude-code
  id: 01m620sx7s0jxrz5mh5dc175y4
  text: |-
    Implementation landed (unit level): MTPSampledVerification enum (.sampleAndMatch / .rejectionSampling) as GenerateParameters.mtpSampledVerification; BatchedMTPVerification.sampled beside the unchanged greedy batch (shared matching helper); StatefulMTPDistributionDrafterModel conformed by both Qwen drafters (seed draft re-drawn from seedHidden; chain via MTPDraftBlockOutput.chainedMTPDrafts, draftMTPTokenBlock deleted); iterator: argmax seed sampler for every stateful drafter call, argmax draft sampler unless rejection mode, stateful drafters exempt from samplingWithoutDraftLogits stand-down, speculateRound split into proposeDrafts/verifyDrafts.
    Discoveries: (1) a 2-token trial never speculates (draft budget = remaining-1), trials need 3 tokens. (2) The chi-square at temp 0.7/top-p 0.95 over 64x64 pairs had NO power: a verifier that always accepts drafts still passed (cells pooled). Fixed with temperature 1.5 + top-k 4 (16 cells); mutation checks now fail the p-value for both a biased sample-and-match and an always-accept rejection verifier.
  timestamp: 2026-10-28T19:46:44.089006+00:00
- actor: claude-code
  id: 01m621w0n9grnw4t25ps4mw2eg
  text: |-
    Measurements (M5 Ultra, Release, IntegrationTesting):
    - qwen38FourBitMTPSampledGeneration() (temp 0.7, top-p 0.95, seed 38), median of 3 runs, 192 tok:
      - sampleAndMatch: math plain 58.5 / MTP 93.0 tok/s (acc 96.9%), code 58.6 / 78.9 (67.5%), prose 58.5 / 76.6 (63.2%)
      - rejectionSampling: math 58.3 / 83.0 (89.1%), code 58.4 / 79.2 (81.0%), prose 58.5 / 73.5 (67.5%)
      Rejection accepts more on code but pays per-row syncs plus two top-p logProbabilities sorts over the 248k vocabulary per draft; sample-and-match verifies a round in one readback. sampleAndMatch is faster overall (248.5 vs 235.7 summed) and stays the default; #expect holds on math (93.0 > 58.5). Test passed.
    - qwen38FourBitMTPGeneration() (temp 0): MTP 92.4 / 82.6 / 80.5 tok/s vs plain 59.3/59.4/59.4, matches plain: true on all three (within 3% of ~93/83/81).
    - qwen38FourBitMTPSystemPromptCache(): passed, no known issues, edit turn prefilled 123 tok with drafter prefilled 123 tok.
    Unit: swift build --build-tests 0 compiler warnings (only the SwiftPM "missing creator for mutated node" build-system line); all 5 bundles green via xcrun xctest (MLXLMTests 897 XCTest + 1939 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5). One pre-existing env-gated skip: MTPGreedyVerificationTests.benchmarkVerification (MTP_GREEDY_BENCHMARK), not touched here and already tracked on the board.
    Existing tests updated for the new contract: Qwen35MTPTests registration now expects !requiresGreedySampling; samplingWithoutDraftLogitsFallsBackToPassthrough uses the stateless token-only StatelessMTPDrafter (stateful drafters now speculate by sample-and-match); testQwenStyleMTPRequiresGreedySampling compares against greedySamplingReason (message reworded: no longer Qwen-specific); ChatSessionQwenTextMTPTests "after a target-only turn..." makes the middle turn target-only with a log-probability request, since a sampled Qwen turn now speculates.
  timestamp: 2026-10-28T20:05:21.705146+00:00
- actor: claude-code
  id: 01m621w4z1mjkfkc5ytacqx0x0
  text: |-
    ### implement — changed
    - evidence: 15 files — Libraries/MLXLMCommon/MTPSampledVerification.swift (new), Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/MLXLMCommon/QwenMTP.swift, Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift, Libraries/MLXLMCommon/Evaluate.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Libraries/BenchmarkHelpers/BenchmarkHelpers.swift, IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift, Tests/MLXLMTests/Qwen35MTPSamplingTests.swift (new), Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift, Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift; swift build --build-tests 0 compiler warnings; 5 bundles green; Release sampled math MTP 93.0 > plain 58.5 tok/s; temp-0 92.4/82.6/80.5 matches plain; system prompt cache pass, edit turn 123 tok
    - next: /review
  timestamp: 2026-10-28T20:05:26.113364+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8280'
title: 'MTP: Qwen MTP at temperature > 0 (sample-and-match with argmax drafts; rejection math is done by PR 459)'
---
## What
Upstream PR 459 (commit 82f273fd) added rejection sampling for drafters that conform to `MTPDistributionDrafterModel` (today only Gemma4Assistant): `DistributionLogitSampler.logProbabilities(logits:)` on `TopPSampler` / `CategoricalSampler`, `speculativeAcceptanceProbability`, `speculativeResidualLogProbabilities`, independent random states, and `MTPSpeculativeSamplingMathTests`. Stateful drafters (Qwen MTP) still require greedy sampling, and the default FoundationModels sampling is random, so Qwen MTP does not run there.

Math check: with a draft that does not depend on the target random stream (argmax), "sample and match" emits exact samples of the target; rejection sampling with a soft q accepts 1 − TV(p, q), which can be higher.

- [x] Qwen path: draft with argmax for the seed and the chained drafts, and verify with sample-and-match (the target sampler and the iterator random state). Set `requiresGreedySampling = false` in both Qwen35MTP.swift files. Keep the exact greedy path at temperature 0. Stateful `LogitProcessor`s stay on the target only.
- [x] Optional: make the Qwen drafter conform to `MTPDistributionDrafterModel` (return processed logits per draft) and remove the stateful-path exclusion in `MTPSpeculativeTokenIterator`, so that the PR 459 rejection path runs for Qwen. Measure both modes and keep the faster one as the default.

## Acceptance Criteria
- [x] At temperature 0, the output is the same as before this task.
- [x] Over many seeded trials on a small vocabulary, the distribution of the first two emitted tokens with Qwen MTP matches plain decode (chi-square p > 0.01), for each mode.
- [x] Release benchmark on the M5 Ultra: at temperature 0.7, top-p 0.95 (a sampled variant of `qwen38FourBitMTPGeneration`), MTP tok/s > plain tok/s on the math category, with an `#expect`.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPSamplingTests.swift (new file; Qwen35MTPTests.swift is ~104 KB): temperature 0 equality, temperature > 0 runs without passthrough, seed repeatability, chi-square.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #sampling
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
- actor: claude-code
  id: 01m6228cq4dkfwbcrbxdhzvw5w
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (6800fd67): 2 findings (2 confirmed, 3 refuted) — Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift:101, Libraries/MLXLMCommon/QwenMTP.swift:102
    - next: implement the 2 open items in the 2026-10-28 15:05 Review Findings section, then re-review
  timestamp: 2026-10-28T20:12:07.268984+00:00
- actor: claude-code
  id: 01m6228vxhc14t5bnwdqnfyzpf
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 15 files
    - test: green (implementer run) — 5 bundles; Release sampled MTP 93.0/78.9/76.6 vs plain 58.5; temp 0 MTP 92.4/82.6/80.5 = plain; cache guard pass, edit turn 123 tok
    - commit: 6800fd67
    - review: findings — Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift:101, Libraries/MLXLMCommon/QwenMTP.swift:102
  timestamp: 2026-10-28T20:12:22.833111+00:00
- actor: claude-code
  id: 01m622pmachjby18kqq9bvzf87
  text: |-
    Review findings (2026-10-28 15:05) resolved:
    1. reuse: the seed-sampler policy now has one definition, `package var mtpDrafterSeedSampler: ArgMaxSampler` at the top of MTPDrafterPrefillFeed.swift, with the explanatory comment moved there. The iterator's `private static var drafterSeedSampler` is deleted and its six seed calls reference `mtpDrafterSeedSampler`. The feed no longer takes a `sampler:` init parameter or stores one (both callers always passed the argmax seed sampler), and `append` draws with `mtpDrafterSeedSampler`, so no second injection path can drift. The `ArgMaxSampler()` in the iterator's `drafterSampler` (chained-draft sampler, which switches with rejection mode) is a separate policy and stays.
    2. magic-numbers-swift: `private static let minimumChainedMTPBlockSize = 2` on the MTPDraftBlockOutput extension in QwenMTP.swift; the precondition and its message use it. swiftlint 0.65.1 `only_rules: [no_magic_numbers]`, `allowed_numbers: [0, 1, -1, 100]` over QwenMTP.swift and MTPDrafterPrefillFeed.swift: 1 violation before (QwenMTP.swift:102:35), 0 after. swift-format lint clean on the three edited files.
    No behaviour change (same ArgMaxSampler everywhere), so the Release benchmarks were not rerun.
  timestamp: 2026-10-28T20:19:53.804795+00:00
- actor: claude-code
  id: 01m622pqkgj6ad2kxj4t2x99fr
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/QwenMTP.swift; swiftlint no_magic_numbers (review config) 0 on both flagged files; swift build --build-tests 0 compiler warnings (only the SwiftPM "missing creator for mutated node" build-system line); xcrun xctest all 5 bundles green — MLXLMTests 897 XCTest + 1939 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 (1 pre-existing env-gated skip, MTPGreedyVerificationTests.benchmarkVerification); 2/2 review findings flipped to [x]; Release benchmarks not rerun (no behaviour change)
    - next: /review
  timestamp: 2026-10-28T20:19:57.168057+00:00
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

## Review Findings (2026-10-28 15:05)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 14 file(s) reviewed, 6 not reviewed.

> 6 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWJ40KYCBZ9SW5CR9AH2BJ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWJ40KYCBZ9SW5CR9AH2BJ.md` — no validator matches this file
> - `.kanban/tasks/01M5QWJXZ5EEBVH7Z9D3W7JKFN.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWJXZ5EEBVH7Z9D3W7JKFN.md` — no validator matches this file
> - `.kanban/tasks/01M5QWS53X7RKJKABDTBTFKTSY.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWS53X7RKJKABDTBTFKTSY.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift:101` `reuse/reuse` — The feed builds its own argmax seed sampler with `ArgMaxSampler()`, although the iterator already defines the same concept as `drafterSeedSampler` ("the sampler of a stateful drafter's seed proposals: argmax"). Two definitions of the seed-sampler policy can drift apart, and a later change to one would silently leave the other behind. Have the feed take the seed sampler from its caller, or move the seed-sampler definition to a shared, non-private location (for example beside `ArgMaxSampler` or next to `MTPDrafterPrefillFeed`) and reference it from both the iterator and the feed. Keep the explanatory comment with the single definition.
- [x] `Libraries/MLXLMCommon/QwenMTP.swift:102` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.

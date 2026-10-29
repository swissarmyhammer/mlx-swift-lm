---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m62mgkf93kcbsx54jwnb4m56
  text: |-
    Picked up (mtp @ 3222fb36). Research:
    - MLXFoundationModels does not import MLXLLM/MLXVLM, so `MLXSpeculativeDecoding` carries an `MTPDrafterRegistration` (MLXLMCommon, public: `.qwen35Text` / `.qwen35VLM`) plus downloader/tokenizer loader, mirroring `SpeculativeDecodingConfig(mtpDrafterConfiguration:...registration:)`, and a closure form for tests.
    - Executor paths: runUnconstrained/runReasoning/runAllowedToolGeneration use `generateTaskRecordingTokens` / `generateProtocolTokensTask` (package, Evaluate.swift); think-then-call phase 1 uses public `generateTokensTask`; phase 2 and schema use GuidedGenerationLoop (separate task ^SJR74AK). Plan: a package `MTPSpeculation` (drafter + blockSize + floor) and one iterator choice helper (MTP when `isCompatible(with:)`, TokenIterator otherwise or on `unsupportedSpeculativeCache`), a package `generateTokensTask(... speculation:)` overload for phase 1.
    - Prompt cache: `ExecutorPromptCachePlan.commitOutcome` takes the ledger from the cache POSITION (`generatedTokens.prefix(position - ledger)`), so an MTP bonus token that no forward fed never enters the ledger; main-cache prefix reuse keeps working, and a warm cache without a continuation cold-starts the resumable Qwen drafter at the cached position (^5fq... cold start work). Drafter state reuse across turns is ^SJQVM27's job; this card adds the completion callback it will consume.
    - Usage: `generatedTokenCount(of:)` adds the stop token only when `stopTokenFedToCache` is set; an MTP stop on the bonus slot is not fed, so usage would be one lower than plain decode. Needs the loop to report the generated stop token itself.
    - Test fixture gaps: `ScalarTokenizer.convertIdToToken` never returns nil, so `TokenizerVocabExtractor` (required-tool / think-then-call grammar tokenizer) loops forever on it; `Qwen35VLMMTPFixture`'s vision tower outputs 16 wide into a 32-wide text model and its image token ids are outside the 64 vocab, so an image turn cannot run on it as is.
    - ChatSession sends media turns to plain decode (`.unsupportedMedia`); the VLM Qwen35 target does feed MTP state for image prompts (prepareContinuation chunk handler with rope deltas), so the executor will speculate on image turns.
  timestamp: 2026-10-29T01:31:10.697902+00:00
- actor: claude-code
  id: 01m62qdmpcgpwnbmm2pgwwa8hb
  text: |-
    implement step: MTP wired into MLXLanguageModel.
    - New `MLXSpeculativeDecoding` (drafter config + blockSize, validated through `ChatSession.validate(mtpBlockSize:)`); `MLXLanguageModel(speculativeDecoding:)`.
    - ModelCache: generic coalesced LoadTask; drafter slot keyed by (modelID, drafterID); remove/evictAll drop or cancel drafters.
    - MLXLMCommon: `MTPSpeculation.makeTokenIterator` (MTP when isCompatible, else TokenIterator; unsupported cache falls back). `generateTaskRecordingTokens` / `generateProtocolTokensTask` / package `generateTokensTask` take `speculation:` and a `drafterContinuation` completion callback for ^SJQVM27 to use. `GenerateCompletionInfo.discardedStopToken` lets usage counting stay correct when a bonus-slot stop token never reaches the cache.
    - Executor: speculation is loaded before container.perform and passed in. Drafter runs inside the same perform with no nested perform. The think-then-call phase-1 drafter state ends with its task (comment + test). On close, phase 1 cancels and keeps draining. warmUp loads the drafter and runs a short greedy MTP stream.
    - Tests: MTPSpeculationTests (7), MTPExecutorTests (11). Greedy parity for text, reasoning, allowed tools, think-then-call, VLM image, `.uncached`; seeded sampling repeats; drafter loads once across sessions and warmUp; eviction frees it; usage is the same. Integration test MTPGenerationTests.swift.
    - Results: all 5 bundles pass (MLXLMTests 1988+920, MLXFoundationModelsTests 388, MLXGuidedGenerationTests 70, CXGrammarTests 7, macros 5); 0 warnings; lint clean on new files.
    - Release: MTPGenerationTests MTP 85.7 vs plain 59.7 tok/s on math. qwen38FourBitMTPGeneration 91.7/82.3/80.3 (plain ~59). qwen38FourBitMTPSystemPromptCache passes (edit turn prefilled 123). qwen38FourBitMTPSampledGeneration passes.
    - Note: an incompatible drafter (e.g. a target loaded as VLM by loadModelContainer) silently falls back to plain decoding. That is worth a follow-up task to report it.
  timestamp: 2026-10-29T02:21:59.372069+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5QWJ40KYCBZ9SW5CR9AH2BJ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: FoundationModels — run MLXLanguageModel text, reasoning and tool turns with the MTP drafter'
---
## What
No FoundationModels path can use MTP today. `Executor.respond` → `runRespond` (Libraries/MLXFoundationModels/MLXLanguageModel.swift ~958-1168) goes to `runTextGeneration` / `runUnconstrained` / `runReasoning` / `runToolGeneration`, and these call `generateTaskRecordingTokens`, `generateProtocolTokensTask`, and `generateTokensTask` (Libraries/MLXLMCommon/Evaluate.swift ~2500, ~2797, ~2835). Each makes a plain `TokenIterator`. `ModelCache` (Libraries/MLXFoundationModels/ModelCache.swift ~46-139) has no drafter slot. The default FoundationModels sampling is random (temperature 0.6, Evaluate.swift ~256), so this task needs the sampling task. The prompt-cache entry work is a separate task.

- [ ] Add an MTP option to `MLXLanguageModel` (for example `speculativeDecoding: MLXSpeculativeDecoding?` with a drafter model ID and `blockSize`), and a `ModelCache` drafter slot keyed by (model ID, drafter ID). Load the drafter through `MTPDrafterModelFactory` with the MLXLLM and MLXVLM Qwen35 MTP registrations, one time, coalesced like model loads. `remove` / `evictAll` of a model also evicts its drafter. `warmUp` (MLXLanguageModel.swift ~442-493) also loads the drafter and runs one short MTP stream.
- [ ] Add an iterator choice to `generateTaskRecordingTokens`, `generateProtocolTokensTask`, and `generateTokensTask`: when a drafter is present and `isCompatible(with:)` is true, build `MTPSpeculativeTokenIterator` with the `state:` input and the drafter-state completion callback; otherwise `TokenIterator`. The think-then-call phase 1 (`generateTokensTask`, ~2786) drops its drafter state at phase 2; write this in a comment and test it.
- [ ] Run the drafter inside the existing `container.perform` of the target (~1077). Do not nest a `perform` on the `MTPDrafterContainer` (MTPDrafterModel.swift ~238-259); take the drafter model out of its container one time, so that no lock order can deadlock.
- [ ] Cancellation: `generateLoopTask` checks cancellation before each `next()` (Evaluate.swift ~3015), and one MTP `next()` can run a full round. Make sure the drain in MLXLanguageModel.swift (~1870, "one forward pass") still holds, or update its comment and test.

## Acceptance Criteria
- [ ] On the shared fixture, a `LanguageModelSession` with `.greedy` gives the same text with and without MTP for plain text, reasoning, allowed-tools, and think-then-call turns, for a turn with an image (VLM target), and with the `.uncached` prompt-cache scope.
- [ ] With the default sampling mode and a fixed seed, an MTP session runs MTP (no passthrough), and two runs with the same seed give the same text.
- [ ] The drafter loads one time across two sessions of the same model and in `warmUp`, and evicting the model frees the drafter.
- [ ] Reported usage (prompt and generated token counts) is the same with and without MTP for the same greedy text.
- [ ] Release integration test on the M5 Max: a greedy FoundationModels session with Qwen3.8-27B-4bit + MTP gives the same tokens as without MTP (or the first difference is at a position where the top-2 target logits differ by < 1e-3), and decode tok/s is higher on the math prompt (`#expect`).

## Tests
- [ ] Tests/MLXFoundationModelsTests/MTPExecutorTests.swift: the paths above, the load count, warm-up, eviction, usage, and cancellation.
- [ ] IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGenerationTests.swift: the real-model test; it skips when `mlx-community/Qwen3.8-27B-MTP-4bit` is not local.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #foundation-models
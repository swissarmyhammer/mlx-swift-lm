---
assignees:
- claude-code
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5SK9FZ2HTB7XJDAPCBRDDM5
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5QWJ40KYCBZ9SW5CR9AH2BJ
position_column: todo
position_ordinal: 8f80
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
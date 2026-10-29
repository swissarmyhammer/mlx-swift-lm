---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m645ckptw7cs2jt5cm4mdy8a
  text: |-
    Implementation landed in c3aa0e25 on worktree-agent-a96eeb70bc03d7a68 (parent 6354610a, ^q9sgwt5).
    - MLXSpeculativeDecoding: new `drafterBytes: Int?` (negative clamps to 0) and `memoryPolicy: SpeculativeDecodingMemoryPolicy?` on both public inits and as public properties. Admission reuses ChatSession's package code: `SpeculativeDecodingMemoryPolicy.admitsSpeculation(mainModelBytes:draftModelBytes:)` + `modelWeightBytes`.
    - MLXLanguageModel.loadMTPSpeculation: with an estimate, weighs target weights + drafterBytes inside the target perform BEFORE the drafter load; not admitted -> returns nil (plain decode) and records `.memoryBudgetExceeded` into the call's SpeculativeDecodingOutcome (metadata "speculativeDecodingFallbackReason": "memoryBudgetExceeded"); `.fail` throws SpeculativeDecodingMemoryError out of respond (and out of warmUp, which logs it). Without an estimate the drafter loads and the policy weighs its loaded parameters (ChatSession's loaded-drafter form) — documented in README and the type doc.
    - Limit: the GenerateCompletionInfo of the plain passes does not carry .memoryBudgetExceeded (its setter is internal to MLXLMCommon and the FM passes run with no speculation); the reason reaches the caller through the response metadata.
    - Macro: #huggingFaceLanguageModel already forwards `speculativeDecoding:` verbatim, so no macro code changed; new LanguageModelMacroMemoryPolicyTests holds the expansion with drafterBytes/memoryPolicy, and the macro doc example shows them.
    Tests: MLXSpeculativeDecodingMemoryPolicyTests (6: stores policy/estimate; fallback -> plain text equal to plain model, 0 drafter loads, reason memoryBudgetExceeded; fail -> SpeculativeDecodingMemoryError and 0 loads; admitted -> 1 load and speculates; no estimate -> loads then weighs, falls back; warm-up under refusing policy loads none). RED was a compile failure on the new init parameters; GREEN after. All 5 bundles: MLXLMTests 920 XCTest + 2035 ST, MLXGuidedGenerationTests 72, MLXFoundationModelsTests 452, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7 — 0 failures, 0 skipped. scripts/verify-docs.sh exit 0.
    Release real-model check (MTPGenerationTests + MTPPromptCacheTests) is NOT run: coordinator rule says wait for "GPU go" before taking bench.lock. A lock-waiter I started earlier never acquired the lock and was killed.
  timestamp: 2026-10-29T15:45:20.090197+00:00
- actor: claude-code
  id: 01m645hrjy08sjyn1btgfkxrqh
  text: |-
    ### implement — changed
    - evidence: commit c3aa0e25 (parent 6354610a) on worktree-agent-a96eeb70bc03d7a68; files: Libraries/MLXFoundationModels/{MLXSpeculativeDecoding.swift, MLXLanguageModel.swift, SpeculativeDecodingOutcome.swift, README.md}, Libraries/MLXHuggingFace/FoundationModelsMacros.swift, Tests/MLXFoundationModelsTests/{MLXSpeculativeDecodingMemoryPolicyTests.swift (new), Qwen35MTPFixture+LanguageModel.swift}, Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroMemoryPolicyTests.swift (new). 5 bundles 0 failures 0 skipped (FM 452, macros 7); verify-docs exit 0; IntegrationTesting Release build-for-testing succeeded.
    - next: real-model Release check waits for "GPU go"; then /review.
  timestamp: 2026-10-29T15:48:08.926186+00:00
position_column: doing
position_ordinal: '8280'
title: 'FoundationModels: memory policy on MLXSpeculativeDecoding so the drafter load falls back to plain decode with a reason'
---
Requested by the Router session (lower priority than the MTP outcome metadata task). ChatSession's SpeculativeDecodingConfig takes a `SpeculativeDecodingMemoryPolicy` (e.g. `.recommendedWorkingSet`); `MLXSpeculativeDecoding` (MLXLanguageModel) has none, so a drafter load can overcommit.

- [x] Add an optional `memoryPolicy: SpeculativeDecodingMemoryPolicy?` (and a drafter byte estimate if needed, matching ChatSession's deferred form) to `MLXSpeculativeDecoding`, reusing the ChatSession admission code rather than copying it.
- [x] Not admitted: `.fallbackToDefault` decodes plain and reports `.memoryBudgetExceeded` (and the Transcript.Response metadata fallback reason); `.fail` throws `SpeculativeDecodingMemoryError`. Neither loads the drafter.
- [x] Forward it through the `#huggingFaceLanguageModel` macro.
- [x] Tests in a new file; README update.

#mtp #foundation-models
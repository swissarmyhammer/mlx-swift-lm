---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8ghf51d560vpk9mtf33s7es
  text: |-
    Research (implement):
    - Phase 1 (runToolCallReasoningPhase) uses generateTokensTask with a fresh cache. generateProtocolTokensTask(decoder: nil) yields the same raw token stream (same RawTokenLoopHandler with no extra stop tokens), so it can plan/commit through withCommittedGeneration without an output change.
    - Phase 1's ledger after ProtocolPassStop = P1 + reasoning (close token included, plain decode feeds it before returning it) + a timing-dependent tail. Phase 2's prompt P2 = P1 + reasoning is then a PREFIX of (or equal to) the ledger. The standard reuse policy refuses that case on every model: ExtendCachedPrefixRule needs a strictly longer prompt, RewindToCommonPrefixRule needs sharedPrefix < prompt count (PromptCacheReusePolicyTests "an identical prompt has no suffix to append and rebuilds" documents it). So a plain-attention model would REBUILD phase 2 without help. Plan: at the phase boundary the slot rewinds the carried caches in place to P2.count - 1 when they can trim (plain attention), so phase 2 extends by one token; a hybrid model cannot trim and its plan restores phase 1's checkpoint (ExecutorPromptCachePlan.restoring, in place).
    - Hybrid checkpoint: phase 1 plans with the tool-aware render's checkpointBoundaries as a plain pass does. When transcriptEnd == prompt end (no generation prompt, as with ToolCallAlphabetTokenizer) the slot takes the prompt-end checkpoint at prefill end; otherwise the checkpoint stands at the end of the last message, a prefix of P1. The entry has one transcript checkpoint slot besides the instructions one, so a separate prompt-end checkpoint would displace the end-of-transcript checkpoint the NEXT turn needs.
    - Next-turn regression found: a plan that restores a checkpoint does not keep it (plan.checkpoint stays nil), and phase 2 cannot retake the end-of-transcript checkpoint because its caches already stand there, so the entry after the turn would lose it. Fix: a restore plan keeps the transcript checkpoint it went back to; the slot's commit prefers the prompt-end checkpoint over a carried one (today the two never coexist).
    - Usage: once phase 2 reuses phase 1's prefix, phase 2's reusedTokenCount can exceed the turn's prompt. The turn's input usage must come from phase 1's plan (its prompt minus the opener).
    - Observability: phase 1's fed prompt count is in its GenerateCompletionInfo (MTPTurn.completions); phase 2's fed count is not observable yet -> record it on GuidedGenerationDiagnosticSink.
  timestamp: 2026-11-28T07:38:43.117317+00:00
- actor: claude-code
  id: 01m8gj4ze3fwkwa43rg6zd9e2q
  text: |-
    Implementation landed (TDD):
    - RED: new ThinkThenCallPromptCacheTests (4 combos: local/clustered x Inkling tiny/attention-only Qwen 3.5) failed as expected: phase 1 fed 96 = whole prompt (want <= 76); usage cached 84 came from phase 2, not phase 1; phase 2 fed 15 > 3 reasoning tokens. Cold-turn parity and cached>=plain-turn already held.
    - GREEN: runToolCallReasoningPhase plans from the slot (checkpointBoundaries of the tool-aware render) and commits through withCommittedGeneration with generateProtocolTokensTask(decoder: nil). After phase 1, ExecutorPromptCacheSlot.rewindLedger(toFeedLastTokenOf:) trims rewindable caches in place to P2.count-1 (new ExecutorPromptCacheEntry.canRewind/rewound); hybrid caches stay and the phase-2 plan restores phase 1's checkpoint. Turn usage = phase 1's usageInput with the opener left out (new appendedTokenCount parameter; cached clamped to total — a test showed the unclamped value traps the process in Usage.Input).
    - Next-turn checkpoint carry (unit tests in ExecutorPromptCacheInstructionsCheckpointTests, each RED first): restore plans and extension plans keep the entry's end-of-transcript checkpoint (ExecutorPromptCacheEntry.checkpoint(serving:)); slot.commit prefers the prompt-end checkpoint over a carried one.
    - Observability: GuidedGenerationDiagnosticSink.fedPromptTokenCounts, recorded in runGuidedLoop.
    - Moved the private AttentionOnlyTarget of ClusterLanguageModelTests to MLXLMTestSupport/Qwen35AttentionOnlyTarget.swift (shared).
    - Deviation to flag: the card offered "a prompt-end checkpoint in phase 1". Phase 1 takes the checkpoints a plain pass takes; with no generation prompt that IS the prompt-end checkpoint, otherwise it is the end-of-last-message checkpoint (a prefix of the phase-1 prompt, costing the few generation-prompt/opener tokens). A separate prompt-end checkpoint would take the entry's one transcript-checkpoint slot from the checkpoint the next turn restores.
  timestamp: 2026-11-28T07:50:38.275618+00:00
- actor: claude-code
  id: 01m8gjpyjr4erfqfare51c1t1v
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests clean (0 warnings); xcrun xctest full bundles, each one process, all exit 0: MLXFoundationModelsTests 744 tests/98 suites; MLXGuidedGenerationTests 96; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 XCTest; MLXLMTests 2849 Swift Testing + 1261 XCTest, 0 failures (3 pre-existing known issues: AttentionKernelToleranceTests, MiniCPM5 ToolTests). swift-format applied. Files: Libraries/MLXFoundationModels/{MLXLanguageModelGeneration,MLXLanguageModel,ExecutorPromptCache,ExecutorPromptCacheEntry,ExecutorPromptCacheSlot}.swift, README.md; Libraries/MLXGuidedGeneration/GuidedGenerationDiagnosticSink.swift; Tests: new ThinkThenCallPromptCacheTests.swift (5 tests x 4 combos), new MLXLMTestSupport/Qwen35AttentionOnlyTarget.swift, ExecutorPromptCacheInstructionsCheckpointTests (+4), ExecutorPromptCacheTests (+1), Qwen35MTPFixture+LanguageModel.swift (helper params), ClusterLanguageModelTests (uses shared target). Follow-up: ^kncsy3m (splice case).
    - next: /review
  timestamp: 2026-11-28T08:00:27.224829+00:00
- actor: claude-code
  id: 01m8gkafs49p6b094fnxxp7t8g
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 warnings, 0 errors); xcrun xctest per bundle, all exit 0. MLXLMTests: 2849 Swift Testing + 1261 XCTest, 0 failures. MLXGuidedGenerationTests: 96. MLXFoundationModelsTests: 744. CXGrammarTests: 7. MLXHuggingFaceMacrosTests: 11 XCTest (3 suites). 0 skipped. swift-format lint clean on all changed files (fixed one LineLength in ExecutorPromptCacheTests.swift via swift-format).
    - note: 3 known issues in MLXLMTests (AttentionKernelToleranceTests near-tie, MiniCPM5 CDATA x2 in ToolTests.swift) are deliberate withKnownIssue markers in files this change does not touch.
    - next: review/commit
  timestamp: 2026-11-28T08:11:07.428670+00:00
position_column: doing
position_ordinal: '80'
title: 'Prompt cache: the think-then-call phase of a required tool turn takes the carried cache'
---
## What
In a required-tool turn with think-then-call (`runToolCallReasoningPhase` in Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift), phase 1 generates its reasoning from a new cache of its own (`generateTokensTask`), so it prefills the whole tool-aware prompt cold on every turn. Since ^nztjxdh the grammar phase (phase 2) plans from the session's prompt cache, but phase 1 still pays a full prefill.

## Fix
- Phase 1 plans from the slot (`planPromptCache`) and commits through `withCommittedGeneration`, as a plain pass does.
- Phase 2 then plans from what phase 1 committed: its prompt (tool-aware prompt + opener + reasoning) extends phase 1's ledger up to the protocol stop; the tail after the `ProtocolPassStop` needs a rewind, which a hybrid model cannot do — take a prompt-end checkpoint in phase 1 (or stop the tail from entering the cache) so phase 2 still reuses on Qwen 3.5 / Inkling 2.
- Output must not change.

## Tests
- A required-tool turn with think-then-call after a plain turn reports cached tokens > 0 for its prompt, local and clustered, and the same tool call as a cold turn.
#cluster
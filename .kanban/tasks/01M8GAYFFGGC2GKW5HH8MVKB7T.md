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
- actor: claude-code
  id: 01m8gknqsz8ava8cgtb6cjjcq3
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (61e8fd4e) — 5 findings (5 confirmed, 2 refuted): Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:224, Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:347, Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:350, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:2100, Tests/MLXFoundationModelsTests/Qwen35MTPFixture+LanguageModel.swift:95
    - next: implement the unchecked items in the 2026-11-28 02:11 Review Findings section; task stays in review
  timestamp: 2026-11-28T08:17:16.095242+00:00
- actor: claude-code
  id: 01m8gkp7sqvh42v3pcdwar65xc
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 13 files, ThinkThenCallPromptCacheTests (5 tests × 4 cases) + 5 unit tests; follow-up ^kncsy3m
    - test: green — all 5 bundles (MLXLMTests 2849+1261, MLXFoundationModelsTests 744, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 61e8fd4e
    - review: findings — ExecutorPromptCacheEntry.swift:224, :347, :350; MLXLanguageModelGeneration.swift:2100; Qwen35MTPFixture+LanguageModel.swift:95
  timestamp: 2026-11-28T08:17:32.471254+00:00
- actor: claude-code
  id: 01m8gmbnv4c70fjc5r6vc9q7mc
  text: |-
    Review findings fixed (behaviour-preserving refactor, no new tests: existing suites cover each path):
    1. ExecutorPromptCacheEntry: one `private static func checkpoint(_:serving:)` holds the serves() filter; instructionsCheckpoint(serving:) and checkpoint(serving:) call it.
    2+3. ExecutorPromptCacheEntry: one `private func rewoundPrefixEntry(caches:tokenCount:)` owns the rewindPromptCache guard and the prefixEntry(state: nil, drafterState: nil, minimumByteCount: byteCount) build; trimmedCopy calls it with checked copies, rewound(keepingTokens:) calls it with its own caches (ExecutorPromptCacheSlot.rewindLedger still calls rewound).
    4. MLXLanguageModelGeneration: new `runProtocolPass(input:parameters:decoder:checkpointBoundaries:promptCache:speculation:context:consume:)` owns planPromptCache + withCommittedGeneration + generateProtocolTokensTask and returns the plan (@discardableResult). All three protocol passes route through it: runAllowedToolGeneration and runReasoning (decoder: protocolDecoder) and runToolCallReasoningPhase (decoder: nil; uses the returned plan for fedTokenCount). The only order change: the plan now runs after the decoder/router/collector construction in those functions, which are pure value constructions, so no behaviour change. makeParameters stays at each caller because the maxTokens source differs (requestedMaxTokens ?? default vs the phase budget). runUnconstrained was NOT routed: it generates through generateTaskRecordingTokens (a text/.chunk stream), not generateProtocolTokensTask, so sharing would change its stream type. No separate task needed: the counterpart edits are done here.
    5. Qwen35MTPFixture+LanguageModel: the existing makeLanguageModel gains `target: (any MLXLMCommon.LanguageModel)? = nil` (assigns context.model when set); the parallel overload is deleted. Callers (ThinkThenCallPromptCacheTests, PromptCacheDivergentCopyTests) compile unchanged.
  timestamp: 2026-11-28T08:29:14.980769+00:00
- actor: claude-code
  id: 01m8gmbs8ze49bsq41jsqyd81q
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift, Tests/MLXFoundationModelsTests/Qwen35MTPFixture+LanguageModel.swift; 5/5 findings checked. swift-format format + lint clean; swift build --build-tests clean (only the swiftbuild graph note "missing creator for mutated node ... mlx-swift_Cmlx.bundle", no compiler warnings). xcrun xctest, each bundle one process, exit 0: MLXFoundationModelsTests 744 tests/98 suites; MLXGuidedGenerationTests 96/23; MLXLMTests 2849 Swift Testing (3 pre-existing known issues) + 1261 XCTest, 0 failures.
    - next: /review
  timestamp: 2026-11-28T08:29:18.495327+00:00
- actor: claude-code
  id: 01m8gmsmjkw8yp2gwx059z2fvz
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (only the pre-existing build-system note "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS"); swift-format lint on the 3 changed files clean. xcrun xctest: MLXLMTests 1261 XCTest + 2849 Swift Testing, 0 failures, 0 skipped (3 pre-existing withKnownIssue: AttentionKernelToleranceTests, MiniCPM5ToolCallParserTests x2); MLXGuidedGenerationTests 96 tests; MLXFoundationModelsTests 744 tests; CXGrammarTests 7 tests; MLXHuggingFaceMacrosTests 11 XCTest + Swift Testing pass.
    - note: the first MLXLMTests run ended silently (exit 0) mid ChatSessionToolCallTests with no summary; the suite alone passed 13/13, and one full rerun with unchanged code completed with the totals above.
    - next: review/commit
  timestamp: 2026-11-28T08:36:52.435982+00:00
- actor: claude-code
  id: 01m8gmw3cap7ahkzbaawpq9kf4
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (d572e211) — 0 findings (attempted 7, failed 0, skipped 0); 3 files reviewed, 2 .kanban files matched no validator; all 5 prior findings checked
    - next: none — task moved to done
  timestamp: 2026-11-28T08:38:13.130285+00:00
- actor: claude-code
  id: 01m8gmwengkseefcbkg22ywnn7
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 3 files, all 5 findings checked
    - test: green — all 5 bundles (MLXLMTests 2849+1261, MLXFoundationModelsTests 744, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11). The first MLXLMTests run stopped silently and exited 0; the rerun completed.
    - commit: d572e211
    - review: clean — 0 findings
  timestamp: 2026-11-28T08:38:24.688064+00:00
position_column: done
position_ordinal: ffffff9380
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

## Review Findings (2026-11-28 02:11)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 12 file(s) reviewed, 5 not reviewed.

> 5 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8GAYFFGGC2GKW5HH8MVKB7T.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8GAYFFGGC2GKW5HH8MVKB7T.md` — no validator matches this file
> - `.kanban/tasks/01M8GJ55H1TRBRP8DWHKNCSY3M.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8GJ55H1TRBRP8DWHKNCSY3M.md` — no validator matches this file
> - `Libraries/MLXFoundationModels/README.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:224` `duplication/duplication` — The new `checkpoint(serving:)` accessor repeats the body of `instructionsCheckpoint(serving:)`. Both are the same `field.flatMap { $0.serves(promptTokens) ? $0 : nil }` over a different stored `PromptCacheCheckpoint?`. The two copies differ only by the field name, so the filter rule could drift between them. Extract a private helper such as `func servingCheckpoint(_ candidate: PromptCacheCheckpoint?, promptTokens: [Int]) -> PromptCacheCheckpoint?` containing the single `flatMap` filter. Have `checkpoint(serving:)` and `instructionsCheckpoint(serving:)` each call it with their own stored property.
- [x] `Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:347` `reuse/reuse` — `rewound(keepingTokens:)` reimplements the body of `trimmedCopy(keepingTokens:)` without the per-cache copy. Both guard on a rewind, call `rewindPromptCache`, and return `prefixEntry(caches:tokens:state:drafterState:minimumByteCount:)` with the same `Array(tokens.prefix(tokenCount))`, `state: nil`, `drafterState: nil`, and `minimumByteCount: byteCount` arguments. Two copies of the same rewind-and-prefix logic will drift apart the next time either one is fixed. Extract the shared tail into one private helper that takes the caches to rewind, for example `rewoundEntry(caches:tokenCount:)`. Have `trimmedCopy` call it with `try caches.map { try $0.checkedCopy() }` and `rewound` call it with `caches`. Alternatively, give `trimmedCopy` a `copying: Bool` parameter and delete `rewound`.
- [x] `Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift:350` `duplication/duplication` — `rewound(keepingTokens:)` repeats the rewind-guard and the `prefixEntry(...)` construction from `trimmedCopy(keepingTokens:)`. Both end by rewinding `caches` to `tokenCount` and building a prefix entry with `state: nil`, `drafterState: nil`, and `minimumByteCount: byteCount`. The two copies must agree on what a rewound entry carries, so a change to one can drift from the other. Extract a private helper such as `rewoundPrefixEntry(caches:tokenCount:)` that performs the `rewindPromptCache` guard and the `prefixEntry` call. Have `trimmedCopy` call it after copying the caches, and have `rewound` call it on `caches` directly.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:2100` `duplication/duplication` — The new reasoning-phase body repeats the plan, commit, and generate preamble that already exists in the plain protocol pass and the protocol pass near line 1699. The copied logic (makeParameters, planPromptCache with the same arguments, withCommittedGeneration, and generateProtocolTokensTask with plan?.input ?? input, plan?.caches, plan?.state, plan?.drafterState, and preparedState/handOff) can drift out of sync across the three sites. Only the decoder argument differs, and here it is nil. Extract a shared helper, for example `runProtocolPass(input:parameters:decoder:promptCache:checkpointBoundaries:speculation:context:consume:)`, that owns the plan, commit, and generate sequence and takes the decoder as a parameter. Call it from the new reasoning phase with `decoder: nil`. Then replace the copies in the existing passes. Those counterpart edits are outside this change, so report them as a separate task.
- [x] `Tests/MLXFoundationModelsTests/Qwen35MTPFixture+LanguageModel.swift:95` `reuse/reuse` — The new `makeLanguageModel(weights:target:capabilities:resolver:tokenizer:)` is a near-match of the existing `makeLanguageModel(weights:capabilities:resolver:eosTokenIds:tokenizer:toolCallFormat:speculativeDecoding:kvCache:)` at line 38. Both build a context with `makeLanguageContext`, both call `Self.makeLanguageModel(serving:...)`, and they differ only in the model the context serves. The existing function could take an optional `target` override instead of the new function being a parallel copy. Two functions with the same name and overlapping parameters now have to be kept in step. Add a `target: (any MLXLMCommon.LanguageModel)? = nil` parameter to the existing `makeLanguageModel` at line 38. When it is set, assign `context.model = target` after `makeLanguageContext`. Then delete the new overload at lines 95-107, or keep it as a thin forwarding call to the extended function. Callers such as `Setup.init` in ThinkThenCallPromptCacheTests.swift can keep their current arguments.

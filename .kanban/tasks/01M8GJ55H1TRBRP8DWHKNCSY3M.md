---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8gqf3168dmpzehw7mqn1183
  text: 'Research: protocol rules reach the executor only through context.configuration.toolCallFormat (MLXLanguageModel.promptCacheReuseRules(of:)); the required frame reads resolved.toolCallFormat (nil in the fixture), and .qwen35 makes no protocol decoder, so a fixture with toolCallFormat: .qwen35 changes only the reuse rules. QwenCommittedTurnRule needs convertTokenToId("<|im_end|>") and a commit after the previous render; with no "<think>" token it splices every turn. The splice only diverges from the render when the render rewrites what the model wrote, thus the test template rewrites the assistant answer. Checkpoint boundaries index the render; PromptCacheCheckpoint.prefillToCheckpoint maps a render boundary b to represented[..<b + (represented.count - render.count)], so phase 2''s transcriptEnd must shift by that delta when phase 2 reads the represented tokens (instructionsEnd stands before any spliced region). Plan: phase 1 outcome carries plan.representedTokens; phase 2 input = represented + reasoning when they differ from phase 1''s prompt (plan nil for media → current path).'
  timestamp: 2026-11-28T09:23:32.518022+00:00
- actor: claude-code
  id: 01m8grmxzwxzzbmfmt2pwm2t1q
  text: 'Implementation landed. RED: the new ThinkThenCallPromptCacheTests.theGrammarPhaseAfterASpliceFeedsNoMoreThanTheReasoning failed in all 4 combinations: the grammar phase fed 87 tokens (hybrid) and 16 tokens (attention-only) against a reasoning of 3 tokens. The tool-call comparison passed before the fix as well. GREEN: all 7 suite tests pass in 4 combinations. Fix: ExecutorPromptCachePlan.splicedPrompt and ExecutorSplicedPrompt (represented tokens, plus representing(_:), which moves transcriptEnd by represented.count - render.count and leaves instructionsEnd unchanged). ToolCallReasoningPhaseOutcome.splicedPrompt now carries the splice. The new MLXLanguageModel.Executor.grammarPhasePrompt(...) builds phase 2 from the represented tokens + reasoning after a splice, and from the render + opener + reasoning otherwise (the same as before). The shared tokenArray(_:shapedLike:) is extracted from continuationInput. Test harness: History .rendered/.spliced in ThinkThenCallPromptCacheTests. The .spliced variant uses a CommittedTurnAlphabetTokenizer (its template drops the last character of each assistant answer and appends the eos token as <|im_end|>) with toolCallFormat .qwen35. The probe and cold sessions replay answer + "a", so their render equals the tokens the warm caches represent. A note for the next agent: ExecutorSplicedPrompt/grammarPhasePrompt unit tests (ExecutorPromptCachePlanTests +5, new GrammarPhasePromptTests 5) were written after the code, against the e2e RED.'
  timestamp: 2026-11-28T09:44:12.540640+00:00
- actor: claude-code
  id: 01m8grn18h5wp9eg1k0wek9fq0
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXFoundationModels/{ExecutorPromptCache.swift, MLXLanguageModelGeneration.swift, README.md}; Tests/MLXFoundationModelsTests/{ThinkThenCallPromptCacheTests.swift (+2 tests x 4 combos), ExecutorPromptCachePlanTests.swift (+5), GrammarPhasePromptTests.swift (new, 5)}. xcrun xctest, one process per bundle, exit 0: MLXFoundationModelsTests 762 tests/99 suites; MLXGuidedGenerationTests 96; MLXLMTests 2849 Swift Testing (3 pre-existing known issues) + 1261 XCTest, 0 failures. swift-format format + lint clean on the changed Swift files.
    - next: /review
  timestamp: 2026-11-28T09:44:15.889728+00:00
- actor: claude-code
  id: 01m8gs98m8z29jet4253h0tmtz
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean. swift-format lint is clean on the changed files. xcrun xctest: MLXLMTests 1261 XCTest + 2849 Swift Testing, 0 failures (3 known issues are withKnownIssue tests in untouched files); MLXGuidedGenerationTests 96; MLXFoundationModelsTests 762; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. 0 skipped.
    - note: the tester first wrote this record on ^md0yqyr by mistake; the orchestrator moved it here.
    - next: commit, review
  timestamp: 2026-11-28T09:55:18.792411+00:00
position_column: doing
position_ordinal: '80'
title: 'Prompt cache: the grammar phase continues a reasoning phase whose plan spliced'
---
## What
Found while implementing ^8mvkb7t. In a required think-then-call turn, phase 1 (runToolCallReasoningPhase in Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift) now plans from the slot. When a protocol rule splices (QwenCommittedTurnRule, DSML, Inkling: a turn after a committed assistant turn), the ledger phase 1 commits holds the tokens the model wrote (representedTokens), not the render. Phase 2 builds its prompt from the render (toolAwareInput + opener + reasoning), so it does not extend that ledger: ExecutorPromptCacheSlot.rewindLedger(toFeedLastTokenOf:) does not apply, a hybrid model's phase-1 checkpoint (taken on the represented tokens) does not serve the render, and phase 2 falls back to the instructions checkpoint or a rebuild.

## Fix idea
When phase 1's plan represented tokens differ from its prompt, build phase 2's text input from phase 1's represented tokens + the reasoning (text-only inputs; a media input carries no plan), so phase 2 continues phase 1's caches exactly, as today's phase 2 continued a splice of the previous turn.

## Test
A plain turn, then a required think-then-call turn under a splicing protocol rule (SplicingRule in PromptCachePlanTestSupport, or a Qwen-format tokenizer): the grammar phase feeds no more than the reasoning (GuidedGenerationDiagnosticSink.fedPromptTokenCounts), hybrid and attention-only, and writes the tool call it wrote before the change.
#cluster
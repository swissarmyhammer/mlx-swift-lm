---
assignees:
- claude-code
position_column: todo
position_ordinal: '8480'
title: 'Prompt cache: the grammar phase continues a reasoning phase whose plan spliced'
---
## What
Found while implementing ^8mvkb7t. In a required think-then-call turn, phase 1 (runToolCallReasoningPhase in Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift) now plans from the slot. When a protocol rule splices (QwenCommittedTurnRule, DSML, Inkling: a turn after a committed assistant turn), the ledger phase 1 commits holds the tokens the model wrote (representedTokens), not the render. Phase 2 builds its prompt from the render (toolAwareInput + opener + reasoning), so it does not extend that ledger: ExecutorPromptCacheSlot.rewindLedger(toFeedLastTokenOf:) does not apply, a hybrid model's phase-1 checkpoint (taken on the represented tokens) does not serve the render, and phase 2 falls back to the instructions checkpoint or a rebuild.

## Fix idea
When phase 1's plan represented tokens differ from its prompt, build phase 2's text input from phase 1's represented tokens + the reasoning (text-only inputs; a media input carries no plan), so phase 2 continues phase 1's caches exactly, as today's phase 2 continued a splice of the previous turn.

## Test
A plain turn, then a required think-then-call turn under a splicing protocol rule (SplicingRule in PromptCachePlanTestSupport, or a Qwen-format tokenizer): the grammar phase feeds no more than the reasoning (GuidedGenerationDiagnosticSink.fedPromptTokenCounts), hybrid and attention-only, and writes the tool call it wrote before the change.
#cluster
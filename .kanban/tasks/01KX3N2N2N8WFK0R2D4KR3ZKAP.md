---
assignees:
- claude-code
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
- 01KX3MN39J77KZTKPEM2SDT6DJ
position_column: todo
position_ordinal: '9380'
title: prewarm(model:transcript:) populates the chunk store
---
## What
`MLXLanguageModel.prewarm(model:transcript:)` (Libraries/MLXFoundationModels/MLXLanguageModel.swift, ~line 990) currently ignores its transcript and runs a fixed dummy prompt — fine for shader JIT, useless for the cache. With the chunk store, transcript prewarming becomes genuinely valuable for the FoundationModelsRouter fork scenario: prewarming the parent transcript populates the shared prefix chunks BEFORE the first fork responds, so every fork's first turn hits.

Change prewarm to (in addition to its existing warm-up effect): tokenize the transcript via the existing TranscriptConverter path, run the text-only/chunkable checks (same gates as makePromptCacheSlot — multimodal transcripts skip), prefill through the model with a fresh cache, and store the result into the chunk store (reusing resolve→prefill→store; a prewarm of an already-cached prefix should be a cheap no-op because resolve finds the chunks and only the capped remainder prefills). Preserve fire-and-forget error behavior — prewarm must never throw or block respond().

## Acceptance Criteria
- [ ] After prewarm(transcript:), a respond() whose prompt extends that transcript reports prefix reuse on its FIRST turn (cachedTokenCount > 0 / reduced fed tokens)
- [ ] Prewarming the same transcript twice does not duplicate chunks (dedup) and the second call does near-zero prefill work
- [ ] Multimodal transcripts skip chunk population (existing isTextOnly gate) without error

## Tests
- [ ] Unit: store-population logic factored so it is testable with the probe model (chunk store populated from a token sequence + fresh cache)
- [ ] Integration (IntegrationTesting, xcodebuild): prewarm parent transcript → forked session's first respond shows cachedTokenCount > 0
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.
---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6313rzccvcbgaa5fwwx860e
  text: 'Also covers the guided-generation path (folded in from duplicate ^8hjz9t1): in ^9xe94tr the Release guided JSON benchmark drafted 0 tokens when the target loaded through loadModelContainer as a VLM with `.qwen35Text`. The fallback report must cover GuidedGenerationLoop / GuidedSpeculation as well as ChatSession and MLXLanguageModel.'
  timestamp: 2026-10-29T05:11:21.836590+00:00
position_column: todo
position_ordinal: 9d80
title: 'MTP: report when a configured drafter silently falls back to plain decode'
---
When `MLXLanguageModel` (and `ChatSession`) are given an MTP drafter that is not compatible with the loaded target (for example the generic `loadModelContainer` loads Qwen3.8 as a VLM while the drafter is the text drafter), generation quietly runs the plain iterator and proposes 0 drafts. Found during ^h91ttb2: the first integration run silently lost MTP.

The user priority is that MTP stays on and fast, so losing it must be visible.

- [ ] Surface the fallback with a stated reason (log once per model/drafter pair, and expose it in completion info / speculative telemetry the way `keptPrefixWindowReason` is exposed)
- [ ] Test: an incompatible drafter yields plain output plus the reported reason, in both ChatSession and MLXLanguageModel
- [ ] Consider making the drafter registration pick the matching text/VLM drafter for the loaded target

#mtp #foundation-models
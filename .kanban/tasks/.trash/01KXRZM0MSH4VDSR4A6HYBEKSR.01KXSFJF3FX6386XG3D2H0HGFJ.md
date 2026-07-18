---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxs1javmb2tqs2c71g53m5fd
  text: 'Correction to the description: the closing net-diff review of 9mv1q33 (942d870..HEAD) confirmed LLMModelFactory.swift''s TrampolineModelFactory is ALSO still non-final (finding LLMModelFactory.swift:609 in the 9mv1q33 15:32 section) — it was NOT addressed under bxndpt6. So `final` should be applied to the trampoline in BOTH LLMModelFactory.swift and VLMModelFactory.swift. That LLMModelFactory item is out of scope for the completed 9mv1q33 (pre-existing, net-zero in its diff) and is captured here.'
  timestamp: 2026-07-17T22:01:19.988595+00:00
position_column: todo
position_ordinal: '80'
title: 'VLMModelFactory.swift hygiene: public-API docs, final trampoline, literal + cross-factory dedup'
---
## What

Deferred from task 9mv1q33 (MiniMax-M2 tool calling), whose iteration-8 case-only rename incidentally pulled `Libraries/MLXVLM/VLMModelFactory.swift` fully into review scope, surfacing pre-existing hygiene debt unrelated to that task. Captured here rather than grinding it inside 9mv1q33. The findings come from the `## Review Findings (2026-07-17 15:32)` engine run on kanban task 01KXKVQYXJBVSMXCJS79MV1Q33 (commit e5c6da8).

## Findings to address (root-fix, file-wide)

- **Missing `///` docs on public API** — nearly every public model-config static property, the public enums `VLMError` and `VLMProcessorTypeRegistry`, `BaseProcessorConfiguration`, the public initializer, `all()`, `_load`, both public typealiases, and `TrampolineModelFactory` / `modelFactory()`.
- **`TrampolineModelFactory` should be `final`** (also flagged in LLMModelFactory — see sibling task; LLMModelFactory's was addressed under bxndpt6).
- **Repeated string/array literals** — `"Describe the image in English"` (×14), `["<|im_end|>"]` (×6), `["<turn|>"]` (×4), `""` (×3); extract named constants mirroring the LLMModelFactory pattern (defaultPrompt*, *EOSToken).
- **Cognitive complexity** — `_load` should be decomposed.
- **Acronym casing** — the EOS-loading locals (`eosTokenIds`/`genEosIds`) that were normalized in LLMModelFactory were reverted here when 9mv1q33 was scoped out; normalize them (`eosTokenIDs`/`genEOSIDs`) as part of this cleanup.

## Cross-cutting (coordinate with the ParoQuantLoader task)

The config-load / base-config-decode / model-create / EOS-token-load / mutableConfiguration blocks are near-verbatim across `LLMModelFactory`, `VLMModelFactory`, and `ParoQuantLoader`. Extract shared helpers into MLXLMCommon so all three factories share one implementation. This is the highest-leverage item; doing it may subsume several of the per-file findings above.

## Acceptance Criteria

- [ ] `swift test` green.
- [ ] A `/review` of the changed VLMModelFactory.swift returns zero findings in the doc/final/literal/complexity classes above.
- [ ] Shared factory logic deduplicated into MLXLMCommon (coordinated with the ParoQuantLoader task) or a clear rationale recorded if left per-file.
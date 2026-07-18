---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: 'ParoQuantLoader.swift hygiene: ParoQuantError docs, cognitive complexity, EOS-local casing'
---
## What

Deferred from task 9mv1q33 (MiniMax-M2 tool calling), whose iteration-8 case-only rename incidentally pulled `Libraries/MLXLMCommon/ParoQuant/ParoQuantLoader.swift` fully into review scope, surfacing pre-existing hygiene debt unrelated to that task. Captured here rather than grinding it inside 9mv1q33. Findings from the `## Review Findings (2026-07-17 15:32)` engine run on kanban task 01KXKVQYXJBVSMXCJS79MV1Q33 (commit e5c6da8).

## Findings to address (root-fix, file-wide)

- **Missing `///` doc** on public `ParoQuantError`.
- **Cognitive complexity** — `convertAutoAWQ` and `loadParoQuantModel` should be decomposed into smaller focused helpers.
- **Acronym casing on EOS-loading locals** — `ParoQuantLoader.swift:330` binds `genEOS` (and an `eosTokenIds` local); normalize to `genEOSIDs`/`eosTokenIDs`. NOTE: 9mv1q33's iter-8 sweep claimed to fix all three loaders' EOS locals but closed only 2 of 3 (LLMModelFactory + VLMModelFactory); this file was the miss, then reverted when 9mv1q33 was scoped out — so it is genuinely still `genEOS` here.

## Cross-cutting (coordinate with the VLMModelFactory task)

The config-load / base-config-decode / model-create / EOS-token-load blocks are near-verbatim across `LLMModelFactory`, `VLMModelFactory`, and `ParoQuantLoader`. Extract shared helpers into MLXLMCommon so all three factories share one implementation. Doing this may subsume the EOS-casing item here.

## Acceptance Criteria

- [ ] `swift test` green.
- [ ] A `/review` of the changed ParoQuantLoader.swift returns zero findings in the doc/complexity/casing classes above.
- [ ] EOS-loading local is `genEOSIDs` / `eosTokenIDs` (or gone, if folded into a shared helper).
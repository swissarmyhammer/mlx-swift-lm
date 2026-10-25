---
assignees:
- claude-code
depends_on:
- 01M5SKBV27GED43EB3T8MCSVWE
- 01M5SKBCEWG4D25SZZNGG7BVM7
- 01M5QWJXZ5EEBVH7Z9D3W7JKFN
- 01M5QWJ40KYCBZ9SW5CR9AH2BJ
position_column: todo
position_ordinal: '8880'
title: 'MTP: draft more than one token per round by chaining the Qwen MTP head'
---
## What
`Qwen35MTPDraftModel.maximumBlockSize = 2` (Libraries/MLXLLM/Models/Qwen35MTP.swift and Libraries/MLXVLM/Models/Qwen35MTP.swift), so each round drafts one token (measured: 1.70–1.96 tokens per verify). vLLM PR 62961 runs Qwen3.8-27B with 3 MTP tokens and reports a mean acceptance length of ~3.1. The multi-token rewind is the previous task; this task only adds the chained drafts.

`draftMTPTokenBlock` (Libraries/MLXLMCommon/QwenMTP.swift) already loops `blockSize − 1` times and chains the MTP hidden. The real limit: `draftBlock` returns `state.seedToken` (one token) every round, because `commitDrafterState` always sets a seed.

### Findings from llama.cpp (common/speculative.cpp, `common_speculative_impl_draft_mtp`)
- Chained drafts 2+ use the MTP head's own output hidden after the shared head norm. Our `Qwen35MTPPredictor` already returns `norm(hidden)`.
- Default draft count `n_max = 3`. Qwen3.6-27B, 3 drafts: acceptance 72–80%, 1.7–2.3× decode on CUDA. On a small GPU 2 drafts was the fastest. The best count depends on the verify cost.
- A confidence gate (`--spec-draft-p-min`) stops the chain when the drafter top-1 probability < p_min. Users report that p_min 0.75 makes 5 drafts the fastest setting.

- [ ] In `draftBlock` (both Qwen35MTP.swift files), return the seed plus up to `blockSize − 2` chained argmax steps from `seedHidden` through `draftMTPTokenBlock`. Stop the chain when the drafter top-1 probability < `pMin` (a new parameter of `SpeculativeDecodingConfig` and `generate(...mtpDrafter:)`, default 0). Track `proposalAppended` for each tentative drafter entry.
- [ ] In `commitDrafterState`, trim the rejected tentative entries and append the accepted ones with the target hidden rows, so that drafter == main during decode stays true.
- [ ] Raise `maximumBlockSize` to 6 and allow `blockSize` and `pMin` in `SpeculativeDecodingConfig(mtpDrafter:blockSize:)`. Set the defaults to blockSize 4 (3 drafts) and pMin 0, as llama.cpp; the benchmark below records whether another setting is faster, and a follow-up task can change the default.

## Acceptance Criteria
- [ ] Greedy output equals plain decode for blockSize 2..6, with pMin 0 and pMin 0.75, on the shared fixture (accept-mode and reject-mode drafters).
- [ ] The drafter-count invariant holds after partial acceptance and after a pMin early stop.
- [ ] Release benchmark on the M5 Max (`qwen38FourBitMTPGeneration` with blockSize 3, 4, 6 and pMin 0, 0.75; `#expect`): the best setting gives ≥ 15% more MTP tok/s on the math category than blockSize 2 with the compiled verify. The benchmark prints the acceptance for each draft depth.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: blockSize 3..6 equality, drafter-count invariant, pMin early stop.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes; then the Release benchmark.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf
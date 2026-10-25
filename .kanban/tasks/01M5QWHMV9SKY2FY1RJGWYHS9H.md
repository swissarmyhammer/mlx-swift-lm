---
assignees:
- claude-code
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5QWHA34WP7DQMWQ4AHNB0J9
position_column: todo
position_ordinal: '8480'
title: 'MTP: benchmark ChatSession multi-turn with MTP and prompt-cache reuse'
---
## What
`qwen38FourBitMTPGeneration` (IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift) measures MTP only on one cold prompt. Add a session benchmark that shows the real use: a system prompt, then follow-up turns that reuse the main and drafter caches.

- [ ] In Libraries/BenchmarkHelpers/BenchmarkHelpers.swift, add `benchmarkMTPSystemPromptCache(container:drafter:...)`, modeled on `benchmarkSystemPromptCache`. Report for each turn: the prefill time, `cachedPromptTokenCount`, the drafter tokens prefilled, decode tok/s, acceptance, and plain-session tok/s for the same turns.
- [ ] Add `qwen38FourBitMTPSystemPromptCache()` to GenerationBenchmarkTests. It skips when the target or `mlx-community/Qwen3.8-27B-MTP-4bit` is not local.
- [ ] `#expect` that each follow-up turn has `cachedPromptTokenCount > 0`, drafter prefill only over the new suffix, and text equal to the plain session.

## Acceptance Criteria
- [ ] The benchmark runs with the Release xcodebuild command in the file header and prints one summary line for each turn.
- [ ] Follow-up turns show MTP decode tok/s > plain decode tok/s on the math prompt category.

## Tests
- [ ] `xcodebuild test ... -only-testing:'IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache()'` passes (Release, as in the file header).

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #benchmark
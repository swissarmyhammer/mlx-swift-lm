---
assignees:
- claude-code
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
- 01M3A29W6YK4F0VMGBCB28DXZ2
- 01M3A1T3CK0SMNZKTTP375ZMCS
position_column: todo
position_ordinal: '8880'
title: Measure a prompt cache file write and read against a prefill, at 4k and 32k tokens, on real weights
---
#prompt-cache

(Answers question a) of the FoundationModelsRouter session, and the `evalLock` question of plan item F3.)

## What

The disk spool is useful only when a write plus a later read cost less than the prefill they save, and when neither stalls other models for too long. `save(arrays:)` holds MLX's process-wide `evalLock` for the whole write (`.build/checkouts/mlx-swift/Source/MLX/IO.swift:61-77`), and the read evaluates the lazy load, which also takes the lock.

Add a real-weights measurement suite `IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift`, in the style of `Qwen35AgenticPromptCacheAssessmentTests.swift` (measurement lines to the unified log with a fixed prefix; the suite downloads nothing).

For each model and for each context of 4 096 and 32 768 tokens:
1. Prefill the context cold; record the seconds.
2. Record `residentByteCount` of the caches (task ^375zmcs).
3. `savePromptCache` to a temporary folder; record the seconds (= the write's `evalLock` hold) and the file size.
4. `loadPromptCacheSnapshot(url:into:)` (tasks ^jvag56k, ^b28dxz2) into fresh `model.newCache(parameters:)`, which evaluates the arrays; record the seconds (= the read's lock hold).
5. One greedy decode step after the restore, and one on the original caches.

Models: `mlx-community/Qwen3-4B-4bit` (`TestFixtures.qwen3ModelID`, pure attention) and `mlx-community/Qwen3.8-27B-mxfp4` (hybrid, the checkpoint of `Qwen35AgenticPromptCacheAssessmentTests`). When a model is not in the local Hugging Face cache, the suite FAILS with a message that names the model (it does not skip: `stable` runs no skipped tests).

## Acceptance Criteria
- [ ] The suite passes on a machine that has both models.
- [ ] For each model and context, one log line holds: prefill seconds, resident bytes, file bytes, write seconds, read seconds, and the longer of the two lock holds.
- [ ] The token of the decode step after the restore equals the token on the original caches, for each model and context.
- [ ] With a model missing from the local cache, the suite fails and names the model.

## Tests
- [ ] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` compiles (`swift build` does not see `IntegrationTesting/`).
- [ ] `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` passes; read the lines with `log show --info --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`.

## Workflow
- Use `/tdd` — write the failing suite first, then make it pass.
- After the run, write the numbers as a comment on this task with a one-line verdict for each model (is write + read less than the prefill? the longest lock hold?), and send them to the FoundationModelsRouter session.
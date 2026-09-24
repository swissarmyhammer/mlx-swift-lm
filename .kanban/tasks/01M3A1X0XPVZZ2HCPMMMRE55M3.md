---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3aps4h4ssxqz9dr3nj2mat8
  text: |-
    Research notes:
    - `PromptCacheSpoolIntegrationTests.swift` is at `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/`, not at the top of the folder.
    - The xcodeproj uses file-system synchronized groups, thus a new test file needs no project edit.
    - The suite drives the prefill with the public `LanguageModel.prepare(_:cache:state:prefill:)` and one forward on the remainder, then `eval(logits, caches)`. `TokenIterator.next()` does not suit: it prefetches the next decode step into the caches.
    - `MLXLanguageModel.modelExistsOnDisk()` (internal, reached through `@testable import`) is the local-cache check. The suite throws `MissingLocalModelError`, which names the model, before any load; a third test holds that message with an absent model id.
    - Validator rules that apply here: no `print` (use `Logger`), no unnamed numeric literal except 0, 1, -1 and 100, no top-level `func`, no `guard` in a test.
  timestamp: 2026-09-24T21:57:37.444981+00:00
- actor: claude-code
  id: 01m3apw8y5n967pry3cp1jw5vh
  text: |-
    ### implement — changed
    - evidence: 1 file — IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift. `xcodebuild build-for-testing` gives TEST BUILD SUCCEEDED with no warning in the new file. `xcodebuild test -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` gives 3 tests passed in 80.75 s.
    - next: test
  timestamp: 2026-09-24T21:59:20.261471+00:00
- actor: claude-code
  id: 01m3aq4p42kf2hetvgs04myfa9
  text: |-
    ### test — red (the failures were there before this change; this change adds no failure and no warning)
    - evidence: `swift build --build-tests` — exit 0, only the known warning `missing creator for mutated node`.
    - `xcodebuild build-for-testing -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` — TEST BUILD SUCCEEDED, no warning in the new file.
    - `xcodebuild test ... -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` — 3 tests in 1 suite passed after 80.750 s.
    - MLXGuidedGenerationTests: 70 passed. MLXFoundationModelsTests: 299 passed. CXGrammarTests: 7 passed. MLXHuggingFaceMacrosTests: 5 passed.
    - MLXLMTests: 694 XCTest tests with 36 failures, and 1207 Swift Testing tests with 101 issues, the same result in 2 runs. These are the numeric tolerance failures that task ^fbhgd7k records. This change adds one file under `IntegrationTesting/`, which no SwiftPM target holds, thus the SwiftPM test products are the same as at HEAD.
    - next: /commit
  timestamp: 2026-09-24T22:03:55.906327+00:00
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
- 01M3A29W6YK4F0VMGBCB28DXZ2
- 01M3A1T3CK0SMNZKTTP375ZMCS
position_column: doing
position_ordinal: '80'
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
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
- actor: claude-code
  id: 01m3aq5xdj816va3683pd77s3b
  text: |-
    ### Measured numbers (run of 2026-09-24, commit b27151a, this machine)

    | model | context | prefill s | resident bytes | file bytes | write s | read s | longest lock hold s | next token equal |
    |---|---|---|---|---|---|---|---|---|
    | qwen3-4b (Qwen3-4B-4bit) | 4096 | 1.567 | 603979776 | 603987892 | 0.122 | 0.020 | 0.122 | yes (20775) |
    | qwen3-4b (Qwen3-4B-4bit) | 32768 | 13.085 | 4831838208 | 4831846569 | 1.187 | 0.161 | 1.187 | yes (17717) |
    | qwen3.8-27b (Qwen3.8-27B-mxfp4) | 4096 | 5.025 | 422379524 | 422396144 | 0.052 | 0.016 | 0.052 | yes (15) |
    | qwen3.8-27b (Qwen3.8-27B-mxfp4) | 32768 | 52.991 | 2301427716 | 2301444623 | 0.224 | 0.075 | 0.224 | yes (15) |

    Verdicts:
    - qwen3-4b: write + read is less than the prefill at both contexts (0.14 s against 1.57 s; 1.35 s against 13.08 s, approximately 10 percent). The longest lock hold is the 32k write, 1.19 s.
    - qwen3.8-27b: write + read is less than the prefill at both contexts (0.07 s against 5.03 s; 0.30 s against 52.99 s, less than 1 percent). The longest lock hold is the 32k write, 0.22 s.

    Note: the read comes directly after the write, thus the file is probably in the OS page cache. A read of a cold file from the SSD can be slower than the read seconds above. The file size is the resident size plus a small header; the pure-attention model stores 4 times more bytes than the hybrid model at 32k.
  timestamp: 2026-09-24T22:04:36.146716+00:00
- actor: claude-code
  id: 01m3aqa394adh59hwmz1pehs6b
  text: |-
    ### review — findings
    - evidence: 1 finding — IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift:61 `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
    - next: implement (iteration 2)
  timestamp: 2026-09-24T22:06:53.220211+00:00
- actor: claude-code
  id: 01m3aqa5j8mckybt949swx028z
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — added PromptCacheSpoolCostAssessmentTests.swift
    - test: red — the new suite passed (3 tests); swift build --build-tests exit 0 with the known warning; 4 SwiftPM bundles passed; MLXLMTests has the 36 + 101 numeric tolerance failures that ^fbhgd7k records, which this change does not touch
    - commit: b27151a
    - review: findings — PromptCacheSpoolCostAssessmentTests.swift:61 code-hygiene/magic-numbers-swift
  timestamp: 2026-09-24T22:06:55.560064+00:00
- actor: claude-code
  id: 01m3aqax825pj4249j4fjnncbq
  text: |-
    ### implement — changed
    - evidence: 1 file — IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift. The two context sizes are now the named constants `shortContextTokenCount` (4_096) and `longContextTokenCount` (32_768); `contextTokenCounts` lists the two names. No other numeric literal in the file stands outside a declaration, except the allowed `0` and `-1` of `logits[0, -1]`.
    - next: test
  timestamp: 2026-09-24T22:07:19.810637+00:00
- actor: claude-code
  id: 01m3aqe9cbgq14s25c4cwc5q7z
  text: |-
    ### test — red (the same MLXLMTests failures as iteration 1, which ^fbhgd7k records; this change adds no failure and no warning)
    - evidence: `swift build --build-tests` — Build complete, only the known warning `missing creator for mutated node`.
    - `xcodebuild test -skipPackagePluginValidation ... -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` — build with no warning in the new file; 3 tests in 1 suite passed after 82.157 s; TEST SUCCEEDED.
    - The SwiftPM bundles were not run again: this iteration changes one file under `IntegrationTesting/`, which no SwiftPM target holds, thus the products are the same as in iteration 1 (4 bundles green; MLXLMTests 36 + 101 numeric tolerance failures, ^fbhgd7k).
    - next: /commit
  timestamp: 2026-09-24T22:09:10.539624+00:00
- actor: claude-code
  id: 01m3aqfm5gym7ssjtzqpdk4gf4
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (51fe9cf) — 0 findings, 0 confirmed, 0 refuted, 7 validators attempted, 0 failed. The one prior finding (PromptCacheSpoolCostAssessmentTests.swift:61) is checked.
    - next: done
  timestamp: 2026-09-24T22:09:54.352716+00:00
- actor: claude-code
  id: 01m3aqfqrk36m9mzngh50d6mst
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — named the two context sizes (shortContextTokenCount, longContextTokenCount)
    - test: red — the suite passed again (3 tests, 82.157 s); swift build --build-tests with the known warning only; the MLXLMTests failures of ^fbhgd7k were there before this change and do not come from it
    - commit: 51fe9cf
    - review: clean — 0 findings; task moved to done

    Second run numbers (51fe9cf): qwen3-4b 4k prefill 0.738 s, write 0.080 s, read 0.021 s; 32k prefill 13.201 s, write 0.533 s, read 0.175 s. qwen3.8-27b 4k prefill 5.133 s, write 0.053 s, read 0.016 s; 32k prefill 55.695 s, write 0.240 s, read 0.075 s. The next tokens are equal at each model and context. The verdicts of the numbers comment do not change.
  timestamp: 2026-09-24T22:09:58.035171+00:00
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
- 01M3A29W6YK4F0VMGBCB28DXZ2
- 01M3A1T3CK0SMNZKTTP375ZMCS
position_column: done
position_ordinal: ff9f80
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
- [x] The suite passes on a machine that has both models.
- [x] For each model and context, one log line holds: prefill seconds, resident bytes, file bytes, write seconds, read seconds, and the longer of the two lock holds.
- [x] The token of the decode step after the restore equals the token on the original caches, for each model and context.
- [x] With a model missing from the local cache, the suite fails and names the model.

## Tests
- [x] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` compiles (`swift build` does not see `IntegrationTesting/`).
- [x] `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` passes; read the lines with `log show --info --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`.

## Workflow
- Use `/tdd` — write the failing suite first, then make it pass.
- After the run, write the numbers as a comment on this task with a one-line verdict for each model (is write + read less than the prefill? the longest lock hold?), and send them to the FoundationModelsRouter session.

## Review Findings (2026-09-24 17:04)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 1 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3A1X0XPVZZ2HCPMMMRE55M3.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3A1X0XPVZZ2HCPMMMRE55M3.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift:61` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
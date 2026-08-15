---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxegtd5s6rsah5rm9k0p00hn
  text: |-
    Implementation complete. Summary per item:

    1. OLMoE/GPTNeoX (CoherenceIntegrationTests.olmoe_1B_7B): confirmed genuine upstream gap -- swift-transformers 1.3.3's TokenizerModel.knownTokenizers has no "GPTNeoXTokenizer" entry (verified by reading the resolved swift-transformers checkout's Tokenizer.swift). Gated with `.disabled(...)` documenting this and noting BPETokenizer would likely work as a bridge if this project ever adds one.

    2/3. glm4EndToEnd / lfm2EndToEnd (ToolCallIntegrationTests): investigated by running both against real cached models.
       - GLM-4-9B-0414-4bit: produces a CORRECT tool call (right function, right complete args) but as untagged plain text ("get_weather\n{...}") because this checkpoint's own chat_template never teaches it the GLM-4.7 `<tool_call>/<arg_key>` envelope GLM4ToolCallParser expects. Real, fixable adapter/parser gap -- but the fix is a new parser (net-new feature, not a bug fix), filed as kanban 8cdyw0k. Gated.
       - LFM2-2.6B-Exp-4bit: produces a malformed/incomplete pythonic call (`properties={"location": "Tokyo"` -- wrong key, missing closing chars). Verified NOT a token-budget truncation artifact (reran with maxTokens 100->300, byte-identical output). Genuine model incapability. Gated.

    4. gemma-3-270m-it-4bit intRoundTrip: root-caused via direct run + log analysis. `ConstraintSetup.reserves(forMaxTokens:)` in MLXLanguageModel.swift computes completionReserve as max(structuralReserve*3, maxTokens/4); for a bare Int schema the maxTokens/4 floor (1024 of 4096) dominates, so the soft-zone completion bias doesn't engage until token 3072 -- letting a degenerate small model repeat a single grammar-legal digit for 75% of the budget before the bias finally forces EOS, producing a 3072-digit "integer" that overflows Swift's Int. This IS in MLXFoundationModels (in-scope), but it's the single reserve-sizing policy for the entire guided-generation/tool-calling Executor path (GenerableRoundTripTests, GuidedGenerationIntegrationTests, GuidedGenerationUsageTests, PrewarmGrammarTests, StopTokenRegressionIntegrationTests, tool-calling phase-2 budgeting, etc. all route through it), so retuning it safely needs a designed replacement + full real-model regression pass -- too broad for this task's scope. Filed as kanban t3nynaj. Split intRoundTrip's parameterized array to exclude gemma-3-270m and added a `.disabled` copy (`intRoundTripGemma3_270m`) with the full root cause documented; verified the other two models (Qwen2.5-3B, Llama-3.2-1B) still pass via the filtered array, and confirmed HardReserveStressTests is unaffected by any future fix to this function since it recomputes the formula locally rather than calling ConstraintSetup.reserves.

    Checkpoint-guard hygiene: converted all 7 checkpoint-absence `Issue.record(...) + return` guards (Gemma4AssistantIntegrationTests x3, MTPDrafterModelFactory x1 free function, Rung4TokenParityTests x3) to `.enabled(if:)` traits, matching the existing MLX_RUN_VLM_INTEGRATION idiom. Verified via real xcodebuild runs that they now report SKIPPED (not FAILED) since the 31B/33GB checkpoints are genuinely absent on this machine.

    -skip-testing: experimentally determined (not guessed) the correct identifier syntax against this exact Xcode/swift-testing toolchain: a BARE `-skip-testing:Target/SuiteName` (no `()`) reliably excludes a whole `@Suite` struct; excluding an individual test (struct method OR free function) requires a trailing `()` on the last path segment, e.g. `-skip-testing:"Target/testFoo()"` -- without it, zero tests match. Documented with a full working invocation in MTPRung4TokenParityTests.swift's header comment, cross-referenced from the other two files.

    Verification: full build-for-testing succeeded; ran all 14 touched tests together via `xcodebuild test-without-building` (checkpoint-guard tests, both gated tool-call tests, gated OLMoE test, gated+ungated guided-generation tests) -- all pass or report the documented SKIP, zero unexplained failures. No production library code touched (Libraries/MLXFoundationModels, MLXLMCommon, MLXGuidedGeneration all unchanged) -- only IntegrationTesting test files.

    Two follow-up tasks filed: 01KXEF8DHN3RTNMK4MAT3NYNAJ (t3nynaj, guided-gen reserve policy) and 01KXEG19H3D94GMD92Q8CDYW0K (8cdyw0k, GLM-4 tool-call parser gap). Adversarial double-check in progress.
  timestamp: 2026-07-13T19:56:14.393605+00:00
- actor: claude-code
  id: 01kxehy6dxpjmq0x5yr2rgskew
  text: |-
    Adversarial double-check: PASS, no findings. Verified against actual source (not just diff text): olmoe gating matches the swift-transformers 1.3.3 checkout's TokenizerModel.knownTokenizers; glm4/lfm2 gating matches the parser implementations and ToolCallFormat detection; intRoundTripModelIDs correctly excludes only gemma-3-270m-it-4bit while sibling parameterized tests (stringRoundTrip/boolRoundTrip/nestedCountConstrainedAcrossModels) keep the full unfiltered modelIDs array; the ConstraintSetup.reserves/GuidedGenerationLoop bias-zone math in the disabled reason was checked byte-for-byte against MLXLanguageModel.swift and GuidedGenerationLoop.swift; all 7 checkpoint-guard .enabled(if:)/try #require conversions use the same hfSnapshotDir(modelId:) predicate as the guard they replaced (no crash risk, no drift between gate and body); git status confirms no file under Libraries/MLXFoundationModels, Libraries/MLXLMCommon, or Libraries/MLXGuidedGeneration was touched; both follow-up kanban tasks' bodies substantively match their referencing .disabled reason strings.

    Task is green and ready for review. Leaving in doing per the implement workflow.
  timestamp: 2026-07-13T20:15:47.133091+00:00
position_column: done
position_ordinal: b580
title: 'Full integration-run follow-ups: 4 genuine failures + checkpoint-guard test hygiene'
---
## What
Full IntegrationTesting run (2026-07-13, HEAD 9118261, single worker, real models, 207 tests / 49 suites / 75 min — log: scratchpad/full-integration.log): 186 passed, 9 env-gated skips, 10 unique failing tests. Seven failures are missing-33GB-checkpoint guards (Gemma4Assistant ×3, MTPDrafterFactory ×1, MTPRung4 ×3 — outside FM scope, self-skipping by design). ALL PromptCache suites passed. The actionable remainder:

1. `CoherenceIntegrationTests.olmoe_1B_7B` — `.unsupportedTokenizer("GPTNeoXTokenizer")`: swift-transformers AutoTokenizer lacks GPTNeoX support. Either add a tokenizer bridge/mapping, pick a different OLMoE-representative model, or gate the case as explicitly-unsupported with a comment.
2. `ToolCallIntegrationTests.glm4EndToEnd` — "Expected at least one tool call, got none": GLM-4 family emitted no tool call. Investigate whether the chat template/tool envelope for glm4 is mishandled by the adapter vs. genuine model incapability; gate or fix accordingly.
3. `ToolCallIntegrationTests.lfm2EndToEnd` — "Expected string 'location' argument": LFM2 produced a tool call with malformed/missing args. Same investigation as above.
4. `MultiModelGuidedGenerationTests.intRoundTrip(modelID: gemma-3-270m-it-4bit)` — DecodingError.dataCorrupted, parsed == nil: guided generation produced unparseable output on the smallest model (other 2 parameterized cases passed). Grammar-constrained decoding should make this structurally impossible — investigate whether the constraint was bypassed/truncated (maxTokens?) for this model, which would be an adapter bug, not a model flake.

TEST HYGIENE (same task): (a) the 7 checkpoint-guard tests record Issues (= suite failures) when the 31B checkpoint is absent — convert to `.enabled(if:)` traits or XCTSkip so environment absence reads as SKIPPED, not FAILED, and CI/full-run triage stays clean. (b) `-skip-testing:IntegrationTestingTests/<Class>` did not match these XCTest classes in this run — determine the correct identifiers (worked for none of the six) and document the working invocation in the suite headers.

## Acceptance Criteria
- [ ] Each of the 4 failures either fixed (test passes) or explicitly gated with a documented reason — no silent Issue.record for environmental absence
- [ ] Checkpoint-guard suites report SKIPPED (not failed) when the 31B checkpoint is absent
- [ ] A documented, verified -skip-testing (or -only-testing) invocation for excluding the MTP/draft-model suites
- [ ] Full run (same invocation, minus gated suites) completes with zero unexplained failures

## Tests
- [ ] Re-run of the affected suites individually; full-run log attached/committed reference

## Workflow
- Use `/tdd` where a code fix is involved; investigation first for items 2-4 (adapter vs. model determination before any gating).

## Review Findings (2026-07-13 20:35)

Scope: `review sha HEAD~1..HEAD` (commit 7d95475 — "test(integration): gate/fix full-run failures, convert checkpoint guards to skip"). Engine returned 22 findings (22 confirmed by the engine's own self-check). All touched files in this commit are pre-existing IntegrationTesting test files; no production code was touched. Triaged each finding against `git diff 7d95475~1..7d95475` to confirm whether the cited lines actually fall inside this commit's diff, and against the review skill's blanket exception for refactoring/restructuring existing test code.

- [x] 12 naming findings in `CoherenceIntegrationTests.swift` (`bitnetB1582B`, `exaone401_2b`, `gemma31Bqat`, `gemma4E2b`, `glm49b`, `granite33_2b`, `granite40hTiny`, `jamba3b4bit`, `lfm21_2b`, `llama32_1b`, `olmo27b`, `olmoe1b7b`) — REJECTED. Verified via `git diff 7d95475~1..7d95475`: this commit's only change to the file is adding a `.disabled(...)` trait to the pre-existing `olmoe_1B_7B` function; none of these 12 function declarations/names were touched. Out-of-scope (pre-existing code) and covered by the blanket "no refactor of existing test code" exception.
- [x] `Gemma4AssistantDraftModelIntegrationTests.swift:14` — fixture-download helper duplication vs `MTPRung4TokenParityTests.swift` — REJECTED. `drafterForwardFixturesOrSkip` (both files) is pre-existing and untouched by this commit (diff only touches the 3 checkpoint-guard blocks later in the file); fixing requires restructuring existing test code. Blanket exception + out-of-scope.
- [x] `Gemma4AssistantDraftModelIntegrationTests.swift:72` — config-loading duplication across the 3 test methods — REJECTED. Pre-existing code, untouched by this commit's diff (only the checkpoint-guard `guard`/`Issue.record` blocks immediately above were converted to `try #require`). Blanket exception + out-of-scope.
- [x] `MTPRung4TokenParityTests.swift:18` — `mtpFixturesDirOrSkip` duplication (same pair as above) — REJECTED. Pre-existing, untouched (diff starts at line 51, well after this helper). Blanket exception + out-of-scope.
- [x] `MultiModelGuidedGenerationTests.swift:104` — new `intRoundTripGemma3_270m` duplicates `intRoundTrip`'s body — REJECTED. This is the deliberate documented-root-cause "disabled twin" pattern described in the task; both suggested fixes (extract a shared helper, or fold gemma back into the parameterized list) require restructuring the pre-existing `intRoundTrip` test. Blanket exception.
- [x] `MultiModelGuidedGenerationTests.swift:240`, `:246`, `:349`, `:357`, `:456` — bias/reserve/constraint-setup duplication across `nestedCountConstrainedAcrossModels`, `itineraryShapedSchemaOnGemma`, `constraintInitWithLargeSchema` — REJECTED. All 5 are pre-existing code outside this commit's two diff hunks (new-file lines 27-44 and 57-141 only). Out-of-scope + blanket exception.
- [x] `MultiModelGuidedGenerationTests.swift:105` — naming: new function `intRoundTripGemma3_270m` doesn't follow lowerCamelCase — FIXED. Unlike `CoherenceIntegrationTests.swift` (whose underscore convention is a pervasive pre-existing style mirroring HF model IDs, out of scope to change), this file's own sibling tests (`intRoundTrip`, `stringRoundTrip`, `boolRoundTrip`, `nestedCountConstrainedAcrossModels`, `itineraryShapedSchemaOnGemma`) are all pure lowerCamelCase with no underscores, and this function was newly added by this commit (not pre-existing). Renamed to `intRoundTripGemma3270m()` in `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/GuidedGeneration/MultiModelGuidedGenerationTests.swift` (function + its doc-comment cross-reference), and updated the one other reference to the old name in kanban task `t3nynaj`'s acceptance criteria.

Outcome: 21/22 findings rejected with documented evidence (pre-existing/out-of-diff-scope test code, covered by the blanket refactor-existing-tests exception); 1/22 fixed directly (mechanical rename of brand-new test code, no logic change). No open findings remain.
---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
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
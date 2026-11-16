---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7k9cpp336fr4n55pmrb6y4f
  text: |-
    Research:
    - `PromptCacheCheckpoint.prefill` calls `prefill.forEachChunk(total:reserving: 0, body)` with no `defaultStepSize`, so it chunks at `PrefillParameters.defaultStepSize` (512). Callers: `PromptCacheCheckpoint.prefillToCheckpoint` <- ChatSession `prefillToCheckpoint` (leading system end, then last message end) and `ExecutorPromptCachePlan.prefillingToCheckpoint` (FM executor). Both pass `parameters.prefill`, so a caller step still flows in.
    - Models with their own default step: Qwen35TextModel (LLM, `Qwen35PrefillPolicy.defaultStepSize`), Qwen35Model wraps it, Qwen35MoEModel subclasses Qwen35Model; MLXVLM.Qwen35 (Qwen35MoE subclasses it); FlintModel (`static let defaultPrefillStepSize = 2048`). Gemma3Text has its own 128 default capped at the sliding window — not named on this card; filed as follow-up.
    - `ForwardingLanguageModel` (MLXScriptedLM) promises to forward every `LanguageModel` requirement, so a new requirement must be forwarded there too.
    - Plan: add a `LanguageModel` requirement `defaultPrefillStepSize` (protocol extension default `PrefillParameters.defaultStepSize`), override in the Qwen 3.5 models and Flint, forward in ForwardingLanguageModel, and pass it as `defaultStepSize:` in the checkpoint feed. Checkpoint positions are untouched (the feed still ends exactly at the boundary, reserving 0).
  timestamp: 2026-11-16T22:59:35.747557+00:00
- actor: claude-code
  id: 01m7kahwbzsjkx3v6zxxwcrhkh
  text: |-
    Implementation landed (not committed):
    - New `LanguageModel` requirement `defaultPrefillStepSize` (protocol extension default `PrefillParameters.defaultStepSize`). `PromptCacheCheckpoint.prefill` passes it as `defaultStepSize:` to `forEachChunk`. A step the caller sets still wins, and the feed still ends exactly at the boundary (`reserving: 0`), so checkpoint positions stay at message boundaries.
    - Overrides: Qwen35TextModel and MLXVLM.Qwen35 (`Qwen35PrefillPolicy.defaultStepSize`), Qwen35Model (forwards to its text model; Qwen35MoEModel and Qwen35MoE inherit), FlintModel (the old `static let defaultPrefillStepSize = 2048` is now the instance `public let`). ForwardingLanguageModel forwards it, because that class promises to forward every requirement. The default `LLMModel.prepare` and the Qwen 3.5 and Flint prepares now read the property, so prepare and the checkpoint feed use one value.
    - Tests: Qwen35PrefillStepTests gets checkpoint feeds for the VLM, text and wrapped models ([2048] instead of [512,1024,1536,2048]; RED before the change), a test that the caller step is kept, and a ForwardingLanguageModel test. FlintTests gets a checkpoint feed test. PrefillParametersTests checks the generic default. `PrefillProgressLog.recordingPrefill(stepSize:)` (test support) replaces the progress-log setup that was repeated.

    Measurement (Qwen3.8-27B-4bit, plain ChatSession first turn, new `qwencli longsession` mode, 3 runs, median of info.promptTime, checkpoint feeds included):
    - agent_16000_0 (18248 tok): before 16.656 s (1096 tok/s), after 14.912 s (1224 tok/s), 10.5% faster
    - agent_32000_2 (29147 tok): before 29.761 s (979 tok/s), after 26.814 s (1087 tok/s), 9.9% faster
    Files: scratchpad/longsession_before_237qeb2.txt, longsession_after_237qeb2.txt, longsession.sh.

    qwencheck.sh against qwen_golden_mn44v3n.json: all 8 golden entries MATCH. No near-tie movement, so no neartie run was needed. SESSION PASS (plain and mtp, turns 2 and 3 warm: cached 139/203 tok, prefilled 32/29), TOOLCHAT plain and mtp PASS, CHECK PASS. Output: scratchpad/qwen_golden_237qeb2.json, after_237qeb2.log.

    Follow-up filed: ^37jss9t (Gemma3Text has its own 128 step plus a sliding-window cap but does not give defaultPrefillStepSize).
  timestamp: 2026-11-16T23:19:53.983191+00:00
- actor: claude-code
  id: 01m7kaj8r0a8g86yxwd0s6pxm3
  text: |-
    ### implement — changed
    - evidence: 12 files — Libraries/MLXLMCommon/LanguageModel.swift, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift, Libraries/MLXLMCommon/Qwen35PrefillPolicy.swift, Libraries/MLXLLM/LLMModel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXVLM/Models/Qwen35.swift, Libraries/MLXScriptedLM/Model/ForwardingLanguageModel.swift, Tests/MLXLMTestSupport/PrefillProgressLog.swift, Tests/MLXLMTests/Qwen35PrefillStepTests.swift, Tests/MLXLMTests/FlintTests.swift, Tests/MLXLMTests/PrefillParametersTests.swift. `swift build --build-tests` clean. xctest: MLXLMTests 2372 Swift Testing + 1134 XCTest passed; MLXGuidedGenerationTests 78; MLXFoundationModelsTests 496; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. 0 failures. Skips and known issues that already existed and are the same as the earlier baseline log xct_8h1_fix_full.log: 4 env-gated LFM2 XCTest skips, the VerifyAttentionBlockPassProfileTests suite skip, 3 known issues. qwencheck CHECK PASS (8/8 MATCH, SESSION PASS). First-turn prefill: 16.66→14.91 s (18k), 29.76→26.81 s (29k). swift-format run.
    - next: /review
  timestamp: 2026-11-16T23:20:06.656291+00:00
- actor: claude-code
  id: 01m7kapwfejqg88m5d162k6vmf
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (16bf4987) — 1 finding (1 confirmed, 0 refuted): Tests/MLXLMTests/FlintTests.swift:271 (reuse/reuse)
    - next: move the shared checkpoint-feed test helper into Tests/MLXLMTestSupport and call it from FlintTests and Qwen35PrefillStepTests, then re-review
  timestamp: 2026-11-16T23:22:37.934776+00:00
- actor: claude-code
  id: 01m7kaq73c5f4x95g3cpe7ht6v
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — defaultPrefillStepSize on LanguageModel, checkpoint feed uses it; first-turn prefill −10.5%/−9.9% at 18k/29k; qwencheck PASS
    - test: green — 5 bundles 0 failures (run by implement)
    - commit: 16bf4987
    - review: findings — Tests/MLXLMTests/FlintTests.swift:271
  timestamp: 2026-11-16T23:22:48.812802+00:00
- actor: claude-code
  id: 01m7kat6k9mqq4236jtbtdm79x
  text: |-
    Finding FlintTests.swift:271 (reuse/reuse) resolved, not committed:
    - New `PrefillProgressLog.progressOfCheckpointFeed(_:tokenCount:stepSize:)` in Tests/MLXLMTestSupport/PrefillProgressLog.swift runs one `PromptCacheCheckpoint.prefill` into a fresh cache with a recording log and returns the processed counts.
    - The feed now reads token 0 repeated (`checkpointFeedToken`) instead of a ramp modulo the vocabulary size. `LanguageModel` has no vocabulary size, and the two old copies used different moduli (`Qwen35FixtureJSON.defaultVocabularySize` and Flint's `vocabularySize`). Token 0 is valid in every vocabulary, and the chunking does not depend on token ids.
    - Deleted `Qwen35PrefillStepTests.progressOfOneChunkCheckpointFeed` and the inline setup in `FlintTests.testCheckpointFeedReadsOneMLXVLMChunkInOneForward`. All five checkpoint-feed tests call the shared helper directly.
  timestamp: 2026-11-16T23:24:26.601123+00:00
- actor: claude-code
  id: 01m7kat8v45khhdfrq1adgvhy1
  text: |-
    ### implement — changed
    - evidence: 3 files — Tests/MLXLMTestSupport/PrefillProgressLog.swift, Tests/MLXLMTests/FlintTests.swift, Tests/MLXLMTests/Qwen35PrefillStepTests.swift. swift-format run (no other files changed). `swift build --build-tests` Build complete, no compiler warnings (only the swiftbuild `missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS` build-system note). `xcrun xctest -XCTest MLXLMTests.FlintTests,MLXLMTests.Qwen35PrefillStepTests` 26 tests, 0 failures.
    - next: /review
  timestamp: 2026-11-16T23:24:28.900773+00:00
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 prefill: checkpoint feeds use the model''s prefill step'
---
## Why
Found while doing ^mn44v3n. `PromptCacheCheckpoint.prefill` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift) feeds the prompt up to each checkpoint boundary with `prefill.forEachChunk(total:reserving: 0)`, which takes the generic default step of 512 when the caller sets none. It never calls `model.prepare`, so the 2048 default that ^mn44v3n gave Qwen 3.5 `prepare` does not reach it. On a hybrid model, ChatSession and the FoundationModels executor feed nearly the whole prompt through this path (leading-system checkpoint, then last-message-end checkpoint), and only the generation-prompt tail reaches `prepare`. So multi-turn Qwen3.8 prefill still runs at step 512.

## What
- Let the checkpoint feed use the model's own default prefill step when the caller sets none (for example a `LanguageModel` requirement or a property with a default of `PrefillParameters.defaultStepSize`, overridden by Qwen 3.5 and Flint), keeping any caller-specified step.
- The checkpoint position must stay the message boundary.

## Acceptance
- [x] A unit test shows that a checkpoint feed of a Qwen 3.5 model with no caller step runs in 2048-token chunks.
- [x] `qwencheck.sh` session check passes (checkpoint restore on follow-ups); first-turn prefill time of a long ChatSession prompt recorded before and after.
- [x] All five bundles green. #qwen-perf

## Review Findings (2026-11-16 17:20)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 12 file(s) reviewed, 6 not reviewed.

> 6 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7JRKZNB0NZ2DECAJQ1YP3B1.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7JRKZNB0NZ2DECAJQ1YP3B1.md` — no validator matches this file
> - `.kanban/tasks/01M7JZ2R1EZG557QCQW237QEB2.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7JZ2R1EZG557QCQW237QEB2.md` — no validator matches this file
> - `.kanban/tasks/01M7K9V072MJY8T4PX537JSS9T.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7K9V072MJY8T4PX537JSS9T.md` — no validator matches this file

- [x] `Tests/MLXLMTests/FlintTests.swift:271` `reuse/reuse` — The new checkpoint-feed test rebuilds the same one-chunk `PromptCacheCheckpoint.prefill` call (token ramp, fresh cache, recording log) that `Qwen35PrefillStepTests.progressOfOneChunkCheckpointFeed` already implements. Two copies of this setup will drift apart, and a fix to the feed setup has to be made twice. Move the checkpoint-feed helper into Tests/MLXLMTestSupport beside `PrefillProgressLog` (for example, a function that takes a model, token count, and optional step size, and returns the processed counts). Have both `FlintTests` and `Qwen35PrefillStepTests` call it, so the feed setup lives in one place.

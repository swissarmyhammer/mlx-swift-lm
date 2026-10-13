---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rzchkqgg8566y5qk2k205c
  text: |-
    ### Research 2026-10-12 (session scratchpad dl/, Python sources)
    - This is probably most of the decode growth from 104 ms at 128 context to about 128 ms at 16k.
    - `DeepSeekV4Cache.swift:210-213, 359-366` runs `concatenated([chunks, added])` on each step, also when no chunk is complete. `DeepSeekV4Attention.swift:516-518` concatenates all pooled chunks behind the window again. The buffer size changes, thus the MLX buffer cache cannot reuse it. At 16k that is about 0.4 GB of new buffers for each step.
    - Python adds a pooled row only when a chunk is complete (`mlx_vlm .../language.py:426-473, 1011-1028`; `pr1195 deepseek_v4.py:1073-1086`).
    - Project only the new token, in BF16. Python uses float32 only in the softmax (`mlx_vlm .../language.py:390-394, 618-654`). Remove the float32 cast before `wkv` / `wgate` (`DeepSeekV4Compressor.swift:223`). This changes numbers, so record the decision.
  timestamp: 2026-10-12T21:13:05.911731+00:00
- actor: claude-code
  id: 01m4vbn89ndnmf74ejszn80ytq
  text: |-
    ### Picked up 2026-10-13, BEFORE measurement (HEAD 9038b29)
    - Bench: scratch dsv4bench, mode `step 4096,32768` (the step mode now takes an optional context list). GPU idle (Device Utilization 0) before the run.
    - ctx=4096: A direct sync 50.9 / 49.6 ms, B pipelined 49.7 / 47.8 ms, C generate 53.8 / 53.1 ms.
    - ctx=32768: A direct sync 79.0 / 77.9 ms, B pipelined 77.0 / 76.0 ms, C generate 76.0 / 76.5 ms.
    - Research: mlx-vlm `PoolingCache.accumulate_windows` (models/cache.py) keeps projected rows and pools a window only when it is full. For a ratio-4 layer it pools the full window ALONE, thus its first half is padded at each decode window and at each prefill call boundary. That does not keep our rule (pooled chunks must not depend on the cut). PR 1195 keeps the projected rows of the previous window for the overlap (`state_kv[:, :ratio]`), which keeps the rule. This port follows PR 1195 there: it keeps one whole chunk of projected rows on a ratio-4 layer.
    - Plan: the chunk cache keeps the PROJECTED rows (wkv and wgate output) in a step buffer, pools only the windows that the call completes, and writes the new chunks into a step-allocated pool buffer. The state format moves to version 2 (a version tag in the meta state). A version-1 file fails closed with a clear error, because the raw rows of version 1 cannot be projected without the compressor.
  timestamp: 2026-10-13T19:26:03.061328+00:00
- actor: claude-code
  id: 01m4ve7adf1xba3g1p354wgj3v
  text: |-
    ### Implementation landed 2026-10-13 (not committed)
    - `DeepSeekV4ChunkCache` (Libraries/MLXLLM/Models/DeepSeekV4Cache.swift) keeps the PROJECTED rows (`wkv`, `wgate` output) in a new step buffer `DeepSeekV4RowBuffer` (256-row steps, write by slice update, as `KVCacheSimple`). The pooled chunks go into a second step buffer of the same type. A call pools only the chunks that it ends (on a ratio-4 layer it starts one chunk early and drops that row, as before, to keep the overlap rule). A call that ends no chunk gives back the SAME chunks array and adds no operation to the pool. A decode step thus does: two projections and two slice updates per branch, nothing more.
    - `DeepSeekV4Compressor` (Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift): new `ProjectedRows`, `project(_:)` and `pooled(_:rope:offset:dtype:)`. `callAsFunction` is now project + pooled; one pooling path.
    - State format version 2: each branch writes 4 slots (chunks, projected values, projected gates, carry position). `DeepSeekV4Cache.metaState` ends in the tag `DeepSeekV4Cache.projectedRows.v2`. `validatePromptCacheRestore` throws `DeepSeekV4Cache.earlierFormatMessage` for a state without the tag (a version-1 file held RAW rows, which cannot become projected rows without the compressor). Fail closed, clear message.
    - 8086fda rule kept: `state` gives the held chunks view itself (evaluated with the token at the end of prefill), the row views read the whole buffers, and a buffer that holds no row is set to nil, thus no lazy buffer is left behind.
    - Compiled decode path (^rnnxh8d) not changed: the cache stage stays outside the traces. Hyper-connection transpose (e08347b) not touched.
    - New tests (DeepSeekV4CacheTests): `aStepThatEndsNoChunkGivesBackTheChunksTheCacheHeld` (both layer kinds), `theCacheKeepsTheProjectedRowsOfTheRunInProgress`, `aPromptCacheOfTheRawRowFormatFailsClosed`. Fixtures of KVCacheByteCountTests and PromptCacheTemplateRestoreTests moved to the 4-slot branch. RED: build failed on the missing `projectedRows` and `earlierFormatMessage`.

    ### Precision decision: BF16 projection (float32 cast before wkv / wgate removed)
    - `project(_:)` runs `wkv` and `wgate` in the activation dtype, as mlx-vlm does. The pooling still runs in float32 (softmax, weighted sum).
    - Greedy parity (`greedyFirstTokensMatchThePythonFixture`, real weights): PASS with BF16. Two-round recall: PASS.
    - Cost of float32 (same code, cast back in, full bench mode, generate ms/tok at ctx 32/2048/8192/32768): BF16 37.0/41.2/48.1/70.8, float32 43.0/46.9/53.3/76.4. About 5-6 ms per step.

    ### Measurements (scratch dsv4bench, GPU idle, `step 4096,32768`, direct sync / pipelined / generate)
    - BEFORE (HEAD 9038b29): 4k 50.9-49.6 / 49.7-47.8 / 53.8-53.1 ms; 32k 79.0-77.9 / 77.0-76.0 / 76.0-76.5 ms.
    - AFTER (BF16): 4k 45.4-45.9 / 44.4-44.6 / 43.8-44.1 ms; 32k 73.4-73.9 / 72.8-72.6 / 73.0-72.7 ms.
    - The step still grows with context (45.6 -> 73.6 ms). New task ^9xmds28 records the next probable cause: the attention on a ratio-4 layer reads ALL pooled chunks behind a mask, where mlx-vlm gathers the top-k.

    ### Prompt-cache check (full mode `32,2048,8192,32768 128 8192 1024`) -- warm == cold turn 2
    - HEAD 9038b29 (measured today, same command): chat NO (warm "Gulls." / cold "Gull."), thinking YES.
    - AFTER, BF16: chat NO (same "Gulls."/"Gull."), thinking NO (first difference at char 112, `"gulls" or "sea...` / `"gulls".`).
    - AFTER, float32 projection: chat YES, thinking NO (char 105, `"gulls".` / `gulls.`).
    - AFTER, BF16, context 4096 (`32 16 4096 48`): chat YES, thinking NO at char 81 (turn 1 there stops at maxTokens before `</think>`, which is task ^mf1mwaw).
    - Reuse on turn 2 is YES in every run.
    - Reading: each flip is a near tie between two tokens. The warm path computes the generated tokens one at a time and the cold path in one block, thus every matrix product of the model rounds a little differently on the two paths. HEAD already fails the chat check with this command. No precision choice makes both modes hold.

    ### Blocker
    The instruction for this step says "warm == cold must still hold in both modes". It does not hold at HEAD with that command (chat NO), and it does not hold after the change (BF16: both NO; float32: thinking NO). A person must decide: accept the BF16 result (parity holds, faster), take float32 (chat holds, thinking does not, about 5 ms slower), or define a check that does not depend on near ties.
  timestamp: 2026-10-13T20:10:52.207097+00:00
- actor: claude-code
  id: 01m4ve7hps1hd1zzr8d6xjp6qs
  text: |-
    ### implement — stuck
    - evidence: 7 files — Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift, Libraries/MLXLLM/Models/DeepSeek V4.swift header (DeepSeekV4.swift), Tests/MLXLMTests/DeepSeekV4CacheTests.swift, Tests/MLXLMTests/KVCacheByteCountTests.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift. `swift build --build-tests` clean; `xcrun xctest MLXLMTests.xctest`: 716 XCTest + 1378 Swift Testing, 0 failures. Real weights: greedy parity PASS, two-round recall PASS. Step ms (direct sync) 4k 50.3 -> 45.6, 32k 78.5 -> 73.6. Card acceptance criteria 4/4 checked.
    - gap: the step instruction "warm == cold must still hold in both modes" (full mode 8192/1024) fails: HEAD chat NO / thinking YES; after (BF16) chat NO / thinking NO; float32 variant chat YES / thinking NO. Near-tie flips; a person must choose the precision or the check.
    - next: human decision on the warm == cold gap, then /review.
  timestamp: 2026-10-13T20:10:59.673667+00:00
- actor: claude-code
  id: 01m4ve9z85vzmg8vr1d01aprbw
  text: |-
    ### Decision by the user — 2026-10-13
    Keep the BF16 projection (no float32 cast before wkv/wgate). Change the warm == cold prompt-cache check so that it does not fail on a near tie: at the first token where the warm and cold turn-2 outputs differ, compute the logits of the cold path at that position and accept the difference only when the warm token is within a measured near-tie margin of the cold top-1 (record the margin and the two logits). Cache reuse (cachedPromptTokenCount > 0, prefilled = only the new turn) stays a hard check.
  timestamp: 2026-10-13T20:12:19.077575+00:00
- actor: claude-code
  id: 01m4vet437wndaey7x70qvqhp4
  text: |-
    ### Decision 2026-10-13 (user, through the coordinator)
    - Keep BF16 projection (current code). The warm == cold check becomes tie-aware.

    ### Tie-aware check in the scratch bench (dsv4bench, main.swift + new TieCheck.swift)
    - At the first token where the warm and the cold turn-2 answers differ, one teacher-forced forward on the cold path: the cold prompt (rebuilt with `container.prepare`, token count checked against the cold run) plus the common prefix of the answer tokens. The check reads the cold top-1, the warm token and the cold token logits.
    - Margin: 2 bf16 units in the last place at the magnitude of the top-1 logit (ulp = 2^(floor(log2|x|) - 7)). Each of the two logits rounds by half a unit, and the one-token and block paths add their own rounding.
    - Verdicts: PASS (same text), FAIL (tie) when top-1 - warm <= margin, FAIL (not a tie) otherwise. Cache reuse is a hard check: cached > 0, cached + prefilled == cold prompt, prefilled <= 64.

    ### Full mode `32,2048,8192,32768 128 8192 1024`, GPU idle, current code (BF16)
    - Speeds (generate ms/tok): 36.9 / 41.1 / 48.2 / 71.9.
    - Chat: reuse PASS (8040 + 10 = 8050; cold prompt rebuilt 8050). First differing token #1: warm "ulls" (75800), cold "ull" (1422). Teacher-forced cold logits: top-1 "ulls" = 41.75, warm = 41.75, cold = 40.0. Gap 0.0, ulp 0.25, margin 0.5 -> FAIL (tie), accepted.
    - Thinking: reuse PASS (8131 + 10 = 8141; rebuilt 8141). First differing token #28: warm `"` (4), cold `".` (3305). Logits: top-1 `".` = 38.75, warm = 38.5, cold = 38.75. Gap 0.25, ulp 0.25, margin 0.5 -> FAIL (tie), accepted. Prefix argmax mismatches on the teacher-forced path: 1 of 28.
    - Note: in chat mode the cold RUN chose "ull", which the one-block teacher-forced forward puts 1.75 below its top-1. The one-block forward and the cold run (chunked prefill, then one-token steps) also differ. The check reads the warm token against the teacher-forced top-1, as specified, and that holds.
  timestamp: 2026-10-13T20:21:08.327887+00:00
- actor: claude-code
  id: 01m4vet85thb1m04xf1d2g46dq
  text: |-
    ### implement — changed
    - evidence: 7 files — Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift, Libraries/MLXLLM/Models/DeepSeekV4.swift, Tests/MLXLMTests/DeepSeekV4CacheTests.swift, Tests/MLXLMTests/KVCacheByteCountTests.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift. xctest 716 XCTest + 1378 Swift Testing, 0 failures. Greedy parity PASS, two-round recall PASS. Step direct sync 4k 50.3 -> 45.6 ms, 32k 78.5 -> 73.6 ms. Prompt-cache check (tie-aware, full mode 8192/1024): chat FAIL (tie) gap 0.0 <= margin 0.5; thinking FAIL (tie) gap 0.25 <= margin 0.5; hard reuse PASS in both modes. BF16 decision by the user.
    - next: /review. Nothing committed.
  timestamp: 2026-10-13T20:21:12.506761+00:00
- actor: claude-code
  id: 01m4vf3e4rdwzakvxq2k99v1a1
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched): only the known warning "missing creator for mutated node" (mlx-swift_Cmlx.bundle), no new warning. xcrun xctest on 5 bundles, all exit 0: CXGrammarTests 7 swift-testing tests, MLXFoundationModelsTests 367, MLXGuidedGenerationTests 70, MLXHuggingFaceMacrosTests 5 XCTest, MLXLMTests 716 XCTest + 1378 swift-testing. 0 failures, 0 skipped. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED. Real-weights integration tests not run, as ordered.
    - next: review
  timestamp: 2026-10-13T20:26:13.528252+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 speed: compressor decode state that does not pool the raw rows again on each step'
---
## What

At each decode step, the pooled cache joins up to 127-255 raw hidden rows with the new token and pools them through the compressor again: `DeepSeekV4Cache.swift:201-221`. Each step also copies all pooled chunks: `DeepSeekV4Cache.swift:359-367`. This cost grows with the context length.

mlx-lm PR 1195 keeps the projected rows and makes a chunk only when a window is full (Metal kernel `dsv4_compressor_overlap_emit`).

## Approach

- Keep the projected (not raw) rows in a buffer that grows in steps, as `KVCacheSimple` does.
- Make a new pooled chunk only when a compress window is full.
- Write into a preallocated chunk buffer, not a new concatenation on each step.
- Keep the rule of `DeepSeekV4CacheTests`: the chunks a block pools must not depend on how the block was cut.
- Keep the prompt-cache state format, or move it to a new version with a restore check.

## Acceptance Criteria

- [x] A decode step does no compressor work when no compress window is full.
- [x] `DeepSeekV4CacheTests` and the other DeepSeek-V4 unit tests pass.
- [x] A prompt cache saved before the change restores, or fails closed with a clear error.
- [x] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after, at 4k and at 32k.
#deepseek-v4 #performance
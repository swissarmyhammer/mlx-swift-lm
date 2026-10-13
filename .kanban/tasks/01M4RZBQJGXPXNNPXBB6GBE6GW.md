---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4v47pe64k333rq76yqxmqn3
  text: |-
    Research: the official DeepSeek code decides the orientation. Ours is wrong.

    Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py (revision 60d8d70770c6776ff598c94bb586a859a38244f1, the main tip on 2026-10-13), `Block.hc_post`, line 685:

        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    The product has shape [b,s,j,k,d] and the sum is over j (axis 2). Thus y[k] = sum over j of comb[j][k] * residual[j], which is `comb^T @ residual` (einsum "bsjk,bsjd->bskd").

    The layout of comb is the same in the official code and in ours: inference/kernel.py line 396 sets comb[j,k] = mixes[2*hc + j*hc + k] (row-major), and lines 401-408 take the softmax over k (the last axis). Our `DeepSeekV4Math.hcSplitFields` reshapes row-major and takes the softmax over the last axis. Thus the transpose is not already in our comb, and our `comb.matmul(residual)` in `DeepSeekV4HyperConnection.expand` reads the matrix the wrong way.

    This agrees with Apple mlx-lm deepseek_v41.py and with mlx-vlm hyper_connection.py. PR 1195 and Thump604 have our (wrong) orientation, and the old greedy fixture came from them.

    Local copies: scratchpad/official/model.py and kernel.py.
  timestamp: 2026-10-13T17:16:18.758251+00:00
- actor: claude-code
  id: 01m4v4yxy8prt3b3v01fdt76hy
  text: |-
    Implementation landed (not committed).

    - `DeepSeekV4HyperConnection.expand` now computes `comb^T @ residual` (`swappedAxes` of the two matrix axes of comb, named by a new private enum `MixingAxis`), with the official source in the doc comment. The Sinkhorn kernel is unchanged: it does the collapse only and gives `comb` in the same layout.
    - Unit tests (TDD): `theExpandAgreesWithTheOfficialReference` (renamed; new numbers from a float64 NumPy transcription of the official `Block.hc_pre`/`Block.hc_post`/`hc_split_sinkhorn_kernel`) and `theExpandTakesTheColumnOfTheMixingMatrixForEachAnswerCopy` (non-symmetric comb, expects row 0). RED: both failed, gap 0.153 and [0.5,0.125,0.25,0.125] in place of [0.5,0.25,0.125,0.125]. GREEN: MLXLMTests 716 XCTest + 1364 Swift Testing, 0 failures, the kernel parity test included. The official transcription also gives the existing collapse, post and comb fixture numbers exactly, and with `comb @ residual` it gives the old expand fixture exactly. Script: scratchpad/scripts/official_hc_fixture.py.
    - Real checkpoint, prompt ids of the parity fixture, 64 greedy tokens:
      - mlx-vlm @ 2b93e778 (transpose, op path and Metal expand kernel both read comb transposed), one full forward for each step = reference R.
      - Swift `TokenIterator` (the path of the integration test): 64 of 64 equal to R.
      - Swift full forward for each step: 37 of 64 equal to R; R has an exact top-2 tie (margin 0.0) at step 37, and at 40 and 54.
      - mlx-vlm cached S=1 decode: leaves R at step 20 (its margin there is 0.25). Thus a cached Python decode is not a fixture source.
      - R text: " We are asked to write one sentence about the sea. ... </think>The sea is a vast, mysterious". The old PR 1189 fixture text was different from token 0.
    - Fixture `IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json` regenerated from R (same prompt ids, same schema). The doc comment of the integration test states the new source, the script, and the measured tie window. Scripts: scratchpad/scripts/vlm_greedy_fixture.py, scratchpad/dsv4bench `ids` mode.
    - Next: the real-weights integration tests one at a time (greedy parity, chat and thinking, tool call, two-round recall).
  timestamp: 2026-10-13T17:29:00.104090+00:00
- actor: claude-code
  id: 01m4v56cfeh5zffbzt074rb02d
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift (doc comments only), IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json (regenerated from mlx-vlm @ 2b93e778, full forward for each step). Official source: deepseek-ai/DeepSeek-V4-Flash @ 60d8d707, inference/model.py line 685 (`Block.hc_post`), comb^T @ residual. Unit: `swift build --build-tests` + `xcrun xctest MLXLMTests.xctest` = 716 XCTest + 1364 Swift Testing, 0 failures (RED before the fix: the 2 expand tests only). Real weights, one at a time via xcodebuild test-without-building: greedyFirstTokensMatchThePythonFixture passed, chatAndThinkingModesBothGenerate passed, aShortToolPromptEmitsOneDSMLToolCall passed (get_stock_level, bay = "bay 7"), twoRoundConversationRecallsTheFirstRound passed (4172). Swift TokenIterator = reference on 64 of 64 tokens. `swift format lint` clean. Bench: 49.5 ms/tok at 32 context.
    - next: /review. Not committed.
  timestamp: 2026-10-13T17:33:04.366022+00:00
- actor: claude-code
  id: 01m4v5f5wqpvehe4q9swxgx4sc
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0; xcrun xctest on 5 bundles, all exit 0. MLXLMTests 716 XCTest and 1364 Swift Testing tests, 0 failures. MLXFoundationModelsTests 367, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 XCTest, 0 failures. 0 skipped. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED. After touching the two changed Swift files, the rebuild showed only the old warning (missing creator for mutated node, mlx-swift_Cmlx.bundle). No new warning. Real-weights tests not run, as told.
    - next: review
  timestamp: 2026-10-13T17:37:52.535019+00:00
- actor: claude-code
  id: 01m4v5htxawnpb0txp16jdz137
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (b4ac904). Counts: 0 findings, 0 confirmed, 0 refuted. 7 validator tasks attempted, 0 failed. The engine reviewed 3 files. It did not review 31 files because no validator matches them (.kanban data and the JSON fixture). The task has no earlier Review Findings section. Acceptance criteria evidence: (1) the research comment records the official source, deepseek-ai/DeepSeek-V4-Flash @ 60d8d707, inference/model.py `Block.hc_post`; (2) the implementation comment names the unit test `theExpandTakesTheColumnOfTheMixingMatrixForEachAnswerCopy` with a non-symmetric comb, RED and then GREEN; (3) the implement comment records the real-weights greedy parity test as passed against mlx-vlm @ 2b93e778 (transpose orientation), 64 of 64 tokens.
    - next: done.
  timestamp: 2026-10-13T17:39:19.594951+00:00
- actor: claude-code
  id: 01m4v5ja7aqvm95e7vmjev12bq
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — DeepSeekV4HyperConnection.swift, DeepSeekV4HyperConnectionTests.swift, DeepseekV4IntegrationTests.swift, greedy-parity fixture (from mlx-vlm 2b93e778)
    - test: green — MLXLMTests 716 XCTest + 1364 Swift Testing, 0 failures; IntegrationTesting build-for-testing succeeded
    - commit: b4ac904
    - review: clean — review sha HEAD~1..HEAD, 0 findings
  timestamp: 2026-10-13T17:39:35.274202+00:00
position_column: done
position_ordinal: ffbb80
title: 'DeepSeek-V4 correctness: check the orientation of the hyper-connection expand (comb or its transpose)'
---
## What

`Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift:296` (expand / `hc_post`) computes `comb @ residual`.

The references do not agree:
- `comb^T @ residual`, or the same as a sum over axis 2:
  - ml-explore/mlx-lm main `mlx_lm/models/deepseek_v41.py:256` (Apple)
  - Blaizzy/mlx-vlm `mlx_vlm/models/deepseek_v4/hyper_connection.py:425`
  - the official DeepSeek code
- `comb @ residual`, the same as ours:
  - mlx-lm PR 1195 `deepseek_v4.py:323`
  - Thump604/mlx-lm `deepseek-v4-support-fixes`

Blaizzy changed to the transpose in mlx-lm PR 1192 after reports of dropped tokens and repetition. See the PR 1192 comments, saved as `c1192.md:573` in the session scratchpad `dl/` folder.

Our greedy parity test (32 of 32 tokens, task ^e7b24ws) used a fixture from mlx-lm PR 1189 / Thump604, which has the same orientation as ours. Thus that test cannot find this fault.

## Approach

1. Read the official DeepSeek-V4 inference code (`hc_post`) and decide the correct orientation. Record the source and the line.
2. If ours is wrong: correct it, make a new parity fixture from a reference with the correct orientation, and run the real-weights checks: greedy text, chat and thinking, tool call, two-round recall.
3. If ours is correct: record why, with the source.

## Decision

Ours was wrong. Official source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py, `Block.hc_post`, line 685: `torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`, which is `comb^T @ residual`. The comb layout (inference/kernel.py lines 396-408) is the same as ours. The expand now reads comb transposed.

## Acceptance Criteria

- [x] The orientation is decided against the official source, and the decision and the source are recorded on this task.
- [x] A unit test pins the orientation with a non-symmetric `comb`.
- [x] If the code changes: greedy output on the real checkpoint is checked against a reference that has the correct orientation.
#deepseek-v4
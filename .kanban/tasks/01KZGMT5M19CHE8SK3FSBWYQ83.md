---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp8tb6xh0hq04w94sj10ba2
  text: |
    ## Note carried from the math-helpers task `p34crp6` (review, 2026-08-10)

    This card says of the Sinkhorn step: "call `hcSplitSinkhorn` from the math-helpers task (already tested there); this task wires it in, it does not reimplement it." That is correct with one qualification.

    **The `eps` term is not pinned by any test on `p34crp6`.** `hcSplitSinkhorn` uses `eps` in three places -- `pre = sigmoid(...) + eps`, `comb = softmax(...) + eps`, and as the division guard inside each row and column normalization. The code holds all three and agrees with the Python. But the 3x3 fixture allows 1e-6 absolute, and removing the `+ eps` on `comb` moves the result by only 8.54e-7. The fixture thus stays green with the term deleted.

    Nothing is wrong in the delivered code. The point for this card is narrower: **do not treat "eps is wired in correctly" as already proven.** The `p34crp6` mutation table lists eight mutations and none of them touches `eps`.

    Two things follow.

    - Pass `hc_eps` from the configuration, as `hcSplitSinkhorn` requires it as an argument with no default. It does not carry a copy of the value.
    - If you want the term pinned, the robust detector is not a tighter tolerance. The measured float32 error on that fixture is 8.44e-8 against the 1e-6 limit, thus a 1e-7 limit would leave only about 1.2x headroom and would be fragile. Prefer a direct test: run `hcSplitSinkhorn` twice on the same input with `eps = 0` and with `eps = 1e-6` and assert the outputs differ. That detects the term itself rather than detecting it by coincidence.
  timestamp: 2026-08-10T16:41:42.877735+00:00
- actor: claude-code
  id: 01kzre0g02m1nswz2jfq8g01wr
  text: |
    ## Research

    Read three sources before writing code.

    - Python reference: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py` -- `hc_split_sinkhorn`, `HyperConnection.hc_pre`, `HyperConnection.hc_post`, `HyperHead.__call__`, and `DeepseekV4Block.__call__` for the call order.
    - Swift reference: `osaurus-ai/vmlx-swift-lm` @ `b166896353b9c95d773de993990c20a0b5ba6905`, `Libraries/MLXLLM/Models/DeepseekV4.swift` -- `DeepseekV4HyperConnection` and `DeepseekV4HyperHead`.
    - The repo idiom: `DeepseekV4AttentionTests.swift`, `DeepseekV4MoETests.swift`, `DeepseekV4MathHelpers.swift`, `DeepseekV4MoE.swift`, `CONTRIBUTING.md`.

    ### Two places where the Swift reference and the Python disagree

    The Python decides both, the same way the `DeepseekV4Attention.swift` header decides its three.

    1. **The two epsilons are swapped on each side.** The Python holds `norm_eps` (which is `rms_norm_eps`) for the `rsqrt` of the reduction and `hc_eps` for the sigmoid and the Sinkhorn steps. The Swift reference passes `hcEps` to the `rmsNorm` of the collapse, and it sets `self.hcEps = config.rmsNormEps` in the head, thus it swaps the two on each side. The DeepSeek-V4-Flash checkpoint gives 1e-6 for both keys, thus the swap is invisible there and visible on a checkpoint whose two keys differ. This file reads each epsilon where the Python reads it.
    2. **The dtype.** The Python runs the whole collapse, the weighted sum and the expand in float32 and casts one time at the end. The Swift reference casts back to the model dtype after the RMS norm and casts `pre`, `post` and `comb` down before the sums. The comment of that file states why float32 is the answer -- the reduction runs over `hcMult * hiddenSize`, which is 16384 on Flash, and bfloat16 loses that sum -- thus it argues against its own code. This file stays in float32.

    ### The parameter names

    The Python names the three tensors `fn`, `base` and `scale`, and its `sanitize` maps `hc_head_fn` onto `model.hc_head.fn`. This file therefore reads the plain names through `@ParameterInfo(key:)`, and the Swift properties are `mixProjection`, `mixBias` and `mixScale`. The Swift reference instead keeps the `hc_head_` prefix as the parameter key of the head, which puts the tensor at `model.hc_head.hc_head_fn`. The weight-load map is task `^pwr8r3h`, thus that task chooses the mapping; the plain names are what the Python hands it.

    ### The API

    - `DeepseekV4HyperConnection.collapse(_:) -> (collapsed:post:comb:)` is `_hc_pre`.
    - `DeepseekV4HyperConnection.expand(blockOutput:residual:post:comb:)` is `_hc_post`.
    - `DeepseekV4HyperHead.callAsFunction(_:)` is `_hc_head_reduce`.

    `ManifoldStream` holds the two steps both layers take -- the mixes projection and the weighted sum over the copies -- so that the two layers do not carry one copy each.
  timestamp: 2026-08-11T12:50:56.130861+00:00
- actor: claude-code
  id: 01kzre17q55j32r23x2e55k5r4
  text: |
    ## The mutation battery

    Eleven mutations, each applied on its own, each reverted after the run. The first pass found ONE test that passed for the wrong reason; the table below is the second pass, after the gap was closed.

    | # | The mutation | The tests that died |
    |---|---|---|
    | M1 | Transpose the mixing matrix in the expand -- `comb.transposed(0, 1, 3, 2)` | `theExpandAgreesWithThePythonReference`, `theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy` |
    | M2 | Add the trailing axis of the weights on the copy axis rather than the width | `theCollapseAgreesWithThePythonReference` -- the run aborted inside it with `[broadcast_shapes] Shapes (1,2,1,4) and (1,2,4,2) cannot be broadcast` |
    | M3 | Sum the copies over the token axis | `theCollapseAgreesWithThePythonReference` |
    | M4 | Collapse with `post` in place of `pre` | `theCollapseAgreesWithThePythonReference` |
    | M5 | Read `hc_eps` in the RMS reduction | `theCollapseAgreesWithThePythonReference`, `theCollapseAnswersTheSinkhornMixingMatrix`, `theCollapseReadsTheRmsNormEpsilonOfTheCheckpoint`, `theCollapseWeighsEachCopyByItsOwnPreWeight`, `theExpandAgreesWithThePythonReference` |
    | M6 | Hard-code the Sinkhorn steps at the Flash default of 20 | `theCollapseAnswersTheSinkhornMixingMatrix`, `theCollapseReadsTheSinkhornIterationsOfTheCheckpoint`, `theExpandAgreesWithThePythonReference` |
    | M7 | Read `rms_norm_eps` in the head sigmoid | `theHeadReduceAgreesWithThePythonReference` |
    | M8 | Drop the sigmoid of the head reduce, weigh by the raw mixes | `theHeadReduceAgreesWithThePythonReference` |
    | M9 | Skip the Sinkhorn steps (in `DeepseekV4MathHelpers.swift`, reverted) | `theCollapseAnswersTheSinkhornMixingMatrix`, `theCollapseReadsTheSinkhornIterationsOfTheCheckpoint`, `theExpandAgreesWithThePythonReference`, `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` |
    | M10 | Drop the `+ eps` on the softmax of `comb` (in `DeepseekV4MathHelpers.swift`, reverted) | `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` |
    | M11 | Drop the `+ eps` on `pre` (in `DeepseekV4MathHelpers.swift`, reverted) | `theSinkhornEpsilonChangesTheMixingMatrix` |

    `DeepseekV4MathHelpers.swift` was mutated for M9 through M11 and written back after each run. `git status` shows the file unchanged.

    ### M10 survived the first pass -- what the card comment warned about

    The card comment asked for a direct test: run `hcSplitSinkhorn` twice, once with `eps = 0` and once with `eps = 1e-6`, and assert the outputs differ. That test is `theSinkhornEpsilonChangesTheMixingMatrix`, and **it does not catch a missing `+ eps` on the softmax of `comb`.** The reason is that `eps` also guards each division inside the row and column normalization. With the softmax term deleted, `eps = 0` and `eps = 1e-6` still answer two different matrices, because the guards still differ. The test stays green, and the term stays unpinned.

    The term needs an input on which it decides the answer. `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` is that input: a mixing-matrix bias whose column 0 holds -50 in every row. The softmax leaves that column at about 6e-23. The `+ eps` lifts it to `eps`, thus the first column normalization answers about a fifth for it and the Sinkhorn steps carry it to 0.2450. With the term deleted the column stays at almost nothing and the NumPy transcription answers 0.0000303. The gap is 0.272 against a limit of 1e-5.

    Both tests stay. M11 needs the eps=0/eps=1e-6 pair, and M10 needs the closed column.

    ### Why the axis alignment is guarded by values and not by shapes

    `comb` is square and doubly stochastic, thus its transpose carries the same shape, the same row sums and the same column sums. The NumPy transcription measured the transposed expand on the fixture: it is 0.153 away from the true answer at the same shape. Two tests read that gap -- the fixture parity test, and a one-hot test that feeds a residual stream holding a value in copy 0 alone and asserts the answer is column 0 of the matrix rather than row 0.

    ### The fixtures are independent

    `hc_reference.py` is a line-by-line NumPy transcription of the four Python routines, run in float64. Every `mx.*` became `np.*` and nothing else changed. No number in the test file was read out of this repository. The inputs are multiples of a sixteenth and an eighth, thus each one is exact in float32 and float64 alike, which is the idiom of `DeepseekV4AttentionTests.swift` and `DeepseekV4MoETests.swift`.
  timestamp: 2026-08-11T12:51:20.421977+00:00
- actor: claude-code
  id: 01kzre285az10010bd32dkvf3k
  text: |
    ### implement — changed
    - evidence: 2 new files, no existing file touched. `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` (`DeepseekV4HyperConnection`, `DeepseekV4HyperHead`, and the shared `ManifoldStream` steps). `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift` (16 tests). `swift test --filter DeepseekV4HyperConnectionTests`: 16 tests, 0 failures. Full `swift test`: 794 tests over 5 targets, 0 failures, 0 warnings. Mutation battery: 11 mutations, 11 killed, each with the named test recorded in the comment above. swiftlint `no_magic_numbers`, `missing_docs`, `cyclomatic_complexity` and `function_body_length` on the two new files: 0 violations. `swift-format` ran on the two new files alone.
    - next: `/review`
  timestamp: 2026-08-11T12:51:53.642345+00:00
- actor: claude-code
  id: 01kzrfape3fef1dr6jgseh1a9a
  text: |-
    ### review — findings
    - scope: `review sha HEAD~1..HEAD` (b4d3cf4)
    - evidence: 1 finding — Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift:313 (counts: findings 1, confirmed 1, refuted 16)
    - out of scope: none. The one finding is on a line this commit adds.
    - next: decide the width of the `Deepseek` to `DeepSeek` rename, then make it.
  timestamp: 2026-08-11T13:13:58.979638+00:00
- actor: claude-code
  id: 01kzrfbcbbweargx3v3b2wwaxe
  text: |-
    ### review — mutation proof of the three claims

    The review engine leaves out test files, thus it cannot say whether a test
    holds. These three checks run the mutation and read which test dies. Baseline
    before each one: `swift test --filter DeepseekV4HyperConnectionTests`, 16 tests,
    0 failures.

    **1. Axis alignment (Bug 1) — the guard holds.**
    Mutation: `comb.asType(.float32).transposed(0, 1, 3, 2).matmul(...)` in
    `expand`. Two named tests died:
    - `theExpandAgreesWithThePythonReference` (DeepseekV4HyperConnectionTests.swift:389)
    - `theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy` (DeepseekV4HyperConnectionTests.swift:432)

    The unmutated `comb.matmul(residual)` reads `(b,s,i,j) x (b,s,j,d)` to
    `(b,s,i,d)`, which is `sum over j of comb[i][j] * residual[j]`, the reading of
    `einsum("bsij,bsjd->bsid", comb, residual)`.

    **2. The epsilon test — the new test closes the hole.**
    Mutation: cut `+ eps` from the `comb` softmax in
    `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift`. Exactly one test died:
    - `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` (DeepseekV4HyperConnectionTests.swift:623)

    `theSinkhornEpsilonChangesTheMixingMatrix`, which is the test this card's own
    comment prescribed, passed through the mutation. The report of the implementer
    is correct: the prescribed test is empty against this mutation, and the new
    test is the one that holds the line.

    `DeepseekV4MathHelpers.swift` went back with `git checkout`;
    `git diff --stat` on it is empty, and `git status --porcelain -- '*.swift'` is
    empty. The suite is 16 of 16 again.

    **3. The two divergences — the port follows the Python.**
    Read from Thump604/mlx-lm @ deepseek-v4-support-fixes,
    `mlx_lm/models/deepseek_v4.py`:
    - `hc_pre` and `HyperHead.__call__` both read `self.norm_eps` in the `rsqrt`,
      and `hc_pre` hands `self.hc_eps` to `hc_split_sinkhorn`, while
      `HyperHead.__call__` adds `self.hc_eps` to the sigmoid. The Swift `collapse`
      passes `normEps` (from `rmsNormEps`) to `MLXFast.rmsNorm` and `hcEps` to
      `hcSplitSinkhorn`; the Swift head passes `normEps` to the reduction and adds
      `hcEps` to the sigmoid. Each epsilon sits where the Python puts it, thus the
      swap of the Swift reference is not in this file.
    - The Python casts to float32 one time at the top and back to the dtype one
      time at the end, in `hc_pre`, in `hc_post` and in `HyperHead.__call__`. The
      Swift does the same: `collapse` casts at `stream.asType(.float32)` and back
      at `collapsed.asType(stream.dtype)`; `expand` holds both terms in float32 and
      casts at `(newTerm + residualTerm).asType(outputDType)`; the head casts back
      at the end of `callAsFunction`. No cast down in the middle.

    One more point, which is not a divergence. The Python reads
    `mixes = (xf @ self.fn.T) * inv`, and the Swift normalizes first and projects
    after. `inv` is one number for each row, thus the two orders give the same
    answer.

    **Acceptance criterion "No changes to existing files" holds.**
    `git show --stat b4d3cf4` lists 6 files: the two new files
    (`Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift`,
    `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift`) and 4 kanban files. No
    existing source file is in the commit.
  timestamp: 2026-08-11T13:14:21.419602+00:00
- actor: claude-code
  id: 01kzrfddfvzbcbvsz54qdzx4cr
  text: |
    ### finish iteration 1 — findings
    - implement: changed — 2 new files, DeepseekV4HyperConnection.swift and DeepseekV4HyperConnectionTests.swift (16 tests); 11 mutations applied, 11 killed
    - test: green — swift test full suite, 1022 tests, 0 failures, 2 known skips
    - commit: b4d3cf4
    - review: findings — Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift:313 (type name must use the canonical spelling `DeepSeek`)
    - note: the review proved the Bug 1 guard. A transposed `comb` killed `theExpandAgreesWithThePythonReference` and `theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy`.
    - note: the eps test this card prescribed is EMPTY against a dropped `+ eps` on the `comb` softmax, because `eps` also guards the divisions of the Sinkhorn steps. The new test `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` catches that fault. Both tests stay; each one catches a different fault.
    - note: the review read the Python and confirmed the two divergences from the Swift reference. The reference swaps the two epsilons and casts down mid-path; this port follows the Python on both.
  timestamp: 2026-08-11T13:15:28.123076+00:00
- actor: claude-code
  id: 01kzrg5dgac4jwd51rhhzs9fvp
  text: |
    ## The rename, and where it stops

    The finding is corrected. Both types this file declares now carry the canonical
    spelling:

    - `DeepseekV4HyperConnection` is now `DeepSeekV4HyperConnection`.
    - `DeepseekV4HyperHead` is now `DeepSeekV4HyperHead`.

    The finding named the second one alone. The first one holds the same cause,
    thus the rename removes the cause from the whole file. The three doc-comment
    references (```` ``DeepSeekV4HyperConnection`` ```` twice and
    ```` ``DeepSeekV4HyperHead`` ```` once) follow the declarations, and the eight
    references in `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift` follow
    them as well. 14 lines changed in each direction, in 2 files, and nothing else.

    ### The mixed reading is expected, and it is temporary

    This file now declares canonical names while it still READS
    `DeepseekV4Configuration` and `DeepseekV4Math`, which other files declare. Two
    things hold that boundary:

    1. The acceptance criterion of this card says "No changes to existing files".
       The commit `b4d3cf4` touches none, and this correction touches none either.
    2. Task `^7x3dq7e` holds the repository-wide rename of the 16 remaining
       declarations under `Libraries/`, and it is ordered to run BEFORE the model
       assembly task `^pwr8r3h`. That task closes the mixed reading.

    No note about this went into the source. The source states the correct names
    and nothing more.

    ### One name this card did not authorize

    The test suite type is still `struct DeepseekV4HyperConnectionTests`, and the
    two file names still spell `Deepseek`. The card names two types and says "Stop
    there", thus this work did not touch them. Neither reason for the boundary
    covers them, however: they are in NEW files of this commit, and `^7x3dq7e`
    scopes its rename to declarations "under `Libraries/`". A comment on `^7x3dq7e`
    records the gap so that a person decides it.
  timestamp: 2026-08-11T13:28:34.570093+00:00
- actor: claude-code
  id: 01kzrg60v6c5mknhs865y2c7sz
  text: |
    ### implement — changed
    - evidence: 2 files, 14 insertions and 14 deletions. `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` (12 lines: 2 declarations and 3 doc-comment references), `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift` (16 lines: 8 references). No other file changed; `git status --porcelain -- '*.swift'` lists these two alone.
    - the finding became: `class DeepSeekV4HyperConnection: Module` and `class DeepSeekV4HyperHead: Module`. The finding named the second one; the first one held the same cause and is corrected as well.
    - tests: full `swift test`, 5 targets — 215 + 425 + 80 + 282 + 7 = 1009 tests, 0 failures, 0 compiler warnings, 2 documented skips (`CompiledDecodeCorrectnessTests`, kanban `akvqxxz`).
    - mutation re-run 1: transpose `comb` in `expand` (`comb.asType(.float32).transposed(0, 1, 3, 2).matmul(...)`). Two tests died, the two the earlier run named: `theExpandAgreesWithThePythonReference` (:389) and `theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy` (:432). The rename did not weaken the axis guard.
    - mutation re-run 2: drop `+ eps` from the `comb` softmax in `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift`. Exactly one test died: `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` (:623).
    - restore: both mutated files went back. `git diff --stat` on `DeepseekV4MathHelpers.swift` is empty, and the file is byte-identical to the copy taken before the mutation.
    - swiftlint on the two changed files: the output is line-for-line identical before and after the rename (15 violations of the swiftlint DEFAULT rules — the repository holds no `.swiftlint.yml`, and the untouched `DeepseekV4MoE.swift` and `DeepseekV4MoETests.swift` trip the same defaults). The rename added none.
    - swift-format: ran on the two changed files alone and made no change; the rename holds the same line widths.
    - next: `/review`
  timestamp: 2026-08-11T13:28:54.374874+00:00
- actor: claude-code
  id: 01kzrgnfvgbvrtqn548ktwmzyv
  text: |-
    ### review — clean

    - evidence: `review sha HEAD~1..HEAD` (d90a235) — 0 findings, 0 confirmed, 0 refuted. The prior finding of 2026-08-11 07:57 is answered and its checklist item is `- [x]`.
    - The two types of `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` now read `DeepSeekV4HyperConnection` (line 166) and `DeepSeekV4HyperHead` (line 313).
    - Files the engine could read: only `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift`. The commit holds 4 files; the engine excludes the test file, and no validator has a glob for `*.md` or `*.jsonl`. Do not read the `attempted: 9` count as a file count.
    - The rename is textual. Each removed line of the Swift diff has an added line that is the same after the two type names are made equal.
    - Mutation proof, each mutation made and then undone:
      1. Transpose of `comb` in `expand` — `theExpandAgreesWithThePythonReference` and `theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy` failed. No other test failed.
      2. `+ eps` removed from the `comb` softmax in `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` — `theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix` failed. No other test failed.
    - After the two mutations were undone, `git status --porcelain -- '*.swift'` gave no output, and `swift test --filter DeepseekV4HyperConnectionTests` passed 16 tests with 0 failures.
    - Out of scope, as the card set: the file still names `DeepseekV4Configuration` and `DeepseekV4Math`, which other files declare. 16 type declarations under `Libraries/` keep the `Deepseek` spelling. Task `^7x3dq7e` holds that repo-wide rename. The engine raised no finding against those references this round.
    - next: none. The card moves to `done`.
  timestamp: 2026-08-11T13:37:21.264173+00:00
- actor: claude-code
  id: 01kzrgp9trdh7sfffrh23wffjk
  text: |
    ### finish iteration 2 — clean
    - implement: changed — 2 files, 14 insertions and 14 deletions; the two types this file declares now read `DeepSeekV4HyperConnection` and `DeepSeekV4HyperHead`
    - test: green — swift test full suite, 1009 tests, 0 failures, 0 warnings, 2 known skips
    - commit: d90a235
    - review: clean — 0 findings, the prior finding is answered and checked, card moved to done
    - note: the scope boundary was a decision, not an oversight. The repo holds 16 more type declarations that spell `Deepseek`, and this card's acceptance criteria say "No changes to existing files". Task `^7x3dq7e` holds the repo-wide rename and must run before the model assembly task `^pwr8r3h`.
    - note: the review re-ran both mutations after the rename. The rename changed no behavior.
  timestamp: 2026-08-11T13:37:47.864459+00:00
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
position_column: done
position_ordinal: e480
title: Port DeepseekV4 mHC hyper-connections (HyperConnection, HyperHead, head reduce)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` with `DeepSeekV4HyperConnection` and `DeepSeekV4HyperHead`. This implements DSV4's **mHC (manifold-constrained hyper-connections)** residual stream — a `hc_mult=4` set of parallel residual copies that are collapsed and expanded per block using a Sinkhorn-normalized mixing matrix. Nothing analogous exists in this repo.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4HyperConnection` at line 1361, `DeepseekV4HyperHead` at line 1514.

Pieces:

1. `_hc_pre` / `_hc_post` — collapse the 4-way residual stream into the block input and expand the block output back out. The gap tracker flags **axis alignment as Bug 1**: getting the mixing axes transposed yields plausible-but-wrong activations that will not crash. Assert on shapes *and* values.
2. Sinkhorn normalization of the `comb` matrix — call `hcSplitSinkhorn` from the math-helpers task (already tested there); this task wires it in, it does not reimplement it.
3. `_hc_head_reduce` — the pre-norm mHC reduction at the top of the stack using the `hc_head_*` parameters, producing the single stream the final norm and LM head consume.

Uses `MLXFast.rmsNorm` (reference lines 912, 1554), already reachable from this target.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~1361-1568 (MIT; header attributes Osaurus AI).
- Bug 1 (axis alignment) and the mHC description: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` exists with both types.
- [x] `_hc_pre` then `_hc_post` round-trips shape: `[B, L, hidden]` to collapsed to expanded `[B, L, hc_mult, hidden]` (or whatever the reference's exact layout is — assert it explicitly). The layout is `(batch, tokens, hc_mult, hidden)` in and `(batch, tokens, hidden)` collapsed, and `theCollapseAndTheExpandKeepTheShapesOfTheWideCheckpoint` asserts each one.
- [x] A value-level test against Python-generated fixtures passes, proving axis alignment (Bug 1) is correct rather than merely shape-compatible.
- [x] `_hc_head_reduce` reduces the 4-way stream to a single `[B, L, hidden]` stream.
- [x] No changes to existing files.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift`, synthetic weights only.
- [x] Test: `_hc_pre`/`_hc_post` shapes for `hc_mult=4`, B=1, L=4, hidden=64.
- [x] Test: value parity against a checked-in Python fixture for a small deterministic input — this is the Bug 1 guard; a transposed implementation must fail it.
- [x] Test: with an identity-like `comb` matrix, `_hc_post(_hc_pre(x))` is close to `x` (sanity path).
- [x] Test: `_hc_head_reduce` output shape and `allFinite`.
- [x] Run: `swift test --filter DeepseekV4HyperConnectionTests` — all pass. 16 tests, 0 failures.

## Workflow
- Use `/tdd` — write the fixture-based value test first; it is the only thing that catches a transposed axis.
#deepseek-v4

## Review Findings (2026-08-11 07:57)

> tool rule 'code-hygiene/no-commented-code-parsed' is unavailable (tool missing: error: unrecognized subcommand 'commented_code'

Usage: sah tool code_context [OPTIONS] [COMMAND]

For more information, try '--help'.); prompt rule 'no-commented-code' ran instead.

> tool rule 'duplication/duplication-parsed' is unavailable (tool missing: error: unrecognized subcommand 'duplication'

  tip: some similar subcommands exist: 'diagnostics', 'duplicates'

Usage: sah tool code_context [OPTIONS] [COMMAND]

For more information, try '--help'.); prompt rules 'duplication', 'rust', 'swift' ran instead.

- [x] `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift:313` — Type name should use canonical spelling 'DeepSeek' (not 'Deepseek'); established mixed-case terms keep their canonical spelling per the casing rule. Rename class from `DeepseekV4HyperHead` to `DeepSeekV4HyperHead`.

### Context for the finding above

The finding names one line. The same cause is in the whole file and in the
files around it. Before you make the rename, read these facts:

- The same file declares `DeepseekV4HyperConnection` with the same spelling.
  The engine did not name that line.
- 16 type declarations under `Libraries/` carry the `Deepseek` spelling, among
  them `DeepseekV4Configuration`, which the two initializers of this file take
  as their argument, and `DeepseekV4Math`, which `collapse` calls.
- A rename of only `DeepseekV4HyperHead` makes that one type the single
  outlier of the family.

The rename crosses files that this commit does not touch, and the card holds
the criterion "No changes to existing files". Decide the width of the rename
before you start it.

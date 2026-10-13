---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4vm5kz45tkq8b91ak5rr7g4
  text: |-
    Research and baseline (HEAD 75e3af6, GPU idle).

    Profile (acceptance item 1). Tool: scratch bench mode `dsv4bench sparseops <chunks>` (scratchpad/dsv4bench/Sources/dsv4bench/SparseOps.swift). It times each op group of the cache stage of ONE ratio-4 layer at L = 1, with the real shapes (64 heads, head dim 512, window 128, index 64 x 128, top-k 512) and synthetic bf16 values. Result in sparseops-9x.txt:
    - N = 1024 chunks (4k) -> N = 8192 chunks (32k), per layer:
      - membership mask of the indexer (`ranked .== chunkIndices`, `.any`, `&& visible`): 0.55 -> 1.81 ms. Largest growth. The compare is [1, 1, 512, N].
      - visibility + `where` before `argPartition`: +0.10 -> +0.64 ms over the plain scores + argPartition.
      - joined Boolean mask [window | chunks] (ones, broadcast, concat): same as the membership mask plus a little.
      - joined keys concat (W + N rows of 512): 0.20 -> 0.22 ms.
      - SDPA over W + N keys with a Boolean mask: 0.34 -> 0.41 ms.
      - indexer score matmul over N: 0.28 -> 0.30 ms.
      - whole masked path: 0.70 -> 2.06 ms (+1.36 ms). Times 21 ratio-4 layers = +28.6 ms, which is the measured step growth from 4k to 32k (45.8 -> 74.3 ms).
      - gather path (scores, argPartition, take top-k, concat W + k, SDPA with no mask): 0.47 -> 0.56 ms.
    - Conclusion: the growth is the selection MASK work (membership compare, visibility, where) and, to a lower degree, SDPA and the copy over all N rows.

    Reference: mlx-vlm `SparseCompressedAttention._attend` gathers top-k rows when pooled > index_topk, with a split softmax for any L. mlx-lm PR 1195 gathers at S == 1, concatenates window + gathered keys, and gives SDPA `mask=None`.

    Decision on scope: the gather runs for a block of ONE token (decode) when the pool holds more than index_topk chunks and the window answers no mask. At L = 1 every pooled chunk is complete and stands behind the query, thus no visibility mask is necessary. A block of more than one token (prefill) keeps the masked path: the prefill indexer mask belongs to ^tj8ds2m.

    Baseline (HEAD), `step 4096,32768`: A direct sync 45.4/46.2 ms at 4k, 74.5/74.1 ms at 32k; C generate 44.0/44.2 and 73.6/72.2 (step-before-9x.txt). Full mode: 37.0 / 41.2 / 48.2 / 71.3 ms/token at 32 / 2k / 8k / 32k, prefill 413.7 / 387.4 / 283.5 tok/s at 2k / 8k / 32k, tie-aware cache check PASS both (full-before-9x.txt).
  timestamp: 2026-10-13T21:54:47.908175+00:00
- actor: claude-code
  id: 01m4vnqfkn3c82mbkqaak8ka7f
  text: |-
    Implementation landed (not committed).

    Change:
    - `DeepSeekV4Indexer.rankedChunks(_:queryResidual:pooledKeys:cos:sin:offset:)` (new, internal): the indices of the top-k chunks, shape (batch, tokens, topK). It builds the visibility mask only when the last chunk ends past the first query, thus at L = 1 it is scores + argPartition and nothing more. `callAsFunction` (the selection mask) reads the same ranking through the private `topChunks(of:visible:)`, so the two answers cannot drift.
    - `DeepSeekV4Attention.attentionOutput`: new branch `decodePicks(...)`. When the block holds ONE token, the window mask is `.none`, the layer holds an indexer, and the pool holds more than index_topk chunks, the layer calls the new `static func attentionOverPickedChunks(queries:windowKeys:pooled:picked:scale:sinks:)`: `takeAlong` of the picked rows, concatenation of window + topK rows, one SDPA with `mask: .none`. Every other block keeps the masked path unchanged. File header and `indexer` doc updated.
    - Kept: hyper-connection transpose, compiled decode (the cache stage stays outside traces), version-2 cache state, BF16 compressor projection, cache-state evaluation at the end of prefill. None of them was touched.

    Tests (TDD; RED was the compile failure for the missing API):
    - `DeepSeekV4AttentionTests.thePickedChunksGiveTheAnswerOfTheMaskedChunks`: batch 2, window 5, 9 pooled chunks, 3 picked chunks for each sequence (unsorted, different per sequence). The gather path equals masked SDPA over [window | all 9 chunks] within 1e-4; a premise check proves that the masked chunks change the answer.
    - `DeepSeekV4IndexerTests.theRankedChunksAreTheChunksTheSelectionMaskPicks`: batch 1 and 2, 1/4/9 tokens, offsets 3/10/40, 5 and 9 chunks (top-k 3): the visible ranked indices equal the chunks of the selection mask. Both the masked and the no-mask branch of `rankedChunks` run.
    - Existing `DeepSeekV4CacheTests.aPrefillFollowedByOneTokenStepsGivesTheAnswerOfOneCall` now runs the gather path (layer 2, 21 to 24 chunks, top-k 3) against the one-call masked path: passes at 2e-3.
    - Shared fixture builder `blockInputs` added in IndexerTests so the new test and `selection` do not copy each other.

    Results (GPU idle, one GPU job at a time):
    - `step 4096,32768` A direct sync: 4k 45.4/46.2 -> 40.4/41.0 ms; 32k 74.5/74.1 -> 42.5/43.5 ms. C generate: 4k 44.0/44.2 -> 40.7/45.0; 32k 73.6/72.2 -> 41.6/43.1 (step-before-9x.txt, step-after-9x.txt).
    - Full mode generate ms/token at 32 / 2k / 8k / 32k: 37.0 / 41.2 / 48.2 / 71.3 -> 36.9 / 38.8 / 39.8 / 41.7. Prefill tok/s unchanged (412.7 / 387.6 / 283.9 vs 413.7 / 387.4 / 283.5). Peak memory unchanged. Tie-aware cache check PASS for both modes; HARD reuse PASS (full-after-9x.txt).
    - sparseops: gather path per ratio-4 layer 0.47 ms at 1024 chunks, 0.56 ms at 8192 chunks (masked path 0.70 -> 2.06 ms).

    Greedy text vs HEAD:
    - Integration test `greedyFirstTokensMatchThePythonFixture()` passed (xcodebuild, alone, 16.2 s). Its prompt is 18 tokens, thus the pool stays below 512 chunks and the gather path does not run there.
    - Full mode text changed at 8k and 32k. New scratch mode `dsv4bench <snapshot> headtie <ctx> <head text> <new text>` (HeadTie.swift) teacher-forces the prompt + common prefix and reads the logits at the first differing token (headtie-9x.txt):
      - 8k: token #5, HEAD " long" 34.75 vs new " very" 34.0, gap 0.75.
      - 32k: token #29, HEAD "." 37.25 vs new "," 37.5, gap 0.25.
      - Both are near ties, inside cutSpreadMargin 4.0 (the cut spread that ^2sebtwba measured).
    - The 2k text and the 32-token text are the same as HEAD.

    ^f24gprg exposure: NOT reduced. The gather runs only at L = 1, where one SDPA call holds 64 x 1 x (window + 512 + 1) scores, far below 2^31 before and after. The exposure is in prefill blocks (L > 1), which keep the masked path. A prefill gather (mlx-vlm style, see the note on ^tj8ds2m) would remove it for ratio-4 layers above 512 chunks, but not for ratio-128 layers or the window part.

    Notes added on ^3q35p00 (item 1 is done for ratio-4 layers above 512 chunks; still open for ratio-128 layers and small pools) and on ^tj8ds2m (rankedChunks is available; prefill gather is an alternative to the scatter).

    Unit suite: `swift build --build-tests` clean (only the existing build-system line "missing creator for mutated node"); `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest, 0 failures; 1380 Swift Testing tests in 110 suites passed; 0 skipped. `swift format lint` on the four changed Swift files: no output.
  timestamp: 2026-10-13T22:22:01.845905+00:00
- actor: claude-code
  id: 01m4vnqm5k3rbyd4n8w8ceybt6
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Tests/MLXLMTests/DeepSeekV4AttentionTests.swift, Tests/MLXLMTests/DeepSeekV4IndexerTests.swift. Decode step A direct sync: 4k 45.8 -> 40.7 ms, 32k 74.3 -> 43.0 ms (GPU idle). Full generate: 2k 41.2 -> 38.8, 8k 48.2 -> 39.8, 32k 71.3 -> 41.7 ms/token. Greedy parity integration test passed; text changes at 8k/32k are near ties (gaps 0.75 and 0.25, margin 4.0). Unit: 716 XCTest + 1380 Swift Testing, 0 failures, 0 skipped.
    - next: /review. Task left in doing; nothing committed.
  timestamp: 2026-10-13T22:22:06.515099+00:00
- actor: claude-code
  id: 01m4vp0s1hdn4jfptmabj70vkq
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched) — only the known warning "missing creator for mutated node" (mlx-swift_Cmlx.bundle), no new warning. xcrun xctest on all 5 bundles: MLXLMTests 716 XCTest + 1380 Swift Testing, 0 failures; MLXFoundationModelsTests 367 tests passed; MLXGuidedGenerationTests 70 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed; 0 skipped. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED, 0 warnings. Real-weights tests not run.
    - next: review
  timestamp: 2026-10-13T22:27:06.417043+00:00
- actor: claude-code
  id: 01m4vp4sx94fg52cm0gv434x39
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (e988a7a). counts: 0 findings, 0 confirmed, 2 refuted, 7 attempted, 0 failed, 0 skipped. The engine reviewed 4 Swift files. It did not review 6 .kanban files because no validator matches them. The task had no prior Review Findings section. Each acceptance criterion has its evidence in the comments: profile at 32k (research comment), gather of topK rows at L = 1 and parity test pass (implementation comment), decode step before and after at 4k and 32k (research and implementation comments).
    - next: none. The task moved to done.
  timestamp: 2026-10-13T22:29:18.377648+00:00
- actor: claude-code
  id: 01m4vp59pbm6gcz0ssb10022qq
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — DeepSeekV4Attention.swift, DeepSeekV4Indexer.swift, DeepSeekV4AttentionTests.swift, DeepSeekV4IndexerTests.swift
    - test: green — MLXLMTests 716 XCTest + 1380 Swift Testing, 0 failures; IntegrationTesting build-for-testing succeeded
    - commit: e988a7a
    - review: clean — review sha HEAD~1..HEAD, 0 findings
  timestamp: 2026-10-13T22:29:34.539411+00:00
position_column: done
position_ordinal: ffc080
title: 'DeepSeek-V4 speed: sparse attention gathers the top-k pooled chunks, not all chunks behind a mask'
---
## What

After task ^cm8fq2q, a decode step does no compressor work when no compress window is full. The decode step still grows with the context: 45.6 ms at 4k and 73.6 ms at 32k (scratch bench `step`, direct sync, GPU idle). mlx-vlm on the same machine is flat at about 31 ms.

On a ratio-4 layer with an indexer, `DeepSeekV4Attention.attentionOutput` concatenates the window keys with ALL pooled chunks (8192 chunks at 32k) and gives SDPA a joined Boolean mask of the same width (`chunkVisibility`, `joinedMask`, `keys`). mlx-vlm (`SparseCompressedAttention._attend`, `_sparse_pooled_attention` in `deepseek_v4/language.py`) gathers only the `index_topk` chunks the indexer selected when the pool holds more than `index_topk` chunks, and splits the softmax between the window and the gathered chunks.

This is a hypothesis from the code. It is not yet measured with a profile.

## Approach

- Profile one decode step at 32k (scratch bench `profile`) and confirm which primitives grow with the chunk count.
- Gather the top-k pooled chunks for each query when the pool holds more than `index_topk` chunks, as mlx-vlm does.
- Keep the greedy parity with the mlx-vlm fixture and the cut rule of `DeepSeekV4CacheTests`.

## Acceptance Criteria

- [x] A profile at 32k names the primitives that grow with the chunk count.
- [x] A decode step at 32k reads at most `index_topk` pooled chunks on a layer with an indexer.
- [x] The 64-token greedy parity integration test passes.
- [x] A decode-step measurement with the GPU otherwise idle shows the speed before and after, at 4k and at 32k. #deepseek-v4 #performance
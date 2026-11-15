---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7g17yaza8nee7nhj1g7k2th
  text: |-
    Research (implement):
    - MLXFast SDPA has no windowed/offset-causal mode: masks are `.causal` (bottom-right aligned, kills only the upper triangle) or `.array`. Steel attention (head dim 128 -> fused kernel, sinks supported) skips key blocks only under do_causal; with an array mask every key tile of 2048 x (512+2048) runs. So `.causal` + key slicing alone cannot express the lower window edge for a block of >1 query rows; a small array mask per band is still needed.
    - Microbench (scratchpad sdpabench, bf16, 32 q heads / 4 kv heads / D 128 / W 513, sinks, M5): L 2048 P 512: array mask 3.42 ms, plain causal (wrong) 1.64 ms, banded batched SDPA Q=128 1.46, Q=256 0.85, Q=512 1.00, Q=1024 1.27 ms. L 2048 P 0: 1.20 vs Q=256 0.79. L 1024 P 512: 0.61 vs Q=256 0.54. Banded output matched the array-mask SDPA bit for bit in bf16. Q = window/2 (256) is the best block.
    - Design: stack the keys of each band of Q query rows (Q + W - 1 keys each, zero-padded at the ends, padding masked) into the batch axis, one SDPA call with one shared [blocks, 1, Q, Q+W-1] mask, sinks passed as before. Flint sliding caches are always RotatingKVCache with keep 0, whose multi-token mask is createCausalMask(n: L, offset: T - L, windowSize: W) over the T rows the update presents, so the band geometry follows from the mask shape [L, T].
  timestamp: 2026-11-15T16:39:27.839865+00:00
- actor: claude-code
  id: 01m7g20ek3v71r4mhdax46ep3k
  text: |-
    Implementation landed (uncommitted):
    - New Libraries/MLXLMCommon/SlidingWindowBands.swift (package struct): blocks of window/2 query rows (256 at W 513); each block's Q+W-1 keys are stacked on the batch axis (zero-padded at the ends, padding masked), one SDPA call with one [blocks,1,Q,Q+W-1] bool mask, sinks unchanged. nil when the queries fit in < 2 blocks.
    - Flint.swift: FlintAttentionMask { sdpa(mode), bands(SlidingWindowBands) }; FlintModelInner.attentionMasks(h:cache:) picks bands for a sliding layer when createAttentionMask gives .array and the cache is nil or a RotatingKVCache with no kept prefix (geometry taken from the mask shape [L, T]); the layer then calls cache.update and the bands instead of attentionWithCacheUpdate.
    - Tests: Tests/MLXLMTests/SlidingWindowBandsTests.swift (4 shapes x 3 dtypes with sinks, without sinks, shape, no-bands for one block, window 1); FlintTests: bands chosen for a multi-block prompt, not for one decode token, wrapped chunked prefill (step 5, window 4) matches token-by-token decode.
    - Real model: f32 (MLX_ENABLE_TF32=0, oQ4e, long_ids 1700 tokens, cold one-shot and 512-chunked through the wrapped rotating cache) vs HEAD Flint.swift built in the same session: max|d| 0.0 on both. oQ4e flintcheck MATCH x4 + TOOL CALL; oQ8e greedy MATCH golden8_fused.json x4. pfbench same session 5985/7909/7319 -> 6561/8895/8098; flintbench prefill 8960/8970/8189 tok/s, decode 6.34/6.68/7.05 ms/tok. ChatSession: oQ4e turn 2 prefilled 902 new tokens after 1213 cached (wrapped), turn 3 12 tokens after 2465; oQ8e 900 after 713, answer correct.
  timestamp: 2026-11-15T16:52:50.915956+00:00
- actor: claude-code
  id: 01m7g27fa5qh7hjb4gce83dqxn
  text: |-
    ### implement — changed
    - evidence: files Libraries/MLXLMCommon/SlidingWindowBands.swift (new), Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/SlidingWindowBandsTests.swift (new), Tests/MLXLMTests/FlintTests.swift. pfbench (@prompt_706, same session) 2k/8k/16k 5985/7909/7319 -> 6561/8895/8098 tok/s; flintbench prefill 8229/7962/7457 -> 8960/8970/8189 tok/s, decode 6.34/6.68/7.05 ms/tok (was 6.34/6.70/7.09). f32 parity (MLX_ENABLE_TF32=0, oQ4e, 1700 tokens cold + 512-chunked through the wrapped cache, vs HEAD built in the same session): max|d| 0.0. flintcheck MATCH x4 + TOOL CALL get_weather; oQ8e greedy MATCH golden8_fused.json x4. ChatSession follow-ups prefill only the new message after the wrap (oQ4e 902 new / 1213 cached, then 12 / 2465; oQ8e 900 / 713). Tests: MLXLMTests 2344 Swift Testing passed (3 known issues) + 1098 XCTest (4 LFM2 env skips, 0 failures); MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. Only build warning is the known upstream "missing creator for mutated node". swift-format run; nothing committed.
    - next: /review
  timestamp: 2026-11-15T16:56:41.029829+00:00
- actor: claude-code
  id: 01m7g2qqsgbw272fp7h25ash1r
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (no warnings besides the pre-existing SwiftPM "missing creator" notice); swift-format lint -s clean on the 4 changed files. xcrun xctest: MLXLMTests 1098 XCTest (4 skipped = LFM2 env baseline, 0 failures) + Swift Testing 2344 tests in 223 suites passed, 3 known issues (baseline); GDNDecodeKernelLayerTests passed. MLXGuidedGenerationTests 76 passed. MLXFoundationModelsTests 494 passed. CXGrammarTests 7 passed. MLXHuggingFaceMacrosTests 11 passed, 0 failures. flintcheck.sh: MATCH x4 (256 greedy ids identical each) and a TOOL CALL get_weather line; golden_base.json untouched.
    - next: review
  timestamp: 2026-11-15T17:05:34.000743+00:00
- actor: claude-code
  id: 01m7g2rnkg36g80m3swj6z8vvb
  text: |-
    ### commit — changed
    - evidence: perf(flint): attend over sliding-window bands at prefill instead of a full array mask (^v6cgcp6); files: Libraries/MLXLMCommon/SlidingWindowBands.swift, Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/SlidingWindowBandsTests.swift, Tests/MLXLMTests/FlintTests.swift, .kanban/ (sha recorded in the commit step record)
    - next: review
  timestamp: 2026-11-15T17:06:04.528945+00:00
- actor: claude-code
  id: 01m7g2y3ta37we2m2bwsaekc12
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (6535f367); 1 finding — Libraries/MLXLMCommon/SlidingWindowBands.swift:220 (code-hygiene/magic-numbers-swift)
    - next: implement the open finding (replace the magic number with a named constant, and every other magic number of that cause in the file), then re-review
  timestamp: 2026-11-15T17:09:02.922031+00:00
- actor: claude-code
  id: 01m7g2yjjw9jns6wmap60ej6fk
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — SlidingWindowBands.swift (new), Flint.swift, SlidingWindowBandsTests.swift (new), FlintTests.swift
    - test: green — MLXLMTests 1098 XCTest + 2344 ST, Guided 76, FM 494, CXGrammar 7, Macros 11; flintcheck MATCH x4 + TOOL CALL
    - commit: 6535f367
    - review: findings — Libraries/MLXLMCommon/SlidingWindowBands.swift:220 (magic-numbers-swift)
  timestamp: 2026-11-15T17:09:18.044642+00:00
- actor: claude-code
  id: 01m7g3h36ry09eh5hfjkc3tdax
  text: 'Finding fix: the review finding `Libraries/MLXLMCommon/SlidingWindowBands.swift:220` `code-hygiene/magic-numbers-swift` ("Magic numbers should be replaced by named constants") was the axis permutation `[0, 2, 1, 3, 4]`. BandAxis now names all five axes of the block-major layout (batch 0, block 1, head 2, row 3, width 4), and `swapHeadAndBlock` is `[batch, head, block, row, width]`, the same permutation. The rule''s own tool (swiftlint 0.65.1 `no_magic_numbers`, allowed [0, 1, -1, 100]) now reports nothing in SlidingWindowBands.swift, SlidingWindowBandsTests.swift or FlintTests.swift. In Flint.swift it still reports 5 literals, at lines 88, 90, 91, 112 and 307 (the defaults 1e-6, 262_144, 10_000, 513, and pow -0.5). git blame dates all five to e7e3ed84, which came before this task. None of them is in a line this task added, so I left them alone. The case shapes in SlidingWindowBandsTests.bandCases are values of a named `static let` declaration. Both the rule''s carve-out and the tool exempt those.'
  timestamp: 2026-11-15T17:19:24.888565+00:00
- actor: claude-code
  id: 01m7g3h6bsqfdyqt5680mjsry3
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLMCommon/SlidingWindowBands.swift (BandAxis named axes; finding flipped to [x]). swiftlint no_magic_numbers is clean on the 3 task files. swift build --build-tests is clean (only the known "missing creator" warning). swift-format was run. xctest: MLXLMTests 1098 XCTest (4 LFM2 skips, 0 failures) + 2344 Swift Testing passed (3 known issues); MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11, 0 failures. flintcheck MATCH x4 + TOOL CALL get_weather; golden_base.json untouched. Nothing committed.
    - next: /review
  timestamp: 2026-11-15T17:19:28.121482+00:00
- actor: claude-code
  id: 01m7g3zrrs2tvc5c7b9qkqhd40
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (only the SwiftPM "missing creator for mutated node" build-system note, no compiler warnings). xcrun xctest per bundle: MLXLMTests 1098 XCTest (0 failures, 4 skipped = the 4 LFM2 env skips) + Swift Testing 2344 tests in 223 suites passed, 3 known issues (baseline), GDNDecodeKernelLayerTests passed; MLXGuidedGenerationTests 76 tests in 17 suites passed; MLXFoundationModelsTests 494 tests in 66 suites passed; CXGrammarTests 7 tests in 5 suites passed; MLXHuggingFaceMacrosTests 11 XCTest, 0 failures. swift-format lint -s on Libraries/MLXLMCommon/SlidingWindowBands.swift clean. flintcheck.sh: MATCH x4 (256 greedy ids identical each) and TOOL CALL get_weather line. golden_base.json untouched.
    - next: review
  timestamp: 2026-11-15T17:27:25.721346+00:00
- actor: claude-code
  id: 01m7g40h22rm8skvjwg1jx03ds
  text: |-
    ### commit — changed
    - evidence: commit subject `refactor(sliding-window): name the band axes in the permutation, found in review of ^v6cgcp6`; files Libraries/MLXLMCommon/SlidingWindowBands.swift and .kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.{md,jsonl}. swift-format made no additional changes.
    - next: none (local commit, not pushed)
  timestamp: 2026-11-15T17:27:50.594982+00:00
position_column: doing
position_ordinal: '80'
title: 'Flint: banded sliding-window prefill attention (the array mask costs ~8-9% of prefill)'
---
## Why
Found in ^tty395a. At 2048-token prefill chunks on oQ4e, an ablation that gave the sliding layers the plain causal mask (numerically wrong, timing only) raised prefill from 8021 to 8762 tok/s at 8k and from 7312 to 7911 at 16k. 35 of 49 Flint layers slide over a window of 513. Each sliding layer of a 2048-token chunk runs SDPA over 2048 x (512 + 2048) scores with an explicit array mask, so no key block is skipped, although each query needs only 513 keys. mlx-vlm 0.7.12 pays the same cost (RotatingKVCache.make_mask array), so this goes beyond the reference.

## What
- Measure first: a Metal trace of one prefill chunk, sliding SDPA vs full SDPA.
- Try splitting the query rows of a sliding layer into sub-blocks, each attending only to the keys of its own window (a smaller array mask per sub-block), or a kernel route that skips key tiles outside the band.
- Keep the attention sinks.

## Acceptance
- Prefill tok/s at 2k, 8k and 16k improves against a same-session baseline (scratchpad pfbench.sh with `@prompt_706.txt`).
- Greedy ids of flintcheck MATCH x4; float32 logit parity holds.
- Decode does not regress.
- All five bundles are green. #flint-perf

## Review Findings (2026-11-15 11:06)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.md` — no validator matches this file
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/SlidingWindowBands.swift:220` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.

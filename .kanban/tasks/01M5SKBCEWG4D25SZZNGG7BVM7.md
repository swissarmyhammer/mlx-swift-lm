---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m614rq61qptbd6ab2acd08zz
  text: |-
    Picked up (M5 Ultra; baseline per ^v51tpya/^cst7wan: Release qwen38FourBitMTPGeneration MTP ~89-91 / 80 / 78-79 tok/s math/code/prose, plain ~59). Research notes:
    - Verify today: MTPSpeculativeTokenIterator sets emit + mtpCacheCheckpointIndexKey = 1 (Qwen maximumNativeTargetCacheRewind 1, blockSize 2, so S = 2). Qwen35TextModel.forward emit branch calls model.forward(applyFinalNorm: false, checkpointAfter:) then model.norm; the inner forward only takes compiled segments for S == 1 && applyFinalNorm, so the verify runs the general per-layer path.
    - The S = 1 schedule (CompiledDecodeSegment / CompiledDecodeSegmentCache) calls Qwen35GatedDeltaNet.forward inside the trace, which already picks the GDN Metal kernel; custom Metal kernels work inside MLX compile. The verify can reuse the same segments and layer bodies with S rows, plus 2 extra outputs per GDN layer for the speculative states (snapshots or op-path checkpoint).
    - Inside a trace the GDN layer falls back to the op path (two-pass split at checkpointAfter) when the kernel cannot run (float32 fixtures, small heads). That fallback bakes checkpointAfter into the graph, so the verify traces are keyed by (tokens, checkpointAfter) and the kind of speculative state is read back from the rank of its recurrent array (snapshots carry one more axis).
    - A Dictionary of segment caches stored on a Module would not be reached by invalidateCompiledTraces() (reflection only casts the stored value), so the keyed caches need their own CompiledTraceInvalidating container.
    - Attention for S queries: Qwen35Attention.attend already routes a plain cache to PrefillAttentionKernel -> DecodeAttentionKernel -> MLX SDPA. PrefillAttentionKernel needs >= 64 queries and DecodeAttentionKernel needs 1 query with no mask, so for a causal 2..6-query verify neither applies and MLX SDPA (vector kernel) runs; the compiled verify keeps that same `attend` route.
    - VLM (Qwen35Language.Model) has its own S = 1 schedule with mrope positionIds; plan is to keep the VLM verify on the general path (task allows it) and add a cold-cache VLM greedy MTP == plain test.
  timestamp: 2026-10-28T11:36:44.993300+00:00
- actor: claude-code
  id: 01m6166mqtwa015r2qdzfj2sfx
  text: |-
    Implementation landed (TDD: RED was the compile failure of the new tests on the missing API — `compiledVerifySegmentCount`, the internal `forward(..., useCompiledDecode:)`; GREEN below).
    - MLXLMCommon/CompiledDecodeSegments.swift: `CompiledVerifyShape` (tokens 2...GatedDeltaDecodeKernel.mostTokens, checkpointAfter inside them; failable init), `KeyedCompiledDecodeSegmentCaches<Owner, Key>` (one CompiledDecodeSegmentCache per key, lazily, under a lock; it is one CompiledTraceInvalidating so invalidateCompiledTraces() reaches every key), `CompiledDecodeSegment.statesPerLinearLayer` and `attentionOutputOffset(outputsPerLinearLayer:)`.
    - MLXLMCommon/GatedDeltaDecodeKernel.swift: `GatedDeltaSpeculativeStates.traceArrays` / `init(traceArrays:)` — a trace returns arrays only; the kind (kernel snapshots vs op-path checkpoint) is read back from the rank of the recurrent array.
    - MLXLLM/Models/Qwen35.swift: `Qwen35TextModelInner.forward` takes `useCompiledDecode` (Qwen3Next pattern). `compiledStep` runs the S = 1 decode schedule (unchanged behaviour) or, for a verify with a checkpoint, `verifyStep` through `compiledVerifySegments` keyed by (S, checkpointAfter). The segment body is shared (`segmentBody(at:_:verify:)`): a verify passes checkpointAfter into `linearLayerBody`, which now returns the speculative states; the GDN layer picks the snapshot kernel inside the trace (op-path split as fallback for float32 / small heads). Each GDN cache gets its states + `saveSpeculativeStates` + `advance(S)` exactly as the general path. A verify leaves the final norm to the caller, so the emit branch of `Qwen35TextModel.forward` (applyFinalNorm: false, then norm) is unchanged. The S-query attention between segments goes through `Qwen35Attention.attend`, the general path's route (Prefill/Decode kernels where they read the call; for a causal 2..6-query verify neither applies — Prefill needs >= 64 queries, Decode needs one query and no mask — so MLX SDPA runs). Eligibility (`scheduleCaches`) is the old decodeStep check refactored without force unwraps; decode S = 1 still requires the `.none` mask.
    - `Qwen35TextModel.forward(_:cache:state:lastTokenOnly:useCompiledDecode:)` is now internal for the test oracle.
    - VLM: kept on the general path (task option); new cold-cache VLM greedy MTP == plain test.
    - Tests (Qwen35MTPTests.swift, suite Qwen35CompiledVerifyTests): bf16 4-layer kernel-eligible model (GDNDecodeKernelFixture.configuration gained `hiddenLayers:`), S = 2..6 x every rollback r = 0..S-1: logits, hidden rows, every cache state after rollback (covers every snapshot slot) within AttentionKernelTolerance.bfloat16; slot counts == S (kernel route); general path compiles no verify segment; invalidation drops verify traces; shape bounds; trace-array round trip; LLM fixture greedy MTP == plain (both drafter modes) with compiled verify count > 0; VLM greedy MTP == plain.
    - Diagnostic (temporary, removed): in the Release benchmark all 1284 verifies took the compiled schedule, CompiledVerifyShape(tokens: 2, checkpointAfter: 1).
    - Unit: `swift build --build-tests` clean (no warnings); MLXLMTests 895 XCTest 0 failures + 1899 Swift Testing passed; MLXFoundationModelsTests 376, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — all passed.
  timestamp: 2026-10-28T12:01:49.818893+00:00
- actor: claude-code
  id: 01m6166yvrzwdv1wgkeyvhzxcs
  text: |-
    Benchmark (Release, M5 Ultra, qwen38FourBitMTPGeneration, 192 tok each, all runs this session):
    - HEAD without the change (git stash of Libraries/Tests): plain 58.5/59.2/58.9, MTP 88.0 / 79.3 / 75.7 tok/s (math/code/prose), matches plain: true x3.
    - With the change, run 1: plain 58.7/59.1/58.4, MTP 90.3 / 80.1 / 77.2; run 2: plain 58.6/58.9/59.0, MTP 89.6 / 79.9 / 76.6; matches plain: true x3 in both. Acceptance and tok/verify identical to HEAD (93.9% 1.96, 73.6% 1.75, 69.0% 1.70).
    - Math gain +1.8% to +2.6% (code +0.8%, prose +1.2% to +2.0%). The >= 8% target is NOT met. No category below the 3% regression floor.

    Why the compiled verify buys only ~2%: the round time hardly depends on acceptance (math 1.96/89.6 = 21.9 ms, prose 1.70/76.9 = 22.1 ms, was 22.3 ms) and is ~1.28x a plain step (17.1 ms). The plain step reads ~15 GB of 4-bit weights in 17.1 ms (~880 GB/s), so it is at or near the memory-bandwidth limit, and a 2-row verify is bounded by the same weight read (~1.10x). Compiling the verify removes kernel launches and fuses elementwise chains, but that CPU/launch time was mostly already hidden behind GPU time: measured saving ~0.4 ms per round. The rest of the round (an estimated ~3 ms) is outside the verify graph: the drafter step (one MTP layer plus the full-vocabulary head) and the per-round host sync on the accept decision before the next draft. Reaching +8% needs work on those (overlap the drafter with the verify readback, or a cheaper draft head), not on the verify graph.

    Regression guard (Release qwen38FourBitMTPSystemPromptCache): passed in 46.7 s, no known issues printed. Edit turn 2: prompt 2292, cached 2169, prefilled 123 tok with MTP on (drafter prefilled 123), 89.2 tok/s, tokens match plain: exact. Other turns exact or near-tie (follow-ups turn 2 near-tie at token 1 gap 0.125; thinking turn 1 near-tie at token 52 gap 0.125, turn 2 near-tie at token 0 gap 0.375). MTP 84.7-90.6 tok/s vs plain ~56 across turns.
  timestamp: 2026-10-28T12:02:00.184421+00:00
- actor: claude-code
  id: 01m61674h1r75qjgbw4p4n0sjp
  text: |-
    ### implement — stuck
    - evidence: 5 files — Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLMCommon/CompiledDecodeSegments.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Tests/MLXLMTests/GDNDecodeKernelFixture.swift, Tests/MLXLMTests/Qwen35MTPTests.swift; every subtask and test item done, unit suites green (MLXLMTests 895 XCTest + 1899 Swift Testing, other 4 bundles green), qwen38FourBitMTPSystemPromptCache passes (edit turn 123 tok, exact); qwen38FourBitMTPGeneration math MTP 88.0 -> 90.3 / 89.6 tok/s (+1.8..2.6%), short of the >= 8% acceptance line.
    - gap: the >= 8% math speedup is not reached; the verify is weight-bandwidth bound and the remaining round time is in the drafter step and per-round host sync, outside the verify graph.
    - next: a person decides — accept the ~2% (keep the compiled verify, drop/relax the 8% line) or open a task on drafter/verify overlap; the code is ready for /review either way.
  timestamp: 2026-10-28T12:02:05.985543+00:00
- actor: claude-code
  id: 01m6168h711wjh96r2dek0z7q2
  text: |-
    ### finish iteration 1 — stuck
    - implement: stuck — code and tests done; perf acceptance (math >= +8%) not met: 88.0 -> 90.3/89.6 tok/s (+~2%). A plain step is already at ~880 GB/s, near the memory bandwidth limit; remaining round cost is the drafter step and the CPU waiting on each round's result.
    - test: green (implementer run) — MLXLMTests 895 XCTest + 1899 Swift Testing; qwen38FourBitMTPSystemPromptCache pass, edit turn 123 tok
    - commit: 9cf705de
    - review: not run — waiting on a human decision about the 8% acceptance line (relax it, or open a task to overlap the drafter with the verify)
  timestamp: 2026-10-28T12:02:51.745856+00:00
- actor: claude-code
  id: 01m617pwsc0r3cgng7wy37r9qc
  text: |-
    ### decision — user relaxed the 8% line
    - User (2026-10-28): "relax the 8% line -- just add up optimizations". Acceptance now: improves on the same-session baseline with output equal to plain. Measured +1.8 to +2.6% on math, so the criterion is met.
    - next: review commit 9cf705de after ^btfktsy finishes (sequential tree).
  timestamp: 2026-10-28T12:28:10.924678+00:00
- actor: claude-code
  id: 01m61fx18ys0m7htmvqmxtw52w
  text: |-
    ### review — findings
    - evidence: review sha 9cf705de~1..9cf705de — 5 findings (5 confirmed, 3 refuted) — Libraries/MLXLLM/Models/Qwen35.swift:915, Libraries/MLXLLM/Models/Qwen35.swift:921, Libraries/MLXLLM/Models/Qwen35.swift:1128, Libraries/MLXLMCommon/CompiledDecodeSegments.swift:133, Tests/MLXLMTests/Qwen35MTPTests.swift:2040
    - next: implement the Review Findings (2026-10-28 09:47) checklist, then re-review
  timestamp: 2026-10-28T14:51:20.734452+00:00
- actor: claude-code
  id: 01m61fxhp15rh7rbxawax4mabr
  text: |-
    ### finish iteration 2 — findings
    - (perf acceptance relaxed by the user; code from iteration 1, commit 9cf705de)
    - review: findings — Libraries/MLXLLM/Models/Qwen35.swift:915, Qwen35.swift:921, Qwen35.swift:1128, Libraries/MLXLMCommon/CompiledDecodeSegments.swift:133, Tests/MLXLMTests/Qwen35MTPTests.swift:2040
  timestamp: 2026-10-28T14:51:37.537543+00:00
- actor: claude-code
  id: 01m61gtj1g8m8edmkkjv44kj94
  text: |-
    Review findings (2026-10-28 09:47) worked, all 5 checked:
    1. GDN layer body duplication: new Libraries/MLXLMCommon/HybridDecodeLayer.swift — `HybridDecodeLayer` protocol (layer gives inputLayerNorm, postAttentionLayerNorm, gatedDeltaMixer, projectAttention, mergeAttention, feedForward) and `GatedDeltaLayerStep`; the extension holds the one shared `linearLayerBody(x:convState:recState:checkpointAfter:)` (returns the speculative states), plus `attentionPreBody(x:)` / `attentionPostBody(x:attention:gate:)` sharing one residual+feed-forward step (the same cause inside Qwen35: three copies of `h + mlp(postNorm(h))`). Qwen35DecoderLayer conforms; its own body copies and `mlpForward` are gone.
    2. Force unwraps: removed every `linearAttn!` / `selfAttn!` in Qwen35.swift via private `gatedDeltaNet` / `fullAttention` accessors (guard + preconditionFailure naming the invariant); the two `mlp as! UnaryLayer` force casts went too (`mlp` is now `Module & UnaryLayer`, the BailingMoe pattern). swiftlint force_unwrapping/force_cast/force_try: 0 in Qwen35.swift and the MLXLMCommon files.
    3. Segment body duplication: `CompiledDecodeSegment.body(_:arguments:layers:finalNorm:verify:)` in CompiledDecodeSegments.swift, generic over `HybridDecodeLayer`; the attention-head output layout moved with it as `CompiledDecodeSegment.AttentionOutput`. Qwen35 `segmentBody` now only embeds segment 0 and calls it. The final norm is applied by the segment with no attention head (always the last one `schedule` builds) on a decode step only — same as the old `index == count - 1` check.
    4. Assign-only `CompiledVerifyShape.tokens`: now the read source of how far each GDN cache advances in `commit` (`verify?.tokens ?? GatedDeltaDecodeKernel.decodeTokens`, the latter made `package`), replacing the parallel `inputs.dim(1)` parameter. Kept in the key: the token count fixes the trace shapes.
    5. Seeded tokens: one file-scope `seededTokens(_:vocabularySize:seed:)` in Qwen35MTPTests.swift; `prompt(length:)` and `tokens(_:of:seed:)` both call it; `verify` adds the batch axis at its call sites.
    TDD: new Tests/MLXLMTests/HybridDecodeLayerTests.swift (6 tests on an arithmetic fake layer: GDN body with/without kept states, attention pre/post bodies, segment body head layout, decode final norm, verify leaves the norm and gives back kept states). RED = compile failure on the missing API; GREEN in the full run.
    Qwen3Next and VLM copies NOT edited (per finding); follow-up task ^5fq4w2k moves them onto the shared functions.
    Not new: SwiftPM prints `warning: missing creator for mutated node: (.../mlx-swift_Cmlx.bundle/Contents/MacOS)` on every `swift build` in this repo (present in all earlier builds in the shell history); it is a build-system message about the Cmlx resource bundle, not a compiler warning. No compiler warnings in swift build or the Release xcodebuild.
  timestamp: 2026-10-28T15:07:28.176373+00:00
- actor: claude-code
  id: 01m61gtpjvm2j6wdqrp418q4h1
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXLMCommon/HybridDecodeLayer.swift (new), Libraries/MLXLMCommon/CompiledDecodeSegments.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Tests/MLXLMTests/HybridDecodeLayerTests.swift (new), Tests/MLXLMTests/Qwen35MTPTests.swift. `swift build --build-tests` 0 compiler warnings; MLXLMTests 895 XCTest 0 failures + 1913 Swift Testing passed (HybridDecodeLayerTests, Qwen35CompiledVerifyTests green); MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 passed. Release qwen38FourBitMTPGeneration (load avg 11.2 before / 7.1 after, other projects' test runs active): plain 59.2 / 59.2 / 59.3, MTP 90.7 / 81.6 / 79.4 tok/s math/code/prose, acceptance 93.9% / 73.6% / 69.0%, matches plain: true x3, TEST SUCCEEDED.
    - next: /review (all 5 findings of 2026-10-28 09:47 checked); follow-up ^5fq4w2k for Qwen3Next/VLM.
  timestamp: 2026-10-28T15:07:32.827866+00:00
depends_on:
- 01M5QWJGM1679S6CWHFCST7WAN
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: compiled multi-token verify schedule for Qwen35 (LLM and VLM targets)'
---
## What
A verify pass has `x.dim(1) >= 2`. The guards in `Qwen35DecoderLayer.callAsFunction` (Libraries/MLXLLM/Models/Qwen35.swift ~797) and `Qwen35TextModelInner.forward` (~988) send it to the general path, which has no compiled segments. Measured: the 2-row matmuls cost only 1.10× a 1-row step (scratchpad rows.py), but a full MTP round costs 1.31× a plain step.

- [x] Add a fixed-shape decode schedule for S = 2..6 to `Qwen35TextModelInner` (reuse `CompiledDecodeSegment` / `CompiledDecodeSegmentCache`, keyed by S). It uses the T-token GDN kernel with snapshot slots (previous task) and emits the drafter hidden rows (the `applyFinalNorm: false` path of `Qwen35TextModel.forward` ~1191).
- [x] Use `DecodeAttentionKernel` / `PrefillAttentionKernel` for the S-query attention step where they apply.
- [x] The MLXVLM Qwen35 model (Libraries/MLXVLM/Models/Qwen35.swift) has its own emit and checkpoint path (`Qwen35VLMMTPRegistration`). Give it the same schedule, or keep it on the general path and test that it still works.

## Acceptance Criteria
- [x] The verify output (logits, hidden rows, caches, and snapshot slots) of the compiled path is within the existing `AttentionKernelTolerance` against the general path, for S = 2..6.
- [x] Greedy MTP output equals plain decode on the shared fixture, for the LLM and VLM targets.
- [x] Release benchmark on the M5 Ultra (`qwen38FourBitMTPGeneration`, `#expect`): MTP tok/s improves against the same-session baseline and the output still matches plain decode. (Relaxed 2026-10-28 by the user from "math ≥ +8%": incremental optimizations add up.) Measured: HEAD 88.0 / 79.3 / 75.7, with the change 90.3 / 80.1 / 77.2 and 89.6 / 79.9 / 76.6 tok/s (math +1.8 to +2.6%); output matches plain on all three.
- [x] Regression guard: Release `qwen38FourBitMTPSystemPromptCache()` still passes with no known issues (edit turn ~123 tokens recomputed with MTP on; token match exact or near-tie per `MTPChatTokenMatch`), and no MTP category drops more than 3% below the baseline.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPTests.swift: compiled S = 2..6 verify against the general path, for rounds that accept and rounds that reject; a VLM-target MTP equality test.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes; then the Release benchmark.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf

## Review Findings (2026-10-28 09:47)

> Scope: `review sha 9cf705de~1..9cf705de` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5SKBCEWG4D25SZZNGG7BVM7.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SKBCEWG4D25SZZNGG7BVM7.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/Qwen35.swift:915` `duplication/duplication` — The changed `linearLayerBody` is a near-copy of the GDN layer body in two other models. The change added `checkpointAfter` and the speculative-state return to this copy only, so the three layer bodies have drifted apart. Move the shared GDN layer body into one function that takes the optional checkpoint and returns the speculative states, and call it from the Qwen35 layer. Leave the Qwen3Next and VLM copies to a separate task that moves them onto the same function.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:921` `code-hygiene/disallowed-constructs-swift` — force_unwrapping: Force unwrapping should be avoided.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:1128` `duplication/duplication` — The changed `segmentBody(at:_:verify:)` is a near-verbatim copy of the segment body in two other models. The change added the verify path (checkpoint states, speculative outputs, and the final-norm guard) to this MLXLLM copy only, so the copies now diverge. Any fix to one side has to be re-applied by hand to the others. Extract one shared segment-body function, parameterized by the verify shape and by the layer and norm modules, and have Qwen35 call it. Qwen3Next and the VLM copy are outside this change, so either route them through the same shared function in a separate task or record the divergence as a known follow-up. Do not edit the counterparts inside this change.
- [x] `Libraries/MLXLMCommon/CompiledDecodeSegments.swift:133` `code-hygiene/dead-code-swift` — var.instance `tokens` is assignOnlyProperty.
- [x] `Tests/MLXLMTests/Qwen35MTPTests.swift:2040` `reuse/reuse` — The new `tokens(_:of:seed:)` helper reimplements the seeded-token generator that already exists as `prompt(length:)`. Both call `MLXRandom.randInt` over a vocabulary range with a seeded key and differ only in the vocabulary source, the seed, and the `[1, count]` shape. Keeping two copies means the token-generation logic can drift between the MTP and compiled-verify suites. Generalize the existing helper instead of adding a parallel one. Move the seeded-token generator to file scope (or a shared fixture) with parameters for vocabulary size, count, and seed, then have `prompt(length:)` and `tokens(_:of:seed:)` both call it. If the shapes must differ, keep the `[1, count]` reshape at the call site in `verify`.

---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7ga8ya0p06mstm3znzbjfp9
  text: |-
    Research (oQ4e, prompt_706, same session; all ablations env-gated in a scratch copy, reverted, Flint.swift == HEAD afterwards).

    Baseline flintbench: decode 6.35 / 6.75 / 7.10 ms/tok at 2k/8k/16k, prefill 8895 / 9064 / 8178 tok/s.

    Where decode time goes (flintcli graph, 2k): wall 6.37, graph build 1.14 ms, asyncEval 5.22 ms of which only 3.6 ms is thread CPU (the rest is MLX's MAX_ACTIVE_TASKS=10 throttle waiting on the GPU). Metal System Trace: ~41 command buffers/token of ~111 us (GPU busy ~4.7 ms), then ~1.1-1.5 ms GPU idle at each token boundary. CPU ~4.8 ms/token and GPU ~4.7 ms/token, neither saturated: a pipeline handoff limit, so a node that dispatches costs CPU build + CPU encode + GPU dispatch.

    Ablations (graph wall, 2k): skip sliding attention -2.0 ms (57 us/layer, 35 layers); skip full attention -0.83 ms (14 layers); skip q/k RMSNorm+RoPE in all layers -0.355 ms (median of 4); skip the cache update and attend over 1 key -0.96 ms; keep the cache write but SDPA over 1 key -0.30 ms => the L=1 cache write path costs ~0.66 ms/token (13.5 us/layer), of which 0.32 ms Swift graph build. Remaining ~30 us/sliding layer is mostly the 8-bit qkv+o_proj weight reads (19.5 MB/layer) — bandwidth, irreducible.

    Per-piece findings:
    - SDPA with sinks DOES take the fused one-pass sdpa_vector kernel (one ScaledDotProductAttention primitive per layer; MLX uses 1-pass below 1024 keys on every device, 2-pass >=1024 on 'd'/'s'). GPU-only (trace) cost: 11.5 us at 1 key, 22.7 us at 513 keys. A faster prototype single-pass kernel (13.7 us at 513 in microbench, ~1 ulp from MLX) gave NO in-model gain (6.42 vs 6.41) — SDPA latency is not on the critical path. DecodeAttentionKernel (no sinks) is no faster at 513 keys (25 us).
    - RotatingKVCache.update at L=1 (wrapped): 4.8 us CPU graph build; KVCacheSimple.update L=1: 6.9 us. Each MLX Swift subscript set costs ~2.5 us (5x a plain op) and adds 2 Reshape nodes (leadingSingletonDimensionsRemoved + reshape back). SliceUpdate is in place (donated). MLX Swift has no public slice-update entry and Cmlx is not an importable product, so the setter overhead cannot be bypassed without an upstream API.
    - View nodes are free: reshaping qkv straight to head-major (no 3 transposes) changed nothing (6.365 vs 6.375).
    - A copy kernel standing in for split+norms+RoPE (upper bound of a fused kernel): -0.165 ms (median of 4); a metalKernel call costs 2.55 us CPU (vs 2.24 for split+reshape+3 transposes).
    - Side discovery (out of scope, global MLX knob): MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200 gives 6.35 -> 5.99 ms/tok; 25/25 gives 7.04.
    - MLX's metallib is compiled by SwiftPM with Metal's default fast math; runtime MLXFast kernels compile with MathMode::Safe. Bit-exact replicas need explicit fma: RMSNorm acc = fma(x,x,acc) then simd_sum, precise::rsqrt, w * T(x*inv); RoPE rx1 = fma(x1,cos,-(x2*sin)), rx2 = fma(x2,cos,x1*sin), inv_freq = fast::exp2(-d*log2(base)). Prototype matched MLX bit-for-bit for bf16/f16/f32 over 20 seeds and offsets 0..100000.

    Plan: one bit-exact Metal kernel FlintDecodeHeadsKernel for a decode step of a fused-qkv layer: from the qkv projection, q/k RMSNorm + RoPE (or none for no_rope layers) and the head-major q/k/v, replacing Split + 2 RMSNorm + 3 Transpose + 2 RoPE (4 dispatches -> 1 on RoPE layers, 2 -> 1 on no-RoPE layers). Cache write and SDPA keep the op path (no public in-place write).
  timestamp: 2026-11-15T19:17:17.760419+00:00
- actor: claude-code
  id: 01m7gd3vav927k904qh2js61wm
  text: |-
    Implementation landed (not committed).
    - New Libraries/MLXLLM/Models/FlintDecodeHeadsKernel.swift: `FlintHeadRotation` (.unrotated / .rotary(base:scale:)) and `FlintDecodeHeadsKernel`, a struct built once per layer. One SIMD group per head turns the fused qkv_proj output of one decode token into head-major q/k/v, doing q and k RMSNorm and RoPE (or none) in one dispatch, bit-exact with MLX rms_single_row + rope_single (explicit fma, fast::exp2, precise::rsqrt; see file header). Layer constants (head counts, eps, RoPE scale and log2 base bits) sit in the Metal header and the kernel name, so a call passes only the T template. The first version passed 9 template args and cost 4.7 us of CPU per call (graph build +0.11 ms/token); the header form recovered most of that.
    - Flint.swift: the protocol gains `headRotation` and `attentionHeads(_:offset:)`; the op path moves to `rotatedHeads(_:offset:)` (unchanged ops). FlintAttention sets headRotation (.unrotated for no_rope layers; .rotary(rope_theta, 1) with a plain RoPE built directly when rope_scaling is nil; nil for scaled RoPE, which keeps the op path). FlintFusedAttention builds the kernel in init(fusing:) and takes it for L=1 with a scalar offset; everything else (prefill, batch offsets, CPU, other head widths, dtype mismatch) takes the op path. MetalSimdWidth.swift: metalSimdWidth/metalSimdWidthDeclaration become `package`.
    - Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift: bit parity against MLX ops for 48 cases (f32/bf16/f16 x rotary/unrotated x offsets 0/1/514/20000 x batch 1/2); nil for a 2-token block and for a norm weight dtype mismatch; init nil for head width 64; layer-level bits of attentionHeads vs rotatedHeads(projectHeads) for f32/bf16/f16 on a RoPE and a no-RoPE layer; graph check that a fused decode step has no RMSNorm node; headRotation for plain, no-RoPE and linear-scaled layers. RED with a nil stub: 48 parity + 2 kernel-graph + 2 rotation cases failed for the expected reasons.
    - Not done, and why: the largest reducible cost measured (the L=1 KV write path, ~0.66 ms/token) is mostly MLX Swift subscript-set overhead, and no public in-place write API exists. A custom kernel cannot write the cache buffer in place, and the SliceUpdate is already in place. Filed ^y5d37gd. SDPA with sinks already takes the fused kernel, so it needs no change. Also filed ^5smnpg3 (float32 decode after chunked prefill vs cold prefill: max|d| 1.30 at HEAD, identical before/after this change) and ^twc3zh6 (MLX command-buffer sizing, -6% decode).
  timestamp: 2026-11-15T20:06:56.603061+00:00
- actor: claude-code
  id: 01m7gd42b2j1ezf83sne4h3ww5
  text: |-
    ### implement — changed
    - evidence: files Libraries/MLXLLM/Models/FlintDecodeHeadsKernel.swift (new), Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXLMCommon/MetalSimdWidth.swift, Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift (new). Where time went (2k): sliding attention 57 us/layer = L=1 KV write path ~13.5 (0.66 ms/token), q/k norm+RoPE ~7, SDPA key reads ~6, rest ~30 (8-bit qkv/o_proj weight reads); decode is a CPU/GPU handoff limit (CPU ~4.8 ms, GPU ~4.7 ms, ~1.3 ms GPU idle per token). flintbench same session: HEAD 6.35/6.74/7.09 -> 6.27/6.63/7.04 ms/tok at 2k/8k/16k. Alternating flintcli graph (median of 4): 2k 6.375 -> 6.27, 16k 7.12 -> 7.05. Prefill (12 alternating runs) 2k 8774 -> 8830, 8k 8934 -> 8945 tok/s (no regression; flintbench 8958/8914/8189 vs 8882/8871/8173 is noise). float32 (MLX_ENABLE_TF32=0): 300 decode-logit rows after a 1700-token prefill (ring wrapped) are bit-identical HEAD vs change; `warm` schedule (chunks, decode, junk-trim, trim/re-decode, 700 chunk) prints identical numbers, so rotated-cache trim/restore is unchanged. flintcheck MATCH x4 + TOOL CALL get_weather (golden_nanqj4p.json; golden_base untouched). oQ8e MATCH x4 vs golden8_fused.json. ChatSession past the wrap: follow-ups prefill 15 then 8 new tokens (cached 714, 1129), text identical HEAD vs change. Tests: MLXLMTests 2353 ST (3 known issues) + 1098 XCTest (4 skipped, 0 failures); MLXGuidedGenerationTests 78; MLXFoundationModelsTests 496; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 — all pass. swift-format applied, lint clean, no new build warnings.
    - gap: the KV-write part of the fusion candidate is not done (no public in-place write; see ^y5d37gd).
    - next: /review
  timestamp: 2026-11-15T20:07:03.778421+00:00
- actor: claude-code
  id: 01m7gd56hhzag8wb9ahjzhhhvd
  text: '16k acceptance check: float32 (MLX_ENABLE_TF32=0) decode logits for 200 teacher-forced steps after a 16384-token prefill of prompt_706 (offsets 16584, ring wrapped many times) are bit-identical between the HEAD binary and this change: max|d| 0.0, greedy argmax equal. Scratchpad flintcli `declogits` mode; files declogits16k_{before,after}.safetensors.'
  timestamp: 2026-11-15T20:07:40.849253+00:00
- actor: claude-code
  id: 01m7gdkzscbjz5qyr1xs6ymq7f
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (1 pre-existing SwiftPM "missing creator" note, no compiler warnings). xcrun xctest per bundle: MLXLMTests 1098 XCTest (0 failures, 4 skipped = LFM2 env baseline) + 2353 Swift Testing tests passed, 3 known issues (baseline), GDNDecodeKernelLayerTests passed; MLXGuidedGenerationTests 78 passed; MLXFoundationModelsTests 496 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 passed. swift-format lint -s clean on Flint.swift, FlintDecodeHeadsKernel.swift, MetalSimdWidth.swift, FlintDecodeHeadsKernelTests.swift. flintcheck.sh: MATCH x4 (256 greedy ids identical each) + TOOL CALL get_weather Paris. golden_base.json untouched.
    - next: review/commit
  timestamp: 2026-11-15T20:15:45.452858+00:00
- actor: claude-code
  id: 01m7gdmst4hgd1mnams0a5g19d
  text: |-
    ### commit — changed
    - evidence: perf(flint): fuse the decode q/k norm, RoPE and head layout into one Metal dispatch (^nanqj4p) — Libraries/MLXLLM/Models/FlintDecodeHeadsKernel.swift, Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXLMCommon/MetalSimdWidth.swift, Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift, .kanban/ (incl. new cards ^y5d37gd, ^5smnpg3, ^twc3zh6). The sha is the HEAD commit carrying this subject line.
    - next: review
  timestamp: 2026-11-15T20:16:12.100499+00:00
- actor: claude-code
  id: 01m7gdv0qszqgz8t0enjhp9wj1
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (fc40251a) — 2 findings (2 confirmed, 2 refuted): Libraries/MLXLLM/Models/Flint.swift:192, Libraries/MLXLLM/Models/Flint.swift:265
    - next: implement the 2 open items in the 2026-11-15 14:16 Review Findings section, then re-review
  timestamp: 2026-11-15T20:19:35.801514+00:00
- actor: claude-code
  id: 01m7gdvgyetke6h48rjqa5a9pa
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — FlintDecodeHeadsKernel.swift (new), Flint.swift, MetalSimdWidth.swift, FlintDecodeHeadsKernelTests.swift (new)
    - test: green — MLXLMTests 1098 XCTest + 2353 ST, Guided 78, FM 496, CXGrammar 7, Macros 11; flintcheck MATCH x4 + TOOL CALL
    - commit: fc40251a
    - review: findings — Libraries/MLXLLM/Models/Flint.swift:192 (dead headRotation), Libraries/MLXLLM/Models/Flint.swift:265 (attentionHeads name clash)
  timestamp: 2026-11-15T20:19:52.398241+00:00
- actor: claude-code
  id: 01m7ge4225py1f3abkg584vz7m
  text: 'Review fixes (iteration 2): removed the `headRotation` requirement from FlintAttentionLayer (no generic reader) and the stored `headRotation` on FlintFusedAttention (written, never read). FlintAttention keeps `headRotation` (read by FlintFusedAttention.init(fusing:) and by the three rotation tests), now with its doc comment. Renamed the method `attentionHeads(_:offset:)` to `attentionInputs(_:offset:)` in the protocol, both implementations (op-path extension and FlintFusedAttention), the callAsFunction call site, the doc link and the two test call sites; `attentionHeads` now names only the head count. Checked every other symbol fc40251a added in FlintDecodeHeadsKernel.swift — each has a reader.'
  timestamp: 2026-11-15T20:24:32.069892+00:00
- actor: claude-code
  id: 01m7geegae4v8bv8e007cx7h3f
  text: |-
    ### implement — changed
    - evidence: files Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift. Both 2026-11-15 14:16 findings flipped to [x]. swift build --build-tests clean (only the pre-existing SwiftPM "missing creator" note). xctest: MLXLMTests 1098 XCTest (4 skipped LFM2 baseline, 0 failures) + 2353 ST passed (3 known issues baseline); MLXGuidedGenerationTests 78; MLXFoundationModelsTests 496; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 — all pass. swift-format applied (no files outside this task changed). flintcheck: MATCH x4 (256 greedy ids each) + TOOL CALL get_weather Paris (out golden_nanqj4p_r2.json; golden_base untouched).
    - next: /review
  timestamp: 2026-11-15T20:30:14.350233+00:00
- actor: claude-code
  id: 01m7gexjhbh4rw762y7x7cd4xy
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 errors, 0 source warnings). xctest one bundle at a time: MLXLMTests 2353 Swift Testing tests / 224 suites passed with 3 known issues + 1098 XCTest, 0 failures, 4 skipped (LFM2 env skips, baseline); GDNDecodeKernelLayerTests passed. MLXGuidedGenerationTests 78 tests passed. MLXFoundationModelsTests 496 tests passed. CXGrammarTests 7 tests passed. MLXHuggingFaceMacrosTests 11 XCTest passed. swift-format lint -s on Flint.swift and FlintDecodeHeadsKernelTests.swift clean. flintcheck.sh: MATCH x4 (256 greedy ids each) and TOOL CALL: get_weather Paris; output written to golden_test.json, golden_base.json untouched.
    - next: review
  timestamp: 2026-11-15T20:38:28.139874+00:00
- actor: claude-code
  id: 01m7geyacacyvek0wx2ehw2ahg
  text: |-
    ### commit — changed
    - evidence: refactor(flint): drop the unused head rotation and name the attention input method apart from the head count, found in review of ^nanqj4p (files: Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift, .kanban/tasks/01M7FHG7TA2MGDAS4Q5NANQJ4P.jsonl, .kanban/tasks/01M7FHG7TA2MGDAS4Q5NANQJ4P.md)
    - next: none; local commit only, not pushed
  timestamp: 2026-11-15T20:38:52.554301+00:00
position_column: doing
position_ordinal: '80'
title: 'Flint: find and cut the cost of sliding-window attention in decode (~63 µs/layer)'
---
## Why
- **Measured:** skipping attention in the 35 sliding layers (window 513, RotatingKVCache) saves ~2.2 ms/token on oQ4e, about 63 µs per layer. The 14 full layers cost ~69 µs each even at 2k context.
- **Expected:** the bytes read are tiny, so most of this must be dispatch overhead: q/k/v projections, q and k RMSNorm, two RoPE calls, the rotating-cache update, the sinks cast, SDPA with sinks, and `o_proj`.

## What
1. **Measure the parts.** Do a micro-ablation or a shader-profiler trace of one layer:
   - the RotatingKVCache update at L=1 (slice or concatenate ops? temporaries?);
   - whether `MLXFast.scaledDotProductAttention` with `sinks` takes the fused vector kernel, or falls back to unfused attention for head dim 128 / GQA 8;
   - the norm and RoPE kernels.
2. **Fix what dominates.** Candidates:
   - Fuse q/k RMSNorm, RoPE and the KV write into one Metal kernel, like the Qwen3.8 decode kernels (`94f35637`, `a6c0dacc`).
   - Write the rotating cache in place.
   - Give SDPA sinks support in the fused kernel if it's missing.

## Acceptance
- Greedy ids are identical on oQ4e. Decode output at 16k context matches the old path, including after the rotating cache wraps.
- ms/token improves against a same-session baseline at 2k and 16k.
- Prompt-cache and ChatSession checkpoint behaviour is unchanged: rotated caches still trim or restore as now.
- All five bundles are green. #flint-perf

## Review Findings (2026-11-15 14:16)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 10 not reviewed.

> 10 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7FHG7TA2MGDAS4Q5NANQJ4P.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FHG7TA2MGDAS4Q5NANQJ4P.md` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.md` — no validator matches this file
> - `.kanban/tasks/01M7GCNJFYATF646QNT5SMNPG3.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7GCNJFYATF646QNT5SMNPG3.md` — no validator matches this file
> - `.kanban/tasks/01M7GCNQTVHPRSGCP27Y5D37GD.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7GCNQTVHPRSGCP27Y5D37GD.md` — no validator matches this file
> - `.kanban/tasks/01M7GCNVYNJNS73KPQGTWC3ZH6.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7GCNVYNJNS73KPQGTWC3ZH6.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/Flint.swift:192` `code-hygiene/dead-code-swift` — var.instance `headRotation` is unused.
- [x] `Libraries/MLXLLM/Models/Flint.swift:265` `swift/naming-clarity` — The new method `attentionHeads(_:offset:)` in the `FlintAttentionLayer` extension shares its base name with the stored property `let attentionHeads: Int` (the query head count) on `FlintAttention`. Inside that type, `attentionHeads` now names both the head count and a function that returns the normed, rotated heads, so a reader cannot tell from the name which one a bare reference means. Line 248 (`let heads = attentionHeads(x, offset: cache?.ropeOffset)`) is one example. Rename the method to a name that says what it returns, such as `rotatedAttentionHeads(_:offset:)` or `attentionInputs(_:offset:)`, in the protocol, both extensions and the call sites, and keep `attentionHeads` as the head count.

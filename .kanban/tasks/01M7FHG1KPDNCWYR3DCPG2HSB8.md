---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7frf2dza28d9b2wrsj4jaqf
  text: 'Handoff from ^g68bhj1: the expert-bias cast and the routed->f32 cast are gone from the decode graph (oQ4e decode step: 191 -> 97 AsType). The per-step casts that remain in the Flint MoE block, and that a compiled router/combine could fuse, are `gate(x).asType(.float32)` before `sigmoid` (47 per token) and `weightedExpertSum(...).asType(x.dtype)` before the shared-expert add (47 per token). `model(...).asType(.float32)` before lm_head runs once per token. Count them with flintcli `dot <promptfile> <ctx> <out.dot>` (scratchpad flintcli), then `grep -o ''label ="[^"]*"'' out.dot | sort | uniq -c`.'
  timestamp: 2026-11-15T14:06:04.223396+00:00
- actor: claude-code
  id: 01m7fszvf3tcj7capayq0p6h2h
  text: 'Research/baseline (oQ4e, prompt_706, ctx 2048, before change): dot graph of one decode step = 3141 nodes; per MoE layer: Matmul(gate)+AsType+Sigmoid+Add(bias)+CustomKernel(router)+Sum+Add(eps)+Divide, then weightedExpertSum (compiled mult + Sum)+AsType+Add(shared), then residual Add. Adds total 239 = 47 bias + 47 eps + 47 shared + 98 residual. flintcli graph: wall 6.88 ms/tok, graph build 1.40 ms, asyncEval 5.46 ms, wait 0.00 (CPU-bound: encode dominates). Plan: (1) shapeless compiled routerScores (asType f32, sigmoid, bias add -> scores+choice, one kernel); (2) router kernel unchanged; (3) normalization moved into a compiled combine together with weightedExpertSum, the asType back, the shared-expert add and the MLP residual add (decoder layer passes h as residual via new FlintFeedForward protocol). The attention residual add has no elementwise neighbour (o_proj matmul before, RMSNorm after) so it stays. Notes: Swift Testing has no name filter under xctest (no SWT_ filter env), so the new suite runs with the whole MLXLMTests bundle. A Module @ParameterInfo cannot be assigned directly in a test (Fatal ''please call update()'').'
  timestamp: 2026-11-15T14:32:42.723192+00:00
- actor: claude-code
  id: 01m7fvpcenz07s51ngtf8xazgj
  text: |-
    Implementation landed (not committed).
    - Flint.swift: new `FlintExpertRouting` (Sendable value type: scores / selected / normalized / combined), `FlintFeedForward` protocol (`callAsFunction(_:residual:)`, default `residual + self(x)`), and the decoder layer now calls `mlp(norm(h), residual: h)`. `FlintSparseMoeBlock` keeps the op path for prefill; a one-token block runs through `compiledDecodeStep` (a `CompiledTrace` over the whole MoE decode step), whose body calls the compiled router scores and the compiled combine (normalization + weighted sum + cast + shared add + MLP residual add).
    - New FlintCompiledDecode.swift (shapeless routerScores, combine, combineRoutedExperts) and Tests/MLXLMTests/FlintCompiledDecodeTests.swift (bit-exact parity f32/bf16, with/without shared expert, both norm_topk_prob settings, block-level decode vs op path, trace compiled only for decode).
    - What did not work alone: the compiled functions without the trace cut nodes 3141 -> 2765 and wall 6.88 -> ~6.45 ms, but graph build stayed flat (1.36-1.48 vs 1.39-1.41 ms): each compiled call has its own CPU cost, about equal to the ops it removes. The per-block CompiledTrace (DeepSeek-V4 precedent) is what cuts graph build.
    - Inside the outer trace MLX splits the router scores into two kernels (AsType+Sigmoid, then the bias add) because the router kernel also reads the scores: final node count 2812, not 2765. Normalization (eps add + divide) is its own kernel, not fused into the weighted multiply.
    - The first-attention residual add has no elementwise neighbour and stays. lm_head's asType(float32) follows RMSNorm and stays.
    - Side finding: my first test version built a module from the global RNG, which shifted the weights of GDNDecodeKernelFixture (which also reads the global RNG) and failed its bf16 Qwen3-Next positive control. My test now uses withRandomState; the fixture fragility is ^3fpsgjv.
    - Prefill at 8k read ~1.3% low against the recorded numbers, but HEAD re-measured in the same session gives 6552/6388/5928, i.e. the same as this change: machine variance.
  timestamp: 2026-11-15T15:02:29.589074+00:00
- actor: claude-code
  id: 01m7fvph5xqwvc01mp8qnwbfh1
  text: |-
    ### implement — changed
    - evidence: files Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXLLM/Models/FlintCompiledDecode.swift (new), Tests/MLXLMTests/FlintCompiledDecodeTests.swift (new). oQ4e decode-step nodes 3141 -> 2812 (AsType 97 -> 3, Add 239 -> 51, Broadcast 216 -> 75). flintcli graph 2k: wall 6.88 -> 6.32-6.39 ms/tok, graph build 1.39-1.41 -> 1.11-1.14 ms, asyncEval 5.46 -> 5.20-5.23 ms. flintbench (two runs) decode 6.32/6.69/7.06 ms/tok at 2k/8k/16k vs HEAD same session 6.88/7.26/7.58 (recorded 6.89/7.27/7.63); prefill 6490-6570/6398-6409/5935-5945 vs HEAD same session 6552/6388/5928. flintcheck MATCH x4 + TOOL CALL get_weather. oQ8e greedy ids MATCH golden8_fused.json x4, chats. ChatSession follow-ups prefill 9 new tokens (oQ8e cached 138, oQ4e cached 184). Tests: MLXLMTests 2337 Swift Testing passed with 3 known issues + 1092 XCTest (4 LFM2 env skips, 0 failures); MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 — all passed. Build warning only the known upstream "missing creator for mutated node". swift-format applied.
    - next: /review
  timestamp: 2026-11-15T15:02:34.429403+00:00
- actor: claude-code
  id: 01m7fw5dfbctq5vpcwfpe4ehv3
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` exit 0, no source warnings. Each bundle run alone with `xcrun xctest`:
      - MLXLMTests: Swift Testing 2337 tests / 222 suites passed, 3 known issues (baseline); XCTest 1092 executed, 0 failures, 4 skipped (the 4 LFM2 env skips, baseline). FlintCompiledDecodeTests and GDNDecodeKernelLayerTests passed.
      - MLXGuidedGenerationTests: 76 tests passed.
      - MLXFoundationModelsTests: 494 tests passed.
      - CXGrammarTests: 7 tests passed.
      - MLXHuggingFaceMacrosTests: 11 XCTest, 0 failures.
      - `swift-format lint -s --configuration .swift-format` on Flint.swift, FlintCompiledDecode.swift, FlintCompiledDecodeTests.swift: clean.
      - flintcheck.sh (GPU free, run alone): MATCH x4 (all 256 greedy ids identical for parity_ids, long_ids, prompt_87, prompt_706) and TOOL CALL get_weather Paris.
    - next: review
  timestamp: 2026-11-15T15:10:42.155043+00:00
- actor: claude-code
  id: 01m7fw695semj3a3xnfky4xwk9
  text: |-
    ### commit — changed
    - evidence: commit "perf(flint): compile the one-token MoE decode step (^pg2hsb8)" (the sha is the commit with this subject on branch stable; it cannot be written into the comment it ships in). Files: Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXLLM/Models/FlintCompiledDecode.swift, Tests/MLXLMTests/FlintCompiledDecodeTests.swift, and all .kanban/ changes (including new card ^3fpsgjv).
    - next: review
  timestamp: 2026-11-15T15:11:10.521300+00:00
depends_on:
- 01M7FHFQT76AN9817CJJ4PQTWR
- 01M7FHFWPMBDPGDSD9BG68BHJ1
position_column: doing
position_ordinal: '80'
title: 'Flint: compile the one-token decode step (router math, combine, residuals)'
---
## Why
- **Router:** on oQ4e the router alone costs ~1.1 ms/token, about 23 µs per MoE layer. That covers sigmoid, bias add, the fused top-k, normalization (sum, add, divide) and the casts.
- **CPU:** building the graph costs 1.5 ms/token, and the GPU sits idle 26% of each step (Metal System Trace).
- **Precedent:** compiling the elementwise chains fuses them and cuts the CPU cost of building the graph. DeepSeek-V4 did this in ebd865dc (`perf(deepseek-v4): compile the one-token decode step`).

## What
- Add compiled functions, used only in the single-token decode shape, for:
  - the router pre and post steps around `fusedRouterTopK`;
  - the weighted expert combine plus the shared-expert add and its casts;
  - the residual adds that can join them.
- Prefill keeps the op path.
- The compiled path must be bit-exact with the op path. Add a test like the DeepSeek-V4 one.

## Acceptance
- Greedy ids are identical on oQ4e.
- The op path and the compiled path are bit-exact in a unit test on a tiny config.
- ms/token and graph-build time improve (flintcli `graph` mode prints build, asyncEval and wait per token).
- All five bundles are green. #flint-perf
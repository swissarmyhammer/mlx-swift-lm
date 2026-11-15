---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7fj14xjn2b141wbgyv7r2kn
  text: |-
    Research (implement):
    - Load path: installWeights -> model.prepareCheckpoint(checkpoint) -> applyCheckpoint, which quantizes each module path that has `<path>.scales` with `checkpoint.perLayerQuantization.quantization(layer: path)`, then update(parameters, verify: .all). So the fusion goes in a FlintModel.prepareCheckpoint override: sanitize, concatenate the parts, write a plan entry for the fused path (copied from the parts), drop the part entries, and swap the modules via update(modules:) (precedent: ParoQuantLoader swaps modules at load).
    - Plan keys of mlx-community flint-oQ4e / oQ8e carry the `language_model.` prefix, which sanitize keeps, so the fused path `language_model.model.layers.N.self_attn.qkv_proj` resolves directly.
    - oQ4e: q/k/v 8-bit gs64 (49 layers), shared gate/up 8-bit gs128 (47), dense gate/up (layers 0,1) 8-bit gs64, switch_mlp default 4-bit gs64. oQ8e: q/k/v 8-bit gs128 on 12 layers else default 8-bit gs64 (all three parts always equal), shared gate/up 8-bit gs128, switch default 8-bit gs64.
    - Fusion only for quantized checkpoints (a plan exists). An unquantized source goes through ModelConversion.convert, which resolves its per-path quantization decisions on the module tree before loadWeights and saves model.parameters(); fusing there would change the saved checkpoint layout.
    - No shared q/k/v fusion helper exists (Phi3/OpenELM/Lille ship qkv_proj natively). FusedGateUpSwitchGLU (MLXLMCommon) splits gate|up halves on the last axis, and its switch slot type differs from SwitchGLU, so the Flint slot becomes a small protocol both conform to.
    - qkv split: reshape the fused output to [B, L, H+2KV, D] and slice heads, so at decode (L=1) the q/k/v slices are contiguous and rms_norm needs no copy.
    - Side effect to note: after fusion a quantized Flint has no q_proj/k_proj/v_proj/gate_proj/up_proj modules, so a LoRA adapter that targets those keys would not find them.
  timestamp: 2026-11-15T12:13:36.562896+00:00
- actor: claude-code
  id: 01m7fjy46ypyjwjm40ams6gw2x
  text: |-
    Implementation landed (Flint.swift, FlintTests.swift):
    - FlintModel.prepareCheckpoint: sanitize, then (only when the checkpoint has a quantization plan) each FlintDecoderLayer.fuseProjections joins q/k/v -> self_attn.qkv_proj, dense mlp gate/up -> mlp.gate_up_proj, shared_expert gate/up -> shared_expert.gate_up_proj, switch_mlp gate/up -> switch_mlp.gate_up_proj (FusedGateUpSwitchGLU). FlintProjectionFusion.fuse joins only when every part resolves to the same (groupSize, bits, mode) in the plan and carries the same tensor set including `scales`; it writes the plan entry for the fused path and drops the part entries, so applyCheckpoint quantizes the fused module with the parts' bits.
    - Module swaps via update(modules:): FlintAttention -> FlintFusedAttention (shares o_proj/norms/sinks/rope), FlintMLP -> FlintFusedMLP, SwitchGLU -> FusedGateUpSwitchGLU. Slots became protocol-typed: `any FlintAttentionLayer` (attention core moved to a protocol extension), `any FlintRoutedExperts`, `UnaryLayer` for mlp/shared_expert (decoder `mlp` is now @ModuleInfo so it can be swapped).
    - Dense layers 0-1 gate/up fuse too (same FlintMLP class, same fallback rule).
    Verification so far:
    - float32 logits (MLX_ENABLE_TF32=0, FLINT_F32=1, flintcli logits on parity_ids, 666 positions) fused vs unfused: max|d| 0.0 on both oQ4e and oQ8e (bit identical).
    - oQ8e greedy golden (4 prompts, 256 ids) fused vs unfused: all MATCH; chat "hello" + haiku text identical.
    - ChatSession multi-turn on oQ4e ("hello", "What is 2+2?", "And times 3?"): prompt 3/8/6 tok with cached 0/185/231 in both builds, identical text.
    Discovery: fusion copies the joined tensors at load. Peak memory footprint (/usr/bin/time -l, chat 200 hello): oQ8e 42.7 GB unfused -> 70.2 GB fused; oQ4e fused 38.6 GB (weights 22 GB). Load time oQ8e 2.8 s -> ~4 s. The excess is transient: the installWeights caller holds the original checkpoint dictionary during prepareCheckpoint, so the parts stay alive until materialize, and freed parts sit in the MLX buffer cache until materializeModelForInference calls Memory.clearCache(). Steady-state RSS unchanged (42.4 GB both). Reducing it needs a loader change in MLXLMCommon (outside this card).
  timestamp: 2026-11-15T12:29:26.110890+00:00
- actor: claude-code
  id: 01m7fkdyaa2010fqwadbzfxszz
  text: |-
    Final checks on the final tree (after swift-format):
    - flintcheck.sh (oQ4e): MATCH on all 4 prompts (parity_ids, long_ids, prompt_87, prompt_706: all 256 greedy ids identical). TOOL CALL: get_weather {"city": "Paris"}.
    - flintbench.sh (oQ4e, median of 3), same-session baseline from unchanged code -> fused:
      ctx 2048: decode 7.57 -> 7.24 ms/tok (2 runs of fused code: 7.07 and 7.24), prefill 6082 -> 6347 tok/s
      ctx 8192: decode 7.91 -> 7.47 ms/tok, prefill 5870 -> 6302 tok/s
      ctx 16384: decode 8.17 -> 7.88 ms/tok, prefill 5528 -> 5738 tok/s
    - oQ8e: loads, `chat 200 hello` answers sensibly; golden ids and chat text identical to the unfused build.
    - Tests: MLXLMTests 1089 XCTest (4 env-gated LFM2 skips that were already there, 0 failures) + 2331 Swift Testing (3 known issues that were already there); MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. All pass. The only build warning is the SwiftPM `missing creator for mutated node ... mlx-swift_Cmlx.bundle` build-graph note, which is not from this code.
    - New follow-up task ^75rm5ds: load peak memory caused by the fused copies.

    ### implement — changed
    - evidence: Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/FlintTests.swift (4 new tests); five bundles green; flintcheck 4/4 MATCH + TOOL CALL; flintbench decode 7.24/7.47/7.88 ms vs baseline 7.57/7.91/8.17
    - next: review
  timestamp: 2026-11-15T12:38:04.362417+00:00
- actor: claude-code
  id: 01m7fky4yk0bnat5bwpbs7pz7q
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean. xctest per bundle: MLXLMTests 1089 XCTest (4 skipped = known LFM2, 0 failures) + 2331 Swift Testing passed (3 known issues, baseline); MLXGuidedGenerationTests 76 passed; MLXFoundationModelsTests 494 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests XCTest 0 failures. swift-format lint -s on Flint.swift and FlintTests.swift: clean (repo-wide lint reports pre-existing AlwaysUseLowerCamelCase findings in other files, e.g. FastVLM.swift, Qwen35.swift, untouched by this change). flintcheck.sh: MATCH x4 (256 ids each), TOOL CALL get_weather present; golden_base.json untouched.
    - next: review
  timestamp: 2026-11-15T12:46:55.443107+00:00
- actor: claude-code
  id: 01m7fkz2xa2ah3c19v8cqpmwb9
  text: |-
    ### commit — changed
    - evidence: local commit on stable, subject "perf(flint): fuse q/k/v and gate/up projections at load (^j4pqtwr)"; files: Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/FlintTests.swift, .kanban/ board changes. swift-format ran; no unrelated files changed. Not pushed. (The sha is reported by the commit step's reply, since a comment inside the commit cannot name its own sha.)
    - next: transient load peak (oQ8e 42.7→70.2 GB) tracked in ^75rm5ds.
  timestamp: 2026-11-15T12:47:26.122529+00:00
position_column: doing
position_ordinal: '80'
title: 'Flint: fuse q/k/v, shared-expert gate/up and routed-expert gate/up projections at load'
---
## Why
Flint decode is bound by kernel count, not bandwidth. On mlx-community/flint-oQ4e (rev 2310ad4e) at 2k context, decode runs at ~7.7 ms/token (~125 tok/s) against a ~2.4 ms bandwidth floor (~2.1 GiB read per token). That is roughly 1,800 kernels per token, with the GPU busy only 74%. Fewer dispatches helps on both the GPU and the CPU side.

## What
In `Libraries/MLXLLM/Models/Flint.swift`:
- **q/k/v:** concatenate `q_proj`, `k_proj` and `v_proj` into one projection (rows, scales and biases along axis 0), then split the output. oQ4e quantizes all three 8-bit/gs64 on all 49 layers.
- **Shared expert:** fuse `gate_proj` and `up_proj` (8-bit/gs128 on all 47 MoE layers).
- **Routed experts:** move `switch_mlp` gate and up to `FusedGateUpSwitchGLU` (both 4-bit/gs64). Sanitize must build `gate_up_proj`.
- **Fallback:** fuse only when the bits, group size and mode of the parts match. Otherwise keep separate layers, because other Flint quants (oQ8e, mlx-lm layouts) may differ. Do it in `sanitize` or at load, so the checkpoints don't change.

## Acceptance
- Greedy token ids match the current code on the flintcli raw and chat prompts, for both oQ4e and oQ8e.
- Logits match in float32 (`MLX_ENABLE_TF32=0`) within the existing parity tolerance.
- Decode ms/token improves against a same-session baseline using scratchpad flintcli `bench <prompt_706.txt> 2048,2048,16384 200`. Any improvement counts; there is no fixed target.
- `FlintTests` are extended with a fused vs unfused equivalence test on a tiny random config.
- All five test bundles are green.
- ChatSession multi-turn prompt-cache reuse is unchanged: follow-ups prefill only the new message.

Note: coordinate model loads with the foundationmodelsacpagent session (SWE-bench run). #flint-perf
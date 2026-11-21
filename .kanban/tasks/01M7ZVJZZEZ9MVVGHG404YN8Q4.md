---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m800y2zkgwehb2k4r909bajb
  text: |-
    Research (picked up):
    - oMLX reference: jundot/omlx omlx/patches/inkling2/dspark.py (model + Markov head + fold), omlx/speculative/dflash/drafter.py (_draft_batched: the block forward), dflash/variants.py, dflash/selection.py select_markov, patches/inkling2/model.py (target capture + mtp_partial_rollback). Others exist: vllm, sglang, max, ollama draft.go, draw-things Inkling2SmallDSpark.swift.
    - Drafter math: per layer a softmax(fuse_weight)[layer] mix of the 4 raw taps -> RMSNorm(dflare_norm) -> k_proj_ctx/v_proj_ctx, k_norm_ctx, RoPE(theta 1e6, non-traditional) at the absolute position. Block = anchor embedding (target embed_norm(embed)) + 7 x mask_embed, bidirectional over itself and the newest sliding_window_size-1 = 1023 context rows (every layer, local or not). draft_hidden = norm(h)[:,1:]; base logits = target lm_head (raw); Markov rnn_hidden head: state r=256, inputs [state, markov_w1(prev), draft_hidden[j], f_rnn], f_rnn = RMSNorm(dflare_norm of markov) of the fuse_weight[3] mix of the newest committed context row; bias = tanh-latent @ (lm_head @ hidden_out).T + lm_head @ hidden_out.bias (folded once), then softcap 30 and the padded-vocab mask.
    - Taps [5, 23, 41, 48]: hidden entering visits 5/23/41 and the final-norm output (visit 48 = number of visits).
    - mlx-community/Inkling-2-Mini-oQ4e-mtp target weights use the oMLX layout (model.layers.N.attn.in_proj, k_conv, loop_down...), which the fork's Inkling2 does not load; the fork loads the HF layout (Inkling-2-Mini-4bit). mtp.safetensors (bf16, top-level keys layers./fuse_weight/markov_head./mask_embed/norm.) is quantization-agnostic, so the drafter loads from the oQ4e repo and drafts for the 4bit target.
    - Fork framework: verify is staged or in place. A recurrent leaf (ArraysCache) refuses staging, so Inkling needs the in-place path: SpeculativeCacheRewindModel + a convolution cache that rewinds from the verify windows (oMLX keeps `_omlx_verify_xp` windows for 1 < L <= 16) and in-place gating that sees recurrent leaves inside CacheList.
  timestamp: 2026-11-21T21:41:55.827883+00:00
- actor: claude-code
  id: 01m804z40entccrdacw4td60qf
  text: |-
    Implementation landed (not committed):
    - Inkling2DSparkDraftModel (MLXVLM/Models/Inkling2DSpark.swift): ResumableMTPDrafterModel; state = 3 RotatingKVCache context rings (maxSize window-1=1023, rewind = block) + a 1-row ring for the Markov context mix; Markov fold (lm_head @ hidden_out) computed once per target under a lock. Loads mtp.safetensors alone (AdditionalWeightFilesProviding + excluded target namespaces). Registered: Inkling2DSparkRegistration, MTPDrafterRegistration.inkling2, MTPDrafterRegistry.inkling2_mini_oq4e_mtp.
    - Target: taps [5,23,41,48] emitted via callAsFunction(text,cache,state) and chunked prepare (defaults to the checkpoint's own mtp_config taps when the state names none, so ChatSession checkpoint feeds work). SpeculativeConvolutionCache (ArraysCache subclass, saves as ArraysCache) keeps verify input windows; trim recuts states. In-place gate now sees recurrent leaves inside CacheList.
    - Verify cost: first measured verify via prefill path L=4 27 ms vs decode 13 ms -> MTP 0.67x/0.79x. Added a sequential decode path: 2..16 tokens of one sequence run through the decode kernels as rows (chained conv states, causal multi-query decode attention with per-row params, two-phase expert conv, per-row tau). Bit-exactness needed: (1) projections/lm_head through a non-contiguous [L,1,D] view so MLX runs one qmv per row (MLX folds rows into qmv_wide otherwise; MultiRowQuantizedMatvec does not run for K=2304, 2304 % 512 != 0); (2) router logits via a small row-independent kernel used by decode too (batched gemv is not row-exact); (3) fused expert kernels for up to 16 rows. Real model: verify L=2/4/8 logits == decode steps bit for bit.
    - Overlap: asyncEval every 4 layer visits inside a verify pass hides the CPU graph build (verify L=4 23.9 -> 17.9 ms).
    - What did not work: MultiRowQuantizedMatvec scope (canRun false for K=2304); router via per-row matmul loop (exact but extra nodes; replaced by the kernel).
  timestamp: 2026-11-21T22:52:23.950538+00:00
- actor: claude-code
  id: 01m8063r3txvyjkvymj9jq153x
  text: |-
    Results (M5, Release IntegrationTesting, target mlx-community/Inkling-2-Mini-4bit + drafter mtp.safetensors from mlx-community/Inkling-2-Mini-oQ4e-mtp @3eea0427; the oQ4e *target* layout is not loadable by the port -> follow-up ^2v3bvtz):
    - Greedy parity (block 4, 160 tokens): 3/3 prompts equal (code 99/180, prose 79/237, arithmetic 95/192 drafts accepted).
    - Verify vs decode steps, real model: L=2/4/8 max |logit diff| = 0.0.
    - Throughput (median of 3, after warm-up):
      short (31-tok prompt, story, 256 tok): plain 81.5 tok/s; block 3 92.2 (1.13x, acc 0.309, 1.62 tok/round); block 4 82.8 (1.02x, acc 0.232, 1.69); block 5 74.4 (0.91x); block 6 66.3 (0.81x); block 8 54.5 (0.67x).
      long (5077-tok code prompt, 128 tok): plain 77.6 tok/s; block 3 111.8 (1.44x, acc 0.576, 2.15 tok/round); block 4 106.1 (1.37x, acc 0.456, 2.35); block 5 92.5 (1.19x); block 6 85.7 (1.10x); block 8 90.7 (1.17x, 3.17 tok/round).
    - Round cost profile (short ctx): decode 13.5 ms, verify L=2 13.0, L=4 17.9, L=8 28.4 ms; draft block 4 2.0 ms, block 8 3.2 ms.
    - Chat (3 turns, MTP block 4 vs plain session): cached 248/330 prompt tokens on follow-ups, drafts accepted every turn, no passthrough, answers identical, decode 110.9/101.9/115.3 tok/s vs plain 82.4/81.6/80.8.
    - Past the 1024-token sliding window parity holds as bf16 near ties only (ring vs temporal key order) -> follow-up ^2r778tc. Cheaper verify rows -> ^f0fe9ym.
    - Existing Inkling2CheckpointIntegrationTests, Inkling2ToolLoopIntegrationTests, Inkling2FoundationModelsToolLoopTests pass. Plain decode router logits now come from a row-independent kernel (also used by decode) so a verify routes bit-exactly; plain short decode 79.5-81.5 tok/s.
    - Naming: the card's `Inkling2DraftModel` (file Libraries/MLXVLM/Models/Inkling2Draft.swift).
  timestamp: 2026-11-21T23:12:24.186573+00:00
- actor: claude-code
  id: 01m8063vnv1ae6qbbd31v99hdg
  text: |-
    ### implement — changed
    - evidence: 13 files — Libraries/MLXVLM/Models/Inkling2Draft.swift (new), Libraries/MLXVLM/Models/Inkling2.swift, Libraries/MLXVLM/Models/Inkling2Kernels.swift, Libraries/MLXVLM/README.md, Libraries/MLXLMCommon/ArraysCache.swift, Libraries/MLXLMCommon/KVCache.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/MTPDrafterModelFactory.swift, Tests/MLXLMTests/Inkling2DraftTests.swift (new), Tests/MLXLMTests/Inkling2Tests.swift, IntegrationTesting/IntegrationTestingTests/Inkling2MTPIntegrationTests.swift (new), IntegrationTesting/IntegrationTestingTests/Inkling2CheckpointIntegrationTests.swift. xctest: MLXLMTests 2556 ST + 1258 XCTest pass (3 pre-existing known issues), MLXGuidedGenerationTests 86, MLXFoundationModelsTests 562, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11; Inkling2MTPIntegrationTests all pass.
    - next: /review
  timestamp: 2026-11-21T23:12:27.835397+00:00
position_column: doing
position_ordinal: '80'
title: 'Inkling 2 Mini: MTP block drafter (mtp.safetensors) in the fork''s speculative framework'
---
Wire Inkling 2 Mini's shipped MTP draft head into the fork's speculative decoding. The head is `mtp.safetensors`, kept in `mlx-community/Inkling-2-Mini-oQ4e-mtp` (15.7 GiB) and the official thinkingmachines/Inkling-2-Mini.

Work on branch `inkling2-mini`. From the earlier assessment:
- It is a block drafter with block size 8, the same kind as the fork's LFM2.5 DSpark drafter. The drafter interfaces, the target-layer-ids hook and the factory registration all exist.
- It needs a new drafter model, `Inkling2DraftModel`. The reference implementation is in oMLX, not mlx-lm; find it. Its pieces:
  - separate context projections
  - extra norms
  - a learned mask embedding
  - a 4×4 layer-fusion weight
  - an "rnn_hidden" Markov head
  - sliding-window layers
- `target_layer_ids` [5, 23, 41, 48] count layer *passes* through the looped stack (48 passes per token). Inkling2TextModel must capture hidden states at those passes and emit the MTP state keys.
- Rejected drafts must rewind the convolution states (ArraysCache) and the rotating attention caches. The fork's Qwen3.5 recurrent path is the precedent.
- The PR's fast decode kernels handle one token only, so a 9-token verify takes the prefill path. Measure the round cost, and add a multi-token verify kernel path if that is needed to make MTP a net win.
- MTP must keep multi-turn prompt-cache reuse working, including after drafter rejections.

Acceptance:
- [x] Unit tests, written first: drafter weight loading, block draft shapes, rewind after a partial accept, and greedy MTP equal to plain greedy on a tiny model.
- [x] Real model (`Inkling-2-Mini-oQ4e-mtp`): MTP greedy token ids equal plain greedy on 3 prompts.
- [x] Measured plain vs MTP decode at short and ~4k context, with the acceptance rate. MTP is faster, or there is a recorded NO-GO with numbers.
- [x] Multi-turn cache reuse works with MTP on.
- [x] All 5 bundles are green.

#inkling
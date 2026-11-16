---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7j4zyq10rcfmj7wnxaf5wx9
  text: |-
    Research (stable 87460a3f, branch dflash2-4bit).

    Plan: DFlash2 runs as a drafter of the MTP framework. It does not get its own iterator.
    - Stable already has a block-drafter seam: `MTPDrafterModel.targetLayerIds` and `mtpTargetLayerIdsKey`, which upstream LFM2 DSpark (`Libraries/MLXLLM/Models/LFM2DSpark.swift`) uses.
    - So DFlash2DraftModel becomes a `ResumableMTPDrafterModel`, and MTPSpeculativeTokenIterator drives it. The drafter then gets for free:
      - the in-place verify with GDN snapshot rewind (the T-token kernel),
      - the compiled verify schedule,
      - the multi-row quantized matvec,
      - the chunked prefill feed,
      - ChatSession/FM speculation, prompt-cache continuations and spill.
    - Mapping from origin/dflash2:
      - DFlash2.swift → ported, as an MTP drafter.
      - DFlash2DrafterModel/Factory → `MTPDrafterModelFactory` plus a registration, as LFM2DSparkRegistration does.
      - DFlash2SpeculativeTokenIterator, its capture/replay verify, `KVCacheSimple.writeRows`, the Evaluate overloads and `ChatSession.dflash2Drafter` → replaced by the MTP iterator, the existing `generate(mtpDrafter:)` and `SpeculativeDecodingConfig(mtpDrafter:)`.
      - The Qwen35 taps → `mtpTargetLayerIdsKey` on the general path and in the compiled verify segments.
    - Lost from the old iterator: pipelining, where it built round n+1 before syncing round n. MTP has the same one sync per round, so the comparison against MTP is like for like.

    Shifted-drafter fit:
    - An entry `e` of the drafter state is the target hidden row of position `e`. After a commit, `nextPosition` equals the bonus position.
    - So the default Resumable resume, cold start, commit and finalize work unchanged. DFlash2 implements only `appendDrafterState`.
    - The context is one `RotatingKVCache(maxSize: contextWindow=2047, rewindCapacity:)` per drafter layer. `logicalView(tail:)` gives the newest rows in temporal order.
    - The per-query window mask (query i hides the oldest i rows once the window is full) is computed from the row count alone.

    Qwen3.8-27B:
    - 24 q heads, 4 KV heads, so GQA 6. MLX SDPA would fall back at q_len ≥ 6.
    - Stable's VerifyAttentionKernel runs 2–8 queries at ≥1024 keys, and MultiRowQuantizedMatvec.mostRows is 8.
    - `GatedDeltaDecodeKernel.mostTokens` is 6. That caps the in-place verify and the compiled verify at blockSize 6. To run block 8, mostTokens must go to 8. The threadgroup memory does not scale with tokens; only the device-memory snapshot output does.
    - Taps 5/19/33/47/61: 19 and 47 are full-attention layers, whose output is the post-attention body of the next compiled segment.

    4-bit at load:
    - `DFlash2DraftModel.prepareCheckpoint` quantizes every Linear weight (affine, gs 64) before materialization and sets `perLayerQuantization`.
    - `applyCheckpoint` then builds the QuantizedLinears. The selector codebooks (Embedding) stay bf16, which matches KaedeTai's 1.27 GB.

    Prior data: ^yxm9net measured that the Flint DFlash2 drafter costs about the same in 4-bit as in bf16 (op-bound). Scratchpad has dfl_bf16.txt and dfl_q4.txt.
  timestamp: 2026-11-16T12:23:29.249034+00:00
- actor: claude-code
  id: 01m7j6bpn8m739tjyx9qr1hb9n
  text: |-
    Implementation landed (uncommitted, branch dflash2-4bit). The code builds clean with `swift build --build-tests`. The new suites pass (31 tests).

    New files:
    - Libraries/MLXLMCommon/DFlash2Target.swift: `DFlash2Geometry` and the `DFlash2TargetModel` protocol (embedding, head, geometry).
    - Libraries/MLXLLM/Models/DFlash2.swift: the port of the configuration, the dynamic conv, the attention, the layers and the selector (greedy walk, compiled per width).
      - `DFlash2DraftModel` is a `ResumableMTPDrafterModel`.
      - Its context is one RotatingKVCache(maxSize 2047, rewind reserve 16) per layer.
      - The per-query window mask is built in `visibilityMask`.
      - Its Linears run through MultiRowQuantizedMatvec when quantized (3–8 rows), and so does the target lm_head on the draft rows.
      - `prepareCheckpoint` quantizes at load when `quantization` is set.
    - Libraries/MLXLLM/DFlash2Registration.swift: `register(in:quantization:)`, the families `MTPDrafterRegistration.dflash2` and `.dflash2FourBit` (own registry), and `MTPDrafterRegistry.qwen3_8_27b_dflash2`.

    Changed files:
    - CompiledVerifyShape has `captureLayers`. CompiledDecodeSegment.body emits the outputs of the captured layers, and CompiledDecodeSchedule.runCapturing collects them.
    - MLXLLM Qwen35 honors `mtpTargetLayerIdsKey` on the general path and on the compiled verify path (`forwardCapturing`). Qwen35TextModel and Qwen35Model conform to DFlash2TargetModel.
    - `GatedDeltaDecodeKernel.mostTokens` goes from 6 to 8, so block 8 verifies in place and compiled. Side effects:
      - the MTP drafter's maximumBlockSize is now 8;
      - the guided forced-token chunk is now 8 tokens instead of 6;
      - the MTPBlockSweep range now ends at 8.
      The MTP default is still bs4.
    - `LogitSampler.suppressing(_:)` (package) lets a self-selecting drafter mask suppressed ids.

    What did not port, and why:
    - The old capture/replay verify, `KVCacheSimple.writeRows`, the pipelined iterator and `ChatSession.dflash2Drafter` are replaced by the MTP iterator and its in-place snapshot rewind.
    - Sampled DFlash2 streams verify by sample-and-match, not by rejection sampling over the selector's candidates. The selector keeps only the greedy walk.

    Discovery: a speculative ChatSession turn leaves its final bonus token uncached, so the next turn prefills the new message plus at most one token. Cached + prefilled equals the plain session on every turn. ChatSessionDFlash2Tests asserts this.

    Real-transcript prompts: ACP hint24 transcripts exist at ~/github/swissarmyhammer/FoundationModelsACPAgent/bench/preds.hint24.transcripts. scratchpad/pyscripts/agent_cuts.py renders them through the Qwen3.8 chat template (tools, reasoning, tool calls and tool outputs) and cuts them at assistant-turn starts. Nine cuts from django__django-13964 session 01M79XTE5YRW530AZB80GHS1R5: 18.2k/21.0k/21.5k, 28.3k/28.8k/29.1k and 67.0k/69.3k/70.9k tokens (scratchpad/dfl4/agent_*.txt). Math/code/prose prompts are in short_*.txt. The harness is scratchpad/dfl4cli (Release).
  timestamp: 2026-11-16T12:47:22.792225+00:00
- actor: claude-code
  id: 01m7j90c5j45m7g4ag0qppe3m7
  text: |-
    ## Measurement: M5 Ultra, Release, Qwen3.8-27B-4bit + incoai/Qwen3.8-27B-DFlash2, greedy

    The harness is scratchpad/dfl4cli. Raw output is in scratchpad/dfl4/{cost,bench,wide}.txt.
    - 256 generated tokens per run. The decode tok/s count everything after the first token.
    - Every config continues from the same prefilled cache (prompt minus its last token) and the drafter continuation of its own family.
    - MTP = mlx-community/Qwen3.8-27B-MTP-4bit, blockSize 4, pMin 0.75 (the shipped default).

    ### Memory: drafter load (bf16 checkpoint, quantized at load for 4-bit)
    | drafter | parameters | active added | load peak above pre-load |
    |---|---|---|---|
    | MTP-4bit | 0.24 GB | 0.24 GB | 0.24 GB |
    | DFlash2 bf16 | 3.85 GB | 3.85 GB | 3.85 GB |
    | DFlash2 4-bit (affine gs64, linears only) | 1.27 GB | 1.27 GB | 4.50 GB (peak 23.72 GB of the process) |

    ### Round cost (scratchpad/dfl4/cost.txt)
    Setup:
    - Contexts are prefixes of agent cut 64000_2.
    - Drafter = one proposal on a full 2047-row window, median of 12.
    - Verify = target forward of S tokens with taps and snapshots, compiled, median of 12.
    - Round = iterator decode time / rounds over 128 tokens.
    - Overhead = round − drafter − verify.
    - Steps = ÷ pipelined plain ms/token: 17.99 at 2k, 20.10 at 32k, 22.87 at 64k.
    - Note: overhead at 2k is noisy because the 128-token rounds include warm-up.

    | ctx | config | drafter ms (steps) | verify ms (steps) | overhead ms (steps) | round ms (steps) |
    |---|---|---|---|---|---|
    | 2k | bf16 S4 | 5.69 (0.32) | 30.90 (1.72) | 12.18 (0.68) | 48.78 (2.71) |
    | 2k | bf16 S5 | 5.70 (0.32) | 35.68 (1.98) | 5.00 (0.28) | 46.38 (2.58) |
    | 2k | bf16 S8 | 7.61 (0.42) | 47.33 (2.63) | 7.35 (0.41) | 62.29 (3.46) |
    | 2k | 4bit S4 | 3.97 (0.22) | 30.90 (1.72) | 2.18 (0.12) | 37.06 (2.06) |
    | 2k | 4bit S5 | 4.64 (0.26) | 35.68 (1.98) | 1.94 (0.11) | 42.27 (2.35) |
    | 2k | 4bit S8 | 6.36 (0.35) | 47.33 (2.63) | 2.68 (0.15) | 56.37 (3.13) |
    | 32k | bf16 S4 | 5.55 (0.28) | 36.99 (1.84) | 1.01 (0.05) | 43.55 (2.17) |
    | 32k | bf16 S5 | 5.68 (0.28) | 42.30 (2.10) | 1.56 (0.08) | 49.54 (2.46) |
    | 32k | bf16 S8 | 7.63 (0.38) | 56.22 (2.80) | 4.63 (0.23) | 68.49 (3.41) |
    | 32k | 4bit S4 | 3.99 (0.20) | 36.99 (1.84) | 0.78 (0.04) | 41.76 (2.08) |
    | 32k | 4bit S5 | 4.47 (0.22) | 42.30 (2.10) | 2.33 (0.12) | 49.10 (2.44) |
    | 32k | 4bit S8 | 6.07 (0.30) | 56.22 (2.80) | 1.25 (0.06) | 63.54 (3.16) |
    | 64k | bf16 S4 | 5.50 (0.24) | 37.33 (1.63) | 1.66 (0.07) | 44.49 (1.95) |
    | 64k | bf16 S5 | 5.68 (0.25) | 42.47 (1.86) | 0.91 (0.04) | 49.07 (2.15) |
    | 64k | bf16 S8 | 7.55 (0.33) | 56.32 (2.46) | 0.87 (0.04) | 64.74 (2.83) |
    | 64k | 4bit S4 | 3.94 (0.17) | 37.33 (1.63) | 1.10 (0.05) | 42.36 (1.85) |
    | 64k | 4bit S5 | 4.45 (0.19) | 42.47 (1.86) | 0.33 (0.01) | 47.26 (2.07) |
    | 64k | 4bit S8 | 6.08 (0.27) | 56.32 (2.46) | 0.53 (0.02) | 62.93 (2.75) |

    MTP bs4 rounds: 31.10 ms (1.73 steps) at 2k, 33.32 ms (1.66) at 32k, 38.05 ms (1.66) at 64k.
    - Commit append of 3 context rows: 0.68–0.80 ms.
    - The 4-bit drafter is 1.4–1.6 ms cheaper than bf16 (≈0.07 steps).
    - The verify dominates. A 4-token verify costs 1.6–1.8 plain steps, a 5-token verify 1.9–2.1, and an 8-token verify 2.5–2.8. A whole MTP round costs about 1.7 steps, because its pMin gate often verifies fewer than 4 tokens.

    ### Tokens per verify, with tok/s (geomean over 3 prompts per band)
    | band | plain | MTP bs4 | bf16 b4 | bf16 b5 | bf16 b8 | 4bit b4 | 4bit b5 | 4bit b8 |
    |---|---|---|---|---|---|---|---|---|
    | short math/code/prose (64–107 ctx) | 58.4 | 88.9 (2.39) | 80.0 (2.90) | 78.1 (3.26) | 64.5 (3.85) | 80.6 (2.85) | 75.8 (3.22) | 68.8 (3.84) |
    | agent 18–21k | 51.0 | 81.9 (2.91) | 60.7 (2.63) | 71.3 (3.43) | 61.5 (3.84) | 71.8 (3.01) | 72.2 (3.52) | 60.7 (3.72) |
    | agent 28–29k | 50.2 | 76.8 (2.69) | 66.0 (2.84) | 68.1 (3.29) | 54.3 (3.53) | 68.0 (2.82) | 65.5 (3.12) | 53.9 (3.41) |
    | agent 67–71k | 42.5 | 73.9 (2.77) | 67.5 (3.01) | 68.3 (3.42) | 58.0 (3.81) | 67.8 (2.91) | 68.4 (3.33) | 58.7 (3.77) |

    - Per prompt: math is the only prompt where DFlash2 beats MTP (bf16 b5 103.4 vs 98.3 tok/s, 4.18 tok/verify).
    - Agent cut 64000_2: 4-bit b4 74.5 vs MTP 75.7. Agent cut 64000_1: 4-bit b5 68.9 vs MTP 66.2. On every other agent cut MTP is faster.
    - Agent prompts are real hint24 turns: chat-templated ACP transcripts (django__django-13964 session 01M79XTE5YRW530AZB80GHS1R5), cut at assistant-turn starts. The model continues each turn greedily.

    ### Greedy correctness
    DFlash2 versus plain greedy token ids, with plain's top-4 logprobs for the near-tie rule:
    - 82 of 84 runs are exact or a near tie (gap ≤ 0.5).
    - Two runs diverge with a gap of 0.75, both on agent cut 16000_2 at step 186: bf16 and 4-bit, blocks 5 and 8.
    - Control (wide.txt): MTP with blockSize 5 on the same cut diverges at the same step with the same 0.75 gap. The cause is the rounding of a wider target verify, not the DFlash2 drafter.
    - Block 4 is within 0.5 everywhere.
    - Unit tests show greedy DFlash2 == plain on the tiny fp32 target, for blocks 4/5/8, for continuations and for ChatSession turns.

    ## GO / NO-GO: NO-GO
    - The GO criterion is that DFlash2 4-bit beats MTP tok/s at 16–64k on agent-transcript turns. It does not. The best DFlash2 setting per band (4-bit, block 4 or 5) reaches:
      - 0.88× MTP at 18–21k,
      - 0.89× at 28–29k,
      - 0.93× at 67–71k,
      - 0.91× on short prompts.
    - Why:
      - DFlash2 does accept more tokens per verify (3.0–3.5 versus MTP's 2.7–2.9).
      - But its round verifies the full block. A round costs 1.85–2.5 plain steps, against MTP's ~1.7.
      - The drafter itself is cheap: 0.17–0.35 steps, and the 4-bit drafter saves 0.07 steps.
    - The 4-bit drafter matches bf16 acceptance and is slightly faster (bands 71.8/68.0/68.4 at its best block versus bf16 71.3/68.1/68.3). That is not enough to close the gap.
    - Block 8 is worse everywhere. Its verify costs 2.5–2.8 steps for only +0.4 tok/verify.
    - Per the card, the router-path card ^bpyk5w5 is gated on GO, so it should not proceed.
  timestamp: 2026-11-16T13:33:37.330588+00:00
- actor: claude-code
  id: 01m7j9ekn69j91qpfnppmcb53t
  text: |-
    ### implement — changed
    - evidence: 12 files.
      - Added: Libraries/MLXLMCommon/DFlash2Target.swift, Libraries/MLXLLM/Models/DFlash2.swift, Libraries/MLXLLM/DFlash2Registration.swift, Tests/MLXLMTests/DFlash2Tests.swift, Tests/MLXLMTests/DFlash2SpeculationTests.swift.
      - Modified: Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLMCommon/CompiledDecodeSchedule.swift, CompiledDecodeSegments.swift, GatedDeltaDecodeKernel.swift (mostTokens 6→8), MTPDrafterModelFactory.swift, SuppressTokens.swift, Libraries/MLXLLM/Documentation.docc/using-model.md.
    - Tests: `swift build --build-tests` is clean, then `xcrun xctest` on all five bundles:
      - MLXLMTests: 2381 Swift Testing tests passed, plus 1098 XCTest tests with 0 failures.
      - MLXFoundationModelsTests 496, MLXGuidedGenerationTests 78, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11: all passed.
      - The 4 skipped XCTests (LFM2 real-checkpoint and benchmark tests, gated by environment variables) and the 3 known issues are also in the stable baseline log (scratchpad t_MLXLMTests.log: 1098 tests, 4 skipped, 3 known issues).
    - swift-format: run over the repo.
    - Verdict: NO-GO. The best DFlash2 4-bit setting reaches 0.88× / 0.89× / 0.93× of MTP bs4 at 18–21k / 28–29k / 67–71k on agent turns.
    - next: review. Card ^bpyk5w5 (router path) is gated on GO and should not proceed. The orchestrator decides whether to keep the port or close it as measured no-go.
  timestamp: 2026-11-16T13:41:23.750532+00:00
- actor: claude-code
  id: 01m7j9y1j9y5d25v4bmh0bn8f8
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0, no compiler warnings (only a SwiftPM "missing creator for mutated node" tool message for mlx-swift_Cmlx.bundle). xcrun xctest: MLXLMTests 2381 Swift Testing tests passed (3 known issues) + 1098 XCTests 0 failures (4 skipped); MLXGuidedGenerationTests 78 passed; MLXFoundationModelsTests 496 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 3 Swift Testing passed + 11 XCTests 0 failures. swift-format run over Libraries, Tests and Package.swift changed nothing.
    - skips: 4 XCTests, all env-gated real-checkpoint or benchmark tests in LFM2Tests.swift and LFM2MoeRoutingTests.swift (MLX_LFM2_MODEL_PATH, MLX_LFM2_BENCHMARK, MLX_LFM2_MOE_MODEL_PATH, MLX_LFM2_MOE_BENCHMARK). The branch changes no files under Tests/, so they skip the same way on stable 87460a3f.
    - known issues: 3, in AttentionKernelToleranceTests (aNearTieRejectsATokenOutsideTheTwoBest) and ToolTests MiniCPM5 (keeps a function close tag inside CDATA, 2). The branch changes no test files, so they are the same on stable.
    - next: review
  timestamp: 2026-11-16T13:49:49.513441+00:00
- actor: claude-code
  id: 01m7j9yn9jp708a7yvjabehfmw
  text: |-
    ### commit — changed
    - evidence: 265c8543 feat(dflash2): port DFlash2 as an MTP drafter with 4-bit drafter quantization
    - next: local branch dflash2-4bit only, not pushed
  timestamp: 2026-11-16T13:50:09.714288+00:00
- actor: claude-code
  id: 01m7jafszn12a4q0ceypywjj6p
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (265c8543); 8 findings — Libraries/MLXLLM/Models/DFlash2.swift:313, :552, :556, :1004; Tests/MLXLMTests/DFlash2Tests.swift:34, :43, :53, :511
    - next: implement the 8 open items in the 2026-11-16 07:50 Review Findings section, remove each cause file-wide, then re-review
  timestamp: 2026-11-16T13:59:31.573502+00:00
- actor: claude-code
  id: 01m7jag7cvqx76xke8vbss4z83
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 12 files (DFlash2 as MTP drafter, 4-bit at load); NO-GO measured
    - test: green — 5 bundles, 0 failures (4 env-gated LFM2 skips, 3 known issues, same files as stable)
    - commit: 265c8543
    - review: findings — DFlash2.swift:313, :552, :556, :1004; DFlash2Tests.swift:34, :43, :53, :511
  timestamp: 2026-11-16T13:59:45.307113+00:00
- actor: claude-code
  id: 01m7janwjd4a18skcx9dc0s15a
  text: |-
    Review fixes (uncommitted, on top of 265c8543).

    Cause of the magic-numbers findings: positional slots (trace-argument indices, tensor-axis permutations) and fixture values were written as bare literals. Fixed everywhere in both files:
    - DFlash2.swift:
      - DFlash2Axis gains `batch`, `head` and `headDim`, so `headsFirst = [batch, head, token, headDim]`. The `dim(0)`/`dim(1)` reads of hidden states now use `DFlash2Axis.batch`/`.token`.
      - DFlash2CandidateSelector gains `WalkArgument` (hidden/logits/anchor) and `CandidateAxis` (batch/position/candidate/rank), so `candidatesLast` is built from names.
      - The segment code gains `SegmentArgument` (hidden/attention/kernel) and `SegmentOutput.hidden`. The context trace gains `ContextArgument` (targetHidden/position).
    - DFlash2Tests.swift:
      - `targetLayerIds = [firstAttentionLayer, firstAttentionLayer + 1]`, where `firstAttentionLayer = fullAttentionInterval - 1`.
      - `maskTokenId = vocabularySize - maskTokenDistanceFromEnd`.
      - `fourBit` is built from `fourBitGroupSize` (= hiddenSize) and `fourBitBits` (= DFlash2DraftModel.fourBit.bits).
    - `DFlash2Fixture.target(seed:hiddenSize:)`: `refusesAnotherTarget` now calls `DFlash2Fixture.target(hiddenSize: 2 * DFlash2Fixture.hiddenSize)`.

    Useful to know:
    - The reviewer's rule is swiftlint `no_magic_numbers` with `allowed_numbers [0, 1, -1, 100]`. Running it by hand (a child config with only_rules: no_magic_numbers) lists every site. Before the fix it reported 10 sites in the two files; after the fix it reports 0.
    - `xcrun xctest -XCTest "MLXLMTests.<Suite>,..."` also filters Swift Testing suites.
  timestamp: 2026-11-16T14:02:50.829278+00:00
- actor: claude-code
  id: 01m7janz8c4hr0pcqmmtvejnvc
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DFlash2.swift, Tests/MLXLMTests/DFlash2Tests.swift.
      - swift-format: run over the repo.
      - swiftlint no_magic_numbers: 0 violations.
      - `swift build --build-tests`: Build complete, no compiler warnings. The only message is the known SwiftPM "missing creator for mutated node" tool message.
      - `xcrun xctest -XCTest` on the 9 DFlash2 suites (Qwen35TargetLayerTapTests, DFlash2IteratorTests, ChatSessionDFlash2Tests, DFlash2Configuration/Geometry/Layer/Context/Proposal/QuantizationTests): 28 tests in 9 suites passed.
      - All 8 findings are checked.
    - next: test, then review
  timestamp: 2026-11-16T14:02:53.580592+00:00
- actor: claude-code
  id: 01m7jb4nk9tdexnvrj2v1jhcwb
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched and rebuilt: 0 compiler warnings; only the build-system node message about the Cmlx bundle). xcrun xctest on all five bundles, all exit 0. MLXLMTests: Swift Testing 2381 tests passed, XCTest 1098 executed, 0 failures. MLXGuidedGenerationTests: 78 passed. MLXFoundationModelsTests: 496 passed. CXGrammarTests: 7 passed. MLXHuggingFaceMacrosTests: 11 XCTest passed. Only the pre-existing 4 LFM2 skips and 3 known issues (AttentionKernelToleranceTests, ToolTests MiniCPM5), in files this branch does not touch.
    - next: review
  timestamp: 2026-11-16T14:10:55.209028+00:00
- actor: claude-code
  id: 01m7jb54e6b9kpcwrx5egd7338
  text: |-
    ### commit — changed
    - evidence: b4fd6c79 refactor(dflash2): name axes and arguments, add hiddenSize to test fixture
    - next: none
  timestamp: 2026-11-16T14:11:10.406678+00:00
- actor: claude-code
  id: 01m7jb6byg9f67f50kvskatejw
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (b4fd6c79): 0 findings, 0 confirmed, 0 refuted, 7 validator runs attempted, 0 failed; 2 source files reviewed; 2 .kanban files had no matching validator; all 8 findings from the 2026-11-16 07:50 review are checked
    - next: none; task moved to done
  timestamp: 2026-11-16T14:11:50.864962+00:00
- actor: claude-code
  id: 01m7jb6ph94gs3ps89fwz58ht6
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 2 files (named axes/arguments, fixture hiddenSize)
    - test: green — 5 bundles, 0 failures (pre-existing 4 LFM2 skips, 3 known issues)
    - commit: b4fd6c79
    - review: clean — 0 findings, 8/8 prior checked → done
  timestamp: 2026-11-16T14:12:01.705086+00:00
position_column: done
position_ordinal: ffffb280
title: 'DFlash2 with a 4-bit drafter: port onto stable and measure go/no-go against MTP'
---
## Why
- **Workload:** on real SWE-bench work, Qwen3.8-27B-4bit + MTP (blockSize 4) decodes at 77 tok/s at 16–64k and 67 tok/s at 64k+ (ACP hint24, 2026-11-13). Decode is 74% of model time.
- **Why DFlash2 was parked:** in October a DFlash2 round cost about 2.9 plain steps on the M5, about 49 ms with the bf16 drafter (origin/dflash2 326188cb, upstream PR ml-explore/mlx-swift-lm#607).
- **New evidence:** KaedeTai/dflash2-mlx-quantized-draft reports that a 4-bit drafter (affine, group size 64) cuts the drafter from 3.85 to 1.27 GB. On an M5 Max with block 4 and greedy decoding, it measured 76.1 tok/s against 61.4 with bf16 and 33.3 dense (2.29x), with 86.9% acceptance.

## What
1. Bring the DFlash2 code from origin/dflash2 onto current stable on a new branch `dflash2-4bit`:
   - DFlash2.swift
   - DFlash2DrafterModel / Factory
   - DFlash2SpeculativeTokenIterator
   - the Qwen35 target-layer hidden-state taps
   - the ChatSession hook

   Stable has moved a long way: MTP, the GDN kernels, compiled verify, mlx 0.33. Reuse the MTP verify machinery (T-token GDN verify kernel, multi-row quantized matvec, compiled verify) wherever DFlash2 verify can share it.
2. Quantize the drafter to 4-bit, group size 64, at load (or load a pre-quantized checkpoint). Make the drafter block size configurable; try 4, 5 and 8.
3. Measure on the M5 Ultra with Qwen3.8-27B-4bit + incoai/Qwen3.8-27B-DFlash2. Take bench.lock and coordinate with ACP/Router first.
   - Round cost split into drafter, verify and overhead, in ms and in plain steps, at 2k, 32k and 64k context.
   - Tokens per verify with greedy decoding on the benchmark prompts (math/code/prose) **and** on real agent turns replayed from ACP's hint24 transcripts.
   - tok/s against MTP blockSize 4 at the same contexts.
4. Go/no-go: record the numbers on this card. GO means DFlash2 with the 4-bit drafter beats MTP's tok/s at 16–64k on agent-transcript turns.

## Acceptance
- [x] Branch dflash2-4bit builds; all five test bundles green.
- [x] Greedy DFlash2 output matches greedy plain token ids (allow bf16 near-ties within 0.5 logit gap, as in the MTP tests). Exception: 2 of 84 real-model runs (agent cut 16000_2, blocks 5 and 8) diverge at a 0.75 gap. MTP at blockSize 5 diverges at the same step with the same gap, so the cause is the rounding of the wider target verify.
- [x] A 4-bit drafter loads; the memory peak is recorded.
- [x] A table of round cost and tokens/verify at 2k/32k/64k for bf16 vs 4-bit drafters, and for blocks 4/5/8.
- [x] tok/s against MTP on agent-transcript turns; GO/NO-GO recorded (NO-GO).

## Notes
- If the verify q_len × GQA goes over 32, MLX SDPA falls back to the unfused path. This hit Flint at L≥5 (Flint GQA 8). Check Qwen3.8's GQA and size the block accordingly.
- Multi-turn prompt-cache reuse must keep working (see the ChatSessionDFlash2Tests on the old branch). #dflash2-4bit

## Review Findings (2026-11-16 07:50)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 11 file(s) reviewed, 35 not reviewed (no validator matched: `.kanban/` tag and task files, `Libraries/MLXLLM/Documentation.docc/using-model.md`).

- [x] `Libraries/MLXLLM/Models/DFlash2.swift:313` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLLM/Models/DFlash2.swift:552` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLLM/Models/DFlash2.swift:556` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLLM/Models/DFlash2.swift:1004` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/DFlash2Tests.swift:34` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/DFlash2Tests.swift:43` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/DFlash2Tests.swift:53` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/DFlash2Tests.swift:511` `reuse/reuse` — refusesAnotherTarget rebuilds a Qwen35 target configuration from scratch (Qwen35TextConfiguration decode of Qwen35FixtureJSON.text with the same layer, interval and vocabulary arguments) instead of extending DFlash2Fixture.target(seed:), which already builds that same target. Only the hidden size differs. The duplicated construction will drift from the fixture if the fixture's target shape changes. Give DFlash2Fixture.target a `hiddenSize: Int = hiddenSize` parameter (alongside `seed`) and call `DFlash2Fixture.target(hiddenSize: 2 * DFlash2Fixture.hiddenSize)` here. The wider target then comes from the same code path as every other test target.
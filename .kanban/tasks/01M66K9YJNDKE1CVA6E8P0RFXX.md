---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m67v7n9v52sxrcf5xdka0wmc
  text: |-
    Research (picked up by implement):
    - Why the array mask: MLXVLM Qwen35 `Qwen35Language.Model.forward` calls `createAttentionMask(h:cache:returnArray: true)` and threads an `MLXArray?` through `DecoderLayer` -> `Attention.attend`, which wraps it in `.array(mask[..., 0..<kvSeqLen])`. Nothing non-causal goes in: the LM's `mask:` parameter (input.text.mask) only feeds `getRopeIndex` for M-RoPE positions, never the attention mask; image tokens attend causally. So the array is always the plain causal mask from `cache.makeMask` (or a rotating/window array a `RotatingKVCache` asks for itself). `returnArray: true` was only there because the layer API took an MLXArray.
    - Second, bigger gap: even with `.causal` the VLM attention calls `attentionWithCacheUpdate` -> `MLXFast.scaledDotProductAttention` for a plain KVCacheSimple. It never routes through `PrefillAttentionKernel.scaledDotProductAttention` (-> Verify -> Decode kernels) as text `Qwen35Attention.attend` does. So the VLM also misses DecodeAttentionKernel on plain 1-row decode (compiled decode's attentionCacheStep passes mask nil -> SDPA).
    - Same array-mask pattern in MLXVLM `Qwen35VLMNextNPredictor` (MTP drafter); the MLXLLM drafter uses `createAttentionMask(h:cache:)`.
    - Plan: VLM mask becomes `MLXFast.ScaledDotProductAttentionMaskMode` end to end (`createAttentionMask(h:cache:)` without returnArray, so caches that need an array — rotating overflow — still return one); share the text route (plain cache, not KVCacheAttentionProtocol -> cache.update + PrefillAttentionKernel route; else attentionWithCacheUpdate) as one package function in MLXLMCommon AttentionUtils used by both models; VLM drafter likewise.
  timestamp: 2026-10-31T02:04:49.851796+00:00
- actor: claude-code
  id: 01m67w13spatj01vhfebgfcqaj
  text: |-
    Implementation landed (uncommitted):
    - MLXLMCommon `PrefillAttentionKernel.attentionWithCacheUpdate(queries:keys:values:cache:scale:mask:)` (new package static): the text model's route moved here unchanged (plain non-KVCacheAttentionProtocol cache -> cache.update + PrefillAttentionKernel.scaledDotProductAttention -> Verify -> Decode -> SDPA; else the free `attentionWithCacheUpdate`). MLXLLM `Qwen35Attention.attend` now calls it (no behaviour change).
    - MLXVLM Qwen35: the attention mask is `MLXFast.ScaledDotProductAttentionMaskMode` end to end (`Attention.callAsFunction/attend`, `DecoderLayer.callAsFunction/forward`, `attentionCacheStep` -> `.none`); `Model.forward` builds it with `createAttentionMask(h:cache:)` (no returnArray), exactly as the text model, so a KVCacheSimple verify gets `.causal` and a RotatingKVCache that overflows still gets its array; `.array` masks keep the old kvSeqLen slice. Attention goes through the shared kernel route, so VLM verifies (3-8 rows) reach VerifyAttentionKernel and VLM 1-row decode reaches DecodeAttentionKernel, like the text path. `Qwen35Language.isUnmasked(_:)` replaces the `== nil` checks of the compiled-decode gates.
    - MLXVLM Qwen35MTP predictor: same `createAttentionMask(h:cache:)` as the MLXLLM drafter.
    - Tests: new Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift (VLM attention layer bit-for-bit = VerifyAttentionKernel output with `.causal` for 3...8 rows, with a #require that the kernel's canRun holds; `.causal` vs the old causal mask array within bf16 tolerance for 3...8 rows; model-level VLM verify forward vs plain decode within bf16 + near-tie). The near-tie check moved from Qwen35VerifyAttentionTests into AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie (shared, no copy). Qwen35VLMCompiledDecodeTests: `attentionMask: nil` -> `.none` (signature change).
    - RED: `swift build --build-tests` failed in the new file ("type 'MLXArray?' has no member 'causal'"). GREEN: MLXLMTests 920 XCTest + 2085 Swift Testing passed (Qwen35VLMVerifyAttentionTests passed); MLXGuidedGenerationTests 72, MLXFoundationModelsTests 460, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7 — 0 fail, 0 skip.
    - Note: `swift test --filter` cannot run GPU Swift Testing here (metallib, see CLAUDE.md) and `xcrun xctest -XCTest` does not filter Swift Testing, so single-suite runs mean a whole-bundle run.
    - Release before-run started: Libraries/ stashed (`git stash` "8p0rfxx-before", patch backup in scratchpad), RoundCost + sweep under bench.lock, stash pops after.
  timestamp: 2026-10-31T02:18:43.894970+00:00
- actor: claude-code
  id: 01m67waec5z5yy606cvwh8ecb3
  text: |-
    BLOCKER (permission denial), measurement not done:
    - The Release "before" run (HEAD libraries; my Libraries/ changes are in `git stash` "8p0rfxx-before", backup patches at scratchpad 8p0rfxx-libraries.patch / 8p0rfxx-full.patch) started at 02:18Z under bench.lock (owner "8p0rfxx <pid>") with two -only-testing targets in ONE xcodebuild: MTPRoundCostBreakdownTests and MTPSWEBenchScaleTests/decodeRateFromTwoToOneHundredThousandContextTokens(path:). Swift Testing runs the two suites IN PARALLEL (their log lines interleave), so those timings are contaminated and must not be used.
    - I tried to stop that xcodebuild (pkill) to re-run the suites one at a time; the auto-mode classifier denied it ("Interfere With Workloads"). Per instructions I stopped there.
    - State left: the nohup chain is still running. When xcodebuild ends, locked-run.sh removes bench.lock/owner + rmdir (trap on EXIT), then the chain runs `git stash pop` in this checkout, restoring the Libraries/ changes (result in scratchpad runs/before.stashpop.txt). Until then the working tree has the tests but HEAD libraries, so `swift build --build-tests` would fail on Qwen35VLMVerifyAttentionTests. If the pop fails, `git apply` the libraries patch.
    - Next (needs a human OK to stop the running xcodebuild, or wait for it to finish): run before/after per suite in separate xcodebuild invocations — RoundCost (router + text, 32k/64k/100k verify rows 1–5), sweep (2k–100k), then after only: qwen38FourBitMTPSystemPromptCache() (edit turn 123), MTPGenerationTests, MTPPromptCacheTests, each separately.
  timestamp: 2026-10-31T02:23:49.637871+00:00
- actor: claude-code
  id: 01m67waf1rbt1vzkh31rf7btc0
  text: |-
    ### implement — stuck
    - evidence: 9 files — Libraries/MLXLMCommon/PrefillAttentionKernel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift (these 4 currently in git stash "8p0rfxx-before", auto-popped when the running before-run ends), Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift (new), Tests/MLXLMTests/AttentionKernelTolerance.swift, Tests/MLXLMTests/Qwen35VerifyAttentionTests.swift, Tests/MLXLMTests/Qwen35VLMCompiledDecodeTests.swift. swift build --build-tests 0 compiler warnings; 5 bundles 0 fail 0 skip. Release measurements (subtask 3) not done: permission denial on stopping a contaminated parallel before-run.
    - next: human decides — allow stopping the running xcodebuild (or wait for it), confirm the stash popped, then run the before/after Release measurements suite by suite.
  timestamp: 2026-10-31T02:23:50.328336+00:00
- actor: claude-code
  id: 01m67zsyfavs6kb199m0j04f1m
  text: |-
    ### test (Release measurements, HEAD 300ca2f3) — green
    Each suite in its own xcodebuild (Release, IntegrationTesting scheme, arm64), under bench.lock; pre-run 1-min load 2.31-2.90 (all < 4). All 5 suites passed.

    1. MTPRoundCostBreakdownTests (602 s, passed). Verify forward by rows 1..4 (the test emits 4 rows, not 5), plain one-token decode, ms:
    - router 32k: plain 22.93; rows 23.06 / 26.73 / 36.09 / 40.15
    - router 64k: plain 24.34; rows 25.00 / 30.96 / 40.81 / 45.24
    - router 100k: plain 28.31; rows 28.47 / 36.48 / 47.74 / 52.26
    - text 32k: plain 20.76; rows 20.82 / 24.49 / 32.06 / 36.34
    - text 64k: plain 23.53; rows 23.55 / 29.23 / 37.59 / 41.74
    - text 100k: plain 26.17; rows 26.27 / 33.37 / 43.38 / 47.41
    (vs ^1abqd92 baseline router 100k 4-row ~67.2 pre-pool-fix, 32k 4-row 41.7 post-fix: now 52.26 / 40.15. Router now ~1.1-1.2x text per row.)

    2. MTPSWEBenchScaleTests decodeRateFromTwoToOneHundredThousandContextTokens(path:) (1155 s, passed; filter without "(path:)" ran 0 tests, so re-ran with it). MTP / plain tok/s (MTP/plain):
    router: 2k 85.2/54.7 (1.56x); 8k 91.5/54.1 (1.69x); 32k 77.5/48.4 (1.60x); 64k 62.2/41.4 (1.50x); 100k 52.4/36.7 (1.43x)
    text:   2k 86.8/55.9 (1.55x); 8k 93.5/55.4 (1.69x); 32k 85.3/48.9 (1.74x); 64k 65.8/42.6 (1.54x); 100k 54.2/37.4 (1.45x)
    Router MTP vs baseline: 2k 85.3->85.2, 8k 87.0->91.5, 32k 61.1->77.5, 64k 59.7->62.2, 100k 52.0->52.4. Router MTP/plain 32k 1.31x->1.60x, 64k 1.48x->1.50x, 100k 1.44x->1.43x. Text MTP vs baseline: 2k 86.8->86.8, 8k 92.7->93.5, 32k 86.6->85.3, 64k 65.8->65.8, 100k 54.1->54.2.

    3. GenerationBenchmarkTests qwen38FourBitMTPSystemPromptCache(): passed (44 s). Edit turn: prompt 2292, cached 2169, prefilled 123 tok (expected 123), MTP 99.0 tok/s vs plain 56.1, text equal to plain.
    4. MTPGenerationTests: passed (greedySessionWritesThePlainTokensFaster, 9.5 s).
    5. MTPPromptCacheTests: passed (7.7 s). Turn 2 recomputed 26 of 233 (expected 26 of 233), MTP 96.1 tok/s vs plain 59.3; turn 1 95.3 vs 60.0.
  timestamp: 2026-10-31T03:24:43.370551+00:00
- actor: claude-code
  id: 01m68010afbjb7nty3f7bzn8b9
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (1b17bca5): 2 findings (2 confirmed, 3 refuted, 14 attempted, 0 failed) — Tests/MLXLMTests/AttentionKernelTolerance.swift:129, Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift:269
    - next: implement the two unchecked items in `## Review Findings (2026-10-30 22:25)`, then re-review
  timestamp: 2026-10-31T03:28:34.639700+00:00
- actor: claude-code
  id: 01m6801z96vjhq3qqc6ycepg21
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — VLM Qwen35 attention through the shared kernel route with .causal
    - test: green — 5 bundles 0 fail/0 skip; Release (one suite per invocation, load < 3): router MTP/plain 32k 1.31 → 1.60x; edit turn 123; FM turn 2 26/233
    - commit: 1b17bca5
    - review: findings — Tests/MLXLMTests/AttentionKernelTolerance.swift:129, Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift:269
  timestamp: 2026-10-31T03:29:06.342170+00:00
- actor: claude-code
  id: 01m681007w9qcwrgs2h86ejr7v
  text: |-
    Review findings (2026-10-30 22:25) worked, tests only, no library change:
    1. AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie: greedy equality is still the first check. Under a near tie the verify token must now be one of decode's two best tokens (`argSort(decode[0])[row, -2...]`); the gap comes from `takeAlong` on those same indices. Both of the two best are accepted rather than only `[-2]`, because on an exact tie argMax and argSort can name different indices for the greedy token. Any token outside the two best fails. New Tests/MLXLMTests/AttentionKernelToleranceTests.swift: `aNearTieRejectsATokenOutsideTheTwoBest` (uses `withKnownIssue`, the first use in this repo, so it fails if the helper records no issue) and `aNearTieAcceptsTheSecondBestToken`. RED on the old helper: "Known issue was not recorded". GREEN after the change.
    2. New Tests/MLXLMTests/VerifyForwardLogits.swift: `VerifyForwardLogits.verify/decode(_:after:forward:)` takes a `Forward = (tokens, LMOutput.State?) -> LMOutput` closure. The text and VLM model types do not share a call signature (`model(LMInput.Text, cache:[KVCache], state:)` vs `languageModel(MLXArray, cache:[KVCache?], state:)`). Each suite now has only a small `forward(_:over:)` closure builder, and the copies are deleted. The text behavior is unchanged: its prompt state is nil, verify starts from `State()`, and the text model returns nil state when it is not emitting, so decode still passes nil on every step.
    - Callers of the near-tie helper (Qwen35VerifyAttentionTests and Qwen35VLMVerifyAttentionTests theVerifyForwardGivesThePlainDecodeLogits) both pass.
    - Note: the `dump validators` file is 754k chars (it inlines very large rule bodies), too big to read whole in one pass. I read the header and rule structure only and followed the usual rules (doc comments, named constants, no copies).
    - `swift build` prints a SwiftPM build-system line, "warning: missing creator for mutated node ... mlx-swift_Cmlx.bundle". It is not a compiler warning and was already there on the first build before my edits. swift-format lint flags line 93 of Qwen35VerifyAttentionTests.swift (the `prefilled` signature), which was already in HEAD; I left it alone.
  timestamp: 2026-10-31T03:45:30.364383+00:00
- actor: claude-code
  id: 01m681011k9jphyw96frvfcpw1
  text: |-
    ### implement — changed
    - evidence: 5 files — Tests/MLXLMTests/AttentionKernelTolerance.swift, Tests/MLXLMTests/AttentionKernelToleranceTests.swift (new), Tests/MLXLMTests/VerifyForwardLogits.swift (new), Tests/MLXLMTests/Qwen35VerifyAttentionTests.swift, Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift. `swift build --build-tests`: 0 compiler warnings. xcrun xctest: MLXLMTests 920 XCTest + 2087 Swift Testing passed (1 known issue, the intended withKnownIssue), MLXGuidedGenerationTests 72, MLXFoundationModelsTests 460, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7. 0 fail, 0 skip. Both review findings are checked.
    - next: /review (re-review HEAD plus the working tree)
  timestamp: 2026-10-31T03:45:31.187234+00:00
position_column: doing
position_ordinal: '8280'
title: 'MTP: route the router path''s (MLXVLM Qwen35) causal verify mask as .causal so verify reaches the decode/verify attention kernels'
---
Found by ^wcmxg07. On the router path, MLXVLM Qwen35 (loadModelContainer → VLM Qwen35, the path the SWE-bench ACP agent runs) builds the causal mask with `createAttentionMask(..., returnArray: true)`. So a multi-row verify passes an ARRAY mask and never reaches DecodeAttentionKernel or VerifyAttentionKernel. Both read only `.causal` or no mask.
- Microbench (16 layers, MTPVerifyForwardProfileTests): MLX SDPA with an array mask is 35–44 ms at 32k and 107–122 ms at 100k for 2–5 rows. The verify route with `.causal` is ~6.5 ms and ~18 ms.
- Round cost after ^wcmxg07: router 100k verify rows 1–5 are 27.67/43.20/56.41/67.16/79.49 ms (unchanged). Text 100k is 27.33/37.22/46.19/51.42/57.03.
- Sweep after: router 100k guard off 1.21x plain, text 1.43x. Router 100k guard on dropped to 0.96x plain, below the 0.97 floor (a known issue).

- [x] Find why the VLM Qwen35 language model asks for an array mask (position ids / mrope? padding?). When the mask is a plain causal one, pass `.causal` (or `.none` for one row), as the text Qwen35 does. — Cause: the VLM layer API took `MLXArray?`, so `Model.forward` forced `returnArray: true`; the LM `mask:` input feeds only M-RoPE (`getRopeIndex`), never attention, so the array was always the plain causal mask. Also the VLM attention called the free `attentionWithCacheUpdate` (MLX SDPA) and never the kernel route. Now `MaskMode` end to end via `createAttentionMask(h:cache:)` (rotating overflow still gets its array) and the shared `PrefillAttentionKernel.attentionWithCacheUpdate` route; VLM MTP predictor likewise.
- [x] Unit test: the VLM attention layer gives the route output bit for bit with the causal mode, and the model-level verify matches plain decode within the bf16 near-tie rule. — Qwen35VLMVerifyAttentionTests: bit-for-bit VerifyAttentionKernel output for 3, 4, 5, 6, 7, 8 rows over 1100 cached keys; `.causal` vs causal mask array ≤ 8e-3 for 3–8 rows; model verify (5 rows) vs plain decode ≤ 8e-3 + near tie. MLXLMTests 920 XCTest + 2085 Swift Testing, MLXGuidedGenerationTests 72, MLXFoundationModelsTests 460, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7: 0 fail, 0 skip.
 - [x] Re-run MTPRoundCostBreakdownTests (router) and the sweep. Router per-row cost should match text. — Done on HEAD 300ca2f3, one suite per xcodebuild; numbers in the "test (Release measurements)" comment. Router rows 1..4 at 100k: 28.47/36.48/47.74/52.26 ms vs text 26.27/33.37/43.38/47.41; router MTP/plain 32k 1.60x, 64k 1.50x, 100k 1.43x.

#mtp #perf

## Review Findings (2026-10-30 22:25)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 8 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M66K9YJNDKE1CVA6E8P0RFXX.jsonl` — no validator matches this file
> - `.kanban/tasks/01M66K9YJNDKE1CVA6E8P0RFXX.md` — no validator matches this file

- [x] `Tests/MLXLMTests/AttentionKernelTolerance.swift:129` `completeness/invariant-propagation` — The near-tie check accepts any verify token whenever the decode gap is below `nearTieLogitGap`, not only decode's greedy token or its second-best token. The doc comment (lines 110-112) says a verify may pick "the second one", but the assertion never checks that. A verify that picks a wrong low-ranked token still passes whenever the top-two gap is small. The change relaxes the assertion for the near-tie input instead of enforcing the rule that a verify row must match one of decode's top two candidates. When the gap is below the threshold, assert that `verifyToken` equals the second-best index of `decode[0]` for that row, e.g. by comparing against `argSort(decode[0], axis: -1)[row, -2]`. Keep the greedy-equality branch as the primary check. Add a test that feeds a verify row whose token is outside the top two and confirms the helper rejects it.
- [x] `Tests/MLXLMTests/Qwen35VLMVerifyAttentionTests.swift:269` `reuse/reuse` — `verifyLogits` reimplements the text suite's `verifyLogits` as a parallel copy. The VLM version sets the same two state flags (`mtpEmitFlagKey`, `mtpVerifyKeepsSnapshotsKey`) and returns the same float32 logits. Only the model type and the cache argument differ. Two copies of the verify-state setup can drift apart, so the shared shape should be extended or parameterized rather than copied. Move `verifyLogits` and `decodeLogits` into `AttentionKernelTolerance` (or a shared test helper) and parameterize them over the model's language-model call, so both suites call one implementation. If the two model types cannot share a signature, make the helper generic over a closure that runs one forward.

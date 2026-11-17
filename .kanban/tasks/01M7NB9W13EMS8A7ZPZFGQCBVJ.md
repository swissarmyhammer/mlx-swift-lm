---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7nc0njg9gez2wc701w5p25h
  text: |-
    Research (worktree agent-ad4d5b5cf6e99e767, reset to stable 03a37849):
    - Drafter checkpoint mlx-community/Qwen3.8-27B-MTP-8bit (downloaded, snapshot 481e0945, 451 MB): config differs from the 4-bit drafter ONLY in quantization.bits 4->8 (group 64, affine); same 31 tensor names, model_type qwen3_5_mtp. Existing Qwen35CheckpointLoadingTests.standaloneFactoryLoadsPublishedLayout already loads standalone drafters at bits 4/5/8 through both factories.
    - FusedQuantizedLinear (q/k/v, gate/up, GDN in_proj): bit-generic; fuses only projections of equal bits/group/mode, keeps first.bits. Existing tests cover 4-bit only (Fixture.bits = 4), plus a mixed 4/8 no-fusion case.
    - MultiRowQuantizedMatvec: reads affine 4 AND 8 bits (g32/64/128) + mxfp4; MultiRowQuantizedMatvecTests already cover 8-bit at op level; the Qwen35 model-level verify-row tests only run 4-bit (MultiRowProjectionFixture.bits = 4). 8-bit block width = 256, so 27B widths (5120, 17408) fit.
    - Qwen35DecodeHeadsKernel and ResidualRMSNormKernel read activations / norm weights only: no weight-bits dependency.
    - QuantizedKVAttentionKernel: kvBits 8 / group 64 only (by design), independent of weight bits; existing tests cover the kernel against the reference and the 4-bit fallback.
    - No hard-coded 4-bit assumption found in Qwen35 VLM/LLM or MLXLMCommon fast paths (grep for bits == / bits: 4).
    - Harness: qwencli / qwenenv.zsh / qwencheck.sh do NOT exist anywhere on disk (searched scratchpad + /private/tmp + ~). Adapted the qh harness of ^2h5wtct (scratchpad/h2h5/qh, modes golden/bench/mtpbench) into scratchpad/q8/qh built against this worktree, adding a `dl` mode (hub download without load) and acceptance-rate output.
  timestamp: 2026-11-17T18:23:55.984262+00:00
- actor: claude-code
  id: 01m7ncfcknegr3bjffpx7sr0pa
  text: |-
    TDD progress:
    - RED: Qwen38Tests.testQwen38EightBitMTPDrafterIsRegistered failed (XCTAssertTrue) before the registry change; GREEN after adding MTPDrafterRegistry.qwen3_8_27b_mtp_8bit to the class and all().
    - 8-bit fast-path tests (fixture MultiRowProjectionFixture gained a `bits` parameter, default 4, plus `eightBits = 8`): fused q/k/v + gate/up (LLM and VLM), fused GDN input projection, VLM prepare fuses all 4 modules at 8 bits, multi-row verify kernel routing (canRun true for 8-bit MLP projections) and verify rows == single-row decode (MLP, GDN in/out, attention q/k/v/gate/o_proj, full-target logits), prefill tail keeps the MLX matvec, compiled decode+verify == unfused model (VLM, LLM), and compiled steps over an 8-bit KV cache (QuantizedKVAttentionKernel group 64 / bits 8, 1024-token prompt so the fused decode kernel reads it). Plus a Swift Testing load of a standalone 8-bit qwen3_8 drafter via Qwen35VLMMTPRegistration (every QuantizedLinear bits == 8).
    - These all PASSED against unchanged library code: the fast paths are bit-generic, no 4-bit-only routing exists. Two first-draft test mistakes corrected: (1) the GDN input group is FusedRows.everyCall by design, so a 64-token prefill may sum in another order at ANY bit width — test now compares 1 and 7 tokens as the existing 4-bit GDN tests do; (2) with a QuantizedKVCache the VLM runs per-layer compiled decode (whole-step segments need plain caches), so the kv8 test checks compiledGDNTraceCount instead of segment counts.
    - Existing 4-bit tests in Qwen35FusedProjectionTests and Qwen35VLMMultiRowProjectionTests refactored into bits-parameterized helpers (no duplication) and still pass.
  timestamp: 2026-11-17T18:31:58.325211+00:00
- actor: claude-code
  id: 01m7ndwf6msvj42vbbdy62681h
  text: |-
    Real-model evidence (M5; harness scratchpad/q8/qh built against worktree agent-ad4d5b5cf6e99e767; bench.lock taken by q8/lock.sh 12:42:07 and released 12:55:59; the runs used foreground pieces under 180 s each):
    - 8-bit target mlx-community/Qwen3.8-27B-8bit (27 GB, snapshot 1159008f) + drafter mlx-community/Qwen3.8-27B-MTP-8bit (snapshot 481e0945) load through VLMModelFactory + MTPDrafterModelFactory.visionLanguage, and the drafter drafts.
    - Greedy plain vs MTP, 256 tokens: code prompt IDENTICAL (bf16 KV and kv8, acceptance 0.926 each). Short prompt diverges at step 152 (bf16 KV) / 116 (kv8). Both are EXACT bf16 ties: gap 0.0000 (step 152: 623 and 47807 both 24.1250; step 116: 1027 and 13255 both 22.7500), measured with the harness `gap` mode. The 4-bit pair diverges on the same short prompt too (step 29 bf16 / 62 kv8), on stable as well.
    - 2k ctx, 256 tokens: bf16 KV plain 31.64 ms/tok (prefill 1273 tok/s) vs MTP 11.76 ms/tok (3.71 tok/verify, acceptance 0.974). kv8 plain 31.89 ms/tok (prefill 1295 tok/s) vs MTP 13.21 ms/tok (3.32 tok/verify, acceptance 0.984).
    - 4-bit regression: qwencheck.sh does not exist. Equivalent check: 4-bit pair golden (short, code, 32k × plain/MTP) vs h2h5/golden_stable_{bf16,kv8}.json = 6/6 + 6/6 identical (12/12).
    Tests: swift build --build-tests clean (no warnings in project sources). xctest: MLXLMTests 1157 XCTest + 2386 Swift Testing passed (3 known issues, existing withKnownIssue), MLXFoundationModelsTests 519 passed, MLXGuidedGenerationTests 78 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 11 passed; 0 failures, 0 skipped. swift-format applied.
  timestamp: 2026-11-17T18:56:35.540935+00:00
- actor: claude-code
  id: 01m7ndwjgeyhysebby0rxs0het
  text: |-
    ### commit — changed
    - evidence: 47729d58 on branch worktree-agent-ad4d5b5cf6e99e767 (worktree .claude/worktrees/agent-ad4d5b5cf6e99e767, based on stable 03a37849); 6 files — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift, Tests/MLXLMTests/{MultiRowProjectionFixture,Qwen35FusedProjectionTests,Qwen35MTPLoadingTests,Qwen35VLMMultiRowProjectionTests,Qwen38Tests}.swift
    - next: the coordinator merges and reviews. Not done here, by instruction: push to stable, and sending the SHA to ACPAgent / Router (card acceptance item 3).
  timestamp: 2026-11-17T18:56:38.926913+00:00
- actor: claude-code
  id: 01m7ne5cw3x4khgbkvb4n83ac0
  text: |-
    ### review — findings
    - evidence: review sha 5207deb5..b5fb7567 — 2 findings (2 confirmed, 0 refuted): Tests/MLXLMTests/Qwen35FusedProjectionTests.swift:284, Tests/MLXLMTests/Qwen38Tests.swift:355
    - next: implement the 2 Review Findings items (both reuse/reuse), then re-review. The push + notify acceptance item is an orchestrator step, not a code finding.
  timestamp: 2026-11-17T19:01:28.067388+00:00
position_column: doing
position_ordinal: '8480'
title: 'Qwen3.8: register and verify the 8-bit MTP drafter (mlx-community/Qwen3.8-27B-MTP-8bit) with the 8-bit target'
---
## Why
ACPAgent (for the user) wants to run `mlx-community/Qwen3.8-27B-8bit` with the 8-bit MTP drafter `mlx-community/Qwen3.8-27B-MTP-8bit`. `MTPDrafterRegistry` in Libraries/MLXLMCommon/MTPDrafterModelFactory.swift only registers `qwen3_8_27b_mtp_4bit`. Router may switch its `large` preset to this pair. Write it TDD.

## What
- Register `qwen3_8_27b_mtp_8bit` (`mlx-community/Qwen3.8-27B-MTP-8bit`) next to the 4-bit entry and in `all()`.
- Check the 8-bit drafter checkpoint's config (quantization bits/group, layer naming) loads through the existing Qwen MTP drafter type (both `shared` and `visionLanguage` registries as the 4-bit uses) — fix any quantization assumption that hard-codes 4-bit (e.g. fused projections, MultiRowQuantizedMatvec, decode-heads kernels that may assume bits 4).
- The 8-bit target `mlx-community/Qwen3.8-27B-8bit` must also go through the fast paths correctly or fall back exactly (fused q/k/v and gate/up at load, multi-row verify kernel) — any kernel that only supports 4-bit must route 8-bit weights to the generic path.

## Acceptance
- [ ] Unit tests first (failing first): registry lookup of the 8-bit id; any 8-bit kernel/fallback routing.
- [ ] Real model (if downloadable within the ~65 GiB weight ceiling, 8-bit 27B ≈ 29 GB plus drafter): target + drafter load and draft through the VLM Qwen35 path; MTP greedy ids equal plain greedy ids on a short prompt; acceptance rate reported; with kvBits 8 too. Real-model runs take scratchpad bench.lock.
- [ ] All 5 test bundles green; swift-format; pushed to stable; SHA sent to ACPAgent (foundationmodelsacpagent-22) and Router (foundationmodelsrouter-da). #mtp-8bit

## Review Findings (2026-11-17 12:57)

> Scope: `review sha 5207deb5..b5fb7567` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 0 not reviewed.

- [x] `Tests/MLXLMTests/Qwen35FusedProjectionTests.swift:284` `reuse/reuse` — `vlmFusionStates(bits:)` rebuilds the vision-language target inline: it decodes the config, runs `seededQuantized` with the `language_model` filter, and calls `prepare()`. That is the same construction as `makeTarget(prepared:matvecOfMLX:bits:)` in `Qwen35VLMMultiRowProjectionTests`, which already takes a `bits` parameter. The copy will drift from the helper it duplicates. Move the target construction into `MultiRowProjectionFixture` (for example `Fixture.visionLanguageTarget(bits:prepared:)`) and have both `makeTarget` and `vlmFusionStates` call it. `vlmFusionStates` then only maps the fusion flags over the modules.
- [x] `Tests/MLXLMTests/Qwen38Tests.swift:355` `reuse/reuse` — The two new 8-bit registry tests each check one property of `qwen3_8_27b_mtp_8bit`: `testQwen38EightBitMTPDrafterIsRegistered` checks `shared.contains(id:)` and `testQwen38EightBitMTPRegistryConfigurationNamesTheEightBitDrafter` checks `.name`. The existing `testQwen38MTPRegistryConfiguration` already asserts both properties for the 4-bit config. The 8-bit checks are a parallel copy rather than an extension of that test. Replace the three tests with one `@Test(arguments:)` or XCTest loop over `(MTPDrafterRegistry.qwen3_8_27b_mtp_4bit, MTPDrafterRegistry.qwen3_8_27b_mtp_8bit)` that checks the name and `shared.contains(id:)` for each config.

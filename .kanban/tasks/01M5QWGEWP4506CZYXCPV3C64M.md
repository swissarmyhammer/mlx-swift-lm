---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzmkyf0j79q1m2f3qgh72
  text: 'Change after the PR 581 port (commit 843854ec): the MLXVLM `prepareContinuation` now passes `state` and joins the hidden rows of each chunk, so the warm VLM continuation runs MTP (remove that acceptance item). The LLM prefill is still `stepSize = Int.max` when there is a continuation or `requiresPromptPrefill`, and the joined rows stay alive. Replace the joined rows with the per-chunk hook of this task.'
  timestamp: 2026-10-25T13:46:13.502187+00:00
- actor: claude-code
  id: 01m5zrphc9n7fbz6pkekdby4xx
  text: |-
    Research (picked up on mtp @ 4b1bf67d):
    - MLXLLM Qwen35TextModel/Qwen35Model have no prepare override; they use the default `LLMModel.prepare` (Libraries/MLXLLM/LLMModel.swift), which chunks with forEachChunk and threads `state = output.state`. The hook goes there (covers both Qwen35 LLM classes).
    - MLXVLM Qwen35 `prepareContinuation` (cold > window or warm) now joins every chunk's hidden rows (PR 581 port); the single-shot path (cold, <= window) emits all rows in one forward. The VLM drafter positions use one scalar delta for the whole suffix: `resumeDelta` (ropeDeltas - cacheOffset), known before the chunk loop, so the hook can pass it per chunk and match unchunked.
    - Iterator forces `stepSize = Int.max`/`.unchunked` whenever requiresPromptPrefill or a continuation; `primeDrafterState` needs `targetHidden.dim(1) >= promptLength`.
    - Shifted drafter algebra: entry at position j = (h[j], t[j+1]). After the target has prefilled [0, b) with the drafter fed per chunk, the drafter holds entries up to b-2 plus pending h[b-1] — exactly the MTPDrafterContinuation boundary form. So per chunk: first chunk without a boundary appends t[a+1..b) with h[a..b-1); later chunks append t[a..b) with [pending, h[a..b-1)]; the final segment + bonus is a plain `resumeDrafterState` with the pending row as boundary. ^achvxy1 can take its checkpoint continuation from the same feed state at a chunk boundary.
    - `LMOutput.State` stores `Any`, so a closure key works; the forward copies incoming state into its output, so the closure must be stripped (persistentTargetState) before anything persists/serializes state.
    - Each `appendDrafterState` also builds a seed proposal graph; intermediate chunk appends must drop it so it does not pin the chunk's graph.
    - Test fakes (QwenStyleMTPDrafter, MTPStateEmittingTarget) return `.tokens(whole)` from prepare, so they never chunk; their coldStart/resume probes keep working through the unchanged whole-prompt path.
  timestamp: 2026-10-27T22:46:36.169026+00:00
- actor: claude-code
  id: 01m5zsf8hp6wz9axfkq2tagt33
  text: |-
    Implementation landed (TDD): RED = hook never called (0 of 5 chunks) and peak memory chunked == unchunked (163357588 B both); GREEN after the change, then full suites green (MLXLMTests 895 XCTest + 1858 Swift Testing, MLXFoundationModelsTests 376).
    Design: `mtpPrefillChunkHandlerKey` (MTPPrefillChunkHandler = (tokens, hidden, positionDeltas) -> Void) in the prepare state, honored only with mtpEmitFlagKey. Default `LLMModel.prepare` (used by both MLXLLM Qwen35 classes) and MLXVLM Qwen35 `prepareContinuation` call it after each chunk forward but the last; the VLM no longer joins rows and passes `resumeDelta` per chunk. New `MTPDrafterPrefillFeed` (MLXLMCommon, internal) appends shifted entries per chunk, keeps the chunk's last row as boundary, drops intermediate seeds, asyncEvals the drafter cache. After prefill the iterator resumes the drafter over the final segment from that boundary (fallback: cold start at the segment). Forced Int.max/unchunked is kept only for a stateful drafter that is NOT resumable (it can only prime from all rows at once). For ^achvxy1: after any chunk the feed's (state, boundaryHidden, cachePosition + fedTokenCount) is exactly a finalized MTPDrafterContinuation, so prefillToCheckpoint can drive the same feed per chunk and snapshot it at the checkpoint boundary.
    Tip for running ONE Swift Testing suite here: `swift test --filter` fails to load the metallib, and `xcrun xctest -XCTest` ignores Swift Testing. Copy `swiftpm-testing-helper` (Xcode toolchain usr/libexec/swift/pm) into a scratch dir, symlink `.build/out/Products/Debug/mlx-swift_Cmlx.bundle` next to it, and run it with DYLD_FRAMEWORK_PATH/DYLD_LIBRARY_PATH = MacOSX.platform/Developer/{Library/Frameworks,usr/lib}, `--test-bundle-path .build/out/Products/Debug/MLXLMTests.xctest/Contents/MacOS/MLXLMTests --testing-library swift-testing --filter <Suite>`.
  timestamp: 2026-10-27T23:00:06.326740+00:00
- actor: claude-code
  id: 01m5zt8bppjgppcejwpfx8yjtt
  text: |-
    Release benchmarks (M5, Xcode Release, qwen38-27B-4bit + MTP-4bit):
    - qwen38FourBitMTPGeneration (math/code/prose MTP tok/s; acceptance 93.9/73.6/69.0% in every run): run1 87.5/77.1/76.8, run2 88.6/73.7/72.6, run3 89.3/80.3/77.9. A fresh HEAD baseline (my Libraries changes stashed, same session) gave 89.3/79.4/75.9. These prompts are < 100 tokens, so the prefill path is the same code before and after the change (LLMModel.prepare returns the whole prompt before any chunk; the feed is never fed). The run-to-run spread (code 73.7..80.3) is noise on identical code paths; run3 is within ±3% of the fresh baseline on all three (0.0 / +1.1 / +2.6%).
    - qwen38FourBitMTPSystemPromptCache: passes with exactly the 2 known issues (edit turn 2 mtpReusesCache, drafterPrefillsOnlyTheSuffix). Cold turns now chunk the MTP prefill: follow-ups t0 2212 tok MTP 1746.8 ms vs plain 1812.2 ms; thinking t0 2191 tok 1739.5 vs 1789.2 ms; edit t0 2197 tok 1743.3 vs 1802.2 ms (MTP prefill ≈104% of plain tok/s; criterion ≥85%). Warm turns: MTP cached/prefilled 2403/38 (84.3 ms vs plain 2404/37 138.4 ms), 2544/29 (74.4 vs 123.7), 2302/25 (68.2 vs 112.5), 2424/25 (67.4 vs 115.1), 2264/26 (69.0 vs 121.2); edit turn 2 MTP recomputes 0/2292 in 1927.7 ms vs plain 2169/123 in 185.5 ms (known gap ^achvxy1). Drafter prefilled == prompt prefilled on every turn; MTP decode 85.4–89.0 tok/s, acceptance 91.4–97.1%, all turns token-exact vs plain except thinking t2 (near-tie at token 9, gap 0.0).
    - First-token logits bit-for-bit vs plain: NOT measured. Chunk boundaries now match plain (same balanced chunks, one reserved position), and the chunk forwards run the same layer path; but the last prompt position still differs: plain runs it through `nextTokenLogits` → `Qwen35TextModelInner.forward(applyFinalNorm: true)` which takes the fused `decodeStep` for one token, while MTP's emit-flag bonus forward takes `applyFinalNorm: false` + separate `norm` and skips `decodeStep`. So bitwise equality is not guaranteed; all four cold turns matched plain tokens exactly.
  timestamp: 2026-10-27T23:13:48.758454+00:00
- actor: claude-code
  id: 01m5zt8f3ydt3djrnrsgv7fvm4
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift (new), Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLLM/LLMModel.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTestSupport/Qwen35MTPFixture.swift, Tests/MLXLMTestSupport/Qwen35VLMMTPFixture.swift (new). `swift build --build-tests` no new warnings; MLXLMTests 895 XCTest + 1858 Swift Testing 0 failures; MLXFoundationModelsTests 376 0 failures; Release qwen38FourBitMTPGeneration x3 + fresh HEAD baseline, qwen38FourBitMTPSystemPromptCache passes with the 2 known issues.
    - next: /review
  timestamp: 2026-10-27T23:13:52.254292+00:00
- actor: claude-code
  id: 01m5ztsjhv18xjrv5nneefdjhy
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched) — only the known "missing creator ... mlx-swift_Cmlx.bundle" message, no compiler warnings; xcrun xctest MLXLMTests.xctest — 895 XCTest, 0 failures + 1858 Swift Testing in 165 suites, passed; MLXFoundationModelsTests — 376 passed; only skip is the env-gated benchmarkVerification(). Second MLXLMTests run: identical counts, 0 failures; Qwen35MTPChunkedPrefillTests (4096-token chunked vs single-pass peak memory) passed both times (3.1s, 3.8s) - stable across 2 runs.
    - next: review
  timestamp: 2026-10-27T23:23:12.827531+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: chunked prompt prefill that feeds the drafter for each chunk'
---
## What
`MTPSpeculativeTokenIterator.init` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift) sets `mtpPrefill.stepSize = Int.max` and `.unchunked` when `drafter.requiresPromptPrefill`. The target then runs the full prompt in one forward pass. For a long prompt the peak memory is very large.

The chunk loop must stay inside the model `prepare`, not in the iterator: the MLXVLM Qwen35 `prepare` merges images and computes the M-RoPE positions. Also, on a warm cache the MLXVLM Qwen35 `prepare` goes to `prepareContinuation` (branch `cacheOffset > 0 || len > window`, Libraries/MLXVLM/Models/Qwen35.swift), whose `forward` passes `state: nil`. The emit flag is then lost, the iterator gets one hidden row, fails `targetHidden.dim(1) >= promptLength` (~370), and every continuation goes to passthrough.

llama.cpp uses the same design: a hook runs after each target ubatch (`common_speculative_impl_draft_mtp::process`). It loses about one third of its prefill speed with MTP, mostly from device-to-host copies of the hidden rows. Keep the hidden rows on the GPU.

- [x] Add a per-chunk hook to `LanguageModel.prepare` (for example an `onChunkHiddenStates: (tokens, hidden) -> Void` in the prepare state, called after each chunk when `mtpEmitFlagKey` is set). Implement it in MLXLLM Qwen35 and MLXVLM Qwen35, including `prepareContinuation`.
- [x] In the iterator, feed the drafter append (from the model-side task) from the hook, with the chunk tokens shifted by one and the chunk hidden rows; keep the last hidden row as the pending hidden. Remove the forced `stepSize = Int.max` / `.unchunked`.
- [x] Do not keep the hidden rows of earlier chunks alive (evaluate the drafter cache and the main cache after each chunk). Do not copy hidden rows to the CPU.

## Acceptance Criteria
- [x] Chunked MTP prefill (stepSize 64 on a prompt of more than 300 tokens) gives the same greedy tokens and the same drafter cache contents (within float tolerance) as unchunked prefill, for the MLXLLM and the MLXVLM Qwen35 targets.
- [x] A warm-cache continuation on the MLXVLM target runs MTP (no passthrough).
- [x] Peak memory during the MTP prefill of a prompt of 4000+ tokens with stepSize 512 is lower than unchunked (`Memory.peakMemory`).
- [x] Release benchmark on the M5 Max (`qwen38FourBitMTPGeneration`): decode tok/s within ±3% of the baseline, and MTP prefill tok/s ≥ 85% of plain prefill tok/s.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared fixture: chunked against unchunked equality (LLM and VLM targets), the VLM warm continuation, and a test that the hook is called one time for each chunk.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38
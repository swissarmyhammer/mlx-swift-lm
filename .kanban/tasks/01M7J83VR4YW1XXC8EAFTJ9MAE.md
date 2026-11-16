---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jmwv3tj6wynzmspsk49f1k
  text: |-
    Research:
    - mx.compile does NOT fuse add + RMSNorm: RMSNorm is a fast primitive (not elementwise), and the decode graph dump (scratchpad q_dec.dot, compiled traces) keeps 162 standalone `Add` nodes next to 161 `RMSNorm`; no `Compiled*Add` node. The adds sit between QuantizedMatmul outputs and RMSNorm, so nothing elementwise is there to fuse. => custom Metal kernel.
    - hidden 5120 > RMS_LOOPED_LIMIT 4096, so MLX runs `rms_looped` with threadgroup = maxTotalThreadsPerThreadgroup (1024), N_READS 4: lane lid sums x[lid*4..+3] then x[4096+lid*4..+3], simd_sum, then simd_sum over 32 local sums. A bit-exact kernel must keep that order. SwiftPM builds the MLX kernels with fast math (fma contraction), runtime kernels without, so write fma explicitly (as Qwen35DecodeHeadsKernel does).
    - Residual adds per layer: `x + mixer` -> postAttentionLayerNorm (inside a layer), `h + ff` -> next layer's inputLayerNorm (crosses layer bodies, inside one CompiledDecodeSegment), last `h + ff` -> final norm (decode) or the caller (verify). Shared bodies live in MLXLMCommon/HybridDecodeLayer.swift + CompiledDecodeSegments.swift, used by VLM Qwen35, LLM Qwen35 and LLM Qwen3Next.
    - Plan: a ResidualRMSNorm kernel in MLXLMCommon returning (sum, normed); a residual-stream value threaded through the segment body so the feed-forward add is deferred into the next norm.
  timestamp: 2026-11-16T17:01:24.474869+00:00
- actor: claude-code
  id: 01m7jpdr04513f8yynjzz30ygf
  text: |-
    Implementation + measurement (GO):
    - New `ResidualRMSNormKernel` (Libraries/MLXLMCommon/ResidualRMSNormKernel.swift): one dispatch writes `sum = residual + addend` (rounded in T first) and `rmsNorm(sum)`. `RMSNorm.normalizingSum(of:adding:)` uses it for an exact `RMSNorm` (subclasses and shapes or dtypes it can't read fall back to the op path).
    - `ResidualStream` (HybridDecodeLayer.swift) holds the residual plus a not-yet-added block output. The shared bodies (linear layer, attention pre/post) and `CompiledDecodeSegment.body` pass it through, so every residual add of a decode/verify segment runs inside the next norm: post-attention norm, the next layer's input norm, the final norm. Only the last add of a verify segment (no final norm there) stays an Add. The old `x:` entry points still exist as wrappers. Covers VLM Qwen35, LLM Qwen35 and Qwen3Next via the shared code.
    - Bit-exactness findings (probe of 256 rows per case):
      1. MLX's `acc / axis_size` is compiled with fast math as `acc * (1/axis_size)`. A true division differs in about 25% of rows for widths that are not a power of two (72, 5120, 6000). The kernel writes `total * (1.0f / WIDTH)`.
      2. In float32 with a looped row (width > 4096), fast math rewrites the output to `x * (w * inv)`. The kernel matches that order for float32 looped rows only. bf16/f16 are unaffected because the cast to T blocks the rewrite.
      3. Reduction order: sequential fma per thread, 1024 threads for looped rows. This matches.
      Parity test: widths 4/72/128/4096/5120/6000 x f32/bf16/f16 x 1/64 rows, all bit-exact.
    - Latent issue filed as ^qjjbzn7: Qwen35DecodeHeadsKernel and FlintDecodeHeadsKernel divide `/ DIM`, so they are exact only for power-of-two head widths.
    - qwencheck: CHECK PASS (all 8 golden id streams match, session reuse PASS, toolchat PASS).
    - qwenbench (scratchpad qwen_bench_ftj9mae.txt), ms/tok at 2k / 32k / 64k:
      - plain: 17.03 / 19.78 / 22.68. vs stable: -6.7% / -4.0% / -3.2%. vs ^6jf4g0c: -1.7% / -1.5% / -1.1%.
      - MTP: 9.94 / 12.86 / 14.02. vs stable: -2.3% / -1.2% / -1.3%. vs ^6jf4g0c: -0.6% / -0.6% / -0.7%.
      - tokens per verify unchanged. Prefill is not touched by this change (1339 / 1160 / 726 tok/s; the 2k -1.5% vs ^6jf4g0c is run noise).
    - Verdict: GO. A small gain on all six decode points; the library change stays.
  timestamp: 2026-11-16T17:28:06.916333+00:00
- actor: claude-code
  id: 01m7jptgkxddgzjk69hkfmzjbq
  text: |-
    ### implement — changed
    - evidence: 5 files, swift-format run. New: Libraries/MLXLMCommon/ResidualRMSNormKernel.swift and Tests/MLXLMTests/ResidualRMSNormKernelTests.swift. Changed: Libraries/MLXLMCommon/HybridDecodeLayer.swift, Libraries/MLXLMCommon/CompiledDecodeSegments.swift, Tests/MLXLMTests/HybridDecodeLayerTests.swift.
    - `swift build --build-tests`: 0 errors, 0 warnings. `xcrun xctest` on all five bundles: 0 failures.
      - MLXLMTests: 2369 tests (was 2361, +8 new).
      - MLXGuidedGenerationTests: 78. MLXFoundationModelsTests: 496. CXGrammarTests: 7. MLXHuggingFaceMacrosTests: 11.
    - The run still has 4 XCTest skips (LFM2 real-checkpoint and benchmark tests gated by env vars), 1 Swift Testing skipped suite (VerifyAttentionBlockPassProfileTests) and 3 known issues. They are identical to the ^6jf4g0c baseline log (xct_MLXLMTests.log), so this change did not add them.
    - qwencheck: CHECK PASS. qwenbench: GO, a small gain on every decode point (see the previous comment).
    - next: /review
  timestamp: 2026-11-16T17:35:05.341633+00:00
depends_on:
- 01M7J83R03FRVF0510R6JF4G0C
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 decode: fuse residual add + RMSNorm'
---
## Why
The decode graph has 162 Add and 161 RMSNorm operations per token. Most of the adds are residual adds that feed the next RMSNorm directly.

## What
Fuse `h = x + r; n = rmsnorm(h)` into one kernel that returns both `h` and `n`. Do it with a custom Metal kernel, or with mx.compile of the pair if that fuses, whichever measures faster. Use it in the decoder layers of Qwen35 (VLM and LLM) for decode and verify.

The result must match the current path exactly: bf16 rounding of `h` happens before the norm, exactly as it does today.

## Acceptance
- [x] `qwencheck.sh`: greedy ids are identical in plain and MTP modes, and cache reuse passes.
- [x] A kernel parity unit test passes. All five bundles are green.
- [x] `qwenbench.sh` delta on this card. If there is no measurable gain, record it as a no-go and revert. #qwen-perf
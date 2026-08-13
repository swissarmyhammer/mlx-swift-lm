# DeepSeek-V4 support

This document records the scope of the DeepSeek-V4 port. It tells you what
this repository supports, and it lists the seven items the port defers. Each
deferred item has a reason and a kanban task reference. A reference of
"unfiled" means that no task exists for the item.

`Tests/MLXLMTests/DeepSeekV4DocsTests.swift` guards this document. If you
remove a deferred item or the provenance record, that suite fails.

## What this repository supports

- The model type `deepseek_v4` loads through `LLMTypeRegistry`
  (`Libraries/MLXLLM/Models/DeepSeekV4.swift`).
- `DeepSeekV4ChatEncoder` (`Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift`)
  makes the prompts. Its output is byte-identical to the output of DeepSeek's
  Python encoder, and this includes DSML tool calls.
- The DSML tool-call parser
  (`Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift`) reads tool
  calls out of model responses.
- The reasoning modes `chat` and `thinking` are connected.

Tests prove these functions on synthetic weights and on encoder fixtures.
Validation against the real weights of `mlx-community/DeepSeek-V4-Flash-4bit`
is not complete; refer to task `^e7b24ws` and to item 2 below.

## Provenance and licenses

The DeepSeek-V4 code came from these repositories, in this sequence:

1. `osaurus-ai/vmlx-swift-lm` — the origin. MIT license, Osaurus AI copyright.
2. `scouzi1966/mlx-swift-lm` — an intermediate copy. MIT license.
3. This repository.

`scouzi1966/maclocal-api` (MIT license) is the integration reference for the
application layer around the library.

The two upstream repositories are not GitHub forks (`fork=false` and
`parent=none` on both). Thus the repositories have no shared git history, and
each port was a manual transcription. `THIRD-PARTY-NOTICES.md` at the
repository root holds the license text of each repository.

## Deferred items

### 1. DSpark speculative decode — unfiled

The reference `scouzi1966/mlx-swift-lm` carries DSpark in
`Libraries/MLXLLM/Models/DeepseekV4.swift`: `DeepseekV4DSparkMarkovHead`
(line 757), `DeepseekV4DSparkConfidenceHead` (777), `DeepseekV4DSparkProposal`
(792), `DeepseekV4DSparkStage` (800), and `DeepseekV4DSparkGenerator` (2355).
This repository decodes the `dspark_*` configuration keys
(`Libraries/MLXLLM/Models/DeepSeekV4Configuration.swift`) and implements none
of the behavior. This repository has its own speculative-decoding machinery
(`Libraries/MLXLMCommon/SpeculativeDecoding.swift` and
`Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`). Compare future
DSpark work against that machinery; do not port the reference code as one
block.

### 2. Activation quantization (`DeepseekV4ActivationQuant`) — verification not complete, task `^e7b24ws`

The reference ships `Libraries/MLXLMCommon/DeepseekV4ActivationQuant.swift`
(153 lines): an e4m3 activation round trip and a symmetric-Q8 matvec.
`osaurus-ai/osaurus` carries a `deepseekV4ActivationQAT` load flag, which
points to an opt-in accuracy or throughput path, not to a load requirement.

Verification outcome: the verification is not complete. The gated real-weights
suite exists at
`IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift`,
but the checkpoint (141 GiB) is absent, and the suite skips. The
suite decides the question when the weights arrive. Task `^e7b24ws` tracks
that run. No test result shows that `mlx-community/DeepSeek-V4-Flash-4bit`
loads without activation quantization.

### 3. JANGTQ / `mxtq` quantization variants — unfiled

`osaurus-ai/vmlx-swift-lm` ships `DeepseekV4JANGTQ.swift` and
`DeepseekV3JANGTQ.swift` for bundles with `weight_format == "mxtq"` (a
TurboQuant codebook). This repository does not support that format. The
`mxtq_bits` configuration keys are skipped.

### 4. DeepSeek-V4-Pro — unfiled

`mlx-community/DeepSeek-V4-Pro` (1.6T total parameters, 49B active) has the
same `model_type`, thus it can possibly load. It is not validated, and its
memory footprint is not practical here. Do not read this document as a
statement of support for that model.

### 5. Fused mxfp4 Metal kernels — deferred in task `^wkv5j6f`; a performance follow-up is unfiled

The reference's `SwitchLayers.swift` carries two hand-written Metal kernels:
`deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and
`deepseek_v4_native_mxfp4_down_sum6`. They are a throughput optimization, not
a correctness requirement. Task `^wkv5j6f` (done) kept the generic
`gatherQuantizedMM` path and deferred the kernels. The reference gates the
kernels behind these environment knobs: `VMLX_DSV4_NATIVE_MXFP4`,
`VMLX_DSV4_MXFP4_ROWS_PER_SIMD`, and `VMLX_DSV4_MXFP4_SIMD_GROUPS`.

Do NOT start a decode-speed task with these kernels. Task `^3gh7rb5` measured
the decode step on the real weights, and the arithmetic of the gathered
matrix multiply is not the cost. Refer to "Decode performance" below.

### 6. `deepseek_v32` — unfiled

This repository has no port of `deepseek_v32`, but upstream `mlx-lm` has one
(`mlx_lm/models/deepseek_v32.py`). It is the closest relative of the sparse
attention of DeepSeek-V4.

Both parts of the DeepSeek-V4 sparse attention now stand in the module tree,
and every tensor of both loads from the published checkpoint:
`Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` (task `^r92pjcr`) picks the
pooled chunks each query reads, and
`Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift` (task `^tty95f4`) pools
those chunks.

The attention path reads neither yet, thus a long prompt still runs dense
attention over every key. The gap is the cache that keeps the pooled chunks
across calls: a decode step carries one token, and a compressor with no such
cache pools nothing out of one token. Task `^ab1eq0r` carries that cache and
the sparse path that reads it.

### 7. Application-layer pieces from `scouzi1966/maclocal-api` — out of scope by design, unfiled

These files are server and CLI concerns of an application around the library,
not library concerns:

- `Sources/AFMKitMLX/Models/DeepseekV4CheckpointConverter.swift`
- `Sources/AFMKitDwarfStar/AFMDwarfStarCheckpoint.swift`
- `Sources/AFMKitMLX/AFMMLXRuntimeAdapter.swift`
- `Sources/AFMKitMLX/AFMMLXMetalSchedulingPolicy.swift`

They are out of scope by design. Their checkpoint-conversion and
Metal-scheduling logic is the best available reference if that function
becomes necessary here.

## Decode performance

Task `^3gh7rb5` profiled one decode step against the real weights of
`mlx-community/DeepSeek-V4-Flash-4bit` on an M3 Ultra (512 GiB), with a
release build and a 64-token context.

The weight files of this checkpoint hold 151,482,475,612 bytes, which is
141 GiB, measured on the local snapshot on 2026-08-13. The routed experts
(`ffn.switch_mlp`) hold 137 GiB of that, which is 97%.

The cost is memory residency, not arithmetic. MLX gives its Metal residency
set a capacity of zero unless the process raises the wired limit, and it takes
only buffers of 1 MB or less from the one heap that set holds. Every weight
tensor of this checkpoint is larger than 1 MB, and each routed-expert weight
tensor holds 1 GiB. Thus every weight buffer sits outside the residency set,
and Metal makes all 141 GiB resident again for each command buffer. One command
buffer is one decode step.

Measured, one decode step over all 43 layers:

| stage | time | share |
|---|---|---|
| routed experts (`ffn.switch_mlp`) | 2039 ms | 99% |
| hyper-connections (`hc_attn`, `hc_ffn`) | 21.8 ms | 1% |
| attention | 14.0 ms | below 1% |
| shared expert | 2.4 ms | below 1% |

The same layer chained 43 times, which runs the same number of operations
against one layer's weights, takes 68 ms. Thus the number of operations is not
the cost, and the fused mxfp4 kernels of deferred item 5 cannot answer this.

The answer is to raise the Metal wired limit BEFORE the weights are
allocated. Measured with a release build and a direct model call: 2.10 s for
one decode step with the default limit, and 0.068 s with the limit raised,
which is 31 times faster. A limit raised AFTER the load changes nothing,
because a buffer joins the residency set when it is made, and a limit that
falls again empties the set.

The integration suite builds for debug and decodes through `TokenIterator`,
which is slower than that release-build profile. Measured there on 2026-08-13,
16 steady steps each: 0.593 s per step with the limit raised, and 2.124 s
without it. The 12,400-token run of
`longGenerationPastTwelveThousandTokensCompletes` therefore takes about 123
minutes, which is inside the 240-minute per-test limit, where the same run
without the fix would need about 7 hours.

`IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift`
raises the limit with a `WiredMemoryTicket` that starts before the shared load
and never ends. A consumer of this library must do the same. The
`wiredMemoryTicket` argument of `MLXLMCommon.generate` starts too late to
help.

Two tests of that suite guard the fix, thus a change that removes it fails in
seconds instead of after hours:

- `wiredMemoryLimitCoversTheWholeCheckpoint` asserts that the manager applied
  the whole request, and that the limit covers the whole 141 GiB checkpoint.
- `decodeStepStaysInsideTheLongGenerationBudget` measures 16 steady-state
  decode steps and asserts that the median step stays inside 0.87 s, which puts
  the 12,400-token run inside three quarters of the 240-minute suite limit.

`swift test` does not compile that suite: no SwiftPM target holds
`IntegrationTesting/`. Compile it with
`xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`.

## Spelling note

This repository spells the family `DeepSeek` (for example, `DeepSeekV4Model`).
The reference repositories spell it `Deepseek`. File names quoted from the
reference repositories keep the reference spelling.

---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8c8b5rxcs258gq8e8751vme
  text: |-
    Research:
    - Confirmed in mlx core (transforms.cpp custom_function): outputs pass through `stop_gradient(out)` with no stream, so `default_stream(default_device())` runs; default streams are thread_local (stream.cpp default_stream_storage), so the first CustomFunction call on each thread makes a new GPU stream. The mlx-c / Swift CustomFunction API has no way to pass a stream, so "pass the stream explicitly" is not available from this repo; taking the plain-ops route.
    - No MLX API reports "a distributed group is active" without side effects (a non-strict `MLXDistributed.initialize` caches a size-1 group under jaccl and would break a later strict init). The group is known where the model shards: `Qwen35DecoderLayer.shard(_:with:group:)` -> `Qwen35GatedDeltaNet.adoptShardedHeads(from:)`, only reached when plan.groupSize > 1. Qwen35 (MLXLLM) is the only TensorParallelShardable model that uses gatedDeltaUpdate (Inkling2 does not; Qwen3Next and VLM Qwen35 are not shardable).
    - Plan: `gatedDeltaOps`/`gatedDeltaUpdate`/`Snapshots.gatedDeltaUpdate` gain a defaulted `recomputeChunks: Bool = true`; false runs the chunk steps as plain ops (same forward, plain-ops VJP keeps every step: correct gradients, more memory). The sharded GDN layer turns it off in adoptShardedHeads.
    - Test: forward through the ops fallback on a fresh Thread (fresh C++ thread-local default stream), compare the GPU-stream count from `mlx_get_streams` before/after.
  timestamp: 2026-11-26T15:42:17.885888+00:00
- actor: claude-code
  id: 01m8c8mypz0p7w8why97r9yht8
  text: |-
    Implementation landed (TDD):
    - RED 1: `GatedDeltaGPUStreamTests` "forward through the ops fallback" on a fresh Thread added GPU stream(s) (CustomFunction path). GREEN after adding `recomputeChunks: Bool = true` to `gatedDeltaUpdate` / `gatedDeltaOps` / `Snapshots.gatedDeltaUpdate`; false runs each chunk as plain `GatedDeltaRecompute.steps` (new `GatedDeltaRecompute.chunk(masked:recomputing:)`).
    - RED 2: a `Qwen35GatedDeltaNet` after `adoptShardedHeads` (Dk = 16 → fallback in inference) added GPU stream [2]. GREEN: layer gets `private(set) var recomputesChunks = true`, set false in `adoptShardedHeads` (only reached for a group of >1 in `Qwen35DecoderLayer.shard`), passed to both gatedDeltaUpdate call sites in `recurrentBody`.
    - Gradient guard: ops gradients (q,k,v,a,b) with recomputeChunks false equal the recomputing path exactly (largest diff 0).
    - Test pitfall for the next agent: measure on a fresh Thread that runs ONLY the forward. Evaluating inputs on that thread first (or eval of already-evaluated arrays, which falls back to `default_stream`) creates the thread's C++ default stream outside the measured span and hides the bug.
    - `swift build --build-tests` prints `warning: missing creator for mutated node: (.../mlx-swift_Cmlx.bundle/Contents/MacOS)` even on a no-op rebuild; it is a SwiftBuild warning about the Cmlx resource bundle, not from these files.
  timestamp: 2026-11-26T15:47:38.335353+00:00
- actor: claude-code
  id: 01m8c93jrhh8tybhxtckt3pqkd
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift (new, 3 tests). `swift build --build-tests` clean apart from the SwiftBuild Cmlx-bundle "missing creator" warning that also appears on a no-op build. xctest: MLXLMTests 1261 XCTest + 2749 Swift Testing passed (3 known issues, all pre-existing withKnownIssue: AttentionKernelToleranceTests, MiniCPM5 CDATA); MLXFoundationModelsTests 618 passed; MLXGuidedGenerationTests 95 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 passed. All existing GatedDelta tests pass, including testGatedDeltaOpsPathIsDifferentiable and the fast-path gradient test.
    - next: /review
  timestamp: 2026-11-26T15:55:37.617403+00:00
- actor: claude-code
  id: 01m8cbfv4mq83jsb3hvpj79wff
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (e8b5ea42), 3 findings (3 confirmed, 0 refuted) — Tests/MLXLMTestSupport/Inkling2TinyModels.swift:93, Tests/MLXLMTestSupport/Inkling2TinyModels.swift:104, Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift:157
    - next: /implement 4hdr1n6 to resolve the three unchecked items in "Review Findings (2026-11-26 10:35)", then re-review
  timestamp: 2026-11-26T16:37:16.564147+00:00
- actor: claude-code
  id: 01m8cbge6hpkc9p0fhz6jg650n
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — GatedDelta.swift, GatedDeltaDecodeKernel.swift, Qwen35.swift, GatedDeltaGPUStreamTests.swift (new)
    - test: green — 5 bundles (MLXLMTests 1261 XCTest + 2749 ST, FM 618, Guided 95, CXGrammar 7, Macros 11), verify-docs 0; flaky ClusterInklingParityTests fixed in Inkling2TinyModels.swift (task-local random state)
    - commit: e8b5ea42
    - review: findings — Tests/MLXLMTestSupport/Inkling2TinyModels.swift:93, Tests/MLXLMTestSupport/Inkling2TinyModels.swift:104, Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift:157
  timestamp: 2026-11-26T16:37:36.081925+00:00
- actor: claude-code
  id: 01m8cc2wd174vee8cac6mkkcky
  text: |-
    Resolved the three findings of "Review Findings (2026-11-26 10:35)":
    - Inkling2TinyModels.swift: changed `[String: MLXArray]()` to `: [String: MLXArray] = [:]`. It was the only empty-collection init in the file; the drafter's `Dictionary(uniqueKeysWithValues:)` is not an empty init. Added the named constants `expertAxis` (0), `rowAxis` (1) and `packedAxis` (2) for the packed NVFP4 weight dims. `swiftlint lint --only-rule no_magic_numbers` reports the finding at line 104 on the HEAD copy of the file and 0 violations on the new one.
    - maxAbsDiff: added one public `maxAbsDiff(_:_:)` (it casts to float32) in the new Tests/MLXLMTestSupport/MaxAbsDiff.swift. Deleted `ContinuationAssertions.maxAbsDiff` and the four private `continuation.maxAbsDiff` wrappers (Qwen3VL, Qwen35, Qwen3VLMoE, GlmOcr continuation tests); their call sites now resolve to the shared function. GatedDeltaGPUStreamTests now uses it instead of `largestDifference`. Gemma2Tests and NanbeigeTests keep their own private pre-existing copies because they were out of scope.
  timestamp: 2026-11-26T16:47:40.449070+00:00
- actor: claude-code
  id: 01m8cc2zcdxfqce5gywbp5vywf
  text: |-
    ### implement — changed
    - evidence: 9 files — Tests/MLXLMTestSupport/MaxAbsDiff.swift (new), Tests/MLXLMTestSupport/Inkling2TinyModels.swift, Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift, ContinuationAssertions.swift, Qwen3VLContinuationTests.swift, Qwen35ContinuationTests.swift, Qwen3VLMoEContinuationTests.swift, GlmOcrContinuationTests.swift. swift-format applied. `swift build --build-tests` reported no warnings other than the known SwiftBuild Cmlx "missing creator" warning. xctest MLXLMTests: 1261 XCTest, 0 failures; 2749 Swift Testing passed (the 3 known issues are pre-existing withKnownIssue). MLXFoundationModelsTests: 618 passed. All 3 findings are checked.
    - next: /review
  timestamp: 2026-11-26T16:47:43.501585+00:00
position_column: doing
position_ordinal: '80'
title: GatedDelta ops fallback uses CustomFunction, which creates a second GPU stream
---
Found by FoundationModelsCluster's GPU-stream guard (wevkejv, 13d83dd).
- MLX's `custom_function` calls `stop_gradient` on the calling thread's C++ default stream, so each call creates a second GPU stream. That stream is the known cause of the FAST_SYNCH hangs (exo#2391).
- `GatedDeltaRecompute.masked/unmasked` (Libraries/MLXLMCommon/GatedDelta.swift:416) are CustomFunctions. They're reached only through `gatedDeltaOps`, which runs when neither fastGatedDelta nor the Metal kernel applies: training (`useKernel: false`), or Dk % 32 != 0.
- Inference with Qwen3.8 and Inkling (Dk = 128) never reaches it, so clustered workers are safe today.

Fix: when a distributed group is active, keep the ops fallback off CustomFunction (plain ops with no VJP wrapper), or pass the stream explicitly. Add a test that asserts one GPU stream after a forward through the fallback, using `mlx_get_streams`. Low priority. It matters only if training or an odd Dk ever runs clustered.

#cluster

## Review Findings (2026-11-26 10:35)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M88ZG19EXA9WYCFTG4HDR1N6.jsonl` — no validator matches this file
> - `.kanban/tasks/01M88ZG19EXA9WYCFTG4HDR1N6.md` — no validator matches this file

- [x] `Tests/MLXLMTestSupport/Inkling2TinyModels.swift:93` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[String: MLXArray]()' with ': [String: MLXArray] = [:]'.
- [x] `Tests/MLXLMTestSupport/Inkling2TinyModels.swift:104` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift:157` `reuse/reuse` — `largestDifference` reimplements the max-absolute-difference helper that the test suite already has in several places. It is a fresh private copy of `abs(a - b).max().item(Float.self)`, so the same logic now lives in another file, and the repo's shared helper is left unused. Call the existing `maxAbsDiff` (or make a shared, file-independent version of it in MLXLMTestSupport and use it here). If the float32 cast matters for bf16 gradients, add it to the shared helper, or keep the cast at the call site: `Self.maxAbsDiff($0.asType(.float32), $1.asType(.float32))`.

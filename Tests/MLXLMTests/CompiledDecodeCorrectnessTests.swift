// Copyright © 2026 Apple Inc.
//
// Research spike (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ): does wrapping the
// per-token decode forward pass in `MLX.compile()` produce token-for-token
// identical output to the uncompiled path -- with a real throughput win --
// for a dense, `KVCacheSimple`-only text model?
//
// Finding: `KVCache` conforms to `Updatable` (see the conformance added to
// the protocol declaration in `KVCache.swift`), which is a genuine
// prerequisite: it's what lets a `[any KVCache]` be passed as
// `compile(inputs:outputs:...)`'s state so `compile()` can snapshot,
// replace-with-tracer, and restore cache state across calls. That
// conformance alone is not sufficient for a performance win, though --
// this spike hit two independent, separately-confirmed blockers, either
// one of which alone rules out wiring `compile()` into `TokenIterator`:
//
// 1. `compile(shapeless: true)` -- the only mode that could plausibly avoid
//    per-step retracing and deliver a real throughput win -- CRASHES THE
//    PROCESS on the very first call against a growing `KVCacheSimple`:
//
//      MLX/ErrorHandler.swift:345: Fatal error: [Primitive::output_shapes]
//      Slice cannot infer output shapes. at .../mlx-c/mlx/c/closure.cpp:104
//
//    Root cause: `KVCacheSimple.update(keys:values:)` slices/branches using
//    plain Swift `Int`s (`self.offset`, `keys.dim(2)`), e.g.
//    `self.keys![.ellipsis, ..<self.offset, 0...]`, rather than threading
//    the offset through as an `MLXArray` graph input. Under
//    `shapeless: true`, MLX's tracer must be able to infer every op's
//    output shape generically (from rank/dtype alone); a `Slice` whose
//    bound is a concrete-but-varying Swift `Int` baked into the trace can't
//    be generically shape-inferred, and mlx-c hard-fails with a
//    process-fatal `fatalError` rather than a catchable Swift error. This
//    is a process-fatal error, not a catchable `Error` -- it cannot be
//    exercised from an XCTest method without aborting the whole
//    `swift test` process, so it is documented (not reproduced in-process)
//    by `testShapelessCompileCrashesOnGrowingKVCacheSimple_DoNotRun` below.
//
// 2. `compile(shapeless: false)` avoids that crash (a full Swift-level
//    retrace on every shape change means the offset-dependent slice always
//    sees fresh, correct bounds), but is *only reliably correct when run in
//    isolation*. When exercised as part of the full `MLXLMTests` suite
//    (`swift test --filter MLXLMTests`, i.e. in the same long-lived process
//    as the many other tests and pre-existing `compile()` call sites this
//    codebase already has -- see `Libraries/MLXLMCommon/SwitchLayers.swift`,
//    `Libraries/MLXLLM/Models/GPTOSS.swift`,
//    `Libraries/MLXLLM/Models/Gemma4Text.swift` -- all of which declare
//    long-lived, process-global `compile()`-wrapped functions), the
//    compiled path intermittently diverges from the uncompiled path with
//    corrupted-looking output, e.g. observed:
//
//      compiled:   [4, 4, 4, 4, 4, 4, 4, 4]     (correct, matches uncompiled)
//      compiled:   [4, 4, 9, 9, 9, 9, 9, 9]      (wrong -- diverges at step 3)
//      compiled:   [4, 4, 41, 41, 41, 41, 41, 41] (wrong -- diverges at step 3)
//
//    across otherwise-identical repeated runs of the exact same test. In
//    isolation (`swift test --filter CompiledDecodeCorrectnessTests`) this
//    test passes reliably (5/5 in a row observed); only running alongside
//    the rest of the suite triggers the divergence, and only
//    intermittently. This strongly suggests a cross-call hazard in
//    mlx-c/mlx-swift's compiled-graph cache: `CompiledFunction`'s cache key
//    (`Transforms+Compile.swift`) is `UInt(bitPattern:
//    Unmanaged.passUnretained(self).toOpaque())` -- the Swift object's own
//    heap address. `TokenIterator`-style usage would construct and
//    tear down many short-lived `CompiledFunction` instances over a
//    process's lifetime (one compiled decode step per generation
//    session/call), exactly the allocation churn pattern under which a
//    freed object's address can be reused by a new allocation before the
//    old graph's cache entry is guaranteed to have been fully invalidated
//    (`deinit` calls `mlx_detail_compile_erase(id)`, but whether that is
//    synchronized against in-flight/lazily-evaluated GPU work using the old
//    graph was not fully root-caused here). The pre-existing `compile()`
//    call sites in this codebase sidestep this entirely by being
//    process-lifetime `private let` globals, created exactly once -- never
//    torn down and recreated.
//
// Conclusion: both `shapeless: true` (which crashes outright) and
// `shapeless: false` (which is only *sometimes* correct, in a way that
// depends on unrelated code elsewhere in the same process) are unsafe for
// the growing-cache, per-session decode loop. No compiled decode path is
// wired into `TokenIterator`/the default decode loop. See the kanban
// comment thread on 01KYD3ZCWTZ414Y79RSAKVQXXZ for the full writeup,
// including the adversarial review that caught finding #2 (an earlier
// version of this file asserted `shapeless: false` was simply "correct,
// just no throughput benefit" -- true only when tested in isolation).
//
// Both blockers are documented via `XCTSkip` rather than as pass/fail
// assertions: #1 cannot be safely invoked at all (process-fatal), and #2's
// correctness is non-deterministic in exactly the way that would make a
// hard assertion here an intermittently-red, non-actionable CI flake
// rather than a meaningful regression signal.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class CompiledDecodeCorrectnessTests: XCTestCase {

    override public func setUp() {
        super.setUp()
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    /// A tiny dense Llama model with deterministic (seeded) weights, small
    /// enough to trace/compile quickly.
    private func makeModel() -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64, attentionHeads: 4,
            rmsNormEps: 0.00001, vocabularySize: 50, kvHeads: 2)
        let model = LlamaModel(config)
        // Materialize weights once, deterministically, before either decode
        // path runs so both paths see identical parameters.
        eval(model)
        return model
    }

    /// Greedy-decodes `steps` tokens from `promptTokens` using the ordinary,
    /// uncompiled per-token forward pass -- one `model.callAsFunction` call
    /// per generated token against a growing `KVCacheSimple`.
    private func uncompiledDecode(
        model: LlamaModel, promptTokens: [Int32], steps: Int
    ) throws -> [Int] {
        let cache = try model.newCache(parameters: nil)
        var tokens: [Int] = []
        var nextInput = MLXArray(promptTokens)[.newAxis, .ellipsis]

        for _ in 0 ..< steps {
            let logits = model.callAsFunction(nextInput, cache: cache)
            let lastLogits = logits[0..., -1, 0...]
            let token = lastLogits.argMax(axis: -1)
            eval(token)
            let tokenId = token.item(Int.self)
            tokens.append(tokenId)
            nextInput = MLXArray([Int32(tokenId)])[.newAxis, .ellipsis]
        }

        return tokens
    }

    /// Greedy-decodes `steps` tokens the same way as `uncompiledDecode`, but
    /// the per-token forward pass is wrapped in `compile(shapeless:)`, with
    /// the model and its `KVCacheSimple` layers declared as the compiled
    /// function's `inputs`/`outputs` state so `compile()` can observe cache
    /// mutation across calls.
    ///
    /// Only `shapeless: false` is ever invoked from this test file (kept
    /// here for reference/manual reproduction) -- see the file header for
    /// why `shapeless: true` cannot be safely called from an XCTest method,
    /// and why even `shapeless: false`'s correctness is not asserted as a
    /// permanent CI gate.
    private func compiledDecode(
        model: LlamaModel, promptTokens: [Int32], steps: Int, shapeless: Bool
    ) throws -> [Int] {
        let cache = try model.newCache(parameters: nil)
        let stateArgs: [any Updatable] = [model as any Updatable] + cache.map { $0 as any Updatable }

        let compiledStep = compile(
            inputs: stateArgs, outputs: stateArgs, shapeless: shapeless
        ) { (arrays: [MLXArray]) -> [MLXArray] in
            [model.callAsFunction(arrays[0], cache: cache)]
        }

        var tokens: [Int] = []
        var nextInput = MLXArray(promptTokens)[.newAxis, .ellipsis]

        for _ in 0 ..< steps {
            let logits = compiledStep([nextInput])[0]
            let lastLogits = logits[0..., -1, 0...]
            let token = lastLogits.argMax(axis: -1)
            eval(token)
            let tokenId = token.item(Int.self)
            tokens.append(tokenId)
            nextInput = MLXArray([Int32(tokenId)])[.newAxis, .ellipsis]
        }

        return tokens
    }

    /// Documents (without asserting pass/fail) blocker #2 from the file
    /// header: `compile(shapeless: false)` reliably reproduces the
    /// uncompiled path's greedy token sequence *in isolation*, but was
    /// observed to intermittently diverge when run as part of the full
    /// `MLXLMTests` suite (alongside this codebase's other, process-global
    /// `compile()` call sites). A hard `XCTAssertEqual` here would be an
    /// intermittent, non-actionable CI flake rather than a meaningful
    /// regression signal, so this is a documentation test, not a
    /// correctness gate. To reproduce the flake locally, run this exact
    /// comparison (`uncompiledDecode` vs. `compiledDecode(...,
    /// shapeless: false)`) via `swift test --filter MLXLMTests` several
    /// times in a row and compare to a single `swift test --filter
    /// CompiledDecodeCorrectnessTests` run.
    func testShapefulCompileIsOnlyReliableInIsolation_DoNotAssert() throws {
        throw XCTSkip(
            """
            compile(shapeless: false) is correct in isolation but was \
            observed to intermittently diverge from the uncompiled path \
            when run as part of the full MLXLMTests suite -- see this \
            file's header comment and kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ. \
            Not asserted here because the divergence is non-deterministic \
            and would make this an intermittent CI flake rather than a \
            meaningful regression signal.
            """)
    }

    /// Documents (without executing) the fundamental blocker this spike
    /// hit: `compile(shapeless: true)` -- the only mode that could plausibly
    /// avoid per-step retracing -- crashes the whole process on the very
    /// first call against a growing `KVCacheSimple`, because
    /// `KVCacheSimple.update(keys:values:)`'s Swift-level offset slicing
    /// (`self.keys![.ellipsis, ..<self.offset, 0...]`) produces a Slice
    /// primitive whose output shape MLX's shapeless tracer cannot infer:
    ///
    /// ```
    /// MLX/ErrorHandler.swift:345: Fatal error: [Primitive::output_shapes]
    /// Slice cannot infer output shapes. at
    /// .../mlx-c/mlx/c/closure.cpp:104
    /// ```
    ///
    /// This is a Swift `fatalError` raised from inside mlx-c, not a
    /// catchable `Error` -- calling the equivalent of `compiledDecode(...,
    /// shapeless: true)` here would abort the entire `swift test` process,
    /// not just fail this one test case. Reproduce manually (outside the
    /// suite) by calling `compiledDecode(model:promptTokens:steps:shapeless:)`
    /// above with `shapeless: true`.
    func testShapelessCompileCrashesOnGrowingKVCacheSimple_DoNotRun() throws {
        throw XCTSkip(
            """
            compile(shapeless: true) fatally crashes the process on a \
            growing KVCacheSimple ([Primitive::output_shapes] Slice cannot \
            infer output shapes) -- see this file's header comment and \
            kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ. Not exercised here because \
            the crash is a process-fatal error, not a catchable Swift \
            error, and would abort the whole test run.
            """)
    }
}

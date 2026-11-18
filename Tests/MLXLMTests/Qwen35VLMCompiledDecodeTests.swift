// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class Qwen35VLMCompiledDecodeTests: XCTestCase {
    private let tokens: [Int32] = [1, 7, 3, 9, 2]
    private let positions: [[Int32]] = [
        [0, 0, 0],
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 8],
        [4, 7, 9],
    ]

    private func configuration(numExperts: Int) throws -> Qwen35Configuration.TextConfiguration {
        let modelType = numExperts > 0 ? "qwen3_5_moe" : "qwen3_5"
        let json = """
            {
                "model_type": "\(modelType)",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": \(numExperts),
                "num_experts_per_tok": \(min(2, numExperts)),
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeModel(
        numExperts: Int, seed: UInt64, dtype: DType = .float16
    ) throws -> Qwen35Language.Model {
        let configuration = try configuration(numExperts: numExperts)
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35Language.Model(configuration)
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            model.train(false)
            return model
        }
    }

    /// A cache factory for the two layers of the test model: a new linear-attention cache,
    /// and a new attention cache from `makeAttention`.
    ///
    /// - Parameter makeAttention: The factory of the attention layer cache.
    /// - Returns: The factory of the caches of the model.
    private func caches(
        _ makeAttention: @escaping () -> KVCache
    ) -> () -> [KVCache?] {
        { [MambaCache(), makeAttention()] }
    }

    private var plainCache: () -> [KVCache?] {
        caches { KVCacheSimple() }
    }

    private var quantizedCache: () -> [KVCache?] {
        caches { QuantizedKVCache(groupSize: 32, bits: 8) }
    }

    private var rotatingCache: () -> [KVCache?] {
        caches { RotatingKVCache(maxSize: 4) }
    }

    private var turboQuantCache: () -> [KVCache?] {
        caches { TurboQuantKVCache(bits: 4, seed: 67) }
    }

    /// TurboQuant runs prefill this many tokens before their decode steps.
    private static let turboQuantPrefillCount = 2

    /// The count of M-RoPE position axes of each token.
    private static let positionAxisCount = 3

    /// The deallocation tests decode this many tokens before they release the model.
    private static let deallocationStepCount = 3

    /// Runs one forward call over the tokens of `range`, with their positions.
    ///
    /// - Parameters:
    ///   - model: The model to run.
    ///   - range: The indices of the tokens and positions to feed.
    ///   - cache: The caches the call writes.
    ///   - compiled: True to allow the compiled decode path.
    /// - Returns: The output of the call, not evaluated.
    private func step(
        _ model: Qwen35Language.Model,
        tokens range: Range<Int>,
        cache: [KVCache?],
        compiled: Bool
    ) -> MLXArray {
        let input = MLXArray(Array(tokens[range])).reshaped(1, range.count)
        let positionIds = MLXArray(
            (0 ..< Self.positionAxisCount).flatMap { axis in
                positions[range].map { $0[axis] }
            }
        ).reshaped(Self.positionAxisCount, 1, range.count)
        eval(input, positionIds)
        return model.forward(
            input,
            cache: cache,
            positionIds: positionIds,
            useCompiledDecode: compiled)
    }

    /// Prefills the first `prefillCount` tokens eagerly in one call, then decodes the other
    /// tokens one at a time.
    ///
    /// - Parameters:
    ///   - model: The model to run.
    ///   - compiled: True to allow the compiled decode path for the decode steps.
    ///   - makeCache: The factory of the caches of the run.
    ///   - prefillCount: The count of tokens of the prefill. `0` decodes each token.
    /// - Returns: The outputs and the final state of the caches.
    private func run(
        _ model: Qwen35Language.Model,
        compiled: Bool,
        cache makeCache: () -> [KVCache?],
        prefillCount: Int = 0
    ) -> CompiledDecodeRun {
        precondition(tokens.count == positions.count)
        let cache = makeCache()
        let prefill = prefillCount > 0 ? [(range: 0 ..< prefillCount, compiled: false)] : []
        let decode = (prefillCount ..< tokens.count).map { index in
            (range: index ..< index + 1, compiled: compiled)
        }
        return CompiledDecodeRun.decoding(prefill + decode, cache: cache) { call in
            step(model, tokens: call.range, cache: cache, compiled: call.compiled)
        }
    }

    /// Runs `model` eagerly and compiled over new caches, and asserts that both runs give the
    /// same outputs, cache state and cache offsets.
    ///
    /// - Parameters:
    ///   - model: The model to run.
    ///   - makeCache: The factory of the caches of each run.
    ///   - prefillCount: The count of tokens of the eager prefill of each run.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    /// - Returns: The compiled run.
    @discardableResult
    private func assertCompiledMatchesEager(
        _ model: Qwen35Language.Model,
        cache makeCache: () -> [KVCache?],
        prefillCount: Int = 0,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> CompiledDecodeRun {
        CompiledDecodeRun.assertCompiledMatchesEager(file: file, line: line) { compiled in
            run(model, compiled: compiled, cache: makeCache, prefillCount: prefillCount)
        }
    }

    /// Asserts that `model` ran no compiled segment, and that its per-layer GDN and MoE
    /// traces ran or did not run.
    ///
    /// - Parameters:
    ///   - model: The model that ran.
    ///   - tracesRan: True when the run must fall back to the per-layer traces, false when it
    ///     must stay on the eager path.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    private func assertNoSegments(
        _ model: Qwen35Language.Model,
        tracesRan: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0, file: file, line: line)
        let traceCounts = [
            ("GDN", model.compiledGDNTraceCount), ("MoE", model.compiledMoETraceCount),
        ]
        for (trace, count) in traceCounts {
            XCTAssertEqual(
                count > 0, tracesRan, "\(trace) traces ran \(count) times", file: file,
                line: line)
        }
    }

    /// The first module of `type` among the modules of `model`.
    ///
    /// - Parameters:
    ///   - type: The type of the module.
    ///   - model: The model that holds the module.
    /// - Returns: The module, or nil when the model holds none of `type`.
    private func firstModule<Found: Module>(
        _ type: Found.Type, in model: Qwen35Language.Model
    ) -> Found? {
        model.modules().lazy.compactMap { $0 as? Found }.first
    }

    /// Asserts that `model` ran its compiled segments, or else its per-layer trace fallback.
    ///
    /// - Parameters:
    ///   - model: The model that ran.
    ///   - usesSegments: True when the run must use the compiled segments.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    private func assertCompiledPath(
        _ model: Qwen35Language.Model,
        usesSegments: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        if usesSegments {
            XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0, file: file, line: line)
        } else {
            assertNoSegments(model, tracesRan: true, file: file, line: line)
        }
    }

    func testDensePlainCacheMatchesEagerExactlyWithDynamicMRoPE() throws {
        let model = try makeModel(numExperts: 0, seed: 31)

        let compiled = assertCompiledMatchesEager(model, cache: plainCache)

        // `ArraysCache.advance` also moves the offset of the recurrent cache,
        // thus both layers hold the 5 tokens.
        XCTAssertEqual(compiled.offsets, [5, 5])
        assertCompiledPath(model, usesSegments: true)
    }

    func testMoEPlainCacheMatchesEagerExactly() throws {
        let model = try makeModel(numExperts: 4, seed: 37)

        assertCompiledMatchesEager(model, cache: plainCache)
        assertCompiledPath(model, usesSegments: true)
    }

    func testLinearLayerSharedHybridBodyMatchesEagerLayerExactly() throws {
        let configuration = try configuration(numExperts: 4)
        let layer = withRandomState(MLXRandom.RandomState(seed: 103)) {
            let layer = Qwen35Language.DecoderLayer(configuration, layerIdx: 0)
            layer.update(parameters: layer.parameters().mapValues { $0.asType(.float16) })
            layer.train(false)
            return layer
        }
        let gatedDeltaNet = try XCTUnwrap(layer.linearAttn)
        let zero = gatedDeltaNet.zeroStates(batch: 1, dtype: .float16)
        let (x, convState, recState) = withRandomState(MLXRandom.RandomState(seed: 107)) {
            (
                MLXRandom.normal([1, 1, configuration.hiddenSize]).asType(.float16),
                MLXRandom.normal(zero.conv.shape).asType(.float16),
                MLXRandom.normal(zero.rec.shape)
            )
        }
        let cache = MambaCache()
        cache[0] = convState
        cache[1] = recState

        let eager = layer.forward(
            x, attentionMask: .none, ssmMask: nil, cache: cache, positionIds: nil,
            keepsSnapshots: false, allowCompiledDecode: false)
        let step = layer.linearLayerBody(x: x, convState: convState, recState: recState)

        XCTAssertEqual(ArraySnapshot(step.output), ArraySnapshot(eager))
        XCTAssertEqual(ArraySnapshot(step.convState), ArraySnapshot(try XCTUnwrap(cache[0])))
        XCTAssertEqual(
            ArraySnapshot(step.recurrentState), ArraySnapshot(try XCTUnwrap(cache[1])))
        XCTAssertNil(step.snapshots)
    }

    func testFirstEmptyCacheTokenStaysEager() throws {
        let model = try makeModel(numExperts: 0, seed: 41)
        let cache = plainCache()

        eval(step(model, tokens: 0 ..< 1, cache: cache, compiled: true))
        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)

        eval(step(model, tokens: 1 ..< 2, cache: cache, compiled: true))
        assertCompiledPath(model, usesSegments: true)
    }

    /// A quantized cache keeps the segments: its attention step writes the same packed keys
    /// and values as the eager path, and the step gives the same output.
    func testQuantizedCacheMatchesEagerExactlyThroughSegments() throws {
        let model = try makeModel(numExperts: 4, seed: 43)

        assertCompiledMatchesEager(model, cache: quantizedCache)
        assertCompiledPath(model, usesSegments: true)
    }

    func testWrappedRotatingCacheMatchesEagerExactly() throws {
        let model = try makeModel(numExperts: 4, seed: 71)

        let compiled = assertCompiledMatchesEager(model, cache: rotatingCache)

        XCTAssertEqual(compiled.offsets, [5, 5])
        XCTAssertEqual(compiled.cacheState.suffix(2).map { $0.shape[2] }, [4, 4])
        assertCompiledPath(model, usesSegments: true)
    }

    func testTurboQuantCacheUsesExactPerLayerFallback() throws {
        let model = try makeModel(numExperts: 4, seed: 73)

        let fallback = assertCompiledMatchesEager(
            model, cache: turboQuantCache, prefillCount: Self.turboQuantPrefillCount)

        XCTAssertEqual(fallback.offsets, [5, 5])
        assertNoSegments(model, tracesRan: true)
    }

    func testBFloat16UsesExactCompiledRoutes() throws {
        let segmentModel = try makeModel(numExperts: 4, seed: 79, dtype: .bfloat16)
        assertCompiledMatchesEager(segmentModel, cache: plainCache)
        assertCompiledPath(segmentModel, usesSegments: true)

        let quantizedModel = try makeModel(numExperts: 4, seed: 83, dtype: .bfloat16)
        assertCompiledMatchesEager(quantizedModel, cache: quantizedCache)
        assertCompiledPath(quantizedModel, usesSegments: true)

        let fallbackModel = try makeModel(numExperts: 4, seed: 103, dtype: .bfloat16)
        assertCompiledMatchesEager(
            fallbackModel, cache: turboQuantCache, prefillCount: Self.turboQuantPrefillCount)
        assertNoSegments(fallbackModel, tracesRan: true)
    }

    func testFloat32StaysOnEagerPathForAllCacheRoutes() throws {
        let segmentModel = try makeModel(numExperts: 4, seed: 89, dtype: .float32)
        assertCompiledMatchesEager(segmentModel, cache: plainCache)
        assertNoSegments(segmentModel, tracesRan: false)

        let layerModel = try makeModel(numExperts: 4, seed: 97, dtype: .float32)
        assertCompiledMatchesEager(layerModel, cache: quantizedCache)
        assertNoSegments(layerModel, tracesRan: false)
    }

    func testTrainingGDNStaysOnEagerPath() throws {
        let model = try makeModel(numExperts: 4, seed: 101)
        let gatedDeltaNet = try XCTUnwrap(
            firstModule(Qwen35Language.GatedDeltaNet.self, in: model))
        gatedDeltaNet.train(true)

        assertCompiledMatchesEager(model, cache: plainCache)
        assertNoSegments(model, tracesRan: false)
    }

    private func assertWeightUpdatesAreVisible(
        cache makeCache: () -> [KVCache?],
        usesSegments: Bool,
        seed: UInt64,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let warm = try makeModel(numExperts: 4, seed: seed)
        let reference = try makeModel(numExperts: 4, seed: seed + 1)
        reference.update(parameters: warm.parameters())

        let baseline = run(warm, compiled: true, cache: makeCache)
        assertCompiledPath(warm, usesSegments: usesSegments, file: file, line: line)

        let updatedParameters = warm.parameters().mapValues { $0 * 1.5 }
        warm.update(parameters: updatedParameters)
        reference.update(parameters: updatedParameters)

        let updated = run(warm, compiled: true, cache: makeCache)
        let expected = run(reference, compiled: true, cache: makeCache)

        XCTAssertNotEqual(updated.outputs, baseline.outputs, file: file, line: line)
        updated.assertMatches(expected, file: file, line: line)
    }

    func testWholeStepSegmentsObserveParameterUpdates() throws {
        try assertWeightUpdatesAreVisible(
            cache: plainCache, usesSegments: true, seed: 47)
    }

    func testQuantizedCacheSegmentsObserveParameterUpdates() throws {
        try assertWeightUpdatesAreVisible(
            cache: quantizedCache, usesSegments: true, seed: 53)
    }

    func testTurboQuantFallbackTracesObserveParameterUpdates() throws {
        try assertWeightUpdatesAreVisible(
            cache: turboQuantCache, usesSegments: false, seed: 107)
    }

    private func assertModelDeallocates(
        cache makeCache: () -> [KVCache?],
        usesSegments: Bool,
        seed: UInt64,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        var model: Qwen35Language.Model? = try makeModel(numExperts: 4, seed: seed)
        var cache: [KVCache?]? = makeCache()

        for index in 0 ..< Self.deallocationStepCount {
            eval(step(model!, tokens: index ..< index + 1, cache: cache!, compiled: true))
        }

        assertCompiledPath(model!, usesSegments: usesSegments, file: file, line: line)

        weak let inner = model
        weak let gatedDeltaNet = firstModule(Qwen35Language.GatedDeltaNet.self, in: model!)
        weak let moe = firstModule(Qwen35Language.SparseMoeBlock.self, in: model!)
        XCTAssertNotNil(gatedDeltaNet, "expected a GDN layer", file: file, line: line)
        XCTAssertNotNil(moe, "expected a MoE block", file: file, line: line)

        model = nil
        cache = nil

        XCTAssertNil(inner, "compiled traces retained the language model", file: file, line: line)
        XCTAssertNil(
            gatedDeltaNet, "compiled traces retained the GDN block", file: file, line: line)
        XCTAssertNil(moe, "compiled traces retained the MoE block", file: file, line: line)
    }

    func testModelDeallocatesAfterWholeStepCompiledDecode() throws {
        try assertModelDeallocates(cache: plainCache, usesSegments: true, seed: 59)
    }

    func testModelDeallocatesAfterQuantizedCacheDecode() throws {
        try assertModelDeallocates(cache: quantizedCache, usesSegments: true, seed: 61)
    }

    func testModelDeallocatesAfterTurboQuantFallbackDecode() throws {
        try assertModelDeallocates(cache: turboQuantCache, usesSegments: false, seed: 109)
    }
}

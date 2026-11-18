import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen3NextCompiledDecodeTests: XCTestCase {
    /// The parts of the tiny Qwen3-Next test model that differ from the shared configuration of
    /// ``GDNDecodeKernelFixture``.
    enum TinyModel {
        /// The decoder layers: a full-attention layer follows each linear-attention layer.
        static let hiddenLayers = 4
        /// The head width of the full-attention layers of the default model.
        static let attentionHeadDim = 8
        /// The hidden size and the expert MLP size of the default model.
        static let hiddenSize = 16
        /// The hidden size and the expert MLP size that a group size of 32 can quantize.
        static let quantizableHiddenSize = 64
        /// The width of each linear-attention key and value head of the default model.
        static let linearHeadDim = 8
        /// The width of each linear-attention head that a group size of 32 can quantize.
        static let quantizableLinearHeadDim = 32
        /// The number of linear-attention key heads and of value heads.
        static let linearHeads = 2
        /// The vocabulary size.
        static let vocabularySize = 32
        /// The base of the rotary position embedding.
        static let ropeTheta = 10000.0
        /// The part of each full-attention head that the rotary embedding turns.
        static let partialRotaryFactor = 0.25
        /// The most positions of the rotary embedding.
        static let maxPositionEmbeddings = 128
        /// The keys of the Qwen3-Next model that the shared fixture does not set.
        static var extraKeys: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "rope_theta": ropeTheta, "partial_rotary_factor": partialRotaryFactor,
                "max_position_embeddings": maxPositionEmbeddings, "norm_topk_prob": true,
                "tie_word_embeddings": true,
            ]
        }
    }

    /// Decodes the configuration of a tiny Qwen3-Next model with four layers. A full-attention
    /// layer follows each linear-attention layer.
    ///
    /// - Parameters:
    ///   - headDim: The head dimension of the full-attention layers.
    ///   - quantizable: True for layer sizes that a group size of 32 can quantize.
    /// - Returns: The configuration.
    /// - Throws: The error of the configuration decoder.
    static func configuration(
        headDim: Int = TinyModel.attentionHeadDim,
        quantizable: Bool = false
    ) throws -> Qwen3NextConfiguration {
        let linearHeadDim =
            quantizable ? TinyModel.quantizableLinearHeadDim : TinyModel.linearHeadDim
        let heads = GDNDecodeKernelFixture.HeadCase(
            batch: 1, keyHeads: TinyModel.linearHeads, valueHeads: TinyModel.linearHeads,
            keyHeadDim: linearHeadDim, valueHeadDim: linearHeadDim)
        return try GDNDecodeKernelFixture.configuration(
            Qwen3NextConfiguration.self, heads, hiddenLayers: TinyModel.hiddenLayers,
            hiddenSize: quantizable ? TinyModel.quantizableHiddenSize : TinyModel.hiddenSize,
            attentionHeadDim: headDim, extraKeys: TinyModel.extraKeys)
    }

    /// A cache factory that keeps the linear-attention caches of the model and gives each
    /// full-attention layer a new cache from `makeAttention`.
    ///
    /// - Parameter makeAttention: The factory of one full-attention layer cache.
    /// - Returns: The factory of the caches of a model.
    private static func attentionCaches(
        _ makeAttention: @escaping () -> KVCache
    ) -> (Qwen3NextModel) throws -> [KVCache] {
        { model in
            try model.newCache(parameters: nil).map { cache in
                cache is MambaCache ? cache : makeAttention()
            }
        }
    }

    private func run(
        _ model: Qwen3NextModel,
        tokens: [Int32],
        compiled: Bool,
        cache makeCache: (Qwen3NextModel) throws -> [KVCache]
    ) throws -> CompiledDecodeRun {
        let cache = try makeCache(model)
        return CompiledDecodeRun.decoding(tokens, cache: cache) { token in
            model.forward(
                MLXArray([token]).reshaped(1, 1),
                cache: cache,
                useCompiledDecode: compiled)
        }
    }

    /// Decodes `tokens` on the general path and on the compiled path, and asserts that both
    /// give the same logits, cache state and cache offsets.
    ///
    /// - Parameters:
    ///   - model: The model to decode with.
    ///   - tokens: The tokens to feed one at a time.
    ///   - makeCache: The factory of the caches of each run.
    ///   - context: The text that the failure messages name, or `nil`.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    /// - Returns: The compiled run.
    /// - Throws: The error of the cache factory.
    @discardableResult
    private func assertCompiledMatchesGeneral(
        _ model: Qwen3NextModel,
        tokens: [Int32],
        cache makeCache: (Qwen3NextModel) throws -> [KVCache] = {
            try $0.newCache(parameters: nil)
        },
        context: String? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> CompiledDecodeRun {
        try CompiledDecodeRun.assertCompiledMatchesEager(context, file: file, line: line) {
            compiled in
            try run(model, tokens: tokens, compiled: compiled, cache: makeCache)
        }
    }

    func testSharedScheduleSplitsAtAttentionWrites() {
        let segments = CompiledDecodeSegment.schedule(
            linearLayers: [true, false, true, true, false])

        XCTAssertEqual(
            segments,
            [
                .init(linearLayers: [0], attentionPreLayer: 1),
                .init(attentionPostLayer: 1, linearLayers: [2, 3], attentionPreLayer: 4),
                .init(attentionPostLayer: 4),
            ])
    }

    func testDecodeConvolutionMatchesGeneralKernelExactly() throws {
        let config = try Self.configuration()
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(29)
            GDNDecodeConvolutionComparison.assertFusedMatchesGeneral(
                Qwen3NextGatedDeltaNet(config), dtype: dtype)
        }
    }

    func testCompiledDecodeMatchesGeneralPathExactly() throws {
        for dtype in [DType.float16, DType.bfloat16] {
            MLXRandom.seed(31)
            let model = Qwen3NextModel(try Self.configuration())
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            let tokens = [Int32(1), 7, 3, 9, 2]

            let compiled = try assertCompiledMatchesGeneral(
                model, tokens: tokens, context: "\(dtype)")

            XCTAssertEqual(compiled.offsets, [5, 5, 5, 5])
            if dtype == .float16 {
                XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
            } else {
                XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
            }
        }
    }

    func testLinearLayerSharedHybridBodyMatchesGeneralLayerExactly() throws {
        MLXRandom.seed(43)
        let config = try Self.configuration()
        let layer = Qwen3NextDecoderLayer(config, layerIdx: 0)
        layer.update(parameters: layer.parameters().mapValues { $0.asType(.float16) })
        let gatedDeltaNet = try XCTUnwrap(layer.linearAttn)
        let zero = gatedDeltaNet.zeroStates(batch: 1, dtype: .float16)
        let x = MLXRandom.normal([1, 1, config.hiddenSize]).asType(.float16)
        let convState = MLXRandom.normal(zero.conv.shape).asType(.float16)
        let recState = MLXRandom.normal(zero.rec.shape)
        let cache = MambaCache()
        cache[0] = convState
        cache[1] = recState

        let general = layer(x, attentionMask: .none, ssmMask: nil, cache: cache)
        let step = layer.linearLayerBody(x: x, convState: convState, recState: recState)

        XCTAssertEqual(ArraySnapshot(step.output), ArraySnapshot(general))
        XCTAssertEqual(ArraySnapshot(step.convState), ArraySnapshot(try XCTUnwrap(cache[0])))
        XCTAssertEqual(ArraySnapshot(step.recurrentState), ArraySnapshot(try XCTUnwrap(cache[1])))
        XCTAssertNil(step.snapshots)
    }

    /// A quantized cache keeps the segments: its attention step writes the same packed keys
    /// and values as the general path, and the step gives the same logits.
    func testQuantizedCacheMatchesGeneralPathExactlyThroughSegments() throws {
        MLXRandom.seed(37)
        let model = Qwen3NextModel(try Self.configuration(headDim: 32))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        let tokens = [Int32(2), 4, 6]
        let quantizedCache = Self.attentionCaches { QuantizedKVCache(groupSize: 32, bits: 8) }

        try assertCompiledMatchesGeneral(model, tokens: tokens, cache: quantizedCache)

        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
    }

    func testTurboQuantCacheKeepsExactGeneralFallback() throws {
        MLXRandom.seed(41)
        let model = Qwen3NextModel(try Self.configuration(headDim: 32))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        let tokens = [Int32(2), 4, 6]
        let turboQuantCache = Self.attentionCaches { TurboQuantKVCache(bits: 4, seed: 67) }

        try assertCompiledMatchesGeneral(model, tokens: tokens, cache: turboQuantCache)

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testQuantizedWeightsCompileAndMatchExactly() throws {
        MLXRandom.seed(39)
        let model = Qwen3NextModel(
            try Self.configuration(headDim: 32, quantizable: true))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        quantize(model: model, groupSize: 32, bits: 4)
        let tokens = [Int32(3), 8, 1, 5]

        try assertCompiledMatchesGeneral(model, tokens: tokens)

        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
    }

    func testModelDeallocatesAfterCompiledDecode() throws {
        MLXRandom.seed(41)
        var model: Qwen3NextModel? = Qwen3NextModel(try Self.configuration())
        model!.update(parameters: model!.parameters().mapValues { $0.asType(.float16) })
        var cache: [KVCache]? = try model!.newCache(parameters: nil)

        for token in [Int32(1), 2, 3] {
            let output = model!(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(output)
        }

        weak let inner = model!.model
        weak let linearAttention = model!.modules()
            .compactMap { $0 as? Qwen3NextGatedDeltaNet }.first
        XCTAssertNotNil(inner)
        XCTAssertNotNil(linearAttention)
        XCTAssertGreaterThan(model!.model.compiledDecodeSegmentCount, 0)

        model = nil
        cache = nil

        XCTAssertNil(inner, "compiled segments retained the model")
        XCTAssertNil(linearAttention, "compiled segments retained their linear-attention layer")
    }

}

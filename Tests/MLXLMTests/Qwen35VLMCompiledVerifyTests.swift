// Copyright © 2026 Apple Inc.
//
// The router path loads Qwen3.8 as the Qwen 3.5 vision-language model. A
// speculative verify of the bonus token and its drafts there runs through the
// compiled decode schedule, one schedule for each token count, as it does in
// the text model. M-RoPE stays outside the traces, with the KV write and the
// attention over the cache, and takes the position ids of the call.
//
// The compiled verify stays inside the tolerance of the general path of the
// layers for each token count and each rollback, at positions an image moved,
// and every step the schedule does not read takes the general path.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Qwen35VLMCompiledVerifyTests {

    /// The decoder layers of the test model: GDN, attention, GDN, attention. The schedule then
    /// holds each kind of segment: one that opens the step, one between two attention layers,
    /// and one that closes it.
    private static let hiddenLayers = 4

    /// The dtype of the test model: one the traces hold exactly.
    private static let dtype = Qwen35CompiledVerifyTests.dtype

    /// The dtype of a model the traces do not hold exactly.
    private static let refusedDtype = DType.float32

    /// The tolerance of the compiled verify against the general path.
    private static let tolerance = AttentionKernelTolerance.bfloat16

    /// The seed of the weights of the test model.
    private static let weightSeed: UInt64 = 20_261_101

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_102

    /// The seed of the verify tokens.
    private static let verifySeed: UInt64 = 20_261_103

    /// The prompt tokens the cache holds before the verify.
    private static let promptLength = 8

    /// The positions an image moved each token after it: nonzero, so a verify that took the
    /// positions after the cache offset would not match.
    private static let positionDelta: Int32 = 3

    /// The vocabulary of the test model.
    private static let vocabularySize = Qwen35MTPFixture.vocabularySize

    /// The factor that gives the test model other weights.
    private static let weightUpdateFactor: Float = 1.5

    /// The quantization of a fused projection.
    private static let projectionGroupSize = 32

    /// The bits of a fused projection.
    private static let projectionBits = 4

    /// The test model, whose GDN layers the kernel runs, on seeded weights.
    private static func kernelModel(dtype: DType = Self.dtype) throws -> Qwen35Language.Model {
        let configuration = try GDNDecodeKernelFixture.configuration(
            Qwen35Configuration.TextConfiguration.self, GDNDecodeKernelFixture.smallHeadCase,
            hiddenLayers: hiddenLayers)
        let model = withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            Qwen35Language.Model(configuration)
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        model.train(false)
        eval(model)
        return model
    }

    /// Whether decoder layer `index` is a GDN layer: each second layer is an attention layer.
    private static func isLinearLayer(_ index: Int) -> Bool {
        index.isMultiple(of: 2)
    }

    /// The caches of the test model, with one plain key cache for each attention layer.
    private static func caches() -> [KVCache] {
        (0 ..< hiddenLayers).map { isLinearLayer($0) ? MambaCache() : KVCacheSimple() }
    }

    /// The M-RoPE positions of `length` tokens after `offset` tokens, moved by
    /// ``positionDelta``.
    private static func positions(offset: Int, length: Int) -> MLXArray {
        qwen35MTPPositionIds(
            offset: offset, length: length, batchSize: 1,
            positionDeltas: MLXArray([positionDelta]))
    }

    /// `count` seeded tokens, shape `(1, count)`.
    private static func tokens(_ count: Int, seed: UInt64) -> MLXArray {
        seededTokens(count, vocabularySize: vocabularySize, seed: seed)[.newAxis]
    }

    /// The caches of `model` after the prompt, on the general path.
    private static func prefilled(_ model: Qwen35Language.Model) -> [KVCache] {
        let cache = caches()
        eval(
            model.forward(
                tokens(promptLength, seed: promptSeed), cache: cache,
                positionIds: positions(offset: 0, length: promptLength), useCompiledDecode: false))
        return cache
    }

    /// The hidden rows of one verify forward of `tokenCount` tokens after the prompt.
    private static func forward(
        _ model: Qwen35Language.Model, tokenCount: Int, cache: [KVCache], compiled: Bool,
        applyFinalNorm: Bool = false, keepsSnapshots: Bool = true
    ) -> MLXArray {
        MultiRowQuantizedMatvec.run(asSpeculativeVerify: true) {
            model.forward(
                tokens(tokenCount, seed: verifySeed), cache: cache,
                positionIds: positions(offset: promptLength, length: tokenCount),
                applyFinalNorm: applyFinalNorm, keepsSnapshots: keepsSnapshots,
                useCompiledDecode: compiled)
        }
    }

    /// Prefills the prompt, runs one verify of `tokenCount` tokens on the compiled or the
    /// general path, then rolls the verify back by `rollback` tokens as a rejected round does.
    private func verify(
        _ model: Qwen35Language.Model, tokenCount: Int, rollingBack rollback: Int, compiled: Bool
    ) -> CompiledVerifyOutcome {
        let cache = Self.prefilled(model)
        let hidden = model.norm(
            Self.forward(model, tokenCount: tokenCount, cache: cache, compiled: compiled))
        return CompiledVerifyOutcome(
            logits: model.embedTokens.asLinear(hidden), hidden: hidden, cache: cache,
            rollingBack: rollback)
    }

    /// Replaces the input projections of `gdn` with 4-bit ones, which fuse.
    private func quantizeProjections(of gdn: Qwen35Language.GatedDeltaNet) throws {
        let quantized = { (linear: Linear) -> Linear in
            QuantizedLinear(
                linear, groupSize: Self.projectionGroupSize, bits: Self.projectionBits)
        }
        try gdn.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(quantized(gdn.inProjQKV)),
                "in_proj_z": .value(quantized(gdn.inProjZ)),
                "in_proj_b": .value(quantized(gdn.inProjB)),
                "in_proj_a": .value(quantized(gdn.inProjA)),
            ]), verify: [])
    }

    // MARK: - The compiled verify

    @Test(
        "a compiled verify stays within tolerance of the general path, for each rollback",
        arguments: Qwen35CompiledVerifyTests.verifyTokenCounts)
    func compiledVerifyMatchesGeneralPath(tokenCount: Int) throws {
        let model = try Self.kernelModel()

        for rollback in 0 ..< tokenCount {
            let general = verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: false)
            let compiled = verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: true)
            try compiled.expectMatches(
                general, tolerance: Self.tolerance, tokenCount: tokenCount, rollback: rollback)
        }
        #expect(model.compiledVerifySegmentCount > 0)
    }

    @Test("a compiled verify applies the final norm a caller asks for")
    func compiledVerifyAppliesTheFinalNorm() throws {
        let model = try Self.kernelModel()
        let tokenCount = CompiledVerifyShape.fewestTokens

        let normed = Self.forward(
            model, tokenCount: tokenCount, cache: Self.prefilled(model), compiled: true,
            applyFinalNorm: true)
        let bare = Self.forward(
            model, tokenCount: tokenCount, cache: Self.prefilled(model), compiled: true)

        let gap = AttentionKernelTolerance.worstGap(normed, model.norm(bare))
        #expect(gap <= Self.tolerance, "worst gap \(gap)")
        #expect(model.compiledVerifySegmentCount > 0)
    }

    @Test("a compiled verify reads the current weights")
    func compiledVerifyReadsUpdatedWeights() throws {
        let model = try Self.kernelModel()
        let tokenCount = CompiledVerifyShape.fewestTokens
        let before = verify(model, tokenCount: tokenCount, rollingBack: 0, compiled: true)

        model.update(
            parameters: model.parameters().mapValues { $0 * Self.weightUpdateFactor })
        eval(model)
        let compiled = verify(model, tokenCount: tokenCount, rollingBack: 0, compiled: true)
        let general = verify(model, tokenCount: tokenCount, rollingBack: 0, compiled: false)

        #expect(AttentionKernelTolerance.worstGap(compiled.logits, before.logits) > Self.tolerance)
        let gap = AttentionKernelTolerance.worstGap(compiled.logits, general.logits)
        #expect(gap <= Self.tolerance, "worst gap \(gap)")
    }

    @Test("greedy MTP decode of a bfloat16 VLM target equals plain decode")
    func visionLanguageTargetMTPEqualsPlainDecode() throws {
        let fixture = try Qwen35VLMMTPFixture(dtype: Self.dtype)
        fixture.target.train(false)

        try Qwen35CompiledVerifyTests.expectGreedyMTPEqualsPlainDecode(on: fixture)

        #expect(fixture.target.languageModel.model.compiledVerifySegmentCount > 0)
    }

    // MARK: - The traces

    @Test("invalidating the compiled traces of the model drops its verify segments")
    func invalidationDropsVerifySegments() throws {
        let model = try Self.kernelModel()
        _ = verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: true)
        #expect(model.compiledVerifySegmentCount > 0)

        model.invalidateCompiledTraces()

        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("replacing a fused projection drops the verify segments that read it")
    func fusedProjectionUpdateDropsVerifySegments() throws {
        let model = try Self.kernelModel()
        let gdn = try #require(
            model.modules().compactMap { $0 as? Qwen35Language.GatedDeltaNet }.first)
        try quantizeProjections(of: gdn)
        #expect(try gdn.prepareFusedInputProjection())
        _ = verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: true)
        #expect(model.compiledVerifySegmentCount > 0)

        try gdn.update(
            parameters: ModuleParameters.unflattened(["in_proj_qkv.weight": gdn.inProjQKV.weight]),
            verify: [])

        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("the compiled verify traces do not retain the model")
    func modelDeallocatesAfterACompiledVerify() throws {
        var model: Qwen35Language.Model? = try Self.kernelModel()
        _ = verify(
            try #require(model), tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0,
            compiled: true)
        weak let released = model

        model = nil

        #expect(released == nil)
    }

    // MARK: - The steps of the general path

    @Test("a verify on the general path compiles no verify segment")
    func generalPathCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        _ = verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: false)
        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("a forward of two tokens that keeps no snapshots is not a verify")
    func forwardWithoutSnapshotsCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        eval(
            Self.forward(
                model, tokenCount: CompiledVerifyShape.fewestTokens, cache: Self.prefilled(model),
                compiled: true, keepsSnapshots: false))
        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test(
        "a verify outside the token range of a compiled schedule takes the general path",
        arguments: [
            GatedDeltaDecodeKernel.decodeTokens, GatedDeltaDecodeKernel.mostTokens + 1,
        ])
    func verifyOutsideTheTokenRangeCompilesNoVerifySegment(tokenCount: Int) throws {
        let model = try Self.kernelModel()
        eval(
            Self.forward(
                model, tokenCount: tokenCount, cache: Self.prefilled(model), compiled: true))
        #expect(model.compiledVerifySegmentCount == 0)
        #expect(model.compiledDecodeSegmentCount == 0)
    }

    @Test("a verify in a dtype the traces do not hold exactly takes the general path")
    func refusedDtypeCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel(dtype: Self.refusedDtype)
        eval(
            Self.forward(
                model, tokenCount: CompiledVerifyShape.fewestTokens, cache: Self.prefilled(model),
                compiled: true))
        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("a verify in training takes the general path")
    func trainingCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        model.train(true)
        eval(
            Self.forward(
                model, tokenCount: CompiledVerifyShape.fewestTokens, cache: Self.prefilled(model),
                compiled: true))
        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("a verify over a cache that owns its attention takes the general path")
    func quantizedCacheCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        let prefilled = Self.prefilled(model)
        let cache = prefilled.indices.map { index -> KVCache in
            Self.isLinearLayer(index)
                ? prefilled[index]
                : QuantizedKVCache(groupSize: Self.projectionGroupSize, bits: Self.projectionBits)
        }
        eval(
            Self.forward(
                model, tokenCount: CompiledVerifyShape.fewestTokens, cache: cache, compiled: true))
        #expect(model.compiledVerifySegmentCount == 0)
    }

    @Test("a verify without GDN states takes the general path")
    func emptyGDNCacheCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        eval(
            Self.forward(
                model, tokenCount: CompiledVerifyShape.fewestTokens, cache: Self.caches(),
                compiled: true))
        #expect(model.compiledVerifySegmentCount == 0)
    }
}

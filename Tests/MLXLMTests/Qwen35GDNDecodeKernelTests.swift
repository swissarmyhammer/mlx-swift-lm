// Copyright © 2026 Apple Inc.
//
// The contract of ``GatedDeltaDecodeKernel`` in the Qwen 3.5 gated delta net.
//
// On one decode token the layer body runs in one Metal kernel. The kernel
// output must stay inside a small tolerance of the op path
// (`recurrentBody`) over the same inputs, and the conv state after the token
// must be the same bits. A positive control shows that the tolerance
// catches a kernel that does not read its recurrent state or its conv state.
// `GDNDecodeKernelLayerTests` holds the same contract for the whole layer of
// each model kind.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Qwen35GDNDecodeKernelTests {

    /// The shared operands of the GDN decode kernel tests.
    typealias Fixture = GDNDecodeKernelFixture

    /// A key head width that the 32 lanes of a SIMD group cannot share.
    private static let narrowKeyHeadDim = 16

    /// The token counts of a speculative verify that the kernel runs.
    static let verifyTokenCounts = Array(2 ... GatedDeltaDecodeKernel.mostTokens)

    /// The operands of one decode step, or of the tokens of one verify.
    struct Operands {
        let layer: Qwen35GatedDeltaNet
        let qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        let convState: MLXArray
        let recurrentState: MLXArray

        /// The operands of some of the tokens, from the given states.
        ///
        /// - Parameters:
        ///   - range: The tokens to keep.
        ///   - convState: The conv state before the first kept token.
        ///   - recurrentState: The recurrent state before the first kept token.
        /// - Returns: The operands of the kept tokens.
        func tokens(
            _ range: Range<Int>, convState: MLXArray, recurrentState: MLXArray
        ) -> Operands {
            Operands(
                layer: layer, qkv: qkv[0..., range], z: z[0..., range], b: b[0..., range],
                a: a[0..., range], convState: convState, recurrentState: recurrentState)
        }

        /// The operands of the first `count` tokens, from the same states.
        ///
        /// - Parameter count: The number of tokens to keep.
        /// - Returns: The operands of the first tokens.
        func prefix(_ count: Int) -> Operands {
            tokens(0 ..< count, convState: convState, recurrentState: recurrentState)
        }
    }

    /// Repeatable random operands for one case in one dtype. The token
    /// inputs are slices of one projected row for each token, as the fused
    /// input projection gives them.
    static func operands(
        _ headCase: Fixture.HeadCase, dtype: DType, tokens: Int = 1
    ) throws -> Operands {
        let layer = try #require(
            Fixture.layer(Fixture.mlxLLMQwen35, headCase, dtype: dtype) as? Qwen35GatedDeltaNet)
        let valueDim = layer.valueDim
        let qkvEnd = layer.convDim
        let zEnd = qkvEnd + valueDim
        let bEnd = zEnd + layer.numVHeads
        let projected = MLXRandom.normal(
            [headCase.batch, tokens, bEnd + layer.numVHeads],
            key: MLXRandom.key(Fixture.Seed.projected)
        ).asType(dtype)
        let states = Fixture.states(of: layer, batch: headCase.batch, dtype: dtype)
        eval(projected)
        return Operands(
            layer: layer,
            qkv: projected[0..., 0..., ..<qkvEnd],
            z: projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                headCase.batch, tokens, layer.numVHeads, layer.headVDim),
            b: projected[0..., 0..., zEnd ..< bEnd],
            a: projected[0..., 0..., bEnd...],
            convState: states.conv, recurrentState: states.recurrent)
    }

    /// The op path over the operands, with a recurrent state and a conv
    /// state of choice.
    static func opPath(
        _ operands: Operands, convState: MLXArray, recurrentState: MLXArray
    ) -> (output: MLXArray, convState: MLXArray, recurrentState: MLXArray) {
        let body = operands.layer.recurrentBody(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
            convState: convState, recState: recurrentState, mask: nil)
        return (body.output, body.convState, body.recurrentState)
    }

    /// The kernel step over the operands, or nil when the kernel refuses
    /// them.
    static func kernelStep(_ operands: Operands) -> GatedDeltaDecodeKernel.Step? {
        operands.layer.decodeKernelStep(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
            convState: operands.convState, recState: operands.recurrentState)
    }

    /// The kernel run of the tokens of a speculative verify, with its
    /// snapshot slots, or nil when the kernel refuses them.
    static func speculativeStep(_ operands: Operands) -> GatedDeltaDecodeKernel.Step? {
        operands.layer.speculativeKernelStep(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
            convState: operands.convState, recState: operands.recurrentState)
    }

    /// The kernel run of the tokens of the operands without snapshot slots:
    /// the run of the accepted tokens only.
    static func plainKernelRun(_ operands: Operands) throws -> GatedDeltaDecodeKernel.Step {
        let inputs = GatedDeltaDecodeKernel.Inputs(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a)
        let weights = operands.layer.decodeKernelWeights
        try #require(
            GatedDeltaDecodeKernel.canRun(
                inputs: inputs, convState: operands.convState,
                recurrentState: operands.recurrentState, weights: weights))
        return GatedDeltaDecodeKernel.step(
            inputs: inputs, convState: operands.convState,
            recurrentState: operands.recurrentState, weights: weights)
    }

    /// Tells if two arrays hold the same bits.
    static func sameBits(_ got: MLXArray, _ expected: MLXArray) -> Bool {
        got.dtype == expected.dtype && arrayEqual(got, expected).item(Bool.self)
    }

    @Test(arguments: Fixture.headCases, Fixture.dtypes)
    func theKernelStaysInsideTheToleranceOfTheOpPath(
        headCase: Fixture.HeadCase, dtype: DType
    ) throws {
        let operands = try Self.operands(headCase, dtype: dtype)
        let step = try #require(Self.kernelStep(operands))
        let reference = Self.opPath(
            operands, convState: operands.convState, recurrentState: operands.recurrentState)

        #expect(step.output.shape == reference.output.shape)
        #expect(step.output.dtype == dtype)
        #expect(abs(reference.output).max().item(Float.self) > 0, "every reference value was zero")
        let outputGap = Fixture.worstGap(step.output, reference.output)
        let stateGap = Fixture.worstGap(step.recurrentState, reference.recurrentState)
        #expect(outputGap <= Fixture.tolerance(dtype), "worst output gap \(outputGap)")
        #expect(stateGap <= Fixture.tolerance(dtype), "worst state gap \(stateGap)")
        #expect(step.convState.dtype == reference.convState.dtype)
        #expect(arrayEqual(step.convState, reference.convState).item(Bool.self))
    }

    /// The positive control of the tolerance test: a path that did not read
    /// the recurrent state, or the conv state, breaks the tolerance.
    @Test(arguments: Fixture.headCases)
    func aPathThatSkipsAStateMovesTheOutputPastTheTolerance(
        headCase: Fixture.HeadCase
    ) throws {
        let dtype = DType.bfloat16
        let operands = try Self.operands(headCase, dtype: dtype)
        let full = Self.opPath(
            operands, convState: operands.convState, recurrentState: operands.recurrentState)
        let noRecurrentState = Self.opPath(
            operands, convState: operands.convState,
            recurrentState: zeros(like: operands.recurrentState))
        let noConvState = Self.opPath(
            operands, convState: zeros(like: operands.convState),
            recurrentState: operands.recurrentState)

        #expect(Fixture.worstGap(noRecurrentState.output, full.output) > Fixture.tolerance(dtype))
        #expect(Fixture.worstGap(noConvState.output, full.output) > Fixture.tolerance(dtype))
    }

    /// The kernel reads one token in float16 or bfloat16, over a float32
    /// recurrent state, and only key heads that the lanes share evenly.
    @Test func theKernelReadsOnlyTheStepsItHolds() throws {
        let headCase = Fixture.smallHeadCase
        let operands = try Self.operands(headCase, dtype: .bfloat16)
        let layer = operands.layer
        let twoTokens = concatenated([operands.qkv, operands.qkv], axis: 1)
        let float32Layer = try Self.operands(headCase, dtype: .float32)
        let narrowKeys = try Self.operands(
            Fixture.HeadCase(
                batch: headCase.batch, keyHeads: headCase.keyHeads,
                valueHeads: headCase.valueHeads, keyHeadDim: Self.narrowKeyHeadDim,
                valueHeadDim: headCase.valueHeadDim),
            dtype: .bfloat16)

        #expect(Self.kernelStep(operands) != nil)
        #expect(Self.kernelStep(narrowKeys) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: twoTokens, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState, recState: operands.recurrentState) == nil)
        #expect(Self.kernelStep(float32Layer) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState,
                recState: operands.recurrentState.asType(.bfloat16)) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState, recState: nil) == nil)
    }

    /// A layer in training keeps the op path: the kernel has no gradient.
    @Test func aTrainingLayerKeepsTheOpPath() throws {
        let operands = try Self.operands(Fixture.smallHeadCase, dtype: .bfloat16)
        operands.layer.train(true)

        #expect(Self.kernelStep(operands) == nil)
    }

    // MARK: - The tokens of a speculative verify

    /// Each snapshot slot of a verify, its conv state, and the output and the
    /// states after the last token stay inside the tolerance of the op path
    /// over the same tokens.
    @Test(arguments: Fixture.headCases, Fixture.dtypes)
    func everySnapshotSlotStaysInsideTheToleranceOfTheOpPath(
        headCase: Fixture.HeadCase, dtype: DType
    ) throws {
        for tokens in Self.verifyTokenCounts {
            let operands = try Self.operands(headCase, dtype: dtype, tokens: tokens)
            let run = try #require(Self.speculativeStep(operands))
            let snapshots = try #require(run.snapshots)
            #expect(snapshots.tokenCount == tokens)

            for token in 1 ... tokens {
                let reference = Self.opPath(
                    operands.prefix(token), convState: operands.convState,
                    recurrentState: operands.recurrentState)
                let slotGap = Fixture.worstGap(
                    snapshots.recurrentState(afterToken: token), reference.recurrentState)
                #expect(
                    slotGap <= Fixture.tolerance(dtype),
                    "slot after token \(token) of \(tokens): worst gap \(slotGap)")
                #expect(
                    Self.sameBits(snapshots.convState(afterToken: token), reference.convState),
                    "conv state after token \(token) of \(tokens)")
            }

            let reference = Self.opPath(
                operands, convState: operands.convState, recurrentState: operands.recurrentState)
            #expect(run.output.shape == reference.output.shape)
            let outputGap = Fixture.worstGap(run.output, reference.output)
            let stateGap = Fixture.worstGap(run.recurrentState, reference.recurrentState)
            #expect(
                outputGap <= Fixture.tolerance(dtype), "\(tokens) tokens: output gap \(outputGap)")
            #expect(stateGap <= Fixture.tolerance(dtype), "\(tokens) tokens: state gap \(stateGap)")
            #expect(
                Self.sameBits(run.convState, reference.convState), "\(tokens) tokens: conv state")
        }
    }

    /// A verify of several tokens computes each token as one decode step of
    /// the kernel computes it: the outputs, the slots and the conv states are
    /// the bits of one kernel step for each token in turn.
    @Test(arguments: Fixture.headCases, Fixture.dtypes)
    func aVerifyHoldsTheBitsOfOneDecodeStepForEachToken(
        headCase: Fixture.HeadCase, dtype: DType
    ) throws {
        let tokens = GatedDeltaDecodeKernel.mostTokens
        let operands = try Self.operands(headCase, dtype: dtype, tokens: tokens)
        let run = try #require(Self.speculativeStep(operands))
        let snapshots = try #require(run.snapshots)

        var convState = operands.convState
        var recurrentState = operands.recurrentState
        for token in 0 ..< tokens {
            let single = try #require(
                Self.kernelStep(
                    operands.tokens(
                        token ..< token + 1, convState: convState, recurrentState: recurrentState)))
            #expect(Self.sameBits(run.output[0..., token ..< token + 1], single.output))
            #expect(
                Self.sameBits(
                    snapshots.recurrentState(afterToken: token + 1), single.recurrentState))
            #expect(Self.sameBits(snapshots.convState(afterToken: token + 1), single.convState))
            convState = single.convState
            recurrentState = single.recurrentState
        }
        #expect(Self.sameBits(run.recurrentState, recurrentState))
        #expect(Self.sameBits(run.convState, convState))
    }

    /// A rollback of `r` tokens selects the slot of the `T - r` accepted
    /// tokens: the cache then holds the states of a run of those tokens only.
    @Test(arguments: Fixture.dtypes)
    func aRollbackGivesTheStatesOfARunOfTheAcceptedTokensOnly(dtype: DType) throws {
        for tokens in Self.verifyTokenCounts {
            let operands = try Self.operands(Fixture.smallHeadCase, dtype: dtype, tokens: tokens)
            for rollback in 1 ..< tokens {
                let run = try #require(Self.speculativeStep(operands))
                let cache = MambaCache()
                cache[0] = run.convState
                cache[1] = run.recurrentState
                cache.saveSpeculativeSnapshots(try #require(run.snapshots))
                cache.advance(tokens)

                #expect(cache.selectSpeculativeSlot(rollingBack: rollback))

                let accepted = try Self.plainKernelRun(operands.prefix(tokens - rollback))
                let kept = tokens - rollback
                #expect(cache.offset == kept)
                // A cache that holds only the bonus token of the pass has no slot left to
                // roll back into, and releases the slots.
                #expect(cache.speculativeSlotCount == SnapshotSlotFixture.slotCount(keeping: kept))
                #expect(Self.sameBits(try #require(cache[0]), accepted.convState))
                #expect(Self.sameBits(try #require(cache[1]), accepted.recurrentState))
            }
        }
    }

    /// The kernel runs the tokens of a verify with slots only for two to
    /// ``GatedDeltaDecodeKernel/mostTokens`` tokens, and never in training.
    @Test func theKernelRunsAVerifyOnlyForTheTokenCountsItHolds() throws {
        let headCase = Fixture.smallHeadCase
        let oneToken = try Self.operands(headCase, dtype: .bfloat16)
        let tooMany = try Self.operands(
            headCase, dtype: .bfloat16, tokens: GatedDeltaDecodeKernel.mostTokens + 1)
        let verify = try Self.operands(headCase, dtype: .bfloat16, tokens: 2)

        #expect(Self.speculativeStep(oneToken) == nil)
        #expect(Self.speculativeStep(tooMany) == nil)
        #expect(Self.speculativeStep(verify) != nil)
        verify.layer.train(true)
        #expect(Self.speculativeStep(verify) == nil)
    }

    // MARK: - A verify through the layer

    /// A verify pass through the layer of each model keeps one snapshot slot
    /// for each token in the cache, and a rollback to the first token selects
    /// the slot after it.
    @Test(arguments: [Fixture.mlxLLMQwen35, Fixture.mlxVLMQwen35])
    func aVerifyThroughTheLayerKeepsOneSlotForEachToken(kind: Fixture.LayerKind) throws {
        let headCase = Fixture.smallHeadCase
        let dtype = DType.bfloat16
        let tokens = GatedDeltaDecodeKernel.mostTokens
        let acceptedTokens = 1
        let layer = try #require(
            Fixture.layer(kind, headCase, dtype: dtype) as? any SpeculativeGDNTestLayer)
        let inputs = MLXRandom.normal(
            [headCase.batch, tokens, Fixture.hiddenSize], key: MLXRandom.key(Fixture.Seed.token)
        ).asType(dtype)
        let states = Fixture.states(of: layer, batch: headCase.batch, dtype: dtype)
        let verifyCache = MambaCache.holding(states)
        let plainCache = MambaCache.holding(states)
        let firstTokenCache = MambaCache.holding(states)

        eval(layer(inputs, mask: nil, cache: verifyCache, keepsSnapshots: true))
        eval(layer(inputs, mask: nil, cache: plainCache, keepsSnapshots: false))
        eval(
            layer(
                inputs[0..., ..<acceptedTokens], mask: nil, cache: firstTokenCache,
                keepsSnapshots: false))

        #expect(verifyCache.speculativeSlotCount == tokens)
        #expect(plainCache.speculativeSlotCount == 0)
        #expect(verifyCache.offset == tokens)
        let verifiedConv = try #require(verifyCache[0])
        let verifiedState = try #require(verifyCache[1])
        let plainState = try #require(plainCache[1])
        #expect(Self.sameBits(verifiedConv, try #require(plainCache[0])))
        let stateGap = Fixture.worstGap(verifiedState, plainState)
        #expect(stateGap <= Fixture.tolerance(dtype), "worst state gap \(stateGap)")

        #expect(verifyCache.selectSpeculativeSlot(rollingBack: tokens - acceptedTokens))
        #expect(verifyCache.offset == acceptedTokens)
        #expect(
            verifyCache.speculativeSlotCount
                == SnapshotSlotFixture.slotCount(keeping: acceptedTokens))
        for (restored, expected) in zip(verifyCache.state, firstTokenCache.state) {
            let gap = Fixture.worstGap(restored, expected)
            #expect(gap <= Fixture.tolerance(dtype), "worst restored state gap \(gap)")
        }
    }
}

/// A gated delta net layer that a speculative verify runs through a cache.
protocol SpeculativeGDNTestLayer: GDNDecodeKernelTestLayer {
    /// The layer over the tokens of `inputs`, through `cache`.
    ///
    /// - Parameters:
    ///   - inputs: The hidden states, shape `(batch, sequence, hiddenSize)`.
    ///   - mask: The token mask, or nil.
    ///   - cache: The cache of the layer.
    ///   - keepsSnapshots: True when the pass is a verify that keeps the
    ///     snapshot slot after each token.
    /// - Returns: The output of the layer.
    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray?, cache: MambaCache?, keepsSnapshots: Bool
    ) -> MLXArray
}

extension Qwen35GatedDeltaNet: SpeculativeGDNTestLayer {}

extension Qwen35Language.GatedDeltaNet: SpeculativeGDNTestLayer {}

extension MambaCache {
    /// A cache that holds the given conv and recurrent states.
    ///
    /// - Parameter states: The conv state and the recurrent state.
    /// - Returns: The cache.
    fileprivate static func holding(_ states: (conv: MLXArray, recurrent: MLXArray)) -> MambaCache {
        let cache = MambaCache()
        cache[0] = states.conv
        cache[1] = states.recurrent
        return cache
    }
}

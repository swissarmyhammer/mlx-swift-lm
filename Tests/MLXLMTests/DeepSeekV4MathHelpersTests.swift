// Copyright © 2026 Apple Inc.
//
// Numeric parity tests for `DeepSeekV4Math`.
//
// Every expected number below comes from the DeepSeek-V4 Python reference,
// `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
// `mlx_lm/models/deepseek_v4.py`. Each routine of that file was transcribed
// into NumPy one line at a time -- `mx.*` became `np.*` and nothing else
// changed -- and run in float64. The comment above each fixture names the
// Python function and the line range it comes from.
//
// A fixture that a Swift function produced itself proves nothing, thus no
// number below was read out of this repository.
//
// Tolerances. MLX evaluates in float32 and the fixtures are float64. The gap
// between the two was measured in NumPy on these exact inputs:
//
//   - Sinkhorn: 8.5e-8 absolute, thus the tests allow 1e-6 absolute.
//   - YaRN inverse frequency: 1.7e-7 relative, thus the tests allow 1e-5
//     relative.
//   - The remaining functions are one or two operations deep and stay inside
//     1e-6 relative, thus the tests allow 1e-5 relative.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4MathHelpersTests {

    // MARK: - Comparison helpers

    /// The largest absolute gap allowed against a Sinkhorn fixture.
    private static let sinkhornTolerance: Float = 1e-6

    /// The largest relative gap allowed against every other fixture.
    private static let relativeTolerance: Float = 1e-5

    /// Reads an array back as a flat list of `Float`, in row-major order.
    private func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// Checks each element against `expected` with an absolute limit.
    private func expectClose(
        _ got: [Float], _ expected: [Float], within tolerance: Float,
        _ what: String, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.count == expected.count, "\(what): length", sourceLocation: sourceLocation)
        for (index, pair) in zip(got, expected).enumerated() {
            let gap = abs(pair.0 - pair.1)
            #expect(
                gap <= tolerance,
                "\(what)[\(index)]: got \(pair.0), expected \(pair.1), gap \(gap)",
                sourceLocation: sourceLocation)
        }
    }

    /// Checks each element against `expected` with a relative limit. A zero
    /// expected value falls back to the limit read as an absolute gap.
    private func expectCloseRelative(
        _ got: [Float], _ expected: [Float], within tolerance: Float,
        _ what: String, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.count == expected.count, "\(what): length", sourceLocation: sourceLocation)
        for (index, pair) in zip(got, expected).enumerated() {
            let limit = pair.1 == 0 ? tolerance : tolerance * abs(pair.1)
            let gap = abs(pair.0 - pair.1)
            #expect(
                gap <= limit,
                "\(what)[\(index)]: got \(pair.0), expected \(pair.1), gap \(gap)",
                sourceLocation: sourceLocation)
        }
    }

    // MARK: - hcSplitSinkhorn

    /// The mHC width the 3x3 fixture uses.
    private static let fixtureHCMult = 3

    /// The DeepSeek-V4-Flash `hc_sinkhorn_iters`.
    private static let sinkhornIterations = 20

    /// The DeepSeek-V4-Flash `hc_eps`.
    private static let sinkhornEps: Float = 1e-6

    /// One row of mixes, `(2 + 3) * 3 = 15` wide.
    private static let sinkhornMixes: [Float] = [
        0.9, -0.4, 0.2,
        0.5, 1.3, -0.7,
        2.0, -1.0, 0.0,
        -0.5, 1.5, 0.5,
        0.25, 0.75, -1.25,
    ]

    /// The three learned scales: pre, post, comb.
    private static let sinkhornScale: [Float] = [0.5, 1.5, 2.0]

    /// The learned bias, `(2 + 3) * 3 = 15` wide.
    private static let sinkhornBase: [Float] = [
        0.1, -0.2, 0.3,
        0.4, -0.5, 0.6,
        0.0, 0.7, -0.7,
        1.0, 0.0, -1.0,
        -0.3, 0.3, 0.9,
    ]

    /// `hc_split_sinkhorn` lines 171-214 with the inputs above, `hc_mult = 3`,
    /// `sinkhorn_iters = 20`, `eps = 1e-6`.
    ///
    /// The matrix is asymmetric -- `max|M - transpose(M)|` is 0.0233 -- thus a
    /// transposed row and column axis gives a different matrix and this
    /// comparison catches it.
    private static let expectedComb: [Float] = [
        0.8537448969, 0.003746895414, 0.1424968511,
        0.02702352937, 0.4771257145, 0.4958556903,
        0.1192305738, 0.5191263901, 0.3616464585,
    ]

    /// `pre` of the same run.
    private static let expectedPre: [Float] = [0.634136591, 0.4013133399, 0.5986886601]

    /// `post` of the same run.
    private static let expectedPost: [Float] = [1.519021834, 1.619996868, 0.7787215321]

    @Test func hcSplitSinkhornMatchesPythonReference() {
        let hcMult = Self.fixtureHCMult
        let mixes = MLXArray(Self.sinkhornMixes, [1, Self.sinkhornMixes.count])
        let scale = MLXArray(Self.sinkhornScale)
        let base = MLXArray(Self.sinkhornBase)

        let (pre, post, comb) = DeepSeekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base, hcMult: hcMult,
            iters: Self.sinkhornIterations, eps: Self.sinkhornEps)
        eval(pre, post, comb)

        #expect(pre.shape == [1, hcMult])
        #expect(post.shape == [1, hcMult])
        #expect(comb.shape == [1, hcMult, hcMult])

        expectClose(floats(pre), Self.expectedPre, within: Self.sinkhornTolerance, "pre")
        expectClose(floats(post), Self.expectedPost, within: Self.sinkhornTolerance, "post")
        expectClose(floats(comb), Self.expectedComb, within: Self.sinkhornTolerance, "comb")
    }

    /// The gap allowed on a Sinkhorn row or column sum. Twenty iterations end
    /// on a column normalization, thus the column sums sit on 1 and the row
    /// sums have converged to within this gap.
    private static let doublyStochasticTolerance: Float = 1e-4

    @Test func hcSplitSinkhornCombIsDoublyStochastic() {
        let hcMult = Self.fixtureHCMult
        let (_, _, comb) = DeepSeekV4Math.hcSplitSinkhorn(
            mixes: MLXArray(Self.sinkhornMixes, [1, Self.sinkhornMixes.count]),
            scale: MLXArray(Self.sinkhornScale),
            base: MLXArray(Self.sinkhornBase),
            hcMult: hcMult,
            iters: Self.sinkhornIterations,
            eps: Self.sinkhornEps)

        let rowSums = comb.sum(axis: -1)
        let columnSums = comb.sum(axis: -2)
        eval(rowSums, columnSums)

        let ones = [Float](repeating: 1, count: hcMult)
        expectClose(floats(rowSums), ones, within: Self.doublyStochasticTolerance, "row sums")
        expectClose(
            floats(columnSums), ones, within: Self.doublyStochasticTolerance, "column sums")
    }

    // MARK: - Partial RoPE

    /// The head width the small RoPE fixture uses.
    private static let fixtureHeadDim = 8

    /// The rotary width the small RoPE fixture uses.
    private static let fixtureRopeDim = 4

    /// Two token vectors of width 8. The first four dimensions carry no
    /// position and the last four take the rotation.
    private static let ropeInput: [Float] = [
        1.0, 2.0, 3.0, 4.0, 0.5, -1.5, 2.5, -0.25,
        -1.0, 0.25, -3.5, 1.75, -0.75, 1.25, -2.0, 3.0,
    ]

    /// `cos(position * invFreq)` for positions 0 and 1 against
    /// `invFreq = [1, 0.01]`, shape `(2, 2)`.
    private static let ropeCos: [Float] = [
        1.0, 1.0,
        0.5403023059, 0.9999500004,
    ]

    /// `sin(position * invFreq)` for the same positions and frequencies.
    private static let ropeSin: [Float] = [
        0.0, 0.0,
        0.8414709848, 0.009999833334,
    ]

    /// `DeepseekV4RoPE.__call__` lines 145-164 on the input above, reached
    /// through the trailing-block split `DeepseekV4Attention.__call__` makes at
    /// lines 602-605.
    ///
    /// Position 0 rotates by zero, thus its whole row is the input. Position 1
    /// keeps its first four dimensions and rotates the last four.
    private static let expectedRoPEForward: [Float] = [
        1.0, 2.0, 3.0, 4.0,
        0.5, -1.5, 2.5, -0.25,
        -1.0, 0.25, -3.5, 1.75,
        -1.45706546, 0.04427464373, -2.029899501, 2.979850335,
    ]

    private func ropeFixtureArrays() -> (x: MLXArray, cos: MLXArray, sin: MLXArray) {
        (
            MLXArray(Self.ropeInput, [1, 2, Self.fixtureHeadDim]),
            MLXArray(Self.ropeCos, [2, Self.fixtureRopeDim / 2]),
            MLXArray(Self.ropeSin, [2, Self.fixtureRopeDim / 2])
        )
    }

    @Test func applyPartialRoPEMatchesPythonReference() {
        let (x, cos, sin) = ropeFixtureArrays()
        let rotated = DeepSeekV4Math.applyPartialRoPE(
            x, cos: cos, sin: sin, ropeDim: Self.fixtureRopeDim)
        eval(rotated)

        #expect(rotated.shape == x.shape)
        expectCloseRelative(
            floats(rotated), Self.expectedRoPEForward, within: Self.relativeTolerance,
            "partial rope forward")
    }

    /// The real DeepSeek-V4 head width.
    private static let deepseekV4HeadDim = 512

    /// The real DeepSeek-V4 rotary width.
    private static let deepseekV4RopeDim = 64

    /// The number of positions the shape tests run.
    private static let deepseekV4SequenceLength = 3

    /// Builds a deterministic rotation for the real DeepSeek-V4 rotary width.
    private func deepseekV4Rotation() -> (x: MLXArray, cos: MLXArray, sin: MLXArray) {
        MLXRandom.seed(20_260_810)
        let sequenceLength = Self.deepseekV4SequenceLength
        let pairCount = Self.deepseekV4RopeDim / 2
        let x = MLXRandom.normal([1, sequenceLength, Self.deepseekV4HeadDim])
        let theta = MLXRandom.uniform(
            low: -Float.pi, high: Float.pi, [sequenceLength, pairCount])
        return (x, cos(theta), sin(theta))
    }

    @Test func applyPartialRoPELeavesTheNoPositionDimensionsAlone() {
        let (x, cosine, sine) = deepseekV4Rotation()
        let noPositionDim = Self.deepseekV4HeadDim - Self.deepseekV4RopeDim

        let rotated = DeepSeekV4Math.applyPartialRoPE(
            x, cos: cosine, sin: sine, ropeDim: Self.deepseekV4RopeDim)
        eval(rotated)

        #expect(rotated.shape == x.shape)

        let keptBefore = floats(x[.ellipsis, 0 ..< noPositionDim])
        let keptAfter = floats(rotated[.ellipsis, 0 ..< noPositionDim])
        #expect(keptBefore == keptAfter, "the leading \(noPositionDim) dimensions must not move")

        let rotatedBefore = floats(x[.ellipsis, noPositionDim...])
        let rotatedAfter = floats(rotated[.ellipsis, noPositionDim...])
        #expect(rotatedBefore != rotatedAfter, "the trailing rotary dimensions must move")
    }

    /// The largest absolute gap allowed on the inverse round trip. The values
    /// are standard normal, thus this is a relative gap of about 1e-6.
    private static let roundTripTolerance: Float = 1e-5

    @Test func inversePartialRoPEUndoesTheForwardRotation() {
        let (x, cosine, sine) = deepseekV4Rotation()

        let rotated = DeepSeekV4Math.applyPartialRoPE(
            x, cos: cosine, sin: sine, ropeDim: Self.deepseekV4RopeDim)
        let restored = DeepSeekV4Math.applyInversePartialRoPE(
            rotated, cos: cosine, sin: sine, ropeDim: Self.deepseekV4RopeDim)
        eval(rotated, restored)

        #expect(floats(rotated) != floats(x), "the forward rotation must change the input")
        expectClose(
            floats(restored), floats(x), within: Self.roundTripTolerance, "inverse round trip")
    }

    // MARK: - yarnInvFreq

    /// `DeepseekV4RoPE.__init__` lines 102-134 with the DeepSeek-V4 compressor
    /// parameters: `dims = 64`, `base = 160000`,
    /// `original_max_position_embeddings = 65536`, `factor = 16`,
    /// `beta_fast = 32`, `beta_slow = 1`. The correction range is
    /// `low = 15`, `high = 25`.
    private static let expectedYarnInvFreq: [Float] = [
        1, 0.6876560219, 0.4728708045, 0.3251724563,
        0.2236067977, 0.153764561, 0.1057371263, 0.07271077167,
        0.05, 0.0343828011, 0.02364354023, 0.01625862282,
        0.01118033989, 0.007688228051, 0.005286856317, 0.003635538584,
        0.002265625, 0.001396801295, 0.0008496897268, 0.000508081963,
        0.0002969777783, 0.0001681799886, 9.086784295e-05, 4.54442323e-05,
        1.953125e-05, 5.372312671e-06, 3.69430316e-06, 2.540409815e-06,
        1.746928107e-06, 1.201285633e-06, 8.260712996e-07, 5.680529037e-07,
    ]

    @Test func yarnInvFreqMatchesPythonReference() {
        let invFreq = DeepSeekV4Math.yarnInvFreq(
            dim: Self.deepseekV4RopeDim,
            base: 160_000,
            originalMaxPositionEmbeddings: 65536,
            factor: 16,
            betaFast: 32,
            betaSlow: 1)
        eval(invFreq)

        #expect(invFreq.shape == [Self.deepseekV4RopeDim / 2])
        expectCloseRelative(
            floats(invFreq), Self.expectedYarnInvFreq, within: Self.relativeTolerance,
            "yarn inverse frequency")
    }

    /// The same reference on parameters that drive the correction range past
    /// the end of the table: `dims = 8`, `base = 10`,
    /// `original_max_position_embeddings = 1e9`, `factor = 4`.
    ///
    /// `ceil(correction_dim(beta_slow))` is 33 here. The
    /// `high = min(high, dims - 1)` clamp of line 128 pulls it to 7, which is
    /// below `low = 26`, thus the whole ramp clips to 1, `smooth` is 0, and
    /// every entry takes the interpolated frequency `invFreq / factor`.
    ///
    /// Without that clamp `high` stays 33, the ramp clips to 0, `smooth` is 1,
    /// and every entry keeps the plain frequency -- four times these numbers.
    /// The fixture thus fails when the clamp is missing.
    private static let expectedYarnInvFreqClamped: [Float] = [
        0.25, 0.1405853313, 0.0790569415, 0.04445698525,
    ]

    @Test func yarnInvFreqClampsTheCorrectionRangeToTheTable() {
        let invFreq = DeepSeekV4Math.yarnInvFreq(
            dim: 8,
            base: 10,
            originalMaxPositionEmbeddings: 1_000_000_000,
            factor: 4,
            betaFast: 32,
            betaSlow: 1)
        eval(invFreq)

        expectCloseRelative(
            floats(invFreq), Self.expectedYarnInvFreqClamped, within: Self.relativeTolerance,
            "clamped yarn inverse frequency")
    }

    // MARK: - sqrtSoftplus

    /// `_score_func` lines 307-313 at moderate inputs.
    private static let sqrtSoftplusInput: [Float] = [-8, -2, -0.5, 0.5, 2, 8]

    /// The float64 answer for the inputs above.
    private static let expectedSqrtSoftplus: [Float] = [
        0.01831410311, 0.3562695764, 0.6885324859,
        0.986953385, 1.458399126, 2.828486416,
    ]

    @Test func sqrtSoftplusMatchesPythonReference() {
        let scores = DeepSeekV4Math.sqrtSoftplus(MLXArray(Self.sqrtSoftplusInput))
        eval(scores)

        expectCloseRelative(
            floats(scores), Self.expectedSqrtSoftplus, within: Self.relativeTolerance,
            "sqrt softplus")
    }

    /// The largest value allowed at `x = -100`. The float64 answer is
    /// `1.93e-22`; anything below this limit is that answer or an underflow to
    /// zero, and both are correct in float32.
    private static let sqrtSoftplusUnderflowLimit: Float = 1e-20

    @Test func sqrtSoftplusStaysFiniteAtTheExtremes() {
        let scores = DeepSeekV4Math.sqrtSoftplus(MLXArray([Float(-100), 0, 100]))
        eval(scores)
        let got = floats(scores)

        for (index, value) in got.enumerated() {
            #expect(value.isFinite, "sqrt softplus[\(index)] is \(value)")
            #expect(!value.isNaN, "sqrt softplus[\(index)] is not a number")
            #expect(value >= 0, "sqrt softplus[\(index)] is \(value), which is below zero")
        }

        // sqrt(softplus(-100)) is 1.93e-22 in float64.
        #expect(got[0] <= Self.sqrtSoftplusUnderflowLimit)
        // sqrt(softplus(0)) is sqrt(log(2)).
        expectCloseRelative(
            [got[1]], [0.8325546112], within: Self.relativeTolerance, "sqrt softplus at zero")
        // sqrt(softplus(100)) is sqrt(100).
        expectCloseRelative(
            [got[2]], [10], within: Self.relativeTolerance, "sqrt softplus at one hundred")
    }

    // MARK: - Clamped SwiGLU

    /// The DeepSeek-V4 `swiglu_limit`.
    private static let swiGLULimit: Float = 10

    /// Gate values for the SwiGLU fixture.
    private static let swiGLUGate: [Float] = [50, 10, -50, -10, 1, 0, 3.5, 1, 1]

    /// Up values for the SwiGLU fixture.
    private static let swiGLUUp: [Float] = [50, 10, -50, -10, 2, 7, -0.5, -50, -10]

    /// `_swiglu_limited` lines 367-371 on the pairs above with `limit = 10`.
    ///
    /// Entries 0 and 1 are equal, which is the high-side saturation: a gate of
    /// 50 and a gate of 10 give one answer, and an unclamped run would give
    /// 2500 at entry 0. Entries 7 and 8 are equal, which is the low-side
    /// saturation of `up`.
    ///
    /// Entries 2 and 3 differ, and that is the reference's asymmetry: the
    /// reference clamps `up` on both sides but clamps `gate` on the high side
    /// only, thus a gate of -50 stays a gate of -50.
    private static let expectedSwiGLU: [Float] = [
        99.99546021, 99.99546021, 9.64374924e-20, 0.00453978687,
        1.462117157, 0, -1.698703596, -7.310585786,
        -7.310585786,
    ]

    @Test func clampedSwiGLUMatchesPythonReference() {
        let output = DeepSeekV4Math.clampedSwiGLU(
            gate: MLXArray(Self.swiGLUGate), up: MLXArray(Self.swiGLUUp),
            limit: Self.swiGLULimit)
        eval(output)

        expectCloseRelative(
            floats(output), Self.expectedSwiGLU, within: Self.relativeTolerance, "clamped swiglu")
    }

    @Test func clampedSwiGLUSaturatesBeyondTheLimit() {
        let output = floats(
            DeepSeekV4Math.clampedSwiGLU(
                gate: MLXArray(Self.swiGLUGate), up: MLXArray(Self.swiGLUUp),
                limit: Self.swiGLULimit))

        // gate 50, up 50 gives the same answer as gate 10, up 10.
        #expect(output[0] == output[1])
        // up -50 gives the same answer as up -10 at the same gate.
        #expect(output[7] == output[8])
        // An unclamped run would give 2500 at entry 0.
        #expect(output[0] < 100)
    }

    // MARK: - reduceRoutedExpertsFP32

    /// The number of routed experts the accumulation fixture uses.
    private static let fixtureExpertCount = 8

    /// One large expert output and seven small ones. `1024 + 1` rounds back to
    /// 1024 in bfloat16, thus a running bfloat16 accumulation drops all seven
    /// small values while the float64 answer is 1031.
    private static let routedExpertValues: [Float] = [1024, 1, 1, 1, 1, 1, 1, 1]

    /// The float64 sum of the values above.
    private static let exactRoutedSum: Float = 1031

    @Test func reduceRoutedExpertsFP32BeatsABFloat16Accumulation() {
        let expertCount = Self.fixtureExpertCount
        let routed = MLXArray(Self.routedExpertValues, [1, expertCount, 1]).asType(.bfloat16)

        let reduced = DeepSeekV4Math.reduceRoutedExpertsFP32(routed)
        eval(reduced)
        #expect(reduced.shape == [1, 1])
        #expect(reduced.dtype == .float32)

        var running = MLXArray(Float(0)).asType(.bfloat16)
        for expert in 0 ..< expertCount {
            running = (running + routed[0, expert, 0]).asType(.bfloat16)
        }
        eval(running)

        let fp32Sum = floats(reduced)[0]
        let bfloat16Sum = floats(running)[0]
        let fp32Error = abs(fp32Sum - Self.exactRoutedSum)
        let bfloat16Error = abs(bfloat16Sum - Self.exactRoutedSum)

        #expect(bfloat16Error > 0, "the bfloat16 accumulation must lose something to compare with")
        #expect(
            fp32Error < bfloat16Error,
            "float32 error \(fp32Error) must beat bfloat16 error \(bfloat16Error)")
        #expect(fp32Sum == Self.exactRoutedSum)
    }

    // MARK: - Pooled-chunk visibility

    /// The number of raw positions one pooled chunk covers in the visibility
    /// tests.
    private static let visibilityChunkWidth = 4

    /// The number of pooled chunks the visibility tests read.
    private static let visibilityChunkCount = 5

    /// The number of queries the visibility tests read.
    private static let visibilityQueryCount = 7

    /// The absolute position of the first query of the visibility tests. It is
    /// not a multiple of ``visibilityChunkWidth``, thus a rule that reads the
    /// offset wrongly cannot agree by luck.
    private static let visibilityOffset = 6

    @Test func pooledChunkVisibilityHidesEveryChunkThatReachesPastItsQuery() {
        let visible = DeepSeekV4Math.pooledChunkVisibility(
            queryCount: Self.visibilityQueryCount,
            offset: Self.visibilityOffset,
            chunkCount: Self.visibilityChunkCount,
            chunkWidth: Self.visibilityChunkWidth)
        eval(visible)

        #expect(visible.shape == [1, Self.visibilityQueryCount, Self.visibilityChunkCount])
        #expect(visible.dtype == .bool)

        // The rule again, in plain Swift: a query at absolute position `q`
        // reads chunk `c` only when the whole chunk stands behind it.
        let read = visible.asType(.int32).asArray(Int32.self)
        for query in 0 ..< Self.visibilityQueryCount {
            let position = Self.visibilityOffset + query
            for chunk in 0 ..< Self.visibilityChunkCount {
                let expected = (chunk + 1) * Self.visibilityChunkWidth <= position + 1
                let actual = read[query * Self.visibilityChunkCount + chunk] != 0
                #expect(actual == expected, "query \(position), chunk \(chunk)")
            }
        }
    }

    @Test func pooledChunkVisibilityHidesEveryChunkFromTheFirstPositions() {
        let visible = DeepSeekV4Math.pooledChunkVisibility(
            queryCount: Self.visibilityChunkWidth - 1,
            offset: 0,
            chunkCount: Self.visibilityChunkCount,
            chunkWidth: Self.visibilityChunkWidth)
        eval(visible)

        #expect(
            visible.asType(.int32).asArray(Int32.self).allSatisfy { $0 == 0 },
            "no chunk has ended before position \(Self.visibilityChunkWidth - 1)")
    }
}

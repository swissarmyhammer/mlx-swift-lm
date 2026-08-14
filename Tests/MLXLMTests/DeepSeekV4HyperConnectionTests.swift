// Copyright © 2026 Apple Inc.
//
// Tests for the DeepSeek-V4 manifold hyper-connections.
//
// Every expected number below comes from the DeepSeek-V4 Python reference,
// `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
// `mlx_lm/models/deepseek_v4.py` -- `hc_split_sinkhorn`,
// `HyperConnection.hc_pre`, `HyperConnection.hc_post` and
// `HyperHead.__call__`. Each routine was transcribed into NumPy one line at a
// time -- `mx.*` became `np.*` and nothing else changed -- and run in float64.
//
// A fixture that a Swift function produced itself proves nothing, thus no
// number below was read out of this repository.
//
// Why the values matter more than the shapes. The mixing matrix `comb` is
// square and doubly stochastic, thus its transpose carries the same shape,
// the same row sums and the same column sums. A collapse or an expand that
// reads the matrix the wrong way round therefore passes every shape check and
// every stochasticity check, and answers plausible numbers that are wrong. The
// NumPy transcription measured the gap: on the fixture below, reading `comb`
// transposed moves the expand by 0.153. The tests read that gap against a
// limit of 1e-5.
//
// The inputs are not random. Both the NumPy transcription and the fixture
// builders below fill the mixing projection with `((row * hcDim + column) % 17
// - 8) / 16` and the bias with `(index % 9 - 4) / 8`, which land on multiples
// of a sixteenth and an eighth and are thus exact in float32 and in float64
// alike. The hidden states are written out one by one and are multiples of a
// sixteenth as well.
//
// Tolerance. MLX evaluates in float32 and the fixtures are float64. The tests
// allow 1e-5, read as an absolute gap or a relative one, whichever is larger.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4HyperConnectionTests {

    // MARK: - The synthetic checkpoint

    /// The number of parallel copies of the residual stream.
    private static let hcMult = 4

    /// The number of one-value-for-each-copy fields the mixes vector holds
    /// ahead of the flattened mixing matrix: one `pre` weight and one `post`
    /// weight for each copy.
    private static let splitFieldCount = 2

    /// The width of the mixes vector, `(2 + hcMult) * hcMult`.
    private static let mixWidth = (splitFieldCount + hcMult) * hcMult

    /// The width of the residual stream of the fixture checkpoint.
    private static let hiddenSize = 2

    /// The width the mixing projection reads, `hcMult * hiddenSize`.
    private static let hcDim = hcMult * hiddenSize

    /// The number of tokens of the fixture.
    private static let tokenCount = 2

    /// The number of Sinkhorn steps of the fixture checkpoint. The number is
    /// not the DeepSeek-V4-Flash default of 20, thus a layer that ignored
    /// `hc_sinkhorn_iters` and read the default would answer a different
    /// mixing matrix.
    private static let sinkhornIterations = 3

    /// The RMS-norm epsilon of the fixture checkpoint. The number is far
    /// larger than the mHC epsilon, thus a collapse that read the wrong one of
    /// the two answers a visibly different mix.
    private static let normEps: Float = 0.0625

    /// The mHC epsilon of the fixture checkpoint.
    private static let hcEps: Float = 1e-6

    /// The width of the residual stream of the wide checkpoint, which the
    /// shape tests read.
    private static let wideHiddenSize = 64

    /// The number of tokens the shape tests read.
    private static let wideTokenCount = 4

    /// The largest gap allowed against a fixture, absolute or relative.
    private static let tolerance: Float = 1e-5

    /// The largest gap allowed where a test measures a limit rather than a
    /// fixture, such as the near-identity round trip.
    private static let looseTolerance: Float = 1e-4

    /// Builds the `config.json` of a synthetic checkpoint.
    ///
    /// - Parameters:
    ///   - hiddenSize: The width of the residual stream.
    ///   - sinkhornIterations: The number of Sinkhorn steps.
    ///   - normEps: The RMS-norm epsilon.
    ///   - hcEps: The mHC epsilon.
    /// - Returns: The decoded configuration.
    private static func configuration(
        hiddenSize: Int = hiddenSize,
        sinkhornIterations: Int = sinkhornIterations,
        normEps: Float = normEps,
        hcEps: Float = hcEps
    ) throws -> DeepSeekV4Configuration {
        let json = """
            {
              "hidden_size": \(hiddenSize),
              "hc_mult": \(hcMult),
              "hc_sinkhorn_iters": \(sinkhornIterations),
              "hc_eps": \(hcEps),
              "rms_norm_eps": \(normEps)
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Fixture builders

    /// Builds an array of the given shape from a list of values.
    private static func array(_ values: [Float], _ shape: [Int]) -> MLXArray {
        MLXArray(values).reshaped(shape)
    }

    /// The period of the mixing-projection ramp.
    private static let projectionPeriod = 17

    /// The value the mixing-projection ramp subtracts, which centres it on
    /// zero.
    private static let projectionOffset = 8

    /// The number the mixing-projection ramp divides by, which lands every
    /// value on a multiple of a sixteenth.
    private static let projectionDivisor: Float = 16

    /// The period of the bias ramp.
    private static let biasPeriod = 9

    /// The value the bias ramp subtracts, which centres it on zero.
    private static let biasOffset = 4

    /// The number the bias ramp divides by, which lands every value on a
    /// multiple of an eighth.
    private static let biasDivisor: Float = 8

    /// Builds a mixing projection of `rows` rows and ``hcDim`` columns.
    ///
    /// - Parameter rows: The number of rows.
    /// - Returns: The projection, shape `(rows, hcDim)`.
    private static func projectionFixture(rows: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(rows * hcDim)
        for row in 0 ..< rows {
            for column in 0 ..< hcDim {
                let step = (row * hcDim + column) % projectionPeriod - projectionOffset
                values.append(Float(step) / projectionDivisor)
            }
        }
        return array(values, [rows, hcDim])
    }

    /// Builds a bias of `count` values.
    ///
    /// - Parameter count: The number of values.
    /// - Returns: The bias, shape `(count)`.
    private static func biasFixture(count: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0 ..< count {
            values.append(Float(index % biasPeriod - biasOffset) / biasDivisor)
        }
        return array(values, [count])
    }

    /// The hidden states of the fixture, shape `(1, 2, 4, 2)`.
    private static let hiddenStates: [Float] = [
        0.5, -0.25, 0.75, 0.125, -0.875, 0.375, 0.25, -0.5,
        -0.125, 0.625, -0.75, 0.875, 0.375, -0.625, 0.0625, 0.25,
    ]

    /// The block output of the fixture, shape `(1, 2, 2)`.
    private static let blockOutput: [Float] = [0.5, -0.25, 0.125, -0.75]

    /// The three learned scales of the fixture, one for each field.
    private static let scales: [Float] = [0.5, 0.25, 0.75]

    /// The learned scale of the head fixture.
    private static let headScales: [Float] = [0.625]

    /// The learned bias of the head fixture.
    private static let headBias: [Float] = [-0.25, 0.125, 0.5, -0.375]

    /// The period of the head projection ramp.
    private static let headProjectionPeriod = 11

    /// The value the head projection ramp subtracts.
    private static let headProjectionOffset = 5

    /// Builds the mixing projection of the head fixture.
    private static func headProjectionFixture() -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(hcMult * hcDim)
        for row in 0 ..< hcMult {
            for column in 0 ..< hcDim {
                let step = (row * hcDim + column) % headProjectionPeriod - headProjectionOffset
                values.append(Float(step) / projectionDivisor)
            }
        }
        return array(values, [hcMult, hcDim])
    }

    /// The hidden states of the fixture as an array.
    private static func hiddenStateFixture() -> MLXArray {
        array(hiddenStates, [1, tokenCount, hcMult, hiddenSize])
    }

    /// Builds a hyper-connection and loads the fixture weights into it.
    ///
    /// - Parameter configuration: The configuration to build the layer from.
    /// - Returns: The layer.
    private static func fixtureHyperConnection(
        configuration: DeepSeekV4Configuration
    ) throws -> DeepSeekV4HyperConnection {
        let hyperConnection = DeepSeekV4HyperConnection(configuration: configuration)
        try hyperConnection.update(
            parameters: ModuleParameters.unflattened([
                ("fn", projectionFixture(rows: mixWidth)),
                ("base", biasFixture(count: mixWidth)),
                ("scale", array(scales, [scales.count])),
            ]),
            verify: [])
        return hyperConnection
    }

    /// Builds a hyper head and loads the fixture weights into it.
    ///
    /// - Parameter configuration: The configuration to build the head from.
    /// - Returns: The head.
    private static func fixtureHyperHead(
        configuration: DeepSeekV4Configuration
    ) throws -> DeepSeekV4HyperHead {
        let head = DeepSeekV4HyperHead(configuration: configuration)
        try head.update(
            parameters: ModuleParameters.unflattened([
                ("fn", headProjectionFixture()),
                ("base", array(headBias, [hcMult])),
                ("scale", array(headScales, [headScales.count])),
            ]),
            verify: [])
        return head
    }

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.uniform(low: -1, high: 1, shape, key: MLXRandom.key(seed))
    }

    /// Builds a hyper-connection of the wide checkpoint with repeatable random
    /// weights.
    private static func wideHyperConnection() throws -> DeepSeekV4HyperConnection {
        let configuration = try configuration(hiddenSize: wideHiddenSize)
        let hyperConnection = DeepSeekV4HyperConnection(configuration: configuration)
        let wideHcDim = hcMult * wideHiddenSize
        try hyperConnection.update(
            parameters: ModuleParameters.unflattened([
                ("fn", randomArray([mixWidth, wideHcDim], seed: 51)),
                ("base", randomArray([mixWidth], seed: 52)),
                ("scale", randomArray([scales.count], seed: 53)),
            ]),
            verify: [])
        return hyperConnection
    }

    // MARK: - Comparison helpers

    /// Reads an array back as a flat list of `Float`, in row-major order.
    private func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// Checks every element of `got` against `expected` and names the worst
    /// element when one is too far away.
    private func expectMatches(
        _ got: MLXArray, _ expected: [Float], _ what: String,
        limit: Float = tolerance, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let values = floats(got)
        guard values.count == expected.count else {
            #expect(
                values.count == expected.count,
                "\(what): got \(values.count) values, expected \(expected.count)",
                sourceLocation: sourceLocation)
            return
        }
        let gaps = zip(values, expected).map { abs($0 - $1) / max(1, abs($1)) }
        let worst = gaps.indices.max { gaps[$0] < gaps[$1] } ?? gaps.startIndex
        #expect(
            gaps[worst] <= limit,
            """
            \(what)[\(worst)]: got \(values[worst]), expected \(expected[worst]), \
            gap \(gaps[worst])
            """,
            sourceLocation: sourceLocation)
    }

    // MARK: - The collapse and the expand agree with the Python

    @Test
    func theCollapseAgreesWithThePythonReference() throws {
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration())

        let (collapsed, _, _) = hyperConnection.collapse(Self.hiddenStateFixture())

        #expect(collapsed.shape == [1, Self.tokenCount, Self.hiddenSize])
        expectMatches(
            collapsed,
            [0.132681459113, -0.069480802154, -0.203221420882, 0.489406275467],
            "the collapsed residual stream")
    }

    @Test
    func theCollapseWeighsEachCopyByItsOwnPreWeight() throws {
        // `post` is the gain the block output takes on the way back out, and
        // `pre` is the weight the collapse reads. The two carry the same shape
        // and both come out of the same mixes vector, thus a collapse that
        // read `post` in place of `pre` would keep every shape. The two lists
        // below are different numbers.
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration())

        let (_, post, _) = hyperConnection.collapse(Self.hiddenStateFixture())

        #expect(post.shape == [1, Self.tokenCount, Self.hcMult])
        expectMatches(
            post,
            [
                0.979483792184, 1.024876540250, 1.267123181750, 1.143749303137,
                1.050837379325, 1.073854591723, 0.999438223637, 1.187709540826,
            ],
            "the post gain of each copy")
    }

    @Test
    func theCollapseAnswersTheSinkhornMixingMatrix() throws {
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration())

        let (_, _, comb) = hyperConnection.collapse(Self.hiddenStateFixture())

        #expect(comb.shape == [1, Self.tokenCount, Self.hcMult, Self.hcMult])
        expectMatches(
            comb,
            [
                0.590084624295, 0.143665093484, 0.144884801764, 0.112767435943,
                0.136913068912, 0.206273795792, 0.497560755446, 0.161911275252,
                0.197746065177, 0.422278138960, 0.164603056872, 0.216190378482,
                0.075255259938, 0.227781967938, 0.192950379851, 0.509129901423,
                0.351760072042, 0.122842927371, 0.302132279339, 0.223130038934,
                0.143401230560, 0.215600705070, 0.249640167126, 0.391614181704,
                0.332987412513, 0.329424865966, 0.173083749132, 0.164315614668,
                0.171850286008, 0.332130501588, 0.275142804257, 0.220939163721,
            ],
            "the Sinkhorn mixing matrix")
    }

    @Test
    func theExpandAgreesWithThePythonReference() throws {
        // This is the guard on the axis alignment. The NumPy transcription
        // reads the same expand with `comb` transposed and answers numbers
        // that are 0.153 away from the list below, at the same shape.
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration())
        let stream = Self.hiddenStateFixture()
        let (_, post, comb) = hyperConnection.collapse(stream)

        let expanded = hyperConnection.expand(
            blockOutput: Self.array(Self.blockOutput, [1, Self.tokenCount, Self.hiddenSize]),
            residual: stream, post: post, comb: comb)

        #expect(expanded.shape == [1, Self.tokenCount, Self.hcMult, Self.hiddenSize])
        expectMatches(
            expanded,
            [
                0.793950685795, -0.376485884744, 0.340712309223, -0.159033532150,
                0.959163147540, -0.359801587275, 0.738789650477, -0.458486953044,
                0.122497700068, -0.593840592871, 0.072697090372, -0.585236116784,
                -0.088586166242, -0.320313216728, -0.005128220010, -0.609490999706,
            ],
            "the expanded residual stream")
    }

    // MARK: - The axis alignment of the expand

    /// A mixing matrix whose rows and columns hold different numbers, thus a
    /// transposed read of it is a different answer.
    private static let asymmetricMixing: [Float] = [
        0.5, 0.25, 0.125, 0.125,
        0.125, 0.5, 0.25, 0.125,
        0.25, 0.125, 0.5, 0.125,
        0.125, 0.125, 0.125, 0.625,
    ]

    /// Column 0 of ``asymmetricMixing``, which is the weight each answer copy
    /// gives to copy 0 of the residual stream.
    private static let asymmetricMixingFirstColumn: [Float] = [0.5, 0.125, 0.25, 0.125]

    @Test
    func theExpandTakesTheRowOfTheMixingMatrixForEachAnswerCopy() throws {
        // `comb[i][j]` weighs copy `j` of the residual stream into copy `i` of
        // the answer. A residual stream that holds a value in copy 0 and zero
        // in every other copy therefore answers column 0 of the matrix, one
        // value for each answer copy. A transposed read answers row 0 instead,
        // and the two lists differ.
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration(hiddenSize: 1))
        let residual = Self.array([1, 0, 0, 0], [1, 1, Self.hcMult, 1])
        let comb = Self.array(Self.asymmetricMixing, [1, 1, Self.hcMult, Self.hcMult])
        let post = MLXArray.zeros([1, 1, Self.hcMult])

        let expanded = hyperConnection.expand(
            blockOutput: MLXArray.zeros([1, 1, 1]),
            residual: residual, post: post, comb: comb)

        #expect(floats(expanded) == Self.asymmetricMixingFirstColumn)
    }

    /// One gain for each copy, which the expand test reads as its `post`.
    private static let postGains: [Float] = [0.5, 0.25, 0.125, 0.0625]

    @Test
    func theExpandAddsTheBlockOutputScaledByThePostGain() throws {
        // With a mixing matrix of zeros the whole answer is the new term, thus
        // copy `i` holds `post[i] * blockOutput`. A block output of one makes
        // the answer the gain list itself.
        let hyperConnection = try Self.fixtureHyperConnection(
            configuration: Self.configuration(hiddenSize: 1))
        let post = Self.array(Self.postGains, [1, 1, Self.hcMult])

        let expanded = hyperConnection.expand(
            blockOutput: Self.array([1], [1, 1, 1]),
            residual: MLXArray.zeros([1, 1, Self.hcMult, 1]),
            post: post,
            comb: MLXArray.zeros([1, 1, Self.hcMult, Self.hcMult]))

        #expect(floats(expanded) == Self.postGains)
    }

    // MARK: - The round trip

    /// The logit that makes a softmax row read as the identity.
    private static let identityDiagonalLogit: Float = 20

    /// The logit that closes the `post` gate, thus the expand adds no new
    /// term.
    private static let closedPostLogit: Float = -30

    /// The width of the residual stream of the round-trip checkpoint.
    private static let roundTripHiddenSize = 3

    @Test
    func theRoundTripAnswersTheStreamWhenTheMixingIsTheIdentity() throws {
        // A mixing projection of zeros leaves the mixes vector at zero, thus
        // the bias alone decides `post` and `comb`. A bias whose comb block
        // holds a large value on the diagonal and zero elsewhere makes the
        // softmax read as the identity, and the Sinkhorn steps hold it there.
        // A bias whose post block is far negative closes the new term. The
        // expand then answers the residual stream it was given.
        let configuration = try Self.configuration(
            hiddenSize: Self.roundTripHiddenSize, normEps: Self.hcEps)
        let hyperConnection = DeepSeekV4HyperConnection(configuration: configuration)
        var bias = [Float](repeating: 0, count: Self.mixWidth)
        let postStart = Self.hcMult
        let combStart = Self.splitFieldCount * Self.hcMult
        for copy in 0 ..< Self.hcMult {
            bias[postStart + copy] = Self.closedPostLogit
            bias[combStart + copy * Self.hcMult + copy] = Self.identityDiagonalLogit
        }
        try hyperConnection.update(
            parameters: ModuleParameters.unflattened([
                ("fn", MLXArray.zeros([Self.mixWidth, Self.hcMult * Self.roundTripHiddenSize])),
                ("base", Self.array(bias, [Self.mixWidth])),
                ("scale", Self.array([1, 1, 1], [Self.scales.count])),
            ]),
            verify: [])
        let stream = Self.randomArray(
            [1, Self.tokenCount, Self.hcMult, Self.roundTripHiddenSize], seed: 61)

        let (collapsed, post, comb) = hyperConnection.collapse(stream)
        let expanded = hyperConnection.expand(
            blockOutput: collapsed, residual: stream, post: post, comb: comb)

        #expect(expanded.shape == stream.shape)
        expectMatches(
            expanded, floats(stream), "the round trip", limit: Self.looseTolerance)
    }

    // MARK: - Shapes on the wide checkpoint

    @Test
    func theCollapseAndTheExpandKeepTheShapesOfTheWideCheckpoint() throws {
        let hyperConnection = try Self.wideHyperConnection()
        let stream = Self.randomArray(
            [1, Self.wideTokenCount, Self.hcMult, Self.wideHiddenSize], seed: 71)

        let (collapsed, post, comb) = hyperConnection.collapse(stream)
        let expanded = hyperConnection.expand(
            blockOutput: collapsed, residual: stream, post: post, comb: comb)

        #expect(collapsed.shape == [1, Self.wideTokenCount, Self.wideHiddenSize])
        #expect(post.shape == [1, Self.wideTokenCount, Self.hcMult])
        #expect(comb.shape == [1, Self.wideTokenCount, Self.hcMult, Self.hcMult])
        #expect(expanded.shape == [1, Self.wideTokenCount, Self.hcMult, Self.wideHiddenSize])
        #expect(floats(expanded).allSatisfy { $0.isFinite })
    }

    // MARK: - The head reduce

    @Test
    func theHeadReduceAgreesWithThePythonReference() throws {
        let head = try Self.fixtureHyperHead(configuration: Self.configuration())

        let reduced = head(Self.hiddenStateFixture())

        #expect(reduced.shape == [1, Self.tokenCount, Self.hiddenSize])
        expectMatches(
            reduced,
            [0.194159236285, 0.044790414795, -0.125088557380, 0.365832729608],
            "the head reduce")
    }

    @Test
    func theHeadReduceAnswersOneFiniteStreamOfTheWideCheckpoint() throws {
        let configuration = try Self.configuration(hiddenSize: Self.wideHiddenSize)
        let head = DeepSeekV4HyperHead(configuration: configuration)
        try head.update(
            parameters: ModuleParameters.unflattened([
                (
                    "fn",
                    Self.randomArray([Self.hcMult, Self.hcMult * Self.wideHiddenSize], seed: 81)
                ),
                ("base", Self.randomArray([Self.hcMult], seed: 82)),
                ("scale", Self.randomArray([Self.headScales.count], seed: 83)),
            ]),
            verify: [])
        let stream = Self.randomArray(
            [1, Self.wideTokenCount, Self.hcMult, Self.wideHiddenSize], seed: 84)

        let reduced = head(stream)

        #expect(reduced.shape == [1, Self.wideTokenCount, Self.wideHiddenSize])
        #expect(floats(reduced).allSatisfy { $0.isFinite })
    }

    // MARK: - The two epsilons

    @Test
    func theSinkhornEpsilonChangesTheMixingMatrix() {
        // `hcSplitSinkhorn` adds `eps` to the softmax of `comb` and to each
        // sum it divides by. The fixture tests above allow 1e-5 and the term
        // moves the matrix by less than that, thus they stay green with the
        // term removed. This test reads the term itself: the same input with
        // two different epsilons must answer two different matrices.
        let mixes = Self.biasFixture(count: Self.mixWidth).reshaped([1, Self.mixWidth])
        let scale = Self.array(Self.scales, [Self.scales.count])
        let bias = Self.biasFixture(count: Self.mixWidth)

        let withoutEps = DeepSeekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: bias,
            hcMult: Self.hcMult, iters: Self.sinkhornIterations, eps: 0)
        let withEps = DeepSeekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: bias,
            hcMult: Self.hcMult, iters: Self.sinkhornIterations, eps: Self.hcEps)

        #expect(floats(withoutEps.comb) != floats(withEps.comb))
        #expect(floats(withoutEps.pre) != floats(withEps.pre))
    }

    /// A logit far enough below its neighbours that the softmax of its row
    /// answers almost nothing for it.
    private static let closedColumnLogit: Float = -50

    /// A mixing-matrix bias whose column 0 is closed and whose other columns
    /// hold no two rows alike.
    private static let closedColumnBias: [Float] = [
        closedColumnLogit, 0, 1, 2,
        closedColumnLogit, 2, 0, 1,
        closedColumnLogit, 1, 2, 0,
        closedColumnLogit, 0, 2, 1,
    ]

    @Test
    func theSinkhornEpsilonLiftsAClosedColumnOfTheMixingMatrix() {
        // `hcSplitSinkhorn` adds `eps` to the softmax of `comb` before the
        // first column normalization. On an ordinary matrix that term moves
        // the answer by less than the fixtures above measure, thus this test
        // reads the one input that makes the term decide the answer: a column
        // whose every logit is far below its neighbours.
        //
        // The softmax leaves that column at about 6e-23. The term lifts it to
        // `eps`, thus the column normalization answers about a quarter for it
        // and the Sinkhorn steps hold it there. With the term removed the
        // column stays at almost nothing, and the NumPy transcription answers
        // 0.0000303 in place of 0.2450 for the first value below.
        var bias = [Float](repeating: 0, count: Self.mixWidth)
        bias.replaceSubrange(
            (Self.splitFieldCount * Self.hcMult) ..< Self.mixWidth,
            with: Self.closedColumnBias)

        let split = DeepSeekV4Math.hcSplitSinkhorn(
            mixes: MLXArray.zeros([1, Self.mixWidth]),
            scale: Self.array([1, 1, 1], [Self.scales.count]),
            base: Self.array(bias, [Self.mixWidth]),
            hcMult: Self.hcMult, iters: Self.sinkhornIterations, eps: Self.hcEps)

        expectMatches(
            split.comb,
            [
                0.244966672800, 0.085934637232, 0.137311751400, 0.531243966432,
                0.218404148377, 0.566117943493, 0.045037072577, 0.174242675094,
                0.264938243524, 0.252637217713, 0.403681378212, 0.077758319655,
                0.271689936220, 0.095309193867, 0.413968804356, 0.216754038946,
            ],
            "the mixing matrix of a closed column")
    }

    @Test
    func theCollapseReadsTheMhcEpsilonOfTheCheckpoint() throws {
        // A layer that held its own epsilon rather than reading `hc_eps` would
        // answer the same numbers for both configurations.
        let stream = Self.hiddenStateFixture()

        let withoutEps = try Self.fixtureHyperConnection(
            configuration: Self.configuration(hcEps: 0))
        let withEps = try Self.fixtureHyperConnection(
            configuration: Self.configuration(hcEps: Self.hcEps))

        #expect(floats(withoutEps.collapse(stream).comb) != floats(withEps.collapse(stream).comb))
    }

    @Test
    func theCollapseReadsTheRmsNormEpsilonOfTheCheckpoint() throws {
        // The RMS reduction of the collapse takes `rms_norm_eps`, not
        // `hc_eps`. A layer that read `hc_eps` there would answer the same
        // numbers for both configurations, because both give the same
        // `hc_eps`.
        let stream = Self.hiddenStateFixture()

        let small = try Self.fixtureHyperConnection(
            configuration: Self.configuration(normEps: Self.hcEps))
        let large = try Self.fixtureHyperConnection(
            configuration: Self.configuration(normEps: Self.normEps))

        #expect(
            floats(small.collapse(stream).collapsed) != floats(large.collapse(stream).collapsed))
    }

    @Test
    func theCollapseReadsTheSinkhornIterationsOfTheCheckpoint() throws {
        // One Sinkhorn step leaves the matrix column-normalized only. Three
        // steps make it doubly stochastic. A layer that ignored
        // `hc_sinkhorn_iters` would answer one matrix for both.
        let stream = Self.hiddenStateFixture()

        let oneStep = try Self.fixtureHyperConnection(
            configuration: Self.configuration(sinkhornIterations: 1))
        let threeSteps = try Self.fixtureHyperConnection(
            configuration: Self.configuration(sinkhornIterations: Self.sinkhornIterations))

        #expect(floats(oneStep.collapse(stream).comb) != floats(threeSteps.collapse(stream).comb))
    }

    @Test
    func theHeadReduceReadsTheRmsNormEpsilonOfTheCheckpoint() throws {
        let stream = Self.hiddenStateFixture()

        let small = try Self.fixtureHyperHead(
            configuration: Self.configuration(normEps: Self.hcEps))
        let large = try Self.fixtureHyperHead(
            configuration: Self.configuration(normEps: Self.normEps))

        #expect(floats(small(stream)) != floats(large(stream)))
    }
}

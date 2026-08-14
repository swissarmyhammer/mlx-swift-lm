// Copyright © 2026 Apple Inc.
//
// Tests for the DeepSeek-V4 mixture-of-experts layer.
//
// Every expected number below comes from the DeepSeek-V4 Python reference,
// `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
// `mlx_lm/models/deepseek_v4.py` -- `_score_func`, `MoEGate.__call__`,
// `_swiglu_limited` and `DeepseekV4MoE.__call__`. Each routine was
// transcribed into NumPy one line at a time -- `mx.*` became `np.*` and
// nothing else changed -- and run in float64.
//
// A fixture that a Swift function produced itself proves nothing, thus no
// number below was read out of this repository.
//
// The dtype of each fixture. Every fixture runs in float32, with one
// exception: ``DeepSeekV4MoETests/theRoutedReductionRunsInFloat32()`` runs in
// bfloat16, because the float32 reduction it measures cannot be seen in
// float32 activations. That test states the bfloat16 rounding of each step
// beside the number it asserts.
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
struct DeepSeekV4MoETests {

    // MARK: - The synthetic checkpoint

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 4

    /// The number of routed experts of the synthetic checkpoint.
    private static let routedExpertCount = 8

    /// The number of routed experts one token reads.
    private static let expertsPerToken = 2

    /// The width of one routed expert.
    private static let expertWidth = 3

    /// The number of tokens in the vocabulary of the synthetic checkpoint.
    private static let vocabSize = 6

    /// The number of first layers that route through the hash table.
    private static let hashLayerCount = 3

    /// The index of a layer that routes through the hash table.
    private static let hashLayer = 0

    /// The index of a layer that routes through the top-k gate.
    private static let topKLayer = 3

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 8

    /// The SwiGLU limit of the synthetic checkpoint.
    private static let swigluLimit: Float = 10

    /// The largest gap allowed against a fixture, absolute or relative.
    private static let tolerance: Float = 1e-5

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// - Parameters:
    ///   - hashLayers: The number of first layers that read the hash table.
    ///   - normalizeTopkProb: True when the gate divides the weights by their
    ///     sum.
    ///   - routedScalingFactor: The factor the gate weights take.
    ///   - routedExperts: The number of routed experts.
    ///   - expertsPerToken: The number of routed experts one token reads.
    ///   - hiddenSize: The width of the residual stream.
    ///   - expertWidth: The width of one routed expert.
    /// - Returns: The decoded configuration.
    private static func configuration(
        hashLayers: Int = hashLayerCount,
        normalizeTopkProb: Bool = false,
        routedScalingFactor: Float = 1,
        routedExperts: Int = routedExpertCount,
        expertsPerToken: Int = expertsPerToken,
        hiddenSize: Int = hiddenSize,
        expertWidth: Int = expertWidth
    ) throws -> DeepSeekV4Configuration {
        let json = """
            {
              "vocab_size": \(vocabSize),
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(layerCount),
              "n_routed_experts": \(routedExperts),
              "num_experts_per_tok": \(expertsPerToken),
              "moe_intermediate_size": \(expertWidth),
              "n_shared_experts": 1,
              "num_hash_layers": \(hashLayers),
              "norm_topk_prob": \(normalizeTopkProb),
              "routed_scaling_factor": \(routedScalingFactor),
              "swiglu_limit": \(swigluLimit)
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Fixture builders

    /// Builds an array of the given shape from a list of values.
    private static func array(_ values: [Float], _ shape: [Int]) -> MLXArray {
        MLXArray(values).reshaped(shape)
    }

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.uniform(low: -1, high: 1, shape, key: MLXRandom.key(seed))
    }

    /// The eight routing logits the gate fixtures read. The values step by a
    /// half, thus each one is exact in float32 and in float64 alike.
    private static let routingLogits: [Float] = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]

    /// `_score_func(logits, "sqrtsoftplus")` on ``routingLogits``.
    private static let routingScores: [Float] = [
        0.8325546112, 0.9869533850, 1.1459763032, 1.3043823358,
        1.4583991261, 1.6058921926, 1.7460204327, 1.8787630022,
    ]

    /// A gate weight that turns the one-hot input below into
    /// ``routingLogits``. Column 0 holds the logits and every other column
    /// holds zero.
    private static func routingGateWeight() -> MLXArray {
        var values = [Float](repeating: 0, count: routedExpertCount * hiddenSize)
        for expert in 0 ..< routedExpertCount {
            values[expert * hiddenSize] = routingLogits[expert]
        }
        return array(values, [routedExpertCount, hiddenSize])
    }

    /// The one-hot input that makes the gate logits equal ``routingLogits``.
    private static func routingInput() -> MLXArray {
        var values = [Float](repeating: 0, count: hiddenSize)
        values[0] = 1
        return array(values, [1, 1, hiddenSize])
    }

    /// The hash table of the synthetic checkpoint, shape
    /// `(vocabSize, expertsPerToken)`. No row holds the top-scoring experts,
    /// thus a layer that read the table where it should read the gate is
    /// visible.
    private static let hashTable: [Int32] = [
        5, 2,
        1, 4,
        0, 3,
        4, 0,
        2, 5,
        3, 1,
    ]

    /// The hash table as an array.
    private static func hashTableArray() -> MLXArray {
        MLXArray(hashTable).reshaped([vocabSize, expertsPerToken])
    }

    /// Builds a gate and loads the routing fixture into it.
    ///
    /// A hash layer declares the hash table alone and every later layer
    /// declares the routing bias alone, thus the fixture carries the one
    /// parameter the layer holds. The load verifies with `[.all]`, which is
    /// the verification `MLXLMCommon.loadWeights` applies, thus a fixture that
    /// filled the other parameter would fail here.
    ///
    /// - Parameters:
    ///   - layer: The index of the decoder layer.
    ///   - bias: The routing bias, one value for each routed expert. A hash
    ///     layer holds no bias and passes over this value.
    ///   - configuration: The configuration to build the gate from.
    /// - Returns: The gate.
    private static func routingGate(
        layer: Int, bias: [Float], configuration: DeepSeekV4Configuration
    ) throws -> DeepSeekV4MoEGate {
        let gate = DeepSeekV4MoEGate(configuration: configuration, layer: layer)
        var fixture: [(String, MLXArray)] = [("weight", routingGateWeight())]
        if configuration.isHashLayer(layer) {
            fixture.append(("tid2eid", hashTableArray()))
        } else {
            fixture.append(("bias", array(bias, [routedExpertCount])))
        }
        try gate.update(parameters: ModuleParameters.unflattened(fixture), verify: [.all])
        return gate
    }

    // MARK: - Comparison helpers

    /// Reads an array back as a flat list of `Float`, in row-major order.
    private func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// Reads an array back as a flat list of `Int32`, in row-major order.
    private func integers(_ array: MLXArray) -> [Int32] {
        array.asType(.int32).asArray(Int32.self)
    }

    /// Checks every element of `got` against `expected`.
    private func expectClose(
        _ got: MLXArray, _ expected: [Float], _ what: String,
        limit: Float = tolerance, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let values = floats(got)
        #expect(values.count == expected.count, "\(what): length", sourceLocation: sourceLocation)
        let worst = zip(values, expected).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 0
        #expect(
            worst <= limit,
            "\(what): got \(values), expected \(expected), gap \(worst)",
            sourceLocation: sourceLocation)
    }

    /// The routed experts and the routing weights of one token, ordered by
    /// the expert index, so that a test never depends on the order
    /// `argPartition` answers in.
    private func routesByExpert(_ indices: MLXArray, _ weights: MLXArray) -> [(Int32, Float)] {
        zip(integers(indices), floats(weights)).sorted { $0.0 < $1.0 }
    }

    // MARK: - The gate: the bias picks, the unbiased score weighs

    @Test
    func gateWeightsAreTheUnbiasedScoresOfTheBiasedSelection() throws {
        // The bias lifts expert 0 far over every other expert, thus the
        // selection moves from the two highest scores, experts 6 and 7, to
        // experts 0 and 7. The WEIGHTS must stay the unbiased scores: a gate
        // that gathered the biased score would answer 10.8325546112 for
        // expert 0 in place of 0.8325546112.
        let bias = [Float(10), 0, 0, 0, 0, 0, 0, 0]
        let gate = try Self.routingGate(
            layer: Self.topKLayer, bias: bias, configuration: Self.configuration())

        let (indices, weights) = gate(Self.routingInput(), inputIds: MLXArray([Int32(0)], [1, 1]))
        let routes = routesByExpert(indices, weights)

        #expect(routes.map(\.0) == [0, 7])
        expectClose(
            MLXArray(routes.map(\.1)),
            [Self.routingScores[0], Self.routingScores[7]],
            "the gathered weights are the unbiased scores")
    }

    @Test
    func gateNormalizesAndScalesTheSelectedWeights() throws {
        // `weights / (weights.sum() + 1e-20) * routed_scaling_factor` on the
        // same two selected experts.
        let bias = [Float(10), 0, 0, 0, 0, 0, 0, 0]
        let configuration = try Self.configuration(
            normalizeTopkProb: true, routedScalingFactor: 1.5)
        let gate = try Self.routingGate(
            layer: Self.topKLayer, bias: bias, configuration: configuration)

        let (indices, weights) = gate(Self.routingInput(), inputIds: MLXArray([Int32(0)], [1, 1]))
        let routes = routesByExpert(indices, weights)

        #expect(routes.map(\.0) == [0, 7])
        expectClose(
            MLXArray(routes.map(\.1)), [0.4605996400, 1.0394003600],
            "the normalized and scaled weights")
    }

    @Test
    func gateAnswersOneWeightForEachSelectedExpert() throws {
        let gate = try Self.routingGate(
            layer: Self.topKLayer,
            bias: [Float](repeating: 0, count: Self.routedExpertCount),
            configuration: try Self.configuration())

        let tokens = MLXArray([Int32(0), 1, 2], [1, 3])
        let input = MLXRandom.uniform(
            low: -1, high: 1, [1, 3, Self.hiddenSize], key: MLXRandom.key(7))
        let (indices, weights) = gate(input, inputIds: tokens)

        #expect(indices.shape == [1, 3, Self.expertsPerToken])
        #expect(weights.shape == [1, 3, Self.expertsPerToken])
        for token in 0 ..< 3 {
            let selected = integers(indices[0, token])
            let selectedWeights = floats(weights[0, token])
            let everyWeightIsPositive = selectedWeights.allSatisfy { $0 > 0 }
            #expect(Set(selected).count == Self.expertsPerToken, "token \(token) repeats an expert")
            #expect(everyWeightIsPositive, "token \(token) holds a weight of zero or less")
        }
    }

    // MARK: - Hash routing

    @Test
    func hashLayerRoutesEachTokenToItsTableRow() throws {
        let gate = try Self.routingGate(
            layer: Self.hashLayer,
            bias: [Float](repeating: 0, count: Self.routedExpertCount),
            configuration: try Self.configuration())
        let tokens = MLXArray([Int32(1), 4, 1], [1, 3])

        let (first, _) = gate(
            MLXRandom.uniform(low: -1, high: 1, [1, 3, Self.hiddenSize], key: MLXRandom.key(11)),
            inputIds: tokens)
        let (second, _) = gate(
            MLXRandom.uniform(low: -4, high: 4, [1, 3, Self.hiddenSize], key: MLXRandom.key(12)),
            inputIds: tokens)

        // Rows 1, 4 and 1 of the table, whatever the hidden state holds.
        #expect(integers(first) == [1, 4, 2, 5, 1, 4])
        #expect(integers(second) == integers(first))
    }

    @Test
    func hashLayerWeighsItsExpertsByTheirOwnScores() throws {
        // The Python gathers the per-token gate scores at the hashed expert
        // ids. A gate that answered one synthetic weight for each route would
        // give the same number twice.
        let gate = try Self.routingGate(
            layer: Self.hashLayer,
            bias: [Float](repeating: 0, count: Self.routedExpertCount),
            configuration: try Self.configuration())

        let (indices, weights) = gate(Self.routingInput(), inputIds: MLXArray([Int32(0)], [1, 1]))

        // Row 0 of the table names experts 5 and 2.
        #expect(integers(indices) == [5, 2])
        expectClose(
            weights, [Self.routingScores[5], Self.routingScores[2]],
            "the hashed experts take their own scores")
    }

    @Test
    func topKLayerSelectsTheHighestScoresRatherThanTheHashTable() throws {
        // Row 0 of the hash table names experts 5 and 2. The two highest
        // scores are experts 6 and 7. A layer past the hash layers must
        // answer 6 and 7.
        let gate = try Self.routingGate(
            layer: Self.topKLayer,
            bias: [Float](repeating: 0, count: Self.routedExpertCount),
            configuration: try Self.configuration())

        let (indices, _) = gate(Self.routingInput(), inputIds: MLXArray([Int32(0)], [1, 1]))

        #expect(integers(indices).sorted() == [6, 7])
    }

    // MARK: - The asymmetric SwiGLU clamp

    /// A dense MLP that answers `clampedSwiGLU(gate: x[0], up: x[1])` in its
    /// first output, thus a test can drive the two projections apart.
    private static func splitProjectionMLP() throws -> DeepSeekV4MLP {
        let mlp = DeepSeekV4MLP(hiddenSize: 2, intermediateSize: 1, swigluLimit: swigluLimit)
        try mlp.update(
            parameters: ModuleParameters.unflattened([
                ("gate_proj.weight", array([1, 0], [1, 2])),
                ("up_proj.weight", array([0, 1], [1, 2])),
                ("down_proj.weight", array([1, 0], [2, 1])),
            ]),
            verify: [])
        return mlp
    }

    /// Runs ``splitProjectionMLP()`` and reads back the clamped activation.
    private func activation(_ mlp: DeepSeekV4MLP, gate: Float, up: Float) -> Float {
        floats(mlp(Self.array([gate, up], [1, 1, 2])))[0]
    }

    @Test
    func theClampHoldsTheHighSideOfTheGate() throws {
        let mlp = try Self.splitProjectionMLP()

        #expect(activation(mlp, gate: 50, up: 1) == activation(mlp, gate: 10, up: 1))
        expectClose(
            MLXArray([activation(mlp, gate: 50, up: 1)]), [9.999546021313],
            "a gate over the limit gives silu(limit)")
    }

    @Test
    func theClampHoldsBothSidesOfTheUpProjection() throws {
        let mlp = try Self.splitProjectionMLP()

        #expect(activation(mlp, gate: 1, up: 50) == activation(mlp, gate: 1, up: 10))
        #expect(activation(mlp, gate: 1, up: -50) == activation(mlp, gate: 1, up: -10))
        expectClose(
            MLXArray([activation(mlp, gate: 1, up: -50)]), [-7.310585786300],
            "an up under the limit gives silu(gate) * -limit")
    }

    @Test
    func theClampLeavesTheLowSideOfTheGateAlone() throws {
        // `silu` already falls to almost nothing at a large negative gate,
        // thus the reference clamps only the high side. A gate of -50 and a
        // gate of -10 give two different numbers, and a symmetric clamp would
        // make them one.
        let mlp = try Self.splitProjectionMLP()

        let deep = activation(mlp, gate: -50, up: 1)
        let shallow = activation(mlp, gate: -10, up: 1)

        #expect(deep != shallow)
        expectClose(MLXArray([deep]), [-9.643749239820e-21], "silu(-50)")
        expectClose(MLXArray([shallow]), [-4.539786870243e-04], "silu(-10)")
    }

    // MARK: - The routed experts

    /// A `switch_mlp` whose every expert answers
    /// `clampedSwiGLU(gate: x[0], up: x[1])` in its first output.
    private static func splitProjectionExperts() throws -> DeepSeekV4SwitchGLU {
        let experts = DeepSeekV4SwitchGLU(
            inputDims: 2, hiddenDims: 1, expertCount: 2, swigluLimit: swigluLimit)
        try experts.update(
            parameters: ModuleParameters.unflattened([
                ("gate_proj.weight", array([1, 0, 1, 0], [2, 1, 2])),
                ("up_proj.weight", array([0, 1, 0, 1], [2, 1, 2])),
                ("down_proj.weight", array([1, 0, 1, 0], [2, 2, 1])),
            ]),
            verify: [])
        return experts
    }

    @Test
    func theRoutedExpertsReadTheClampOfTheCheckpoint() throws {
        // The routed path takes its limit from the same `swiglu_limit`. A
        // routed path built with no limit would answer 50 here.
        let experts = try Self.splitProjectionExperts()
        let indices = MLXArray([Int32(0)], [1, 1, 1])

        let saturated = experts(Self.array([50, 1], [1, 1, 2]), indices)
        let atTheLimit = experts(Self.array([10, 1], [1, 1, 2]), indices)

        #expect(floats(saturated) == floats(atTheLimit))
        expectClose(saturated[0, 0, 0, 0], [9.999546021313], "a routed gate over the limit")
    }

    @Test
    func theSortedRoutingPathAgreesWithTheUnsortedPath() throws {
        // `gatherSort` runs only once the route count reaches 64. A block of
        // 32 tokens routed to 2 experts each is exactly 64 routes and takes
        // the sorted path; the same tokens fed one at a time take the
        // unsorted path. A wrong `scatterUnsort` gives one token the answer
        // of another, and the two paths then disagree.
        let tokenCount = 32
        let experts = DeepSeekV4SwitchGLU(
            inputDims: Self.hiddenSize, hiddenDims: Self.expertWidth,
            expertCount: Self.routedExpertCount, swigluLimit: Self.swigluLimit)
        try experts.update(
            parameters: ModuleParameters.unflattened([
                (
                    "gate_proj.weight",
                    Self.randomArray(
                        [Self.routedExpertCount, Self.expertWidth, Self.hiddenSize], seed: 21)
                ),
                (
                    "up_proj.weight",
                    Self.randomArray(
                        [Self.routedExpertCount, Self.expertWidth, Self.hiddenSize], seed: 22)
                ),
                (
                    "down_proj.weight",
                    Self.randomArray(
                        [Self.routedExpertCount, Self.hiddenSize, Self.expertWidth], seed: 23)
                ),
            ]),
            verify: [])

        let input = Self.randomArray([1, tokenCount, Self.hiddenSize], seed: 24)
        let indices =
            (MLXRandom.randInt(
                0 ..< Self.routedExpertCount,
                [1, tokenCount, Self.expertsPerToken], key: MLXRandom.key(25))).asType(.uint32)

        let sorted = experts(input, indices)
        #expect(sorted.shape == [1, tokenCount, Self.expertsPerToken, Self.hiddenSize])

        for token in 0 ..< tokenCount {
            let one = experts(
                input[0..., token ..< (token + 1), 0...],
                indices[0..., token ..< (token + 1), 0...])
            expectClose(one.flattened(), floats(sorted[0, token].flattened()), "token \(token)")
        }
    }

    // MARK: - The shared expert

    /// A mixture-of-experts layer whose routed path answers zero, and whose
    /// shared expert answers `clampedSwiGLU(gate: x[0], up: x[1])` in its
    /// first output. The routed `down_proj` holds zeros, thus the layer
    /// output IS the shared-expert output.
    private static func sharedExpertOnlyLayer() throws -> DeepSeekV4MoE {
        let configuration = try configuration(hiddenSize: 2, expertWidth: 1)
        let moe = DeepSeekV4MoE(configuration: configuration, layer: topKLayer)
        try moe.update(
            parameters: ModuleParameters.unflattened([
                ("gate.weight", zeros([routedExpertCount, 2])),
                ("gate.bias", zeros([routedExpertCount])),
                ("switch_mlp.gate_proj.weight", zeros([routedExpertCount, 1, 2])),
                ("switch_mlp.up_proj.weight", zeros([routedExpertCount, 1, 2])),
                ("switch_mlp.down_proj.weight", zeros([routedExpertCount, 2, 1])),
                ("shared_experts.gate_proj.weight", array([1, 0], [1, 2])),
                ("shared_experts.up_proj.weight", array([0, 1], [1, 2])),
                ("shared_experts.down_proj.weight", array([1, 0], [2, 1])),
            ]),
            verify: [.all])
        return moe
    }

    @Test
    func theSharedExpertReadsTheClampOfTheCheckpoint() throws {
        // The training-time reference of the checkpoint decides this point.
        // The checkpoint binds to the `deepseek_v4` model of Hugging Face
        // `transformers`, whose `DeepseekV4SparseMoeBlock.__init__` builds
        // the shared expert as `DeepseekV4MLP(config)`, and that MLP clamps
        // with `config.swiglu_limit`. A shared expert built with a limit of
        // zero would answer silu(50), which is almost 50, in place of
        // silu(10) here. Task `^kp1pnj4` records the decision.
        let moe = try Self.sharedExpertOnlyLayer()
        let tokens = MLXArray([Int32(0)], [1, 1])

        let saturated = moe(Self.array([50, 1], [1, 1, 2]), inputIds: tokens)
        let atTheLimit = moe(Self.array([10, 1], [1, 1, 2]), inputIds: tokens)

        #expect(floats(saturated) == floats(atTheLimit))
        expectClose(saturated[0, 0, 0], [9.999546021313], "a shared gate over the limit")
    }

    // MARK: - The whole layer

    /// Builds a whole mixture-of-experts layer with repeatable random
    /// weights.
    ///
    /// The gate of a hash layer declares the hash table alone and the gate of
    /// every later layer declares the routing bias alone, thus the fixture
    /// carries the one parameter the layer holds. The load verifies with
    /// `[.all]`, which is the verification `MLXLMCommon.loadWeights` applies.
    ///
    /// - Parameters:
    ///   - layer: The index of the decoder layer.
    ///   - sharedScale: The factor every shared-expert weight takes. A scale
    ///     of zero turns the shared expert off.
    /// - Returns: The layer.
    private static func randomLayer(layer: Int, sharedScale: Float = 1) throws -> DeepSeekV4MoE {
        let checkpointConfiguration = try configuration()
        let moe = DeepSeekV4MoE(configuration: checkpointConfiguration, layer: layer)
        var fixture: [(String, MLXArray)] = [
            ("gate.weight", randomArray([routedExpertCount, hiddenSize], seed: 31))
        ]
        if checkpointConfiguration.isHashLayer(layer) {
            fixture.append(("gate.tid2eid", hashTableArray()))
        } else {
            fixture.append(("gate.bias", randomArray([routedExpertCount], seed: 32)))
        }
        fixture.append(
            contentsOf: [
                (
                    "switch_mlp.gate_proj.weight",
                    randomArray([routedExpertCount, expertWidth, hiddenSize], seed: 33)
                ),
                (
                    "switch_mlp.up_proj.weight",
                    randomArray([routedExpertCount, expertWidth, hiddenSize], seed: 34)
                ),
                (
                    "switch_mlp.down_proj.weight",
                    randomArray([routedExpertCount, hiddenSize, expertWidth], seed: 35)
                ),
                (
                    "shared_experts.gate_proj.weight",
                    randomArray([expertWidth, hiddenSize], seed: 36) * sharedScale
                ),
                (
                    "shared_experts.up_proj.weight",
                    randomArray([expertWidth, hiddenSize], seed: 37) * sharedScale
                ),
                (
                    "shared_experts.down_proj.weight",
                    randomArray([hiddenSize, expertWidth], seed: 38) * sharedScale
                ),
            ])
        try moe.update(parameters: ModuleParameters.unflattened(fixture), verify: [.all])
        return moe
    }

    @Test
    func theLayerKeepsTheShapeOfItsInput() throws {
        let moe = try Self.randomLayer(layer: Self.topKLayer)
        let input = Self.randomArray([1, 4, Self.hiddenSize], seed: 41)
        let tokens = MLXArray([Int32(0), 1, 2, 3], [1, 4])

        let output = moe(input, inputIds: tokens)

        let everyValueIsFinite = floats(output).allSatisfy { $0.isFinite }
        #expect(output.shape == input.shape)
        #expect(everyValueIsFinite)
    }

    @Test
    func theSharedExpertAddsToTheRoutedResult() throws {
        let input = Self.randomArray([1, 4, Self.hiddenSize], seed: 41)
        let tokens = MLXArray([Int32(0), 1, 2, 3], [1, 4])

        let routedOnly = try Self.randomLayer(layer: Self.topKLayer, sharedScale: 0)
        let withShared = try Self.randomLayer(layer: Self.topKLayer, sharedScale: 1)

        let difference = floats(
            withShared(input, inputIds: tokens) - routedOnly(input, inputIds: tokens))
        let sharedExpertMovedTheOutput = difference.contains { abs($0) > 1e-3 }

        #expect(sharedExpertMovedTheOutput, "the shared expert changed nothing")
    }

    @Test
    func theRoutedReductionRunsInFloat32() throws {
        // Two experts of width 1 over a residual stream of width 1. The two
        // experts hold the same projections apart from the sign of
        // `down_proj`, thus their two weighted outputs almost cancel. The
        // router logits 0 and 0.0078125 are each exact in bfloat16 and give
        // the routing weights 0.8325546112 and 0.8349018265.
        //
        // `clampedSwiGLU` answers in the dtype of its gate, thus one expert
        // output is the bfloat16 number 0.73046875. The two weighted outputs
        // are then 0.6081551261 and -0.6098696936.
        //
        // A float32 sum gives -0.001714567475, which rounds to the bfloat16
        // number -0.001716613770. A sum in bfloat16 rounds each weighted
        // output to 0.609375 first and answers exactly zero.
        let configuration = try Self.configuration(
            hashLayers: 0, routedExperts: 2, expertsPerToken: 2, hiddenSize: 1, expertWidth: 1)
        let moe = DeepSeekV4MoE(configuration: configuration, layer: 0)
        func bfloat16(_ values: [Float], _ shape: [Int]) -> MLXArray {
            Self.array(values, shape).asType(.bfloat16)
        }
        try moe.update(
            parameters: ModuleParameters.unflattened([
                ("gate.weight", bfloat16([0, 0.0078125], [2, 1])),
                ("gate.bias", bfloat16([0, 0], [2])),
                ("switch_mlp.gate_proj.weight", bfloat16([1, 1], [2, 1, 1])),
                ("switch_mlp.up_proj.weight", bfloat16([1, 1], [2, 1, 1])),
                ("switch_mlp.down_proj.weight", bfloat16([1, -1], [2, 1, 1])),
                ("shared_experts.gate_proj.weight", bfloat16([0], [1, 1])),
                ("shared_experts.up_proj.weight", bfloat16([0], [1, 1])),
                ("shared_experts.down_proj.weight", bfloat16([0], [1, 1])),
            ]),
            verify: [])

        let output = moe(bfloat16([1], [1, 1, 1]), inputIds: MLXArray([Int32(0)], [1, 1]))
        let value = floats(output)[0]

        #expect(abs(value) > 1e-4, "the reduction collapsed to zero, thus it ran in bfloat16")
        #expect(
            abs(value - -0.001716613770) <= 2e-5,
            "got \(value), expected -0.001716613770")
    }
}

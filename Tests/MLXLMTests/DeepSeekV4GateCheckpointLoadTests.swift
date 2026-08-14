// Copyright © 2026 Apple Inc.
//
// Regression: the DeepSeek-V4 routing gate declared BOTH `tid2eid` and `bias`
// on every layer, thus no published checkpoint could load.
//
// The DeepSeek-V4 Python reference -- `Thump604/mlx-lm` @
// `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py`,
// `MoEGate.__init__` -- declares one of the two on each layer and never both:
// a hash layer holds `tid2eid`, and every later layer holds
// `e_score_correction_bias`. The published checkpoint carries exactly that
// set: `ffn.gate.tid2eid` on layers 0 to `num_hash_layers - 1`, and
// `ffn.gate.bias` on every layer after them.
//
// `MLXLMCommon.loadWeights` loads with `model.update(parameters:verify:
// [.all])`. That verification throws `UpdateError.keyNotFound` for a module
// parameter no weight fills, and it throws for a weight no module parameter
// takes. A gate that declares both parameters fails the first way on every
// real checkpoint, and a gate that declares neither fails the second way.
//
// The checkpoint below states the routing tensors BY RULE rather than reading
// them out of the module tree, thus it says what a real checkpoint carries and
// does not follow the tree it loads into. Every other tensor comes from a
// donor model, as `GLM4LmHeadTiedLoadTests` builds its checkpoint.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4GateCheckpointLoadTests {

    // MARK: - The synthetic checkpoint

    /// The number of tokens in the vocabulary of the synthetic checkpoint.
    private static let vocabSize = 12

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 16

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 4

    /// The number of first layers that route through the hash table. It is
    /// more than one and fewer than ``layerCount``, thus the checkpoint holds
    /// both kinds of layer.
    private static let hashLayerCount = 2

    /// The index of a layer that routes through the hash table.
    private static let hashLayer = 0

    /// The index of a layer that routes through the top-k gate.
    private static let topKLayer = hashLayerCount

    /// The number of routed experts of the synthetic checkpoint.
    private static let routedExpertCount = 8

    /// The number of routed experts one token reads.
    private static let expertsPerToken = 2

    /// The number of tokens each forward pass below carries.
    private static let tokenCount = 5

    /// The batch of each forward pass below.
    private static let batchSize = 1

    /// The seed the weight filler starts at.
    private static let weightSeed: UInt64 = 20_260_812

    /// The low end of every random weight.
    private static let weightLow: Float = -0.5

    /// The high end of every random weight.
    private static let weightHigh: Float = 0.5

    /// The path suffix of the routing bias of one layer.
    private static let biasSuffix = "ffn.gate.bias"

    /// The path suffix of the hash table of one layer.
    private static let hashTableSuffix = "ffn.gate.tid2eid"

    /// The `config.json` of the synthetic checkpoint.
    private static func configuration() throws -> DeepSeekV4Configuration {
        let json = """
            {
              "vocab_size": \(vocabSize),
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(layerCount),
              "num_attention_heads": 4,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "qk_rope_head_dim": 4,
              "q_lora_rank": 8,
              "rms_norm_eps": 1e-6,
              "max_position_embeddings": 64,
              "o_groups": 2,
              "o_lora_rank": 4,
              "n_routed_experts": \(routedExpertCount),
              "n_shared_experts": 1,
              "num_experts_per_tok": \(expertsPerToken),
              "moe_intermediate_size": 8,
              "num_hash_layers": \(hashLayerCount),
              "norm_topk_prob": true,
              "routed_scaling_factor": 1.0,
              "swiglu_limit": 10.0,
              "hc_mult": 2,
              "hc_sinkhorn_iters": 4,
              "hc_eps": 1e-6,
              "rope_theta": 10000.0,
              "compress_ratios": [],
              "use_attn_sink": true,
              "tie_word_embeddings": false
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    /// The expert the hash table names for one token on one route.
    ///
    /// - Parameters:
    ///   - token: The token identifier.
    ///   - route: The route of that token.
    /// - Returns: The expert identifier.
    private static func hashedExpert(token: Int, route: Int) -> Int32 {
        Int32((token + route) % routedExpertCount)
    }

    /// The hash table of one layer, shape `(vocabSize, expertsPerToken)`.
    private static func hashTable() -> MLXArray {
        var values = [Int32]()
        for token in 0 ..< vocabSize {
            for route in 0 ..< expertsPerToken {
                values.append(hashedExpert(token: token, route: route))
            }
        }
        return MLXArray(values).reshaped([vocabSize, expertsPerToken])
    }

    /// Tells whether one parameter path names a routing tensor of a gate.
    private static func isRoutingTensor(_ path: String) -> Bool {
        path.hasSuffix(biasSuffix) || path.hasSuffix(hashTableSuffix)
    }

    /// Builds the checkpoint a published DeepSeek-V4 conversion carries.
    ///
    /// Every tensor that is not a routing tensor comes from a donor model, and
    /// the routing tensors are stated by rule: a hash layer carries the hash
    /// table alone, and every later layer carries the routing bias alone.
    ///
    /// - Returns: The tensors, by module path.
    private static func publishedCheckpoint() throws -> [String: MLXArray] {
        let donor = DeepSeekV4Model(try configuration())
        var checkpoint = [String: MLXArray]()
        var seed = weightSeed
        for (path, value) in donor.parameters().flattened().sorted(by: { $0.0 < $1.0 })
        where !isRoutingTensor(path) {
            seed += 1
            checkpoint[path] = MLXRandom.uniform(
                low: weightLow, high: weightHigh, value.shape, key: MLXRandom.key(seed))
        }
        for layer in 0 ..< layerCount {
            let prefix = "model.layers.\(layer)."
            if layer < hashLayerCount {
                checkpoint[prefix + hashTableSuffix] = hashTable()
            } else {
                checkpoint[prefix + biasSuffix] = MLXRandom.uniform(
                    low: weightLow, high: weightHigh, [routedExpertCount],
                    key: MLXRandom.key(weightSeed))
            }
        }
        return checkpoint
    }

    /// A run of token identifiers, shape `(batch, tokens)`.
    private static func tokens() -> MLXArray {
        MLXArray((0 ..< tokenCount).map { Int32($0 % vocabSize) })
            .reshaped([batchSize, tokenCount])
    }

    /// A repeatable random hidden state, shape `(batch, tokens, hiddenSize)`.
    private static func hiddenState(seed: UInt64) -> MLXArray {
        MLXRandom.uniform(
            low: weightLow, high: weightHigh, [batchSize, tokenCount, hiddenSize],
            key: MLXRandom.key(seed))
    }

    // MARK: - The load

    /// The shape of the production failure: `loadWeights` verifies with
    /// `[.all]`, and the published checkpoint carries `tid2eid` on the hash
    /// layers alone and `bias` on the later layers alone. Before the fix this
    /// threw `keyNotFound`, because every gate declared both.
    @Test("A checkpoint that splits the routing tensors by layer loads")
    func aCheckpointThatSplitsTheRoutingTensorsByLayerLoads() throws {
        let model = DeepSeekV4Model(try Self.configuration())

        try model.update(
            parameters: ModuleParameters.unflattened(try Self.publishedCheckpoint()),
            verify: [.all])
        eval(model)
    }

    @Test("A hash gate declares the table alone and a top-k gate the bias alone")
    func eachGateDeclaresTheRoutingTensorOfItsOwnKind() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let paths = Set(model.parameters().flattened().map(\.0))

        for layer in 0 ..< Self.layerCount {
            let prefix = "model.layers.\(layer)."
            let hasTable = paths.contains(prefix + Self.hashTableSuffix)
            let hasBias = paths.contains(prefix + Self.biasSuffix)
            let isHashLayer = layer < Self.hashLayerCount
            #expect(hasTable == isHashLayer, "layer \(layer) declares the hash table")
            #expect(hasBias == !isHashLayer, "layer \(layer) declares the routing bias")
        }
    }

    // MARK: - The routing after the load

    @Test("The loaded hash layer routes each token to its own table row")
    func theLoadedHashLayerRoutesEachTokenToItsTableRow() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        try model.update(
            parameters: ModuleParameters.unflattened(try Self.publishedCheckpoint()),
            verify: [.all])
        let gate = model.model.layers[Self.hashLayer].mixtureOfExperts.gate

        let inputIds = Self.tokens()
        let (indices, _) = gate(Self.hiddenState(seed: 101), inputIds: inputIds)
        eval(indices)

        var expected = [Int32]()
        for token in 0 ..< Self.tokenCount {
            for route in 0 ..< Self.expertsPerToken {
                expected.append(
                    Self.hashedExpert(token: token % Self.vocabSize, route: route))
            }
        }
        #expect(indices.asType(.int32).asArray(Int32.self) == expected)
    }

    @Test("The loaded top-k layer selects the highest biased scores")
    func theLoadedTopKLayerSelectsTheHighestBiasedScores() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        try model.update(
            parameters: ModuleParameters.unflattened(try Self.publishedCheckpoint()),
            verify: [.all])
        let gate = model.model.layers[Self.topKLayer].mixtureOfExperts.gate

        // A top-k gate reads the hidden state, thus two different states give
        // two different selections. A gate that read a hash table instead
        // would answer the same experts for both.
        let inputIds = Self.tokens()
        let (first, _) = gate(Self.hiddenState(seed: 201), inputIds: inputIds)
        let (second, _) = gate(Self.hiddenState(seed: 202), inputIds: inputIds)
        eval(first, second)

        #expect(
            first.asType(.int32).asArray(Int32.self)
                != second.asType(.int32).asArray(Int32.self),
            "the selection must follow the hidden state")
    }
}

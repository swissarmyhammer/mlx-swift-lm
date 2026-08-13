// Copyright © 2026 Apple Inc.
//
// Tests for the DeepSeek-V4 decoder layer, the decoder stack, and the
// weight-load filter.
//
// Every piece these tests assemble already carries its own parity tests
// against a NumPy transcription of the DeepSeek-V4 Python reference. This file
// tests the WIRING those parity tests cannot see: the order of the two halves
// of a decoder layer, the manifold expand each half writes back through, the
// reduction at the top of the stack, the module path of each tensor, and the
// key map the load filter applies.
//
// The order of the two halves comes from the Python reference,
// `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
// `mlx_lm/models/deepseek_v4.py`, `DeepseekV4Block.__call__`: the attention
// half runs first, and each half reads the stream the half before it wrote.
// `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` states that order
// on its own, out of the pieces, thus a production file that swapped the two
// halves stops agreeing with it.
//
// The key map comes from the same file, `Model.sanitize`, and from the
// `quantization` block of the published `DeepSeek-V4-Flash-4bit` checkpoint,
// which `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` holds.
// That block names checkpoint key paths, and `quantize(model:filter:)` hands
// its filter the flattened module path, thus the two must agree.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4ModelTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - The synthetic checkpoint

    /// The number of tokens in the vocabulary of the synthetic checkpoint.
    private static let vocabSize = 12

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 16

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 4

    /// The number of routed experts of the synthetic checkpoint.
    private static let routedExpertCount = 8

    /// The number of routed experts one token reads.
    private static let expertsPerToken = 2

    /// The number of parallel copies of the residual stream.
    private static let hcMult = 2

    /// The axis of a `(batch, tokens, copies, width)` residual stream that
    /// holds the parallel copies.
    private static let copyAxis = 2

    /// The number of first layers that route through the hash table.
    private static let hashLayerCount = 1

    /// The index of a layer that routes through the hash table.
    private static let hashLayer = 0

    /// The index of a layer that routes through the top-k gate. It is the
    /// first layer past the hash layers.
    private static let topKLayer = hashLayerCount

    /// The number of tokens one prefill block carries.
    private static let promptLength = 8

    /// The number of decode steps ``decodeAdvancesTheCacheOneStepForEachToken``
    /// takes.
    private static let decodeStepCount = 3

    /// The batch of every forward pass below.
    private static let batchSize = 1

    /// The largest gap allowed between two float32 results of the same
    /// arithmetic in a different order.
    private static let tolerance: Float = 1e-5

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// `compress_ratios` is empty, thus no layer holds a compressor and every
    /// layer turns its positions with `rope_theta`. The compressor and the
    /// indexer are out of scope for this file, and
    /// `DeepSeekV4CompressorTests` covers them.
    ///
    /// - Parameter tieWordEmbeddings: True when the checkpoint ties the
    ///   language-model head to the embedding table.
    /// - Returns: The decoded configuration.
    private static func configuration(
        tieWordEmbeddings: Bool = false
    ) throws -> DeepSeekV4Configuration {
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
              "hc_mult": \(hcMult),
              "hc_sinkhorn_iters": 4,
              "hc_eps": 1e-6,
              "rope_theta": 10000.0,
              "compress_ratios": [],
              "use_attn_sink": true,
              "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Fixture builders

    /// The low end of every random weight.
    private static let weightLow: Float = -0.5

    /// The high end of every random weight.
    private static let weightHigh: Float = 0.5

    /// The seed the weight filler starts at.
    private static let weightSeed: UInt64 = 20_260_811

    /// The suffix of the hash table each mixture gate holds.
    private static let hashTableSuffix = "tid2eid"

    /// Builds a model of the synthetic configuration and loads a repeatable
    /// random weight into every parameter it declares.
    ///
    /// The load runs through `update(parameters:verify:[.all])`, which is the
    /// verification `MLXLMCommon.loadWeights` applies, thus a module tree that
    /// declared a parameter no checkpoint could fill would fail here.
    ///
    /// Every hash table takes expert identifiers rather than a random value,
    /// because a hash layer reads its table as an index.
    ///
    /// - Parameter tieWordEmbeddings: True when the checkpoint ties the
    ///   language-model head to the embedding table.
    /// - Returns: The loaded model.
    private static func loadedModel(
        tieWordEmbeddings: Bool = false
    ) throws -> DeepSeekV4Model {
        let model = DeepSeekV4Model(try configuration(tieWordEmbeddings: tieWordEmbeddings))
        var checkpoint: [String: MLXArray] = [:]
        var seed = weightSeed
        for (key, value) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            seed += 1
            checkpoint[key] =
                key.hasSuffix(hashTableSuffix)
                ? hashTable(shape: value.shape)
                : MLXRandom.uniform(
                    low: weightLow, high: weightHigh, value.shape, key: MLXRandom.key(seed))
        }
        try model.update(parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
        return model
    }

    /// Builds a hash table of the given shape whose every value names a routed
    /// expert of the synthetic checkpoint.
    private static func hashTable(shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { Int32($0 % routedExpertCount) }).reshaped(shape)
    }

    /// A run of token identifiers of the given length.
    private static func tokens(count: Int, from first: Int = 0) -> MLXArray {
        MLXArray((0 ..< count).map { Int32((first + $0) % vocabSize) })
            .reshaped([batchSize, count])
    }

    /// A repeatable random residual stream, shape
    /// `(batch, tokens, hcMult, hiddenSize)`.
    private static func residualStream(tokenCount: Int, seed: UInt64) -> MLXArray {
        MLXRandom.uniform(
            low: weightLow, high: weightHigh,
            [batchSize, tokenCount, hcMult, hiddenSize],
            key: MLXRandom.key(seed))
    }

    /// The values of an array, in row-major order.
    private static func floats(_ array: MLXArray) -> [Float] {
        eval(array)
        return array.asType(.float32).reshaped(-1).asArray(Float.self)
    }

    // MARK: - Module paths

    @Test func theModulePathsAreTheCheckpointKeyPaths() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let modulePaths = Set(model.leafModules().flattened().map(\.0))
        let parameterPaths = Set(model.parameters().flattened().map(\.0))

        // The `quantization` block of the published checkpoint names these
        // module paths, and `quantize(model:filter:)` matches against the
        // flattened module path. `DeepSeekV4QuantizationPlanTests` pins the
        // same names from the plan's side.
        for path in [
            "model.embed_tokens",
            "lm_head",
            "model.layers.0.attn.wq_a",
            "model.layers.0.attn.wq_b",
            "model.layers.0.attn.wkv",
            "model.layers.0.attn.wo_a",
            "model.layers.0.attn.wo_b",
            "model.layers.0.ffn.switch_mlp.gate_proj",
            "model.layers.0.ffn.switch_mlp.up_proj",
            "model.layers.0.ffn.switch_mlp.down_proj",
            "model.layers.0.ffn.shared_experts.gate_proj",
        ] {
            #expect(modulePaths.contains(path), "module path \(path)")
        }

        // The manifold tensors and the two norms are parameters of their own
        // rather than leaf modules.
        for path in [
            "model.norm.weight",
            "model.hc_head.fn",
            "model.hc_head.base",
            "model.hc_head.scale",
            "model.layers.0.attn_norm.weight",
            "model.layers.0.ffn_norm.weight",
            "model.layers.0.hc_attn.fn",
            "model.layers.0.hc_attn.base",
            "model.layers.0.hc_attn.scale",
            "model.layers.0.hc_ffn.fn",
            "model.layers.0.hc_ffn.base",
            "model.layers.0.hc_ffn.scale",
            "model.layers.0.attn.attn_sink",
            "model.layers.0.ffn.gate.weight",
            // A hash layer holds the table alone and a later layer holds the
            // bias alone, which is the set the published checkpoint carries.
            "model.layers.\(Self.topKLayer).ffn.gate.bias",
            "model.layers.\(Self.hashLayer).ffn.gate.tid2eid",
        ] {
            #expect(parameterPaths.contains(path), "parameter path \(path)")
        }
    }

    @Test func theStackHoldsOneLayerAndOneCacheEntryForEachConfiguredLayer() throws {
        let model = DeepSeekV4Model(try Self.configuration())

        #expect(model.model.layers.count == Self.layerCount)
        #expect(model.kvHeads.count == Self.layerCount)
        #expect(model.loraLayers.count == Self.layerCount)
        #expect(model.newCache(parameters: nil).count == Self.layerCount)
    }

    // MARK: - The decoder layer

    @Test func theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf() throws {
        let configuration = try Self.configuration()
        let model = try Self.loadedModel()
        let layer = try #require(model.model.layers.first)
        let stream = Self.residualStream(tokenCount: Self.promptLength, seed: 101)
        let inputIds = Self.tokens(count: Self.promptLength)

        // The Python `DeepseekV4Block.__call__`, one line at a time. The
        // attention half reads the stream the caller handed in; the mixture
        // half reads the stream the attention half wrote.
        let attentionCollapse = layer.attentionConnection.collapse(stream)
        let attended = layer.attention(
            layer.attentionNorm(attentionCollapse.collapsed), mask: .causal, cache: nil)
        let afterAttention = layer.attentionConnection.expand(
            blockOutput: attended, residual: stream,
            post: attentionCollapse.post, comb: attentionCollapse.comb)
        let mixtureCollapse = layer.ffnConnection.collapse(afterAttention)
        let mixed = layer.ffn(layer.ffnNorm(mixtureCollapse.collapsed), inputIds: inputIds)
        let expected = layer.ffnConnection.expand(
            blockOutput: mixed, residual: afterAttention,
            post: mixtureCollapse.post, comb: mixtureCollapse.comb)

        let output = layer(stream, mask: .causal, cache: nil, inputIds: inputIds)

        #expect(
            output.shape == [
                Self.batchSize, Self.promptLength, configuration.hcMult, configuration.hiddenSize,
            ])
        let got = Self.floats(output)
        let want = Self.floats(expected)
        #expect(got.count == want.count)
        for (index, value) in got.enumerated() {
            #expect(abs(value - want[index]) <= Self.tolerance, "value \(index)")
        }
    }

    @Test func theDecoderLayerHandsTheTokenIdentifiersToTheHashRoutingGate() throws {
        let model = try Self.loadedModel()
        let layer = model.model.layers[Self.hashLayer]
        #expect(layer.ffn.gate.isHashLayer, "the premise: layer \(Self.hashLayer) routes by hash")

        // One stream, two runs of token identifiers. A hash layer names its
        // experts from the identifiers alone, thus the two answers differ only
        // when the identifiers reach the gate.
        let stream = Self.residualStream(tokenCount: Self.promptLength, seed: 202)
        let first = layer(
            stream, mask: .causal, cache: nil,
            inputIds: Self.tokens(count: Self.promptLength, from: 0))
        let second = layer(
            stream, mask: .causal, cache: nil,
            inputIds: Self.tokens(count: Self.promptLength, from: 1))

        let firstValues = Self.floats(first)
        let secondValues = Self.floats(second)
        #expect(firstValues.count == secondValues.count)
        #expect(
            zip(firstValues, secondValues).contains { abs($0 - $1) > Self.tolerance },
            "the two runs must differ, because the hash table routes by token identifier")
    }

    // MARK: - The decoder stack

    @Test func theStackReducesTheParallelCopiesBeforeTheFinalNorm() throws {
        let configuration = try Self.configuration()
        let model = try Self.loadedModel()
        let inner = model.model
        let inputIds = Self.tokens(count: Self.promptLength)

        var stream = MLX.repeated(
            inner.embedTokens(inputIds).expandedDimensions(axis: Self.copyAxis),
            count: configuration.hcMult, axis: Self.copyAxis)
        for layer in inner.layers {
            stream = layer(stream, mask: .causal, cache: nil, inputIds: inputIds)
        }
        let expected = inner.norm(inner.hcHead(stream))

        let output = inner(inputIds, cache: nil)

        #expect(
            output.shape == [Self.batchSize, Self.promptLength, configuration.hiddenSize],
            "the head reduces the copy axis away before the final norm")
        let got = Self.floats(output)
        let want = Self.floats(expected)
        #expect(got.count == want.count)
        for (index, value) in got.enumerated() {
            #expect(abs(value - want[index]) <= Self.tolerance, "value \(index)")
        }
    }

    @Test func prefillGivesOneLogitRowForEachToken() throws {
        let model = try Self.loadedModel()
        let logits = model(Self.tokens(count: Self.promptLength), cache: nil)

        #expect(logits.shape == [Self.batchSize, Self.promptLength, Self.vocabSize])
        #expect(Self.floats(logits).allSatisfy { $0.isFinite })
    }

    @Test func decodeAdvancesTheCacheOneStepForEachToken() throws {
        let model = try Self.loadedModel()
        let cache = model.newCache(parameters: nil)

        let prefill = model(Self.tokens(count: Self.promptLength), cache: cache)
        eval(prefill)
        #expect(cache[0].offset == Self.promptLength)

        for step in 0 ..< Self.decodeStepCount {
            let logits = model(Self.tokens(count: 1, from: step), cache: cache)
            #expect(logits.shape == [Self.batchSize, 1, Self.vocabSize])
            #expect(Self.floats(logits).allSatisfy { $0.isFinite })
            #expect(cache[0].offset == Self.promptLength + step + 1)
        }

        for entry in cache {
            #expect(entry.offset == Self.promptLength + Self.decodeStepCount)
        }
    }

    // MARK: - Tied word embeddings

    @Test func aTiedCheckpointDeclaresNoLanguageModelHead() throws {
        let tied = DeepSeekV4Model(try Self.configuration(tieWordEmbeddings: true))
        let untied = DeepSeekV4Model(try Self.configuration(tieWordEmbeddings: false))

        #expect(tied.lmHead == nil)
        #expect(untied.lmHead != nil)
        #expect(!tied.parameters().flattened().contains { $0.0.hasPrefix("lm_head.") })
    }

    @Test func aCheckpointThatNamesNoTieBuildsItsOwnLanguageModelHead() throws {
        // `tie_word_embeddings` is absent here, as it is from a `config.json`
        // that states nothing about it. The default must be the untied one,
        // which is what the published DeepSeek-V4-Flash checkpoint states.
        let json = """
            {
              "vocab_size": \(Self.vocabSize),
              "hidden_size": \(Self.hiddenSize),
              "num_hidden_layers": 1
            }
            """
        let configuration = try JSONDecoder().decode(
            DeepSeekV4Configuration.self, from: Data(json.utf8))

        #expect(!configuration.tieWordEmbeddings)
        #expect(DeepSeekV4Model(configuration).lmHead != nil)
    }

    @Test func aTiedCheckpointProjectsThroughItsEmbeddingTable() throws {
        let model = try Self.loadedModel(tieWordEmbeddings: true)
        let inputIds = Self.tokens(count: Self.promptLength)

        let expected = model.model.embedTokens.asLinear(model.model(inputIds, cache: nil))
        let logits = model(inputIds, cache: nil)

        #expect(logits.shape == [Self.batchSize, Self.promptLength, Self.vocabSize])
        let got = Self.floats(logits)
        let want = Self.floats(expected)
        #expect(got.count == want.count)
        for (index, value) in got.enumerated() {
            #expect(abs(value - want[index]) <= Self.tolerance, "value \(index)")
        }
    }

    @Test func anUntiedCheckpointProjectsThroughItsOwnLanguageModelHead() throws {
        let model = try Self.loadedModel(tieWordEmbeddings: false)
        let head = try #require(model.lmHead)
        let inputIds = Self.tokens(count: Self.promptLength)

        let expected = head(model.model(inputIds, cache: nil))
        let logits = model(inputIds, cache: nil)

        let got = Self.floats(logits)
        let want = Self.floats(expected)
        #expect(got.count == want.count)
        for (index, value) in got.enumerated() {
            #expect(abs(value - want[index]) <= Self.tolerance, "value \(index)")
        }
    }

    @Test func aTiedCheckpointLoadsWithoutItsOwnLanguageModelHead() throws {
        // The shape of the production failure this guards: `loadWeights` calls
        // `update(parameters:verify:[.all])`, and a tied checkpoint omits
        // `lm_head.weight` altogether.
        let untied = try Self.loadedModel(tieWordEmbeddings: false)
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in untied.parameters().flattened() where !key.hasPrefix("lm_head.") {
            checkpoint[key] = value
        }

        let tied = DeepSeekV4Model(try Self.configuration(tieWordEmbeddings: true))
        try tied.update(parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(tied)
    }

    // MARK: - The load filter

    /// A one-value array, which stands in for a checkpoint tensor whose values
    /// no test reads.
    private static func marker(_ value: Float) -> MLXArray {
        MLXArray([value])
    }

    @Test func sanitizeDropsTheMultiTokenPredictionHead() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let sanitized = model.sanitize(weights: [
            "mtp.0.attn.wq_a.weight": Self.marker(1),
            "mtp.0.hc_head_fn": Self.marker(2),
            "mtp.0.ffn.experts.0.w1.weight": Self.marker(3),
            "norm.weight": Self.marker(4),
        ])

        #expect(!sanitized.keys.contains { $0.contains("mtp") })
        #expect(sanitized["model.norm.weight"] != nil)
    }

    @Test func sanitizeDropsALayerTheConfigurationDoesNotDeclare() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let beyond = Self.layerCount
        let sanitized = model.sanitize(weights: [
            "layers.\(beyond).attn.wq_a.weight": Self.marker(1),
            "layers.\(Self.layerCount - 1).attn.wq_a.weight": Self.marker(2),
        ])

        #expect(sanitized["model.layers.\(beyond).attn.wq_a.weight"] == nil)
        #expect(sanitized["model.layers.\(Self.layerCount - 1).attn.wq_a.weight"] != nil)
    }

    @Test func sanitizeGivesEveryCheckpointKeyItsModulePath() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let sanitized = model.sanitize(weights: [
            "embed.weight": Self.marker(1),
            "norm.weight": Self.marker(2),
            "head.weight": Self.marker(3),
            "hc_head_fn": Self.marker(4),
            "hc_head_base": Self.marker(5),
            "hc_head_scale": Self.marker(6),
            "layers.1.attn_norm.weight": Self.marker(7),
            "layers.1.ffn_norm.weight": Self.marker(8),
            "layers.1.hc_attn_fn": Self.marker(9),
            "layers.1.hc_ffn_scale": Self.marker(10),
            "layers.1.attn.wq_a.weight": Self.marker(11),
            "layers.1.ffn.gate.weight": Self.marker(12),
            "layers.1.ffn.shared_experts.w1.weight": Self.marker(13),
            "layers.1.ffn.shared_experts.w2.weight": Self.marker(14),
            "layers.1.ffn.shared_experts.w3.weight": Self.marker(15),
        ])

        let expected: [String: Float] = [
            "model.embed_tokens.weight": 1,
            "model.norm.weight": 2,
            "lm_head.weight": 3,
            "model.hc_head.fn": 4,
            "model.hc_head.base": 5,
            "model.hc_head.scale": 6,
            "model.layers.1.attn_norm.weight": 7,
            "model.layers.1.ffn_norm.weight": 8,
            "model.layers.1.hc_attn.fn": 9,
            "model.layers.1.hc_ffn.scale": 10,
            "model.layers.1.attn.wq_a.weight": 11,
            "model.layers.1.ffn.gate.weight": 12,
            "model.layers.1.ffn.shared_experts.gate_proj.weight": 13,
            "model.layers.1.ffn.shared_experts.down_proj.weight": 14,
            "model.layers.1.ffn.shared_experts.up_proj.weight": 15,
        ]
        #expect(Set(sanitized.keys) == Set(expected.keys))
        for (key, value) in expected {
            let tensor = try #require(sanitized[key], Comment(rawValue: key))
            #expect(Self.floats(tensor) == [value], Comment(rawValue: key))
        }
    }

    @Test func sanitizeConvergesBothHyperConnectionSpellings() throws {
        // The mlx-community `DeepSeek-V4-Flash-4bit` checkpoint spells each
        // per-layer hyper-connection `<half>_hc.<field>` and writes the
        // `model.` prefix, where the raw checkpoint spells it
        // `hc_<half>_<field>` without the prefix. The Python reference,
        // `ml-explore/mlx-lm` PR 1189 `Model.sanitize`, sends the two
        // spellings to the one module path `hc_<half>.<field>`.
        let model = DeepSeekV4Model(try Self.configuration())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.attn_hc.base": Self.marker(1),
            "model.layers.0.attn_hc.fn": Self.marker(2),
            "model.layers.0.attn_hc.scale": Self.marker(3),
            "model.layers.2.ffn_hc.base": Self.marker(4),
            "model.layers.2.ffn_hc.fn": Self.marker(5),
            "model.layers.2.ffn_hc.scale": Self.marker(6),
            "layers.1.hc_attn_fn": Self.marker(7),
            "layers.1.hc_ffn_base": Self.marker(8),
        ])

        let expected: [String: Float] = [
            "model.layers.0.hc_attn.base": 1,
            "model.layers.0.hc_attn.fn": 2,
            "model.layers.0.hc_attn.scale": 3,
            "model.layers.2.hc_ffn.base": 4,
            "model.layers.2.hc_ffn.fn": 5,
            "model.layers.2.hc_ffn.scale": 6,
            "model.layers.1.hc_attn.fn": 7,
            "model.layers.1.hc_ffn.base": 8,
        ]
        #expect(Set(sanitized.keys) == Set(expected.keys))
        for (key, value) in expected {
            let tensor = try #require(sanitized[key], Comment(rawValue: key))
            #expect(Self.floats(tensor) == [value], Comment(rawValue: key))
        }
    }

    @Test func sanitizeMapsTheScoreCorrectionBiasOntoTheGateBias() throws {
        // The mlx-community checkpoint spells the routing bias of a top-k
        // gate `ffn.gate.e_score_correction_bias`, the score-correction name
        // of the DeepSeek-V3 lineage. `DeepSeekV4MoEGate` holds that tensor
        // under the key `bias` and adds it to the expert scores only for the
        // top-k selection, which is the place the Python reference adds its
        // `e_score_correction_bias`. Each spelling goes through its own
        // `sanitize` call, because the two names converge on one module path.
        let model = DeepSeekV4Model(try Self.configuration())
        let checkpointSpelling = model.sanitize(weights: [
            "model.layers.\(Self.topKLayer).ffn.gate.e_score_correction_bias":
                Self.marker(1)
        ])
        let rawSpelling = model.sanitize(weights: [
            "layers.\(Self.topKLayer).ffn.gate.bias": Self.marker(2)
        ])

        let modulePath = "model.layers.\(Self.topKLayer).ffn.gate.bias"
        #expect(Set(checkpointSpelling.keys) == [modulePath])
        let mapped = try #require(checkpointSpelling[modulePath])
        #expect(Self.floats(mapped) == [1])
        let unchanged = try #require(rawSpelling[modulePath])
        #expect(Self.floats(unchanged) == [2])
    }

    @Test func sanitizeStacksThePerExpertWeightsIntoTheSwitchLayer() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< Self.routedExpertCount {
            weights["layers.0.ffn.experts.\(expert).w1.weight"] = Self.marker(Float(expert))
            weights["layers.0.ffn.experts.\(expert).w2.weight"] = Self.marker(
                Float(expert) + Float(Self.routedExpertCount))
            weights["layers.0.ffn.experts.\(expert).w3.weight"] = Self.marker(
                Float(expert) + Float(2 * Self.routedExpertCount))
        }

        let sanitized = model.sanitize(weights: weights)

        #expect(!sanitized.keys.contains { $0.contains(".experts.") })
        let expected: [String: Float] = [
            "gate_proj": 0, "down_proj": Float(Self.routedExpertCount),
            "up_proj": Float(2 * Self.routedExpertCount),
        ]
        for (projection, first) in expected {
            let key = "model.layers.0.ffn.switch_mlp.\(projection).weight"
            let note = Comment(rawValue: key)
            let stacked = try #require(sanitized[key], note)
            #expect(stacked.shape == [Self.routedExpertCount, 1], note)
            // The stack keeps the expert order, thus expert `e` sits at row
            // `e`. A stack in another order would still carry this shape.
            #expect(
                Self.floats(stacked)
                    == (0 ..< Self.routedExpertCount).map { first + Float($0) }, note)
        }
    }

    @Test func sanitizeKeepsTheHashTableOfARoutingLayer() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let table = MLXArray(
            (0 ..< (Self.vocabSize * Self.expertsPerToken)).map {
                Int64($0 % Self.routedExpertCount)
            }
        ).reshaped([Self.vocabSize, Self.expertsPerToken])

        let sanitized = model.sanitize(weights: [
            "layers.\(Self.hashLayer).ffn.gate.tid2eid": table
        ])

        let loaded = try #require(
            sanitized["model.layers.\(Self.hashLayer).ffn.gate.tid2eid"])
        #expect(loaded.shape == [Self.vocabSize, Self.expertsPerToken])
        #expect(loaded.dtype == .int64)
    }

    @Test func sanitizeDropsTheLanguageModelHeadOfATiedCheckpoint() throws {
        let tied = DeepSeekV4Model(try Self.configuration(tieWordEmbeddings: true))
        let untied = DeepSeekV4Model(try Self.configuration(tieWordEmbeddings: false))
        let weights = [
            "embed.weight": Self.marker(1),
            "head.weight": Self.marker(2),
        ]

        #expect(tied.sanitize(weights: weights)["lm_head.weight"] == nil)
        #expect(untied.sanitize(weights: weights)["lm_head.weight"] != nil)
        #expect(tied.sanitize(weights: weights)["model.embed_tokens.weight"] != nil)
    }

    @Test func sanitizeLeavesAnAlreadyConvertedCheckpointAlone() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let converted = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened())

        let sanitized = model.sanitize(weights: converted)

        #expect(Set(sanitized.keys) == Set(converted.keys))
    }
}

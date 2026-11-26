// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// A tiny checkpoint (`config.json` and `model.safetensors`, a few megabytes) on seeded random
/// weights for each model family that shards, with a configuration that splits across two
/// processes.
///
/// `ShardCheck tiny-checkpoint` writes one, so `scripts/shard-check.sh --tiny` compares the
/// logits of two ring processes with the unsharded logits for every family, with no download.
public enum TinyShardCheckpoint {

    /// The tokens of the smallest tiny vocabulary: the unpadded vocabulary of the tiny Inkling 2
    /// target.
    private static let smallestVocabularySize = 48

    /// The tokens of ``prompt``.
    private static let promptLength = 8

    /// The step between consecutive ids of ``prompt``, modulo the smallest vocabulary. It is
    /// prime to that vocabulary, so the ids differ.
    private static let promptStride = 7

    /// The prompt token ids of a check of a tiny checkpoint: ids every tiny vocabulary holds.
    /// The default prompt of `ShardCheck` is pre-tokenized for the vocabulary of a real
    /// checkpoint, which is far larger.
    public static let prompt = (1 ... promptLength).map {
        $0 * promptStride % smallestVocabularySize
    }

    /// A failure to make a tiny checkpoint.
    public enum Failure: Error, Equatable, CustomStringConvertible {
        /// No tiny checkpoint is known for this model type.
        case unknownFamily(String)
        /// The JSON of a fixture is not an object.
        case notAnObject(String)

        /// The failure, for the console.
        public var description: String {
            switch self {
            case .unknownFamily(let family):
                "no tiny checkpoint for the model type \(family)"
            case .notAnObject(let json):
                "the fixture JSON is not an object: \(json)"
            }
        }
    }

    /// The `config.json` of the tiny checkpoint of `family`.
    ///
    /// - Parameter family: A `model_type` of ``RemoteModelExecutor/shardableModelFamilies``.
    /// - Returns: The JSON text.
    /// - Throws: ``Failure/unknownFamily(_:)`` for a model type with no tiny checkpoint.
    public static func configurationJSON(of family: String) throws -> String {
        let configuration = try recipe(of: family).configuration(modelType: family)
        let data = try JSONSerialization.data(
            withJSONObject: configuration, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// The model of the tiny checkpoint of `family`, on seeded random weights.
    ///
    /// - Parameter family: A `model_type` of ``RemoteModelExecutor/shardableModelFamilies``.
    /// - Returns: The model, as a worker loads it from the checkpoint.
    /// - Throws: ``Failure/unknownFamily(_:)`` for a model type with no tiny checkpoint, or the
    ///   error of the model.
    public static func makeModel(of family: String) async throws -> any LanguageModel {
        try await recipe(of: family).makeModel(json: configurationJSON(of: family))
    }

    /// Writes the tiny checkpoint of `family`, `config.json` and `model.safetensors`, to
    /// `directory`, which it creates when it does not exist.
    ///
    /// - Parameters:
    ///   - family: A `model_type` of ``RemoteModelExecutor/shardableModelFamilies``.
    ///   - directory: The directory of the checkpoint.
    /// - Throws: ``Failure/unknownFamily(_:)`` for a model type with no tiny checkpoint, or the
    ///   error of the model or of a file write.
    public static func write(family: String, to directory: URL) async throws {
        let json = try configurationJSON(of: family)
        let model = try await makeModel(of: family)
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        eval(Array(arrays.values))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: directory.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
    }

    /// The recipe of `family`.
    private static func recipe(of family: String) throws -> Recipe {
        guard let recipe = Recipe.byFamily[family] else { throw Failure.unknownFamily(family) }
        return recipe
    }
}

/// How the tiny checkpoint of one model family is made.
private enum Recipe {
    /// A Qwen 3.5 configuration that wraps the text configuration, 4-bit affine.
    case qwen35(experts: Bool)
    /// A flat Qwen 3.5 text configuration, 4-bit affine.
    case qwen35Text
    /// A rotated 4-bit prism pack over the dense Qwen 3.5 text configuration.
    case prismHadamardQwen35
    /// The tiny Inkling 2 target with NVFP4 experts, with or without its loop.
    case inkling2(loop: Bool)

    /// The recipe of each family of ``RemoteModelExecutor/shardableModelFamilies``.
    static let byFamily: [String: Recipe] = [
        "qwen3_5": .qwen35(experts: false),
        "qwen3_8": .qwen35(experts: false),
        "qwen3_5_moe": .qwen35(experts: true),
        "qwen3_8_moe": .qwen35(experts: true),
        "qwen3_5_text": .qwen35Text,
        "qwen3_8_text": .qwen35Text,
        "prism_hadamard_qwen35": .prismHadamardQwen35,
        "inkling2": .inkling2(loop: false),
        "inkling2_loop": .inkling2(loop: true),
    ]

    /// The seed of the random weights of a Qwen 3.5 model.
    private static let qwen35Seed: UInt64 = 5

    /// The seed of the random weights and signs of a prism pack.
    private static let prismSeed: UInt64 = 6

    /// The `config.json` of the recipe, as a JSON object, naming `modelType`.
    func configuration(modelType: String) throws -> [String: Any] {
        switch self {
        case .qwen35(let experts):
            return [
                "model_type": modelType,
                "quantization": try JSONObject.parse(TinyQwen35.quantizationJSON),
                "text_config": try TinyQwen35.textConfiguration(experts: experts),
            ]
        case .qwen35Text:
            var configuration = try TinyQwen35.textConfiguration(experts: false)
            configuration["model_type"] = modelType
            configuration["quantization"] = try JSONObject.parse(TinyQwen35.quantizationJSON)
            return configuration
        case .prismHadamardQwen35:
            return try TinyPrismPack.configuration(modelType: modelType)
        case .inkling2(let loop):
            var configuration = try JSONObject.parse(Inkling2TinyModels.targetJSON)
            configuration["model_type"] = modelType
            if !loop, var text = configuration["text_config"] as? [String: Any] {
                text["loop_config"] = nil
                configuration["text_config"] = text
            }
            return configuration
        }
    }

    /// The model of the configuration `json` on seeded random weights, as a worker loads it.
    func makeModel(json: String) async throws -> any LanguageModel {
        switch self {
        case .qwen35, .qwen35Text:
            let model = try await Self.createModel(json: json, seed: Self.qwen35Seed)
            try Self.quantize(model, json: json)
            return model
        case .prismHadamardQwen35:
            let model = try await Self.createModel(json: json, seed: Self.prismSeed)
            withRandomState(MLXRandom.RandomState(seed: Self.prismSeed)) {
                TinyPrismPack.randomizeRotations(of: model)
            }
            return model
        case .inkling2:
            return try Inkling2TinyModels.target(json: json)
        }
    }

    /// The model of the LLM factory for the configuration `json`, on the random initial
    /// weights of the random state `seed`.
    private static func createModel(json: String, seed: UInt64) async throws
        -> any LanguageModel
    {
        let configuration = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        return try await withRandomState(MLXRandom.RandomState(seed: seed)) {
            try await LLMTypeRegistry.shared.createModel(
                configuration: Data(json.utf8), modelType: configuration.modelType)
        }
    }

    /// Quantizes every layer of `model` that can quantize, as the quantization of the
    /// configuration `json` states. The loader quantizes the layers whose scales the checkpoint
    /// holds.
    private static func quantize(_ model: any LanguageModel, json: String) throws {
        let configuration = try JSONDecoder().decode(BaseConfiguration.self, from: Data(json.utf8))
        guard let plan = configuration.perLayerQuantization else { return }
        MLXNN.quantize(model: model) { path, _ in plan.quantization(layer: path)?.asTuple }
    }
}

/// The text configuration of the tiny Qwen 3.5 checkpoints: the small fixture of
/// ``Qwen35FixtureJSON``, widened so its heads, its MLP and its experts split across two
/// processes in whole 4-bit groups of 64.
private enum TinyQwen35 {

    /// The widths and counts of the dense text configuration. Two of the four layers are
    /// gated delta (recurrent) layers and two are full attention.
    static let denseShapeJSON = """
        {
          "hidden_size": 256, "num_hidden_layers": 4, "full_attention_interval": 2,
          "intermediate_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 64, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
          "linear_key_head_dim": 64, "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
          "vocab_size": 512, "tie_word_embeddings": false, "max_position_embeddings": 4096
        }
        """

    /// The mixture of experts of every layer of the expert text configuration.
    static let expertShapeJSON = """
        {
          "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
          "shared_expert_intermediate_size": 256
        }
        """

    /// The quantization of the checkpoints.
    static let quantizationJSON = #"{"group_size": 64, "bits": 4}"#

    /// The text configuration, dense or with experts, as a JSON object.
    static func textConfiguration(experts: Bool) throws -> [String: Any] {
        var configuration = try JSONObject.parse(Qwen35FixtureJSON.text(mtpLayers: 0))
        configuration.merge(try JSONObject.parse(denseShapeJSON)) { $1 }
        if experts {
            configuration.merge(try JSONObject.parse(expertShapeJSON)) { $1 }
        }
        return configuration
    }
}

/// The tiny prism pack: the manifest of `PrismHadamardQwen35` beside the dense text
/// configuration of ``TinyQwen35``, with a rotated 4-bit projection in every layer. The blocks
/// of the projections are 32, 64 and 128 wide, and the input share of each process holds
/// whole blocks.
private enum TinyPrismPack {

    /// The fields of the pack beside its manifest and its text configuration.
    static let packJSON = """
        {
          "base_model_type": "qwen3_5", "schema_version": 2,
          "tensor_namespace": "mlx-vlm-qwen3_5", "gdn_activation_layout": "grouped",
          "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}
        }
        """

    /// The rotated modules of the manifest: those of the model, and the projections of each
    /// layer by the layer kind they stand in, with paths relative to the layer.
    static let rotatedModulesJSON = """
        {
          "model": [
            {"path": "model.embed_tokens", "block": 64, "embedding": true, "dtype": "float16"},
            {"path": "lm_head", "block": 64, "dtype": "float16"}
          ],
          "every_layer": [
            {"path": "mlp.gate_proj", "block": 64}, {"path": "mlp.up_proj", "block": 64},
            {"path": "mlp.down_proj", "block": 128}
          ],
          "linear_attention": [
            {"path": "linear_attn.in_proj_qkv", "block": 64},
            {"path": "linear_attn.in_proj_z", "block": 32},
            {"path": "linear_attn.out_proj", "block": 128}
          ],
          "full_attention": [
            {"path": "self_attn.q_proj", "block": 64}, {"path": "self_attn.k_proj", "block": 64},
            {"path": "self_attn.v_proj", "block": 64}, {"path": "self_attn.o_proj", "block": 64}
          ]
        }
        """

    /// The `config.json` of the pack, naming `modelType`.
    static func configuration(modelType: String) throws -> [String: Any] {
        let text = try TinyQwen35.textConfiguration(experts: false)
        var configuration = try JSONObject.parse(packJSON)
        configuration["model_type"] = modelType
        configuration["vocab_size"] = text["vocab_size"]
        configuration["text_config"] = text
        configuration["modules"] = try modules(of: text)
        return configuration
    }

    /// The manifest entries of the rotated modules of every layer of `text`.
    private static func modules(of text: [String: Any]) throws -> [[String: Any]] {
        guard
            let table = try JSONObject.parse(rotatedModulesJSON) as? [String: [[String: Any]]],
            let layers = text["num_hidden_layers"] as? Int,
            let interval = text["full_attention_interval"] as? Int
        else { throw TinyShardCheckpoint.Failure.notAnObject(rotatedModulesJSON) }
        let perLayer = (0 ..< layers).flatMap { layer in
            // Qwen 3.5 makes the last layer of each interval full attention.
            let kind = (layer + 1) % interval == 0 ? "full_attention" : "linear_attention"
            return (table["every_layer", default: []] + table[kind, default: []]).map {
                entry -> [String: Any] in
                var entry = entry
                entry["path"] = "model.layers.\(layer).\(entry["path"] ?? "")"
                return entry
            }
        }
        return table["model", default: []] + perLayer
    }

    /// Gives every rotated projection of `model` random quantized weights and random signs,
    /// and every rotated embedding random signs, from the current random state. The factory
    /// makes rotated projections with zero weights and unit signs.
    static func randomizeRotations(of model: any LanguageModel) {
        for (_, leaf) in model.leafModules().flattened() {
            if let rotated = leaf as? HadamardQuantizedLinear {
                randomize(rotated)
            } else if let embedding = leaf as? HadamardQuantizedEmbedding {
                embedding.update(
                    parameters: ModuleParameters.unflattened([
                        "signs": randomSigns(count: embedding.signs.dim(0))
                    ]))
            }
        }
    }

    /// Random weights of unit output variance in the quantization of `projection`, and random
    /// signs.
    private static func randomize(_ projection: HadamardQuantizedLinear) {
        let (outputs, inputs) = projection.shape
        let weight = MLXRandom.normal([outputs, inputs]) / sqrt(Float(inputs))
        let (packed, scales, biases) = MLX.quantized(
            weight, groupSize: projection.groupSize, bits: projection.bits, mode: projection.mode)
        var parameters = [
            "weight": packed, "scales": scales, "signs": randomSigns(count: inputs),
        ]
        parameters["biases"] = biases
        projection.update(parameters: ModuleParameters.unflattened(parameters))
    }

    /// `count` signs, each +1 or -1 with even odds, as float32.
    private static func randomSigns(count: Int) -> MLXArray {
        MLX.where(MLXRandom.bernoulli([count]), Float(1), Float(-1))
    }
}

/// The JSON objects of the fixtures.
private enum JSONObject {
    /// The JSON object of `json`.
    ///
    /// - Throws: ``TinyShardCheckpoint/Failure/notAnObject(_:)`` when `json` holds another
    ///   value, or the error of the JSON parse.
    static func parse(_ json: String) throws -> [String: Any] {
        guard
            let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
                as? [String: Any]
        else { throw TinyShardCheckpoint.Failure.notAnObject(json) }
        return object
    }
}

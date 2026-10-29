// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

/// A Qwen 3.5 MTP drafter checkpoint in a temporary directory, with the drafter factory that
/// loads it.
///
/// The checkpoint holds a small drafter in one of the layouts that Qwen 3.5 checkpoints
/// publish: embedded in the target checkpoint (the drafter tensors under a prefix) or
/// standalone (a `<family>_mtp` model type), for the text target or the vision-language
/// target, plain or quantized. The caller removes ``directory`` when it is done.
public struct Qwen35MTPCheckpointFixture {

    /// The width of the hidden state of the checkpoint drafter.
    public static let hiddenSize = 64

    /// The width of the MLP of the checkpoint drafter.
    private static let intermediateSize = 128

    /// The width of each attention head of the checkpoint drafter.
    private static let headDimension = 32

    /// The group size of each quantized tensor.
    public static let quantizationGroupSize = 32

    /// The bits of the quantized input projection, whatever the bits of the other tensors.
    public static let inputProjectionBits = 8

    /// The model type of the Qwen 3.5 checkpoints.
    public static let defaultFamily = "qwen3_5"

    /// The tensor prefix of a drafter embedded in a converted checkpoint.
    public static let convertedPrefix = "mtp."

    /// The module path of the drafter's input projection.
    private static let inputProjectionPath = "mtp.fc"

    /// The directory that holds `config.json` and `model.safetensors`.
    public let directory: URL

    /// The factory that loads a drafter from ``directory``.
    public let factory: MTPDrafterModelFactory

    /// Writes the checkpoint of a small drafter and registers its model type with a fresh
    /// factory.
    ///
    /// - Parameters:
    ///   - vision: Whether the drafter is the vision-language drafter.
    ///   - standalone: Whether the checkpoint is a standalone `<family>_mtp` checkpoint, or the
    ///     drafter tensors embedded in a target checkpoint.
    ///   - prefix: The tensor prefix of the drafter in the checkpoint.
    ///   - bits: The bits of the quantized tensors, or `nil` for a plain checkpoint.
    ///   - family: The model type of the target family.
    /// - Throws: The error of the configuration, of the drafter or of a file write.
    public init(
        vision: Bool, standalone: Bool, prefix: String = "", bits: Int? = nil,
        family: String = defaultFamily
    ) async throws {
        let config = try Self.configuration(
            vision: vision, standalone: standalone, bits: bits, prefix: prefix, family: family)
        let registry = ModelTypeRegistry<any MTPDrafterModel>()
        let modelType = standalone ? family + "_mtp" : family
        await registry.registerModelType(modelType) { data in
            try Self.makeDrafter(configuration: data, vision: vision)
        }
        let model = try Self.makeDrafter(configuration: config, vision: vision)
        if let bits {
            quantize(model: model) { path, _ in
                (
                    Self.quantizationGroupSize,
                    path == Self.inputProjectionPath ? Self.inputProjectionBits : bits, .affine
                )
            }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            try config.write(to: directory.appendingPathComponent("config.json"))
            try save(
                arrays: Self.checkpointWeights(of: model, prefix: prefix, standalone: standalone),
                metadata: standalone ? ["format": "mlx"] : [:],
                url: directory.appendingPathComponent("model.safetensors"))
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        self.directory = directory
        self.factory = MTPDrafterModelFactory(
            typeRegistry: registry, modelRegistry: MTPDrafterRegistry.shared)
    }

    /// The tensors of `model` as the checkpoint stores them: under `prefix`, with each norm
    /// weight at one (or at zero, the unshifted convention, under the converted prefix), and
    /// with a target embedding beside an embedded drafter.
    private static func checkpointWeights(
        of model: any MTPDrafterModel, prefix: String, standalone: Bool
    ) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (name, array) in model.parameters().flattened() {
            let serialized = prefix + name.dropFirst(convertedPrefix.count)
            let isNorm = name.contains("norm")
            weights[serialized] = isNorm ? MLXArray.ones(array.shape) : array
            if prefix == convertedPrefix, isNorm {
                weights[serialized] = MLXArray.zeros(array.shape)
            }
        }
        if !standalone {
            weights["model.embed_tokens.weight"] = MLXArray.zeros([
                Qwen35FixtureJSON.defaultVocabularySize, hiddenSize,
            ])
        }
        return weights
    }

    /// The `config.json` of a checkpoint drafter.
    ///
    /// - Parameters:
    ///   - vision: Whether the drafter is the vision-language drafter.
    ///   - standalone: Whether the checkpoint is a standalone `<family>_mtp` checkpoint.
    ///   - bits: The bits of the quantized tensors, or `nil` for a plain checkpoint.
    ///   - prefix: The tensor prefix of the drafter in the checkpoint.
    ///   - family: The model type of the target family.
    /// - Returns: The JSON data.
    /// - Throws: The error of the JSON serialization.
    public static func configuration(
        vision: Bool, standalone: Bool, bits: Int?, prefix: String = convertedPrefix,
        family: String = defaultFamily
    ) throws -> Data {
        let json =
            vision
            ? Qwen35FixtureJSON.visionLanguage(mtpLayers: 1)
            : Qwen35FixtureJSON.wrappedText(modelType: defaultFamily)
        let sized = json.replacingOccurrences(
            of: "\"hidden_size\": \(Qwen35FixtureJSON.defaultHiddenSize)",
            with: "\"hidden_size\": \(hiddenSize)"
        )
        .replacingOccurrences(
            of: "\"intermediate_size\": 32", with: "\"intermediate_size\": \(intermediateSize)"
        )
        .replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": \(headDimension)")
        guard
            var root = try JSONSerialization.jsonObject(with: Data(sized.utf8))
                as? [String: Any]
        else {
            throw ModelFactoryError.invalidConfiguration("The fixture JSON is not an object.")
        }
        root["model_type"] = standalone ? family + "_mtp" : family
        if var text = root["text_config"] as? [String: Any] {
            text["model_type"] = family + "_text"
            root["text_config"] = text
        }
        if standalone { root["vision_config"] = [String: Any]() }
        if let bits {
            root["quantization"] = [
                "group_size": quantizationGroupSize, "bits": bits,
                (standalone ? "fc" : prefix + "fc"): [
                    "group_size": quantizationGroupSize, "bits": inputProjectionBits,
                ],
            ]
        }
        return try JSONSerialization.data(withJSONObject: root)
    }

    /// The drafter of a checkpoint configuration.
    private static func makeDrafter(configuration: Data, vision: Bool) throws
        -> any MTPDrafterModel
    {
        if vision {
            let config = try JSONDecoder().decode(
                VisionLanguageConfiguration.self, from: configuration)
            return MLXVLM.Qwen35VLMNextNDraftModel(
                config.textConfiguration,
                checkpointPolicy: .init(modelType: config.modelType, preconvertedNorms: false))
        }
        return MLXLLM.Qwen35MTPDraftModel(
            try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: configuration))
    }

    /// The fields of a vision-language checkpoint configuration that the drafter reads.
    private struct VisionLanguageConfiguration: Decodable {
        let modelType: String
        let textConfiguration: MLXVLM.Qwen35Configuration.TextConfiguration

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case textConfiguration = "text_config"
        }
    }
}

/// A downloader that refuses to download: a configuration that names a local directory needs
/// none.
public struct UnusedDownloader: Downloader {

    /// Makes the downloader.
    public init() {}

    /// Refuses to download.
    ///
    /// - Parameters:
    ///   - id: The repository of the request.
    ///   - revision: The revision of the request.
    ///   - patterns: The file patterns of the request.
    ///   - useLatest: Whether the request asks for the latest revision.
    ///   - progressHandler: The progress callback of the request.
    /// - Returns: Never.
    /// - Throws: `ModelFactoryError.invalidConfiguration`, always.
    public func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        throw ModelFactoryError.invalidConfiguration("A local MTP checkpoint downloads nothing.")
    }
}

/// A tokenizer loader that refuses to load: an MTP drafter checkpoint borrows the tokenizer of
/// its target, thus a drafter load that asks for one is a mistake.
public struct UnusedTokenizerLoader: TokenizerLoader {

    /// Makes the loader.
    public init() {}

    /// Refuses to load a tokenizer.
    ///
    /// - Parameter url: The directory of the checkpoint.
    /// - Returns: Never.
    /// - Throws: `ModelFactoryError.invalidConfiguration`, always.
    public func load(from url: URL) async throws -> any Tokenizer {
        throw ModelFactoryError.invalidConfiguration(
            "MTP checkpoints must borrow the target tokenizer.")
    }
}

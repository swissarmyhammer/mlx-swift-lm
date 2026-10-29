// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5-family multimodal MTP drafter model types.
///
/// These creators live in the target-specific `visionLanguage` registry so a
/// standalone MTP config with an empty `vision_config` still selects the
/// M-RoPE-aware drafter deterministically.
public enum Qwen35VLMMTPRegistration {
    public static func register() async {
        let registry = MTPDrafterTypeRegistry.visionLanguage

        for modelType in ["qwen3_5", "qwen3_5_moe", "qwen3_8", "qwen3_8_moe"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35MTPTextConfiguration.self, from: data)
                    return Qwen35VLMNextNDraftModel(config.textConfiguration)
                })
        }

        for modelType in ["qwen3_5_mtp", "qwen3_8_mtp"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35MTPTextConfiguration.self, from: data)
                    return Qwen35VLMNextNDraftModel(
                        config.textConfiguration,
                        checkpointPolicy: .init(
                            modelType: config.modelType, preconvertedNorms: true))
                })
        }
    }
}

extension SpeculativeDecodingConfig {

    /// Initialize speculative decoding with a Qwen3.5-family multimodal MTP drafter that
    /// `ChatSession` loads from `configuration` on the first turn that speculates, once the
    /// memory policy admits `mtpDrafterBytes`.
    ///
    /// The load registers ``Qwen35VLMMTPRegistration`` and loads the drafter through
    /// `MTPDrafterModelFactory.visionLanguage`, for a target that `VLMModelFactory` loaded.
    ///
    /// - Parameters:
    ///   - configuration: the drafter checkpoint, as a model id or a local directory
    ///   - mtpDrafterBytes: estimated resident parameter bytes of the drafter
    ///   - downloader: the downloader that fetches a remote checkpoint
    ///   - tokenizerLoader: the tokenizer loader the factory takes; the drafter borrows the
    ///     tokenizer of its target
    ///   - blockSize: total number of tokens in each verification block
    ///   - minimumDraftProbability: the drafter top-1 probability below which the drafter stops
    ///     a round's chain early; 0 drafts the whole block every round.
    ///   - memoryPolicy: optional memory policy that each turn checks before it speculates
    /// - Throws: ``SpeculativeDecodingConfigurationError`` when `blockSize` or
    ///   `minimumDraftProbability` is out of range.
    public init(
        qwen35VLMMTPDrafter configuration: ModelConfiguration,
        mtpDrafterBytes: Int,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        blockSize: Int = defaultMTPBlockSize,
        minimumDraftProbability: Float = defaultMTPMinimumDraftProbability,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) throws {
        try self.init(
            mtpDrafterConfiguration: configuration, mtpDrafterBytes: mtpDrafterBytes,
            from: downloader, using: tokenizerLoader, factory: .visionLanguage,
            registerDrafterTypes: Qwen35VLMMTPRegistration.register, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability, memoryPolicy: memoryPolicy)
    }
}

private struct Qwen35MTPTextConfiguration: Decodable {
    let modelType: String
    let textConfiguration: Qwen35Configuration.TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
    }
}

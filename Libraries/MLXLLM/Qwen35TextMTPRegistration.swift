// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5-family text MTP drafter model types.
///
/// Standalone Qwen MTP checkpoints do not identify the verifier architecture:
/// their `vision_config` is empty for both text and multimodal use. Text and
/// multimodal drafters therefore use separate registries instead of relying
/// on config-shape predicates or registration order.
public enum Qwen35TextMTPRegistration {
    public static func register() async {
        let registry = MTPDrafterTypeRegistry.shared

        for modelType in ["qwen3_5_text", "qwen3_8_text"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35TextConfiguration.self, from: data)
                    return Qwen35MTPDraftModel(config)
                })
        }

        for modelType in ["qwen3_5", "qwen3_5_moe", "qwen3_8", "qwen3_8_moe"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35Configuration.self, from: data)
                    return Qwen35MTPDraftModel(config)
                })
        }

        for modelType in ["qwen3_5_mtp", "qwen3_8_mtp"] {
            await registry.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35Configuration.self, from: data)
                    return Qwen35MTPDraftModel(config, preconvertedNorms: true)
                })
        }
    }
}

extension SpeculativeDecodingConfig {

    /// Initialize speculative decoding with a Qwen3.5-family text MTP drafter that
    /// `ChatSession` loads from `configuration` on the first turn that speculates, once the
    /// memory policy admits `mtpDrafterBytes`.
    ///
    /// The load registers ``Qwen35TextMTPRegistration`` and loads the drafter through
    /// `MTPDrafterModelFactory.shared`.
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
        qwen35TextMTPDrafter configuration: ModelConfiguration,
        mtpDrafterBytes: Int,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        blockSize: Int = defaultMTPBlockSize,
        minimumDraftProbability: Float = defaultMTPMinimumDraftProbability,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) throws {
        try self.init(
            mtpDrafterConfiguration: configuration, mtpDrafterBytes: mtpDrafterBytes,
            from: downloader, using: tokenizerLoader, factory: .shared,
            registerDrafterTypes: Qwen35TextMTPRegistration.register, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability, memoryPolicy: memoryPolicy)
    }
}

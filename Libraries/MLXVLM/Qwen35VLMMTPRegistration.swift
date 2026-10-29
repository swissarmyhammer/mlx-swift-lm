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

extension MTPDrafterRegistration {

    /// The Qwen3.5-family multimodal MTP drafters: a deferred `SpeculativeDecodingConfig` that
    /// takes this registers ``Qwen35VLMMTPRegistration`` and loads its drafter through
    /// `MTPDrafterModelFactory.visionLanguage`, for a target that `VLMModelFactory` loaded.
    public static let qwen35VLM = MTPDrafterRegistration(
        factory: .visionLanguage, registerDrafterTypes: Qwen35VLMMTPRegistration.register)
}

private struct Qwen35MTPTextConfiguration: Decodable {
    let modelType: String
    let textConfiguration: Qwen35Configuration.TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
    }
}

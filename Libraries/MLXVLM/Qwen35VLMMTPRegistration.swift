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
        await registry.registerDrafter(
            modelTypes: ["qwen3_5", "qwen3_5_moe", "qwen3_8", "qwen3_8_moe"],
            configuration: Qwen35MTPTextConfiguration.self,
            create: { Qwen35VLMNextNDraftModel($0.textConfiguration) })
        await registry.registerDrafter(
            modelTypes: ["qwen3_5_mtp", "qwen3_8_mtp"],
            configuration: Qwen35MTPTextConfiguration.self,
            create: {
                Qwen35VLMNextNDraftModel(
                    $0.textConfiguration,
                    checkpointPolicy: .init(modelType: $0.modelType, preconvertedNorms: true))
            })
    }
}

extension MTPDrafterRegistration {

    /// The Qwen3.5-family multimodal MTP drafters: a deferred `SpeculativeDecodingConfig` that
    /// takes this registers ``Qwen35VLMMTPRegistration`` and loads its drafter through
    /// `MTPDrafterModelFactory.visionLanguage`, for a target that `VLMModelFactory` loaded.
    /// `matchingTarget(_:_:)` with `qwen35Text` of MLXLLM covers a target that loaded as a text
    /// model too.
    public static let qwen35VLM = MTPDrafterRegistration(
        factory: .visionLanguage, registerDrafterTypes: Qwen35VLMMTPRegistration.register,
        draftsForTarget: { $0 is Qwen35 })
}

private struct Qwen35MTPTextConfiguration: Decodable {
    let modelType: String
    let textConfiguration: Qwen35Configuration.TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
    }
}

// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Liquid AI's `Lfm2DSparkDraftModel` checkpoints with the shared
/// block-drafter factory.
public enum LFM2DSparkRegistration {
    /// The architecture that tells a DSpark `config.json` from a Qwen 3 one of the same
    /// `model_type`.
    static let architecture = "Lfm2DSparkDraftModel"

    /// Registers the DSpark drafter for the `qwen3` model type of the shared drafter registry,
    /// for a `config.json` whose `architectures` name `architecture`.
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerDrafter(
            modelTypes: ["qwen3"], configuration: LFM2DSparkConfiguration.self,
            matches: { $0.architectures.contains(architecture) },
            create: { LFM2DSparkDraftModel($0) })
    }
}

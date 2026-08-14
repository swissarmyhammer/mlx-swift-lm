// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4 support: `model_type == "deepseek_v4"` must be loadable end to
// end. These tests pin the type-registry entry — the detection rule that the
// chat-encoder wiring keys on (card ^mjrzkgm).

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4RegistryTests {

    @Test("the type registry has a deepseek_v4 entry")
    func typeRegistryContainsDeepSeekV4() async {
        #expect(await LLMTypeRegistry.shared.contains("deepseek_v4"))
    }

    @Test("createModel builds a DeepSeekV4Model from the checkpoint config")
    func createModelBuildsADeepSeekV4Model() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(DeepSeekV4SyntheticCheckpoint.configJSON.utf8),
            modelType: "deepseek_v4")
        #expect(model is DeepSeekV4Model)
    }

    /// The published checkpoint is a model registry entry, not only a type
    /// registry entry. The two registries answer different questions: the type
    /// registry maps `model_type` onto a model class, and the model registry
    /// maps a Hub id onto a configuration a caller can load by name.
    @Test("the model registry holds the published DeepSeek-V4-Flash-4bit id")
    func modelRegistryContainsDeepSeekV4Flash() {
        #expect(LLMRegistry.shared.contains(id: "mlx-community/DeepSeek-V4-Flash-4bit"))

        let configuration = LLMRegistry.deepseek_v4_flash_4bit
        #expect(configuration.name == "mlx-community/DeepSeek-V4-Flash-4bit")
        #expect(configuration.defaultPrompt == "Why is the sky blue?")
    }
}

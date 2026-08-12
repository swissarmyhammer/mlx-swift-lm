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

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

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
}

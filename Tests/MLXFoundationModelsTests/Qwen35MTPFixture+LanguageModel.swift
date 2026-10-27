// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import MLXLMTestSupport

@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension Qwen35MTPFixture {

    /// A FoundationModels language model whose load gives the container of the fixture's
    /// target, so that an executor test can run a `LanguageModelSession` on it.
    ///
    /// The model takes a fresh identity, thus the process-wide model cache keeps it apart
    /// from every other test.
    ///
    /// - Parameter weights: The directory that makes the model available, from
    ///   `makeScriptedWeightsDirectory()`.
    /// - Returns: The model.
    func makeLanguageModel(weights: URL) -> MLXLanguageModel {
        let container = makeModelContainer()
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: "probe/qwen35-mtp-fixture-\(UUID().uuidString)"),
            capabilities: [],
            weightsLocation: { _ in weights },
            load: { _, _ in container })
    }
}

#endif

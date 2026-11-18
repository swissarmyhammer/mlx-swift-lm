// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The warm-up that `prewarm` starts checks for cancellation between its stages, thus a
/// cancelled warm-up stops before its next stage instead of running to its end.
@Suite("A cancelled warm-up stops before its next stage")
struct WarmUpCancellationTests {

    /// Stops a load that reached its loader.
    private struct LoadStopped: Error {}

    @Test("a warm-up in a cancelled task throws CancellationError before it loads the model")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aWarmUpInACancelledTaskThrowsBeforeItLoads() async {
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "test/warm-up-cancelled-\(UUID().uuidString)"),
            capabilities: [],
            weightsLocation: { _ in FileManager.default.temporaryDirectory },
            load: { _, _ in throw LoadStopped() })
        let warmUp = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await model.warmUp()
        }

        await #expect(throws: CancellationError.self) {
            try await warmUp.value
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

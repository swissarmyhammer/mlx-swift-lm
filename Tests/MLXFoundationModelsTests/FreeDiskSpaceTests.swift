// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

@Suite("MLXLanguageModel free disk space")
struct FreeDiskSpaceTests {

    /// The plain free space that the reading reports: the value that macOS gave for this
    /// volume when it reported zero for the important-usage key.
    private static let plainFreeSpace = 238_000_000_000

    @Test("the free disk space takes the plain free space when the important-usage value is zero")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theFreeDiskSpaceTakesThePlainFreeSpaceWhenImportantUsageIsZero() {
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "org/repo"),
            capabilities: [],
            weightsLocation: { _ in FileManager.default.temporaryDirectory },
            load: { _, _ in throw CocoaError(.featureUnsupported) })

        let freeBytes = model.freeDiskSpaceBytes(reading: { _ in
            ExecutorPromptCacheVolumeCapacity(importantUsage: 0, available: Self.plainFreeSpace)
        })

        #expect(freeBytes == Int64(Self.plainFreeSpace))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

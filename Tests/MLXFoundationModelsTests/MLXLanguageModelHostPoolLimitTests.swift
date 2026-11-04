// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2) && os(macOS)

extension FoundationModelsCacheTests {

    /// The MLX buffer-reuse pool limit belongs to the host app: a load leaves it alone.
    @Suite("MLXLanguageModel host pool limit")
    struct HostPoolLimit {

        /// Stops a load before it reaches a downloader.
        private struct LoadStopped: Error {}

        /// Whether a load of `model` reached its loader, which stops it with ``LoadStopped``.
        @available(macOS 27.0, *)
        private static func loadReachesTheLoader(of model: MLXLanguageModel) async -> Bool {
            do {
                try await model.preload()
                return false
            } catch {
                return error is LoadStopped
            }
        }

        /// The body runs in a process of its own. Any other test of this binary that loads a
        /// model would spend an override that fires once per process, and this test would then
        /// see no change.
        @available(macOS 27.0, *)
        @Test("A load leaves the host's Memory.cacheLimit in place")
        func loadLeavesTheHostPoolLimit() async {
            await #expect(processExitsWith: .success) {
                let hostLimit = Memory.cacheLimit - 1
                Memory.cacheLimit = hostLimit
                let model = MLXLanguageModel(
                    configuration: ModelConfiguration(id: "org/host-pool-limit"),
                    capabilities: [],
                    weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                    load: { _, _ in throw LoadStopped() })

                let reachedTheLoader = await Self.loadReachesTheLoader(of: model)

                let limitStands = Memory.cacheLimit == hostLimit
                exit(reachedTheLoader && limitStands ? EXIT_SUCCESS : EXIT_FAILURE)
            }
        }
    }
}

#endif

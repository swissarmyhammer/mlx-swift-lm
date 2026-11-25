// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Synchronization
import Testing

@testable import MLXFoundationModels

extension FoundationModelsCacheTests {

    /// A local load and a clustered load of one checkpoint hold two entries of the model cache,
    /// and evicting the clustered one tears down its workers.
    @Suite("The model cache keeps clustered and local loads apart")
    struct ClusteredModelCache {

        /// The plan the clustered model of each test names.
        private static let planID = "plan-a"

        /// The containers each model can load: the first load, and a reload after an eviction.
        /// Each clustered container has a worker of its own, as a worker serves one session.
        private static let loadsPerModel = 2

        /// A local and a clustered model of one checkpoint, and the worker behind the first load
        /// of the clustered one.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private struct Loads {
            let local: MLXLanguageModel
            let clustered: MLXLanguageModel
            let worker: InProcessRemoteModelWorker

            /// Makes the two models of one new checkpoint identity. Each load of a model gives a
            /// new container, thus a test tells a cached container from a reloaded one.
            init() throws {
                let fixture = try Qwen35MTPFixture()
                let configuration = ModelConfiguration(
                    id: "probe/clustered-cache-\(UUID().uuidString)")
                let workers = (0 ..< ClusteredModelCache.loadsPerModel).map { _ in
                    fixture.makeWorker()
                }
                let localContainers = ContainerQueue(
                    (0 ..< ClusteredModelCache.loadsPerModel).map { _ in
                        ModelContainer(context: fixture.makeModelContext())
                    })
                let clusteredContainers = ContainerQueue(
                    try workers.map { worker in
                        ModelContainer(
                            context: try fixture.makeClusteredModelContext(worker: worker))
                    })
                local = MLXLanguageModel(
                    configuration: configuration, capabilities: [],
                    weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                    load: { _, _ in try localContainers.next() })
                clustered = MLXLanguageModel(
                    configuration: configuration, capabilities: [],
                    loadIdentity: .clustered(planID: ClusteredModelCache.planID),
                    weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
                    load: { _, _ in try clusteredContainers.next() })
                self.worker = workers[0]
            }

            /// Loads both models into the cache.
            func preloadBoth() async throws {
                try await local.preload()
                try await clustered.preload()
            }
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("a local and a clustered load of one model id give two containers")
        func aLocalAndAClusteredLoadGiveTwoContainers() async throws {
            let loads = try Loads()

            let local = try await loads.local.loadContainer()
            let clustered = try await loads.clustered.loadContainer()

            #expect(local !== clustered)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("a local model keeps the model id of its configuration")
        func aLocalModelKeepsTheModelIDOfItsConfiguration() throws {
            let loads = try Loads()

            #expect(loads.local.modelID == loads.local.configuration.name)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("evicting the clustered load leaves the local entry")
        func evictingTheClusteredLoadLeavesTheLocalEntry() async throws {
            let loads = try Loads()
            try await loads.preloadBoth()
            let before = try await loads.local.loadContainer()

            await loads.clustered.evict()

            #expect(try await loads.local.loadContainer() === before)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("evicting the local load leaves the clustered entry")
        func evictingTheLocalLoadLeavesTheClusteredEntry() async throws {
            let loads = try Loads()
            try await loads.preloadBoth()
            let before = try await loads.clustered.loadContainer()

            await loads.local.evict()

            #expect(try await loads.clustered.loadContainer() === before)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("evicting the clustered load closes its transport")
        func evictingTheClusteredLoadClosesItsTransport() async throws {
            let loads = try Loads()
            try await loads.preloadBoth()

            await loads.clustered.evict()

            #expect(loads.worker.closeCount == 1)
        }

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        @Test("evicting the local load leaves the transport open")
        func evictingTheLocalLoadLeavesTheTransportOpen() async throws {
            let loads = try Loads()
            try await loads.preloadBoth()

            await loads.local.evict()

            #expect(loads.worker.closeCount == 0)
        }

        #if os(macOS)
        /// The body runs in a process of its own: `evictAll()` drops the state of every
        /// test that runs beside it. See ``FoundationModelsCacheTests``.
        @available(macOS 27.0, *)
        @Test("evictAll() closes the transport of a clustered load")
        func evictAllClosesTheTransportOfAClusteredLoad() async {
            await #expect(processExitsWith: .success) {
                let loads = try Loads()
                try await loads.preloadBoth()

                await MLXLanguageModel.evictAll()

                #expect(loads.worker.closeCount == 1)
            }
        }
        #endif
    }
}

/// The containers a loader hands out, one for each load. The model cache runs the loader on
/// its own task, thus a `Mutex` holds the queue.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class ContainerQueue: Sendable {
    private let containers: Mutex<[ModelContainer]>

    /// Makes the queue of `containers`, in the order of the loads.
    init(_ containers: [ModelContainer]) {
        self.containers = Mutex(containers)
    }

    /// The container of the next load.
    ///
    /// - Throws: ``ContainerQueueExhausted`` when every container was handed out.
    func next() throws -> ModelContainer {
        try containers.withLock { queue in
            guard !queue.isEmpty else { throw ContainerQueueExhausted() }
            return queue.removeFirst()
        }
    }
}

/// Thrown when a model loads more times than its test planned.
private struct ContainerQueueExhausted: Error {}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)

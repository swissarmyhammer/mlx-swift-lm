// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2) && os(macOS)

/// A model in the shared cache, with one prompt cache entry in the shared store.
///
/// Each of the two holds one array of ``bufferBytes``, and nothing else holds them. An array made
/// from bytes takes its buffer from the MLX allocator with no GPU command, thus releasing the last
/// reference puts the buffer in the pool at once and no completion handler can delay it.
@available(macOS 27.0, *)
private struct EvictionPoolFixture {

    /// The bytes of each array. The size is far above any buffer the fixture itself leaves.
    private static let bufferBytes = 1 << 24

    /// The arrays the fixture holds: one in the model, one in the prompt cache entry.
    private static let bufferCount = 2

    /// The key of the entry's array in the model state.
    private static let stateKey = LMOutput.Key<MLXArray>("test.evicted-state")

    /// A fresh array that no other reference holds.
    private static func makeBuffer() -> MLXArray {
        MLXArray([UInt8](repeating: 1, count: bufferBytes), [bufferBytes])
    }

    /// The model the tests evict.
    let model: MLXLanguageModel

    /// The MLX active bytes with the model and the entry in place.
    let activeBytesBeforeEviction: Int

    /// Loads a model that holds one array and checks in a prompt cache entry that holds another.
    static func make() async throws -> EvictionPoolFixture {
        let modelID = "org/eviction-pool"
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [],
            weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
            load: { _, _ in
                makeScriptedContainer(
                    modelID: modelID, rounds: [], weights: EvictionPoolFixture.makeBuffer())
            })
        try await model.preload()
        var state = LMOutput.State()
        state[Self.stateKey] = Self.makeBuffer()
        await ExecutorPromptCacheStore.current.checkIn(
            ExecutorPromptCacheKey(modelID: modelID, sessionID: "evicted-session"),
            ExecutorPromptCacheEntry(caches: [KVCacheSimple()], tokens: [1], state: state))
        return EvictionPoolFixture(model: model, activeBytesBeforeEviction: Memory.activeMemory)
    }

    /// Whether both arrays were live before the eviction, and both have left MLX now: neither
    /// their live bytes nor pool buffers of their size remain.
    var bothBuffersLeftMLX: Bool {
        activeBytesBeforeEviction >= Self.bufferCount * Self.bufferBytes
            && Memory.activeMemory < Self.bufferBytes
            && Memory.cacheMemory < Self.bufferBytes
    }
}

/// Eviction promises to free the GPU memory, and the MLX pool keeps every freed buffer until a
/// clear runs. Each body runs in a process of its own: the pool is process-global, so a parallel
/// suite would move the readings, and `evictAll()` here would wipe the state of that suite.
@Suite("MLXLanguageModel eviction returns the pool to the system")
struct MLXLanguageModelEvictionPoolTests {

    @available(macOS 27.0, *)
    @Test("evict() leaves no buffer of the model or its prompt cache in the pool")
    func evictClearsThePoolAfterReleasingTheModel() async {
        await #expect(processExitsWith: .success) {
            let fixture = try await EvictionPoolFixture.make()

            await fixture.model.evict()

            exit(fixture.bothBuffersLeftMLX ? EXIT_SUCCESS : EXIT_FAILURE)
        }
    }

    @available(macOS 27.0, *)
    @Test("evictAll() leaves no buffer of a model or its prompt cache in the pool")
    func evictAllClearsThePoolAfterReleasingEveryModel() async {
        await #expect(processExitsWith: .success) {
            let fixture = try await EvictionPoolFixture.make()

            await MLXLanguageModel.evictAll()

            exit(fixture.bothBuffersLeftMLX ? EXIT_SUCCESS : EXIT_FAILURE)
        }
    }
}

#endif

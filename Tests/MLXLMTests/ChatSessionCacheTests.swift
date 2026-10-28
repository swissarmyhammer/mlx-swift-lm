// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMTestSupport
import MLXNN
import MLXOptimizers
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

final class ChatSessionCacheTests: XCTestCase, ChatSessionFixtures {
    // MARK: - KV Cache

    func testCurrentCacheNilBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheAfterGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testInitWithKVCache() async throws {
        // build a cache from an initial session
        let container = ModelContainer(context: model())
        let initial = ChatSession(container, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        try await initial.withCache { [targetLength, generationParameters] cache in
            XCTAssertNotNil(cache)

            if let cache {
                // restore the cache into a new session and verify generation continues
                let restored = ChatSession(
                    container,
                    cache: cache.map { $0.copy() },
                    generateParameters: generationParameters)
                let result = try await restored.respond(to: "hello again")
                XCTAssertGreaterThan(result.count, targetLength, result)
            }
        }
    }

    func testSaveCacheThrowsBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        let url = TemporaryPromptCacheFile.makeURL()
        do {
            try await session.saveCache(to: url)
            XCTFail("expected ChatSessionError.noCacheAvailable")
        } catch ChatSessionError.noCacheAvailable {
            // expected
        }
    }

    func testSaveAndRestoreCache() async throws {
        let ctx = model()
        let initial = ChatSession(ctx, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        let url = TemporaryPromptCacheFile.makeURL()
        try await initial.saveCache(to: url)

        let promptCache = try loadPromptCacheSnapshot(url: url)
        let restored = ChatSession(
            ctx, promptCache: promptCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testSaveCachePreservesRestoredState() async throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let stateKey = LMOutput.Key<MLXArray>("test.chatSessionState")
        let stateValue = MLXArray([Int32(2), 4, 6])
        var state = LMOutput.State()
        state[stateKey] = stateValue
        let promptCache = PromptCacheSnapshot(cache: [cache], metadata: [:], state: state)
        let session = ChatSession(
            model(), promptCache: promptCache, generateParameters: generationParameters)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try await session.saveCache(to: url)
        let restored = try loadPromptCacheSnapshot(url: url)
        let restoredState = try XCTUnwrap(restored.state?[stateKey])

        XCTAssertTrue(allClose(restoredState, stateValue, rtol: 0, atol: 0).item(Bool.self))
    }

    // MARK: - Carrying model state

    /// Stateful continuation models cannot safely rewind arbitrary model state
    /// when speculative proposals are rejected, so the session must use normal
    /// generation without loading a draft model.
    /// A model that hands back an anchor on every prefill still speculates, and
    /// the anchor survives the speculative turn: the iterator seeds it, threads
    /// it through verification, and hands it back for the next turn to persist.
    func testSpeculativeDecodingCarriesModelStateProducedByItsTurns() async throws {
        let context = Self.makeStateProducingModel()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: parameters
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(to: "hello") {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }
        XCTAssertNotNil(try XCTUnwrap(info).speculativeDecodingTelemetry)

        // A second turn must succeed with the anchor the speculative turn carried.
        _ = try await session.respond(to: "hello again")

        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try await session.saveCache(to: url)

        let restored = try loadPromptCacheSnapshot(url: url)
        XCTAssertNotNil(
            restored.state?[StateProducingModel.anchorKey],
            "the session dropped the model state produced by its turns")
    }

    /// The end-to-end path the snapshot API exists for: a session that carries
    /// model state saves it with its cache, and a new session restored from
    /// that snapshot keeps generating.
    func testStateProducingSessionSurvivesSaveAndRestore() async throws {
        let context = Self.makeStateProducingModel()
        let session = ChatSession(context, generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")

        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try await session.saveCache(to: url)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        XCTAssertNotNil(
            snapshot.state?[StateProducingModel.anchorKey],
            "saveCache dropped the model state")

        let restored = ChatSession(
            Self.makeStateProducingModel(), promptCache: snapshot,
            generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    /// A snapshot can be built in process, not only read from disk, so a caller
    /// holding a cache and its state has a way to keep the two together.
    func testSessionAcceptsInProcessPromptCacheSnapshot() async throws {
        let context = Self.makeStateProducingModel()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)

        // Warm a cache directly, the way a caller building a prefix cache in
        // process would, and pair it with its state without going through disk.
        let cache = try context.model.newCache(parameters: parameters)
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("cached prefix")]))
        let iterator = try TokenIterator(
            input: input, model: context.model, cache: cache, parameters: parameters)
        XCTAssertTrue(cache.contains { $0.offset > 0 })

        let snapshot = PromptCacheSnapshot(cache: cache, state: iterator.state)
        XCTAssertNotNil(snapshot.state?[StateProducingModel.anchorKey])

        let restored = ChatSession(
            context, promptCache: snapshot, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testForkContinuesTheConversationOfItsSource() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        let branch = await session.fork()
        let branchReply = try await collectGeneration(branch.streamDetails(to: "second"))
        let sourceReply = try await collectGeneration(session.streamDetails(to: "third"))

        // Both continue from the first prompt and the tokens it generated.
        XCTAssertEqual(branchReply.info.cachedPromptTokenCount, firstPromptLength + 3)
        XCTAssertEqual(sourceReply.info.cachedPromptTokenCount, firstPromptLength + 3)
    }

    func testForkWithHistoryPrefillsOnlyWhatItsCacheDoesNotHold() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let instructions = "shared instructions"
        let session = ChatSession(
            model(processor: processor),
            instructions: instructions,
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)
        let sourceState = await session.withCache(Self.cachedValues)

        let fork = await session.fork(history: [])
        let forkReply = try await collectGeneration(fork.streamDetails(to: "second"))
        let forkRenderedLength = await lengthIterator.next()
        let forkPromptLength = try XCTUnwrap(forkRenderedLength)

        // The system marker, the instructions, their end marker and the user marker.
        let sharedPrefixLength = instructions.unicodeScalars.count + 3
        XCTAssertEqual(forkReply.info.cachedPromptTokenCount, sharedPrefixLength)
        XCTAssertEqual(forkReply.info.promptTokenCount, forkPromptLength - sharedPrefixLength)

        // The fork rewrote rows the source holds, and the source did not see it.
        let stateAfterFork = await session.withCache(Self.cachedValues)
        XCTAssertEqual(stateAfterFork, sourceState)
        let sourceReply = try await collectGeneration(session.streamDetails(to: "third"))
        XCTAssertEqual(sourceReply.info.cachedPromptTokenCount, firstPromptLength + 3)
    }

    func testForkStartsColdWhenTheTranscriptCarriesMedia() async throws {
        let (_, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        let answer = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))

        // The source's tokens exactly, over a different image.
        let fork = await session.fork(history: [
            .user("inspect this", images: [.array(MLXArray([Float(1)]))]),
            .assistant(answer),
        ])
        let reply = try await collectGeneration(fork.streamDetails(to: "describe it"))

        XCTAssertEqual(reply.info.cachedPromptTokenCount, 0)

        // A branch keeps the transcript its cache was built from, media included.
        let branch = await session.fork()
        let branchReply = try await collectGeneration(branch.streamDetails(to: "describe it"))
        XCTAssertGreaterThan(branchReply.info.cachedPromptTokenCount, 0)
    }

    func testForkStartingColdCopiesNoCache() async throws {
        let (_, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let copies = CopyCounter()
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        context.model = CopyCountingLanguageModel(context.model, copies: copies)
        let session = ChatSession(context, generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")

        _ = await session.fork(history: [
            .user("inspect this", images: [.array(MLXArray([Float(0)]))])
        ])
        XCTAssertEqual(copies.value, 0)

        // A text transcript reuses the cache, so the same session copies every layer.
        _ = await session.fork(history: [])
        let layerCount = await session.withCache { $0?.count ?? 0 }
        XCTAssertEqual(copies.value, layerCount)
    }

    func testForkWithHistoryKeepsOnlyTheSharedPrefixOnceItResponds() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        // Plain caches on every layer, so no sliding window drops the shared prefix.
        var context = Self.makeModel(processor: processor)
        context.model = CopyCountingLanguageModel(context.model, copies: CopyCounter())
        let session = ChatSession(
            context,
            instructions: "shared instructions",
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: String(repeating: "a long document ", count: 64))
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        let fork = await session.fork(history: [])
        let reply = try await collectGeneration(fork.streamDetails(to: "hi"))
        XCTAssertEqual(reply.info.cachedPromptTokenCount, "shared instructions".count + 3)

        // The fork holds the shared prefix and room to grow, not the document.
        let capacities =
            await fork.withCache { cache in
                (cache ?? []).compactMap { ($0 as? KVCacheSimple)?.keys?.dim(2) }
            } ?? []
        XCTAssertFalse(capacities.isEmpty)
        for capacity in capacities {
            XCTAssertLessThan(capacity, firstPromptLength)
        }
    }

    func testForkSharesTheDraftModelItsSourceHasNotLoaded() async throws {
        let loadCounter = DraftModelLoadCounter()
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                await loadCounter.increment()
                return ModelContainer(context: Self.makeModel())
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        let fork = await session.fork()
        _ = try await session.respond(to: "hello")
        _ = try await fork.respond(to: "hello")

        let loadCount = await loadCounter.value
        XCTAssertEqual(loadCount, 1)
    }

    func testCurrentCacheNilForHistorySessionBeforeGeneration() async throws {
        // .history state should behave like .empty: no cache until first generation
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(), history: history, generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheNonNilForHistorySessionAfterGeneration() async throws {
        // after generation from .history state, cache transitions to .kvcache
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(),
            history: history,
            generateParameters: generationParameters)
        _ = try await session.respond(to: "hello again")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testCurrentCacheNilAfterClear() async throws {
        // clear() resets to .empty; currentCache() should return nil again
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
        await session.clear()
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCacheStatusPreservesTheRealizedRequestWhenParametersChange() async throws {
        let initialConfiguration = KVCacheConfiguration(
            strategy: .turboQuant(.qualityFirst))
        let session = ChatSession(
            model(),
            cache: [KVCacheSimple()],
            generateParameters: GenerateParameters(kvCache: initialConfiguration))

        session.generateParameters.kvCache = KVCacheConfiguration(strategy: .fullPrecision)

        let status = try await session.cacheStatus()
        XCTAssertEqual(status.phase, .realized)
        XCTAssertEqual(status.requestSource, .typed)
        XCTAssertEqual(status.requestedConfiguration, initialConfiguration)
    }

    func testSessionRetainsCacheReplacementTriggeredDuringDecode() async throws {
        let (_, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let affine = try AffineKVCacheConfiguration(
            bits: 4, groupSize: 64, compressionStart: 8)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(
                maxTokens: 3,
                kvCache: KVCacheConfiguration(
                    strategy: .affine(affine), compatibility: .allowPartial),
                temperature: 0))

        _ = try await session.respond(to: "hello")
        let observedFirstOffset = try await session.withCache { cache in
            let cache = try XCTUnwrap(cache)
            let quantized = try XCTUnwrap(
                cache.first { $0 is QuantizedKVCache } as? QuantizedKVCache)
            return quantized.offset
        }
        let firstOffset = try XCTUnwrap(observedFirstOffset)
        let firstStatus = try await session.cacheStatus()
        XCTAssertEqual(firstStatus.processedTokenCount, firstOffset)
        XCTAssertEqual(firstStatus.phase, .realized)
        XCTAssertEqual(firstStatus.requestSource, .typed)
        XCTAssertEqual(firstStatus.requestedStrategy, .affine)
        XCTAssertGreaterThan(firstStatus.compressedLayerCount, 0)
        XCTAssertTrue(
            firstStatus.layers.contains {
                $0.state == .active && $0.resolvedStrategy == .affine
            })

        _ = try await session.respond(to: "hello again")
        try await session.withCache { cache in
            let cache = try XCTUnwrap(cache)
            let quantized = try XCTUnwrap(
                cache.first { $0 is QuantizedKVCache } as? QuantizedKVCache)
            XCTAssertGreaterThan(quantized.offset, firstOffset)
        }
    }

    func testModelContainerReportsCappedSlidingAndFullAttentionCaches() async throws {
        let container = ModelContainer(context: model())
        let parameters = GenerateParameters(maxTokens: 1, maxKVSize: 64)

        let status = try await container.cacheStatus(parameters: parameters)

        XCTAssertEqual(status.requestSource, .legacy)
        XCTAssertEqual(status.requestedConfiguration?.capacity?.maxTokens, 64)
        XCTAssertEqual(status.capacityDisposition, .fullyApplied)
        XCTAssertEqual(status.layers.count, 8)
        XCTAssertTrue(status.attentionMaxSizes.allSatisfy { $0 == 64 })
    }

    func testModelContainerPreservesNativeSlidingWindowForTypedCapacity() async throws {
        let container = ModelContainer(context: model())
        let capacity = try KVCacheConfiguration.Capacity(
            maxTokens: 64, preservedPrefixTokens: 3)
        let parameters = GenerateParameters(
            maxTokens: 1,
            kvCache: KVCacheConfiguration(capacity: capacity))

        let status = try await container.cacheStatus(parameters: parameters)

        XCTAssertEqual(status.requestSource, .typed)
        XCTAssertEqual(status.requestedConfiguration?.capacity, capacity)
        XCTAssertEqual(status.capacityDisposition, .fullyApplied)
        XCTAssertEqual(status.capacityAppliedLayerCount, 1)
        XCTAssertEqual(status.layers.count, 8)
        XCTAssertEqual(
            status.attentionMaxSizes.compactMap { $0 },
            [512, 512, 512, 512, 512, 64, 512, 512])
    }

    func testSessionReportsRealizedRawCacheInsteadOfPlannedLayout() async throws {
        let rawCache: [KVCache] = Array(repeating: 0, count: 8).map { _ in KVCacheSimple() }
        let session = ChatSession(
            model(),
            cache: rawCache,
            generateParameters: GenerateParameters(maxTokens: 1, maxKVSize: 64))

        let status = try await session.cacheStatus()

        XCTAssertEqual(status.phase, .realized)
        XCTAssertEqual(status.requestSource, .legacy)
        XCTAssertEqual(status.requestedConfiguration?.capacity?.maxTokens, 64)
        XCTAssertEqual(status.capacityDisposition, .ignored)
        XCTAssertTrue(status.attentionMaxSizes.allSatisfy { $0 == nil })
    }

    func testSessionRebuildsStructuredCacheWhenMaxKVSizeChanges() async throws {
        let session = ChatSession(
            model(), generateParameters: GenerateParameters(maxTokens: 1))

        _ = try await session.respond(to: "first")
        let initial = try await session.cacheStatus()
        XCTAssertEqual(initial.capacityDisposition, .notRequested)
        XCTAssertTrue(initial.attentionMaxSizes.contains(512))
        XCTAssertTrue(initial.attentionMaxSizes.contains(nil))

        session.generateParameters = GenerateParameters(maxTokens: 1, maxKVSize: 64)
        let stale = try await session.cacheStatus()
        XCTAssertEqual(stale.phase, .realized)
        XCTAssertNil(stale.requestedConfiguration)

        _ = try await session.respond(to: "second")
        let limited = try await session.cacheStatus()
        XCTAssertEqual(limited.capacityDisposition, .fullyApplied)
        XCTAssertEqual(limited.requestSource, .legacy)
        XCTAssertTrue(limited.attentionMaxSizes.allSatisfy { $0 == 64 })
        await session.withCache { cache in
            XCTAssertEqual(cache?.count, 8)
            XCTAssertTrue(cache?.allSatisfy { $0.maxSize == 64 } == true)
        }

        session.generateParameters = GenerateParameters(maxTokens: 1)
        _ = try await session.respond(to: "third")
        let restored = try await session.cacheStatus()
        XCTAssertEqual(restored.capacityDisposition, .notRequested)
        XCTAssertTrue(restored.attentionMaxSizes.contains(512))
        XCTAssertTrue(restored.attentionMaxSizes.contains(nil))
    }

    func testSessionRebuildsWhenTypedRequestRestoresNativeWindows() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 1, maxKVSize: 64))

        _ = try await session.respond(to: "legacy")
        let legacy = try await session.cacheStatus()
        XCTAssertEqual(legacy.capacityDisposition, .fullyApplied)
        XCTAssertEqual(legacy.requestSource, .legacy)
        XCTAssertTrue(legacy.attentionMaxSizes.allSatisfy { $0 == 64 })

        let capacity = try KVCacheConfiguration.Capacity(maxTokens: 64)
        session.generateParameters = GenerateParameters(
            maxTokens: 1,
            kvCache: KVCacheConfiguration(
                capacity: capacity,
                compatibility: .allowPartial))

        _ = try await session.respond(to: "typed")
        let typed = try await session.cacheStatus()
        XCTAssertEqual(typed.requestSource, .typed)
        XCTAssertEqual(typed.capacityDisposition, .fullyApplied)
        XCTAssertEqual(typed.capacityAppliedLayerCount, 1)
        XCTAssertEqual(
            typed.attentionMaxSizes.compactMap { $0 },
            [512, 512, 512, 512, 512, 64, 512, 512])
    }
}

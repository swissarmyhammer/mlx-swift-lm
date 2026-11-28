// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// The KV-cache configurations the tests of this file request.
///
/// The test models write caches whose head dimension the affine group size does not divide, so
/// each configuration allows a partial application: the layers it cannot compress stay in full
/// precision, and the generation still runs.
enum KVCacheOptionFixtures {

    /// Affine 8-bit KV-cache compression.
    static let eightBit = KVCacheConfiguration(
        strategy: .affine(.eightBit), compatibility: .allowPartial)

    /// Affine 4-bit KV-cache compression.
    static let fourBit = KVCacheConfiguration(
        strategy: .affine(.fourBit), compatibility: .allowPartial)

    /// Affine 8-bit KV-cache compression that must reach an attention layer, for a model whose
    /// caches the affine group size divides.
    static let compressingEightBit = KVCacheConfiguration(strategy: .affine(.eightBit))

    /// A key and value width the affine group size divides, thus a KV-cache configuration
    /// compresses a scripted cache of this width.
    static let quantizableHeadDimension = AffineKVCacheConfiguration.eightBit.groupSize
}

/// An `MLXLanguageModel` hands its KV-cache configuration to every generation it runs: the
/// passes of `respond`, the guided passes, and the warm-up.
@Suite("A model applies its KV-cache configuration to generation and warm-up", .serialized)
struct KVCacheConfigurationOptionTests {

    /// The token budget of the guided pass.
    private static let guidedMaxTokens = 16

    /// The schema of the guided pass.
    private static let guidedSchema = #"{"type":"boolean"}"#

    /// The token budget of each pass that ``makeParameters`` builds parameters for.
    private static let maxTokens = 32

    /// The round size of the speculative warm-up.
    private static let blockSize = 3

    @Test("the initializer stores the KV-cache configuration")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theInitializerStoresTheKVCacheConfiguration() {
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "org/repo"),
            kvCache: KVCacheOptionFixtures.eightBit,
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: stubLoad())

        #expect(model.kvCache == KVCacheOptionFixtures.eightBit)
    }

    @Test("a model with no KV-cache configuration keeps full precision")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aModelWithNoKVCacheConfigurationKeepsFullPrecision() {
        #expect(makeStubModel("org/repo").kvCache == nil)
    }

    @Test("makeParameters sets the KV-cache configuration it is given")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeParametersSetsTheKVCacheConfigurationItIsGiven() {
        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: Self.maxTokens, requestedTemperature: nil, samplingConfiguration: nil,
            kvCache: KVCacheOptionFixtures.eightBit)

        #expect(parameters.kvCache == KVCacheOptionFixtures.eightBit)
    }

    @Test("makeParameters takes the KV-cache configuration of the served model")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeParametersTakesTheKVCacheConfigurationOfTheServedModel() {
        let parameters = MLXLanguageModel.Executor.$kvCache.withValue(
            KVCacheOptionFixtures.fourBit
        ) {
            MLXLanguageModel.Executor.makeParameters(
                maxTokens: Self.maxTokens, requestedTemperature: nil, samplingConfiguration: nil)
        }

        #expect(parameters.kvCache == KVCacheOptionFixtures.fourBit)
    }

    @Test("makeParameters outside a respond keeps full precision")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeParametersOutsideARespondKeepsFullPrecision() {
        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: Self.maxTokens, requestedTemperature: nil, samplingConfiguration: nil)

        #expect(parameters.kvCache == nil)
    }

    @Test("the warm-up parameters carry the KV-cache configuration")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theWarmUpParametersCarryTheKVCacheConfiguration() {
        let parameters = MLXLanguageModel.warmUpParameters(
            prefillStepSize: nil, kvCache: KVCacheOptionFixtures.eightBit)

        #expect(parameters.kvCache == KVCacheOptionFixtures.eightBit)
    }

    @Test("the speculative warm-up parameters carry the KV-cache configuration")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theSpeculativeWarmUpParametersCarryTheKVCacheConfiguration() {
        let parameters = MLXLanguageModel.speculativeWarmUpParameters(
            blockSize: Self.blockSize, kvCache: KVCacheOptionFixtures.eightBit)

        #expect(parameters.kvCache == KVCacheOptionFixtures.eightBit)
    }

    @Test("a warm-up builds its caches under the KV-cache configuration of the model")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aWarmUpBuildsItsCachesUnderTheKVCacheConfiguration() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let recording = PrefillStepRecordingModel(ownDefaultStepSize: nil)
        let configuration = ModelConfiguration(id: "probe/kv-cache-warm-up-\(UUID().uuidString)")
        let container = ModelContainer(
            context: ModelContext(
                configuration: configuration, model: recording,
                processor: FixedPromptInputProcessor(), tokenizer: GuidedByteTokenizer()))
        let model = MLXLanguageModel(
            configuration: configuration, kvCache: KVCacheOptionFixtures.eightBit,
            weightsLocation: { _ in weights }, load: { _, _ in container })

        try await model.warmUp()
        await model.evict()

        #expect(recording.requestedKVCaches == [KVCacheOptionFixtures.eightBit])
    }

    @Test("a guided pass builds its caches under the KV-cache configuration of the served model")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aGuidedPassBuildsItsCachesUnderTheKVCacheConfiguration() async throws {
        let modelID = "probe/kv-cache-guided-\(UUID().uuidString)"
        let recording = PrefillStepRecordingModel(ownDefaultStepSize: nil)
        let context = ModelContext(
            configuration: ModelConfiguration(id: modelID), model: recording,
            processor: FixedPromptInputProcessor(), tokenizer: GuidedByteTokenizer())
        let grammar = try await MLXLanguageModel.Executor.guidedGrammar(
            kind: .json, source: Self.guidedSchema, reserveSchemaJSON: Self.guidedSchema,
            maxTokens: Self.guidedMaxTokens, modelID: modelID, context: context)
        let input = try await context.processor.prepare(input: UserInput(prompt: ""))

        try MLXLanguageModel.Executor.$kvCache.withValue(KVCacheOptionFixtures.eightBit) {
            _ = try MLXLanguageModel.Executor.runGuidedLoop(
                input: input, transcriptBoundary: nil, grammar: grammar,
                maxTokens: Self.guidedMaxTokens, promptCache: ExecutorPromptCacheSlot(nil),
                speculation: nil, context: context
            ) { _ in }
        }

        #expect(recording.requestedKVCaches == [KVCacheOptionFixtures.eightBit])
    }
}

/// A prompt cache built under one KV-cache configuration never serves a turn under another, and
/// a second turn under the same configuration still reuses the prefix of the first.
@Suite("The prompt cache keeps KV-cache configurations apart")
struct KVCacheConfigurationPromptCacheTests: PromptCachePlanFixtures {

    /// The session every turn of this suite belongs to.
    private static let sessionID = "kv-cache-session"

    /// A store in a folder of its own, whose writer writes nothing.
    ///
    /// These tests read the memory tier alone. A writer that writes nothing thus leaves no file,
    /// and two stores never share a file name.
    private func makeStore() -> ExecutorPromptCacheStore {
        PromptCacheTestStore.withoutWrites(folderNamed: "KVCacheConfigurationPromptCacheTests")
    }

    /// A scripted session model under `modelID` whose KV-cache configuration is `kvCache`.
    ///
    /// Two models of one `modelID` share one loaded container, thus they differ in their
    /// KV-cache configuration alone.
    ///
    /// - Parameters:
    ///   - modelID: the identity of the model.
    ///   - weights: the directory that makes the model available.
    ///   - kvCache: the KV-cache configuration of the model, or `nil` for full precision.
    ///   - recurrentLayerCount: the recurrent layers beside the attention layer. The default of
    ///     zero gives a model that is not hybrid.
    ///   - headDimension: the width of each key and value the model writes.
    /// - Returns: the model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModel(
        modelID: String, weights: URL, kvCache: KVCacheConfiguration?,
        recurrentLayerCount: Int = 0, headDimension: Int = ScriptedCacheShape.headDimension
    ) -> MLXLanguageModel {
        ScriptedSessionModel.make(
            modelID: modelID, weights: weights,
            scripts: Array(
                repeating: ScriptedSessionModel.scriptedResponse,
                count: ScriptedSessionModel.maximumPassCount),
            cacheLayerCount: ScriptedSessionModel.cacheLayerCount,
            processor: PromptBytesInputProcessor(), recurrentLayerCount: recurrentLayerCount,
            kvCache: kvCache, headDimension: headDimension)
    }

    /// A scripted session model whose 8-bit configuration compresses its attention cache.
    ///
    /// - Parameters:
    ///   - weights: the directory that makes the model available.
    ///   - recurrentLayerCount: the recurrent layers beside the attention layer.
    /// - Returns: the model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeCompressingModel(
        weights: URL, recurrentLayerCount: Int = 0
    ) -> MLXLanguageModel {
        makeModel(
            modelID: "probe/kv-cache-compressing-\(UUID().uuidString)", weights: weights,
            kvCache: KVCacheOptionFixtures.compressingEightBit,
            recurrentLayerCount: recurrentLayerCount,
            headDimension: KVCacheOptionFixtures.quantizableHeadDimension)
    }

    /// Runs a first turn of the session under `first`, then a second turn under `second`, and
    /// gives the prompt tokens the second turn reused.
    ///
    /// - Parameters:
    ///   - first: the KV-cache configuration of the first turn.
    ///   - second: the KV-cache configuration of the second turn.
    /// - Returns: the reused prompt tokens of the second turn.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func reusedTokensOfASecondTurn(
        first: KVCacheConfiguration?, second: KVCacheConfiguration?
    ) async throws -> Int {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let modelID = "probe/kv-cache-session-\(UUID().uuidString)"
        let firstModel = makeModel(modelID: modelID, weights: weights, kvCache: first)
        let secondModel = makeModel(modelID: modelID, weights: weights, kvCache: second)
        let store = makeStore()

        try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID),
            model: firstModel, inside: store)
        let reused = try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID, turns: 2),
            model: secondModel, inside: store)
        await secondModel.evict()
        return reused
    }

    @Test("a full-precision session and an 8-bit session name two caches")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aFullPrecisionSessionAndAnEightBitSessionNameTwoCaches() {
        let request = makeExecutorRequest(
            transcript: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID))
        let fullPrecision = MLXLanguageModel.Executor.sessionCacheKey(
            for: request, modelID: "test/kv-cache", kvCache: nil)
        let eightBit = MLXLanguageModel.Executor.sessionCacheKey(
            for: request, modelID: "test/kv-cache", kvCache: KVCacheOptionFixtures.eightBit)

        #expect(fullPrecision != eightBit)
    }

    @Test("an 8-bit session and a 4-bit session name two caches")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEightBitSessionAndAFourBitSessionNameTwoCaches() {
        let request = makeExecutorRequest(
            transcript: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID))
        let eightBit = MLXLanguageModel.Executor.sessionCacheKey(
            for: request, modelID: "test/kv-cache", kvCache: KVCacheOptionFixtures.eightBit)
        let fourBit = MLXLanguageModel.Executor.sessionCacheKey(
            for: request, modelID: "test/kv-cache", kvCache: KVCacheOptionFixtures.fourBit)

        #expect(eightBit != fourBit)
    }

    @Test("a full-precision cache never serves an 8-bit turn")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aFullPrecisionCacheNeverServesAnEightBitTurn() async throws {
        let reused = try await reusedTokensOfASecondTurn(
            first: nil, second: KVCacheOptionFixtures.eightBit)

        #expect(reused == 0)
    }

    @Test("an 8-bit cache never serves a full-precision turn")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEightBitCacheNeverServesAFullPrecisionTurn() async throws {
        let reused = try await reusedTokensOfASecondTurn(
            first: KVCacheOptionFixtures.eightBit, second: nil)

        #expect(reused == 0)
    }

    @Test("a 4-bit cache never serves an 8-bit turn")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aFourBitCacheNeverServesAnEightBitTurn() async throws {
        let reused = try await reusedTokensOfASecondTurn(
            first: KVCacheOptionFixtures.fourBit, second: KVCacheOptionFixtures.eightBit)

        #expect(reused == 0)
    }

    @Test("a second 8-bit turn reuses the prefix of the first")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSecondEightBitTurnReusesThePrefixOfTheFirst() async throws {
        let reused = try await reusedTokensOfASecondTurn(
            first: KVCacheOptionFixtures.eightBit, second: KVCacheOptionFixtures.eightBit)

        #expect(reused > 0)
    }

    @Test("an 8-bit pass checks in the caches its generation compressed")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEightBitPassChecksInTheCachesItsGenerationCompressed() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeCompressingModel(weights: weights)
        let store = makeStore()

        try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID), model: model,
            inside: store)
        let entry = await store.peek(
            PromptCacheTestKey.make(
                modelID: .init(model.modelID), sessionID: .init(Self.sessionID),
                kvCache: model.kvCache))
        await model.evict()

        #expect(entry?.caches.first is QuantizedKVCache)
    }

    @Test("a second 8-bit turn of a hybrid model reuses the prefix of the first")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSecondEightBitTurnOfAHybridModelReusesThePrefixOfTheFirst() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeCompressingModel(weights: weights, recurrentLayerCount: 1)
        let store = makeStore()

        try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID), model: model,
            inside: store)
        let reused = try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID, turns: 2),
            model: model, inside: store)
        await model.evict()

        #expect(reused > 0)
    }

    @Test("releasing a session releases its cache under every KV-cache configuration")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func releasingASessionReleasesItsCacheUnderEveryKVCacheConfiguration() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeModel(
            modelID: "probe/kv-cache-release-\(UUID().uuidString)", weights: weights,
            kvCache: KVCacheOptionFixtures.eightBit)
        let store = makeStore()
        for kvCache in [nil, KVCacheOptionFixtures.eightBit, KVCacheOptionFixtures.fourBit] {
            await store.checkIn(
                PromptCacheTestKey.make(
                    modelID: .init(model.modelID), sessionID: .init(Self.sessionID),
                    kvCache: kvCache),
                entry(tokens: [1, 2, 3]))
        }

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.releasePromptCache(sessionID: Self.sessionID)
        }

        #expect(await store.retainedSessionCount == 0)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

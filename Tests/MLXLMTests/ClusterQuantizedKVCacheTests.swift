// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

/// A cluster model whose workers quantize their KV caches decodes the tokens the local model
/// decodes under the same KV-cache configuration.
///
/// The worker is an ``InProcessRemoteModelWorker`` over the target of the tiny Qwen 3.5 fixture,
/// with a full-attention head the affine group size divides, so the configuration quantizes the
/// attention cache of each rank.
@Suite("A cluster model quantizes the KV caches of its workers", .serialized)
struct ClusterQuantizedKVCacheTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the prompt of each turn of a chat session.
    private static let chatPrompts = [[3, 14, 15], [9, 2, 6], [5, 35, 8]]

    /// The tokens each generation decodes.
    private static let maxTokens = 16

    /// A full-attention head width the affine group size divides.
    private static let quantizableHeadDimension = AffineKVCacheConfiguration.eightBit.groupSize

    /// The bit widths of the legacy `kvBits` field the tests run.
    private static let quantizedBits = [8, 4]

    /// A compression start four decode steps past the prompt, so the caches convert inside the
    /// decode.
    private static let lateCompressionStart = 12

    /// Greedy parameters of ``maxTokens`` tokens that quantize the KV cache to `kvBits` bits from
    /// `quantizedKVStart`.
    private static func parameters(kvBits: Int, quantizedKVStart: Int = 0) -> GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens, kvBits: kvBits, quantizedKVStart: quantizedKVStart,
            temperature: 0)
    }

    /// Greedy parameters of ``maxTokens`` tokens under the typed configuration `kvCache`.
    private static func parameters(kvCache: KVCacheConfiguration) -> GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, kvCache: kvCache, temperature: 0)
    }

    /// The fixture whose attention caches quantize, with a drafter of `mode`.
    private static func fixture(
        drafter mode: Qwen35MTPFixture.DrafterMode = .copiesTarget
    ) throws -> Qwen35MTPFixture {
        try Qwen35MTPFixture(drafter: mode, headDimension: quantizableHeadDimension)
    }

    /// The input of `tokens`.
    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    /// The tokens and completion of one generation on `context` with `parameters`.
    private static func decode(
        _ tokens: [Int], context: ModelContext, parameters: GenerateParameters,
        cache: [KVCache]? = nil
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: input(tokens), cache: cache, parameters: parameters, context: context))
    }

    /// The tokens and completion of one MTP generation on the cluster model of `context`, with
    /// the drafter of its rank 0.
    private static func speculate(
        _ tokens: [Int], context: ModelContext, parameters: GenerateParameters
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        let model = try #require(context.model as? ClusterLanguageModel)
        return try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: input(tokens), parameters: parameters, context: context,
                mtpDrafter: try #require(model.mtpDrafter)))
    }

    /// The caches a plain local generation of `parameters` left: the quantized replacements of
    /// the caches the model made.
    private static func localGenerationCaches(
        _ fixture: Qwen35MTPFixture, parameters: GenerateParameters
    ) throws -> [KVCache] {
        var iterator = try TokenIterator(
            input: input(prompt), model: fixture.target, parameters: parameters)
        while iterator.next() != nil {}
        return iterator.cache
    }

    /// The remote cache of a plain generation of `parameters` on a worker of `followerCount`
    /// followers, and the worker.
    private static func clusteredGeneration(
        _ fixture: Qwen35MTPFixture, parameters: GenerateParameters, followerCount: Int = 0
    ) async throws -> (cache: RemoteKVCache, worker: InProcessRemoteModelWorker) {
        let worker = fixture.makeWorker(followerCount: followerCount)
        let context = try fixture.makeClusteredModelContext(worker: worker)
        let cache = try context.model.newCache(parameters: parameters)
        _ = try await decode(prompt, context: context, parameters: parameters, cache: cache)
        return (try #require(cache.first as? RemoteKVCache), worker)
    }

    /// Runs the turns of ``chatPrompts`` on `local` and on `clustered`, then expects that the
    /// clustered session writes the local turns and reuses its prompt cache after the first turn.
    private static func expectTheLocalTurns(
        local: ChatSession, clustered: ChatSession, fixture: Qwen35MTPFixture,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async throws {
        var localTurns: [ChatTurn] = []
        var clusteredTurns: [ChatTurn] = []
        for prompt in chatPrompts {
            localTurns.append(try await ChatTurn(local, prompt: fixture.text(for: prompt)))
            clusteredTurns.append(try await ChatTurn(clustered, prompt: fixture.text(for: prompt)))
        }

        #expect(clusteredTurns == localTurns, sourceLocation: sourceLocation)
        #expect(
            clusteredTurns.dropFirst().allSatisfy { $0.cachedPromptTokenCount > 0 },
            sourceLocation: sourceLocation)
    }

    // MARK: - The new cache

    @Test("a new cache of kvBits carries the quantization to the worker")
    func aNewCacheOfKVBitsCarriesTheQuantization() throws {
        let fixture = try Self.fixture()
        let worker = fixture.makeWorker(followerCount: 1)
        let model = try ClusterLanguageModel.connect(over: worker)

        _ = try model.newCache(parameters: Self.parameters(kvBits: 8))

        let quantizations = worker.requests.compactMap { request in
            if case .newCache(_, let quantization) = request.op { quantization } else { nil }
        }
        #expect(
            quantizations == [
                RemoteKVCacheQuantization(
                    bits: 8, groupSize: AffineKVCacheConfiguration.eightBit.groupSize,
                    compressionStart: 0, compatibility: .allowPartial)
            ])
    }

    @Test("every rank holds the quantized bytes rank 0 holds")
    func everyRankHoldsTheQuantizedBytesOfRankZero() async throws {
        let generation = try await Self.clusteredGeneration(
            Self.fixture(), parameters: Self.parameters(kvBits: 8), followerCount: 1)

        #expect(
            generation.worker.followerStatuses(of: generation.cache.id).map(\.residentByteCount)
                == [generation.cache.residentByteCount])
    }

    @Test("the remote cache reports the bytes of the quantized local caches")
    func theRemoteCacheReportsTheQuantizedBytes() async throws {
        let fixture = try Self.fixture()
        let parameters = Self.parameters(kvBits: 8)
        let local = try Self.localGenerationCaches(fixture, parameters: parameters)
        try #require(local.contains { $0 is QuantizedKVCache })

        let clustered = try await Self.clusteredGeneration(fixture, parameters: parameters)

        #expect(clustered.cache.residentByteCount == local.reduce(0) { $0 + $1.residentByteCount })
    }

    @Test("a generation quantizes the caches of the worker with one op")
    func aGenerationQuantizesWithOneOp() async throws {
        let generation = try await Self.clusteredGeneration(
            Self.fixture(), parameters: Self.parameters(kvBits: 8))

        let quantizeOps = generation.worker.requests.filter { request in
            if case .quantize = request.op { true } else { false }
        }
        #expect(quantizeOps.count == 1)
    }

    @Test("a maxKVSize is refused")
    func aMaxKVSizeIsRefused() throws {
        let model = try ClusterLanguageModel.connect(over: Self.fixture().makeWorker())

        #expect(throws: ClusterModelError.unsupported(.cacheConfiguration)) {
            try model.newCache(parameters: GenerateParameters(maxKVSize: Self.maxTokens))
        }
    }

    @Test("a TurboQuant configuration is refused")
    func aTurboQuantConfigurationIsRefused() throws {
        let model = try ClusterLanguageModel.connect(over: Self.fixture().makeWorker())

        #expect(throws: ClusterModelError.unsupported(.cacheConfiguration)) {
            try model.newCache(
                parameters: Self.parameters(
                    kvCache: KVCacheConfiguration(strategy: .turboQuant(.balanced))))
        }
    }

    // MARK: - Parity

    @Test("a greedy generation of kvBits writes the local tokens", arguments: quantizedBits)
    func aGreedyGenerationOfKVBitsWritesTheLocalTokens(bits: Int) async throws {
        let fixture = try Self.fixture()
        let parameters = Self.parameters(kvBits: bits)
        let local = try await Self.decode(
            Self.prompt, context: fixture.makeModelContext(), parameters: parameters)

        let clustered = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            parameters: parameters)

        #expect(clustered.tokens == local.tokens)
    }

    @Test("a greedy generation that quantizes inside the decode writes the local tokens")
    func aGreedyGenerationThatQuantizesLateWritesTheLocalTokens() async throws {
        let fixture = try Self.fixture()
        let parameters = Self.parameters(kvBits: 8, quantizedKVStart: Self.lateCompressionStart)
        let local = try await Self.decode(
            Self.prompt, context: fixture.makeModelContext(), parameters: parameters)

        let clustered = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            parameters: parameters)

        #expect(clustered.tokens == local.tokens)
    }

    @Test("a greedy generation of a typed affine configuration writes the local tokens")
    func aGreedyGenerationOfATypedAffineConfigurationWritesTheLocalTokens() async throws {
        let fixture = try Self.fixture()
        let parameters = Self.parameters(
            kvCache: KVCacheConfiguration(strategy: .affine(.eightBit)))
        let local = try await Self.decode(
            Self.prompt, context: fixture.makeModelContext(), parameters: parameters)

        let clustered = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            parameters: parameters)

        #expect(clustered.tokens == local.tokens)
    }

    @Test("a chat session of kvBits 8 reuses its prompt cache as the local session does")
    func aChatSessionOfKVBitsReusesItsPromptCacheAsTheLocalSessionDoes() async throws {
        let fixture = try Self.fixture()
        let parameters = Self.parameters(kvBits: 8)
        let local = ChatSession(fixture.makeModelContext(), generateParameters: parameters)
        let clustered = ChatSession(
            try fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            generateParameters: parameters)

        try await Self.expectTheLocalTurns(local: local, clustered: clustered, fixture: fixture)
    }

    // MARK: - MTP

    @Test("a greedy MTP generation of kvBits 8 writes the local MTP tokens")
    func aGreedyMTPGenerationOfKVBitsWritesTheLocalMTPTokens() async throws {
        let fixture = try Self.fixture(drafter: .independent)
        let parameters = Self.parameters(kvBits: 8)
        let local = try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: Self.input(Self.prompt), parameters: parameters,
                context: fixture.makeModelContext(), mtpDrafter: fixture.drafter,
                blockSize: Qwen35MTPFixture.blockSize))

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(
                worker: fixture.makeWorker(drafting: true, followerCount: 1)),
            parameters: parameters)

        #expect(speculative.tokens == local.tokens)
    }

    @Test("a greedy MTP generation of kvBits 8 proposes drafts")
    func aGreedyMTPGenerationOfKVBitsProposesDrafts() async throws {
        let fixture = try Self.fixture()

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)),
            parameters: Self.parameters(kvBits: 8))

        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }

    @Test("after the rejects of an MTP generation of kvBits 8 every rank holds rank 0's offset")
    func afterTheRejectsOfAQuantizedMTPGenerationEveryRankHoldsTheOffset() async throws {
        let fixture = try Self.fixture(drafter: .independent)
        let parameters = Self.parameters(kvBits: 8)
        let worker = fixture.makeWorker(drafting: true, followerCount: 1)
        let context = try fixture.makeClusteredModelContext(worker: worker)
        let model = try #require(context.model as? ClusterLanguageModel)
        let cache = try model.newCache(parameters: parameters)
        _ = try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: Self.input(Self.prompt), cache: cache, parameters: parameters,
                context: context, mtpDrafter: try #require(model.mtpDrafter)))
        let remote = try #require(cache.first as? RemoteKVCache)

        #expect(worker.followerStatuses(of: remote.id).map(\.offset) == [remote.offset])
    }

    @Test("a chat session with MTP and kvBits 8 reuses its prompt cache as the local one does")
    func aChatSessionWithMTPAndKVBitsReusesItsPromptCacheAsTheLocalOneDoes() async throws {
        let fixture = try Self.fixture(drafter: .independent)
        let parameters = Self.parameters(kvBits: 8)
        let local = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: parameters)
        let clustered = ChatSession(
            try fixture.makeClusteredModelContext(
                worker: fixture.makeWorker(drafting: true, followerCount: 1)),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: parameters)

        try await Self.expectTheLocalTurns(local: local, clustered: clustered, fixture: fixture)
    }
}

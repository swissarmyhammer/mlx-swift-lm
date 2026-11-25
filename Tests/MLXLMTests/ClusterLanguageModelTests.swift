// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLMCommon

/// A model whose forward pass runs in a worker decodes the tokens the local model decodes.
///
/// The worker is an ``InProcessRemoteModelWorker`` over the target of the tiny Qwen 3.5
/// fixture: a real model, with one recurrent and one attention layer, on seeded weights.
@Suite("A cluster model decodes as the local model", .serialized)
struct ClusterLanguageModelTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each generation decodes.
    private static let maxTokens = 16

    /// The tokens fed after ``prompt``.
    private static let followUp = [12, 7, 30, 21]

    /// Greedy parameters of ``maxTokens`` tokens.
    private static let greedy = GenerateParameters(maxTokens: maxTokens, temperature: 0)

    /// The input of `tokens`.
    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    /// The tokens and completion of one greedy generation on `context`.
    private static func decode(
        _ tokens: [Int], context: ModelContext, cache: [KVCache]? = nil,
        parameters: GenerateParameters = greedy
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: input(tokens), cache: cache, parameters: parameters, context: context))
    }

    // MARK: - Generation

    @Test("a greedy generation writes the local tokens")
    func aGreedyGenerationWritesTheLocalTokens() async throws {
        let fixture = try Qwen35MTPFixture()
        let local = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.maxTokens, speculative: false)

        let clustered = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: fixture.makeWorker()))

        #expect(clustered.tokens == local.tokens)
    }

    @Test("a long prompt prefills in forwards of at most one chunk and writes the local tokens")
    func aLongPromptPrefillsInChunksAndWritesTheLocalTokens() async throws {
        let fixture = try Qwen35MTPFixture()
        let prompt = (0 ..< 2_500).map { $0 % Qwen35MTPFixture.vocabularySize }
        var parameters = Self.greedy
        parameters.prefill = PrefillParameters(
            stepSize: RemoteModelProtocol.maximumChunkTokenCount)
        let worker = fixture.makeWorker()

        let local = try await Self.decode(
            prompt, context: fixture.makeModelContext(), parameters: parameters)
        let clustered = try await Self.decode(
            prompt, context: fixture.makeClusteredModelContext(worker: worker),
            parameters: parameters)

        let forwardLengths = worker.requests.compactMap { request -> Int? in
            guard case .forward(let tokens, _, _) = request.op else { return nil }
            return tokens.count
        }
        #expect(forwardLengths.allSatisfy { $0 <= RemoteModelProtocol.maximumChunkTokenCount })
        #expect(forwardLengths.prefix(4).reduce(0, +) == prompt.count)
        #expect(clustered.tokens == local.tokens)
    }

    @Test("a chat session reuses its prompt cache as the local session does")
    func aChatSessionReusesItsPromptCacheAsTheLocalSessionDoes() async throws {
        let fixture = try Qwen35MTPFixture()
        let local = try fixture.makeChatSession(speculative: false, maxTokens: Self.maxTokens)
        let clustered = ChatSession(
            try fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            generateParameters: Self.greedy)

        var localTurns: [ChatTurn] = []
        var clusteredTurns: [ChatTurn] = []
        for prompt in [[3, 14, 15], [9, 2, 6], [5, 35, 8]] {
            localTurns.append(try await ChatTurn(local, prompt: fixture.text(for: prompt)))
            clusteredTurns.append(try await ChatTurn(clustered, prompt: fixture.text(for: prompt)))
        }

        #expect(clusteredTurns == localTurns)
        #expect(clusteredTurns.dropFirst().allSatisfy { $0.cachedPromptTokenCount > 0 })
    }

    @Test("a reconciled prompt cache trims on the worker and decodes the local tokens")
    func aReconciledPromptCacheTrimsOnTheWorker() async throws {
        let model = try AttentionOnlyTarget.make()
        let worker = InProcessRemoteModelWorker(
            model: model, vocabularySize: Qwen35MTPFixture.vocabularySize)
        let fixture = try Qwen35MTPFixture()
        var localContext = fixture.makeModelContext()
        localContext.model = model
        var clusteredContext = localContext
        clusteredContext.model = try ClusterLanguageModel.connect(over: worker)
        let cached = Self.prompt + [12, 7, 30, 21]
        let next = Self.prompt + [40, 41]

        var decodes: [[Int]] = []
        for context in [localContext, clusteredContext] {
            let cache = try context.model.newCache(parameters: nil)
            _ = try PromptCacheCheckpoint.prefill(
                cached, into: cache, state: nil, model: context.model, prefill: .init())
            let reuse = try #require(
                reconcilePromptCache(promptTokens: next, cachedTokens: cached, caches: cache))
            decodes.append(
                try await Self.decode(
                    reuse.fedTokens(of: next), context: context, cache: cache
                ).tokens)
        }

        #expect(decodes[1] == decodes[0])
        #expect(worker.requests.contains { if case .trim = $0.op { true } else { false } })
    }

    @Test("a cluster context loads a checkpoint directory without its weights")
    func aClusterContextLoadsACheckpointDirectoryWithoutItsWeights() async throws {
        let directory = try Self.makeCheckpointDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let suppressed = 5
        try Data(#"{"suppress_tokens": [\#(suppressed)]}"#.utf8)
            .write(to: directory.appendingPathComponent("generation_config.json"))
        let fixture = try Qwen35MTPFixture()

        let context = try await loadClusterModelContext(
            from: directory, using: PseudoWordTokenizerLoader(), transport: fixture.makeWorker())

        let model = try #require(context.model as? ClusterLanguageModel)
        #expect(model.suppressedTokenIds == [suppressed])
    }

    @Test("a cluster context refuses a worker that runs another model type")
    func aClusterContextRefusesAWorkerOfAnotherModelType() async throws {
        let directory = try Self.makeCheckpointDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try Qwen35MTPFixture()
        let worker = InProcessRemoteModelWorker(
            model: fixture.target, vocabularySize: Qwen35MTPFixture.vocabularySize,
            modelType: "inkling2")

        await #expect(
            throws: ClusterModelError.modelTypeMismatch(expected: "qwen3_5_text", found: "inkling2")
        ) {
            try await loadClusterModelContext(
                from: directory, using: PseudoWordTokenizerLoader(), transport: worker)
        }
    }

    /// A checkpoint directory of the text configuration of the fixture, with no weights.
    private static func makeCheckpointDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("cluster-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(
            Qwen35FixtureJSON.text(
                mtpLayers: 0, hiddenLayers: Qwen35MTPFixture.hiddenLayers,
                fullAttentionInterval: Qwen35MTPFixture.fullAttentionInterval,
                hiddenSize: Qwen35MTPFixture.hiddenSize,
                vocabularySize: Qwen35MTPFixture.vocabularySize, tieWordEmbeddings: false
            ).utf8
        ).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    // MARK: - The op stream

    @Test("every op carries the next sequence number")
    func everyOpCarriesTheNextSequenceNumber() async throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()

        _ = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: worker))

        #expect(worker.requests.map(\.sequence) == Array(1 ... UInt64(worker.requests.count)))
    }

    @Test("a generation runs inside a request that the next generation ends")
    func aGenerationRunsInsideARequest() async throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()
        let context = try fixture.makeClusteredModelContext(worker: worker)

        _ = try await Self.decode(Self.prompt, context: context)
        _ = try await Self.decode(Self.prompt, context: context)

        let ops = worker.requests.map(\.op)
        let firstForward = try #require(
            ops.firstIndex { if case .forward = $0 { true } else { false } })
        #expect(ops.filter { $0 == .begin }.count == 2)
        #expect(ops.filter { $0 == .end }.count == 1)
        #expect(ops[..<firstForward].contains(.begin))
    }

    @Test("a copy keeps its own position when the original trims")
    func aCopyKeepsItsOwnPositionWhenTheOriginalTrims() async throws {
        let model = try ClusterLanguageModel.connect(
            over: InProcessRemoteModelWorker(
                model: try AttentionOnlyTarget.make(),
                vocabularySize: Qwen35MTPFixture.vocabularySize))
        let cache = try model.newCache(parameters: nil)
        _ = model.nextTokenLogits(
            Self.input(Self.prompt).text[text: .newAxis], cache: cache, state: nil)

        let copy = cache[0].copy()
        let trimmed = cache[0].trim(3)

        #expect(trimmed == 3)
        #expect(cache[0].offset == Self.prompt.count - 3)
        #expect(copy.offset == Self.prompt.count)
    }

    @Test("the cache storage of a hybrid remote cache returns to its checkpoint by copy")
    func theCacheStorageOfAHybridRemoteCacheReturnsToItsCheckpoint() async throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())
        let storage = KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)
        _ = model.nextTokenLogits(
            Self.input(Self.prompt).text[text: .newAxis], cache: storage.cache, state: nil)
        storage.commitProcessedTokens(Self.prompt.count)
        let checkpoint = try #require(storage.checkpoint())
        _ = model.nextTokenLogits(
            Self.input(Self.followUp).text[text: .newAxis], cache: storage.cache, state: nil)
        storage.commitProcessedTokens(Self.followUp.count)

        let restored = storage.restore(checkpoint)

        #expect(restored)
        #expect(storage.cache[0].offset == Self.prompt.count)
    }

    @Test("a reset empties the cache on the worker")
    func aResetEmptiesTheCacheOnTheWorker() async throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())
        let cache = try model.newCache(parameters: nil)
        _ = model.nextTokenLogits(
            Self.input(Self.prompt).text[text: .newAxis], cache: cache, state: nil)

        try #require(cache[0] as? RemoteKVCache).reset()

        #expect(cache[0].offset == 0)
    }

    @Test("a cache that goes away frees its caches on the worker")
    func aCacheThatGoesAwayFreesItsCachesOnTheWorker() async throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()
        let model = try ClusterLanguageModel.connect(over: worker)
        var cache: [KVCache]? = try model.newCache(parameters: nil)
        #expect(cache?.count == 1)
        #expect(worker.liveCacheCount == 1)

        cache = nil

        #expect(worker.liveCacheCount == 0)
    }

    @Test("the status of a hybrid cache keeps checkpoints by copy")
    func theStatusOfAHybridCacheKeepsCheckpointsByCopy() throws {
        let fixture = try Qwen35MTPFixture()

        let status = RemoteCacheStatus(describing: try fixture.target.newCache(parameters: nil))

        #expect(status.canLoseRewind)
    }

    @Test("the status of an attention cache rewinds by trim")
    func theStatusOfAnAttentionCacheRewindsByTrim() throws {
        let status = RemoteCacheStatus(
            describing: try AttentionOnlyTarget.make().newCache(parameters: nil))

        #expect(status.isTrimmable)
        #expect(!status.canLoseRewind)
        #expect(status.rewindLimit == .always)
    }

    @Test("a prompt checkpoint copies a remote cache with a sliding window")
    func aPromptCheckpointCopiesARemoteCacheWithASlidingWindow() throws {
        let model = try ClusterLanguageModel.connect(
            over: InProcessRemoteModelWorker(
                model: SlidingWindowCacheModel(), vocabularySize: Qwen35MTPFixture.vocabularySize))
        let cache = try model.newCache(parameters: nil)

        #expect(cache[0].isTrimmable)
        #expect(PromptCacheCheckpoint.applies(to: cache))
    }

    // MARK: - Refusals

    @Test("an image input is refused")
    func anImageInputIsRefused() throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])))

        #expect(throws: ClusterModelError.unsupported(.mediaInput)) {
            try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil, prefill: .init())
        }
    }

    @Test("a KV-cache configuration is refused")
    func aKVCacheConfigurationIsRefused() throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())

        #expect(throws: ClusterModelError.unsupported(.cacheConfiguration)) {
            try model.newCache(parameters: GenerateParameters(kvBits: 8))
        }
    }

    @Test("an MTP generation is refused")
    func anMTPGenerationIsRefused() throws {
        let fixture = try Qwen35MTPFixture()
        let context = try fixture.makeClusteredModelContext(worker: fixture.makeWorker())

        #expect(throws: ClusterModelError.unsupported(.speculativeDecoding)) {
            try generateTokens(
                input: Self.input(Self.prompt), parameters: Self.greedy, context: context,
                mtpDrafter: fixture.drafter)
        }
    }

    @Test("a draft-model generation is refused")
    func aDraftModelGenerationIsRefused() throws {
        let fixture = try Qwen35MTPFixture()
        let context = try fixture.makeClusteredModelContext(worker: fixture.makeWorker())

        #expect(throws: ClusterModelError.unsupported(.speculativeDecoding)) {
            try generateTokens(
                input: Self.input(Self.prompt), parameters: Self.greedy, context: context,
                draftModel: fixture.target)
        }
    }

    @Test("a chat session with speculative decoding is refused")
    func aChatSessionWithSpeculativeDecodingIsRefused() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = ChatSession(
            try fixture.makeClusteredModelContext(worker: fixture.makeWorker()),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: Self.greedy)

        await #expect(throws: ClusterModelError.unsupported(.speculativeDecoding)) {
            try await session.respond(to: fixture.text(for: Self.prompt))
        }
    }

    @Test("a prompt cache file cannot hold a remote cache")
    func aPromptCacheFileCannotHoldARemoteCache() throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())
        let cache = try model.newCache(parameters: nil)

        #expect {
            try preparePromptCacheSave(cache: cache)
        } throws: { error in
            (error as? KVCacheError)?.message
                == "RemoteKVCache has no serializedClassName, so it cannot be saved"
        }
    }

    @Test("a worker of another protocol version is refused")
    func aWorkerOfAnotherProtocolVersionIsRefused() throws {
        let fixture = try Qwen35MTPFixture()
        let worker = InProcessRemoteModelWorker(
            model: fixture.target, vocabularySize: Qwen35MTPFixture.vocabularySize,
            protocolVersion: RemoteModelProtocol.version + 1)

        #expect(
            throws: ClusterModelError.protocolVersionMismatch(
                expected: RemoteModelProtocol.version, found: RemoteModelProtocol.version + 1)
        ) {
            try ClusterLanguageModel.connect(over: worker)
        }
    }

    // MARK: - Failures

    @Test("a failed op fails every later op of the model")
    func aFailedOpFailsEveryLaterOpOfTheModel() throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()
        let model = try ClusterLanguageModel.connect(over: worker)
        let cache = try model.newCache(parameters: nil)
        worker.failOps(after: 0)

        let trimmed = cache[0].trim(1)

        #expect(trimmed == 0)
        #expect(
            throws: ClusterModelError.transportFailed(String(describing: InProcessWorkerFault()))
        ) {
            try model.newCache(parameters: nil)
        }
    }

    // MARK: - Teardown

    @Test("closing a cluster model closes its transport one time")
    func closingAClusterModelClosesItsTransportOneTime() throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()
        let model = try ClusterLanguageModel.connect(over: worker)

        model.close()
        model.close()

        #expect(worker.closeCount == 1)
    }

    @Test("an op after close throws closed")
    func anOpAfterCloseThrowsClosed() throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())

        model.close()

        #expect(throws: ClusterModelError.closed) {
            try model.newCache(parameters: nil)
        }
    }

    @Test("a forward that fails mid-decode cancels the generation")
    func aForwardThatFailsMidDecodeCancelsTheGeneration() async throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker()
        let context = try fixture.makeClusteredModelContext(worker: worker)
        let decodedTokens = 4
        // newCache, begin, the prompt forward and the first decode forwards succeed.
        worker.failOps(after: 3 + decodedTokens)

        let decode = try await Self.decode(Self.prompt, context: context)

        #expect(decode.info.stopReason == .cancelled)
        #expect(decode.tokens.count < Self.maxTokens)
    }
}

/// What one chat turn wrote and how much of its prompt the cache served.
private struct ChatTurn: Equatable {

    /// The text of the reply.
    let text: String

    /// The prompt tokens a reused cache served.
    let cachedPromptTokenCount: Int

    /// Runs one turn of `session`.
    init(_ session: ChatSession, prompt: String) async throws {
        var text = ""
        var cached: Int?
        for try await item in session.streamDetails(to: prompt) {
            text += item.chunk ?? ""
            cached = item.info?.cachedPromptTokenCount ?? cached
        }
        self.text = text
        self.cachedPromptTokenCount = try #require(cached)
    }
}

/// A tiny Qwen 3.5 target whose two layers are both full attention, so every cache trims.
private enum AttentionOnlyTarget {

    /// The seed of the weights.
    private static let seed: UInt64 = 21

    /// Makes the target.
    static func make() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, hiddenLayers: 2, fullAttentionInterval: 1,
                    hiddenSize: Qwen35MTPFixture.hiddenSize,
                    vocabularySize: Qwen35MTPFixture.vocabularySize, tieWordEmbeddings: false
                ).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35TextModel(configuration)
        }
        eval(model)
        return model
    }
}

/// A model whose one cache is a sliding window, which rewinds until its window wraps. Only its
/// cache is read.
private final class SlidingWindowCacheModel: Module, LanguageModel {

    /// The window of the cache.
    private static let window = 8

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [RotatingKVCache(maxSize: Self.window)]
    }
}

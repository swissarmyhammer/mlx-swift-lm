// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// A cluster model speculates with the MTP drafter of its rank 0, and decodes the tokens a plain
/// generation decodes.
///
/// The worker is an ``InProcessRemoteModelWorker`` whose rank 0 holds the drafter: the Qwen 3.5
/// MTP head of the tiny Qwen fixture, or the DSpark drafter of the tiny Inkling 2. A worker with
/// a follower stands for a cluster of two ranks, the second of which holds no drafter.
@Suite("A cluster model speculates with the drafter of its rank 0", .serialized)
struct ClusterMTPDecodingTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each generation decodes.
    private static let maxTokens = 24

    /// Greedy parameters of ``maxTokens`` tokens.
    private static let greedy = GenerateParameters(maxTokens: maxTokens, temperature: 0)

    /// The input of `tokens`.
    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    /// The drafter rank 0 of the cluster model of `context` holds.
    private static func remoteDrafter(of context: ModelContext) throws -> RemoteMTPDrafter {
        let model = try #require(context.model as? ClusterLanguageModel)
        return try #require(model.mtpDrafter)
    }

    /// The tokens and completion of one greedy MTP generation on the cluster model of `context`.
    private static func speculate(
        _ tokens: [Int], context: ModelContext, cache: [KVCache]? = nil
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: input(tokens), cache: cache, parameters: greedy, context: context,
                mtpDrafter: remoteDrafter(of: context)))
    }

    /// The tokens and completion of one greedy plain generation on `context`.
    private static func decode(_ tokens: [Int], context: ModelContext) async throws
        -> Qwen35MTPFixture.GreedyDecode
    {
        try await Qwen35MTPFixture.collect(
            try generateTokens(input: input(tokens), parameters: greedy, context: context))
    }

    // MARK: - Qwen 3.5 MTP

    @Test("the worker reports the drafter of its rank 0")
    func theWorkerReportsTheDrafterOfItsRankZero() throws {
        let fixture = try Qwen35MTPFixture()
        let context = try fixture.makeClusteredModelContext(
            worker: fixture.makeWorker(drafting: true))

        let drafter = try Self.remoteDrafter(of: context)

        #expect(drafter.capabilities.drafterID == fixture.drafter.drafterID.rawValue)
    }

    @Test("a worker without a drafter offers none")
    func aWorkerWithoutADrafterOffersNone() throws {
        let fixture = try Qwen35MTPFixture()
        let model = try ClusterLanguageModel.connect(over: fixture.makeWorker())

        #expect(model.mtpDrafter == nil)
    }

    @Test(
        "a greedy MTP generation writes the clustered plain tokens",
        arguments: Qwen35MTPFixture.DrafterMode.allCases)
    func aGreedyMTPGenerationWritesTheClusteredPlainTokens(
        mode: Qwen35MTPFixture.DrafterMode
    ) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try await Self.decode(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: fixture.makeWorker()))

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)))

        #expect(speculative.tokens == plain.tokens)
    }

    @Test(
        "a greedy MTP generation writes the local tokens",
        arguments: Qwen35MTPFixture.DrafterMode.allCases)
    func aGreedyMTPGenerationWritesTheLocalTokens(mode: Qwen35MTPFixture.DrafterMode)
        async throws
    {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let local = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.maxTokens, speculative: false)

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)))

        #expect(speculative.tokens == local.tokens)
    }

    @Test("a greedy MTP generation accepts the drafts the local MTP generation accepts")
    func aGreedyMTPGenerationAcceptsTheLocalDrafts() async throws {
        let fixture = try Qwen35MTPFixture()
        let local = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.maxTokens, speculative: true)

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)))

        #expect(speculative.info.acceptedDraftTokens == local.info.acceptedDraftTokens)
    }

    @Test("a greedy MTP generation reports the drafts it proposed")
    func aGreedyMTPGenerationReportsTheDraftsItProposed() async throws {
        let fixture = try Qwen35MTPFixture()

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)))

        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }

    @Test("a greedy MTP generation reports no fallback")
    func aGreedyMTPGenerationReportsNoFallback() async throws {
        let fixture = try Qwen35MTPFixture()

        let speculative = try await Self.speculate(
            Self.prompt,
            context: fixture.makeClusteredModelContext(worker: fixture.makeWorker(drafting: true)))

        #expect(speculative.info.passthroughReason == nil)
    }

    @Test("a round runs as one op that carries the drafts and the logits of each verified token")
    func aRoundRunsAsOneOp() async throws {
        let fixture = try Qwen35MTPFixture()
        let worker = fixture.makeWorker(drafting: true)

        _ = try await Self.speculate(
            Self.prompt, context: fixture.makeClusteredModelContext(worker: worker))

        let decodeForwards = worker.requests.filter { request in
            guard case .forward(let tokens, _, _) = request.op else { return false }
            return tokens.count == 1
        }
        #expect(decodeForwards.isEmpty)
    }

    // MARK: - Every rank

    /// Runs a greedy MTP generation of the drafter that rejects drafts on a cluster of two ranks,
    /// and finalizes it, as the generation loop does.
    ///
    /// - Returns: The finalized iterator, the tokens it emitted, its cache and its worker.
    private static func finalizedRejectingGeneration() throws -> (
        iterator: ClusterMTPTokenIterator, tokens: [Int], cache: RemoteKVCache,
        worker: InProcessRemoteModelWorker
    ) {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let worker = fixture.makeWorker(drafting: true, followerCount: 1)
        let model = try ClusterLanguageModel.connect(over: worker)
        let cache = try model.newCache(parameters: nil)
        var iterator = try ClusterMTPTokenIterator(
            input: input(prompt), model: model, drafter: try #require(model.mtpDrafter),
            cacheStorage: KVCacheStorage(cache, plan: .disabled), parameters: greedy,
            blockSize: nil, minimumDraftProbability: 0)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        iterator.finalizeGeneration()
        return (iterator, tokens, try #require(cache.first as? RemoteKVCache), worker)
    }

    @Test("the generation of the rejecting drafter rejects drafts")
    func theGenerationOfTheRejectingDrafterRejectsDrafts() throws {
        let generation = try Self.finalizedRejectingGeneration()

        #expect(generation.iterator.acceptedDraftTokens < generation.iterator.proposedDraftTokens)
    }

    @Test("after rejects the cache holds the prompt and each emitted token a forward fed")
    func afterRejectsTheCacheHoldsEachFedToken() throws {
        let generation = try Self.finalizedRejectingGeneration()

        let unfed = generation.iterator.hasCachedLastToken ? 0 : 1
        #expect(generation.cache.offset == Self.prompt.count + generation.tokens.count - unfed)
    }

    @Test("after rejects every rank holds the offset of rank 0")
    func afterRejectsEveryRankHoldsTheOffsetOfRankZero() throws {
        let generation = try Self.finalizedRejectingGeneration()

        #expect(
            generation.worker.followerStatuses(of: generation.cache.id).map(\.offset)
                == [generation.cache.offset])
    }

    @Test("after rejects no rank failed an op")
    func afterRejectsNoRankFailedAnOp() throws {
        let generation = try Self.finalizedRejectingGeneration()

        #expect(generation.worker.followerFailures.isEmpty)
    }

    @Test("a follow-up generation on the kept cache writes the local tokens")
    func aFollowUpGenerationOnTheKeptCacheWritesTheLocalTokens() async throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let context = try fixture.makeClusteredModelContext(
            worker: fixture.makeWorker(drafting: true, followerCount: 1))
        let cache = try context.model.newCache(parameters: nil)
        let first = try await Self.speculate(Self.prompt, context: context, cache: cache)
        let followUp = [first.tokens[first.tokens.count - 1], 12, 7, 30]
        let fed = try #require(cache.first).offset - Self.prompt.count

        let second = try await Self.speculate(followUp, context: context, cache: cache)

        let local = try await fixture.decodeGreedily(
            prompt: Self.prompt + first.tokens.prefix(fed) + followUp,
            maxTokens: Self.maxTokens, speculative: false)
        #expect(second.tokens == local.tokens)
    }

    // MARK: - Inkling 2 DSpark

    /// The rows of the vocabulary projection of the tiny Inkling 2, padding included.
    private static let inklingVocabularyRows = 50

    /// The tokens of the tiny Inkling 2 vocabulary that are not padding.
    private static let inklingVocabulary = 48

    /// The `config.json` of the tiny Inkling 2 with its DSpark drafter.
    private static let inklingJSON = Inkling2TinyModels.draftJSON(
        vocabularySize: inklingVocabularyRows, unpaddedVocabularySize: inklingVocabulary)

    /// The context of the tiny Inkling 2, whose model is `model`.
    private static func inklingContext(model: any LanguageModel) -> ModelContext {
        let tokenizer = ScalarTokenizer(vocabularySize: inklingVocabulary)
        let configuration = ModelConfiguration(id: "tiny-inkling2-dspark")
        return ModelContext(
            configuration: configuration, model: model,
            processor: ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer)
    }

    @Test("a greedy DSpark generation on Inkling 2 writes the local tokens", arguments: [5, 40])
    func aGreedyDSparkGenerationWritesTheLocalTokens(promptLength: Int) async throws {
        let target = try Inkling2TinyModels.target(json: Self.inklingJSON, bfloat16: true)
        let worker = InProcessRemoteModelWorker(
            model: target, vocabularySize: Self.inklingVocabulary,
            drafter: try Inkling2TinyModels.drafter(json: Self.inklingJSON), followerCount: 1)
        let prompt = (0 ..< promptLength).map { ($0 * 7 + 3) % Self.inklingVocabulary }
        let local = try await Self.decode(prompt, context: Self.inklingContext(model: target))

        let speculative = try await Self.speculate(
            prompt,
            context: Self.inklingContext(model: try ClusterLanguageModel.connect(over: worker)))

        #expect(speculative.tokens == local.tokens)
    }

    @Test("a greedy DSpark generation on Inkling 2 proposes drafts")
    func aGreedyDSparkGenerationProposesDrafts() async throws {
        let target = try Inkling2TinyModels.target(json: Self.inklingJSON, bfloat16: true)
        let worker = InProcessRemoteModelWorker(
            model: target, vocabularySize: Self.inklingVocabulary,
            drafter: try Inkling2TinyModels.drafter(json: Self.inklingJSON))

        let speculative = try await Self.speculate(
            Self.prompt.map { $0 % Self.inklingVocabulary },
            context: Self.inklingContext(model: try ClusterLanguageModel.connect(over: worker)))

        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }
}

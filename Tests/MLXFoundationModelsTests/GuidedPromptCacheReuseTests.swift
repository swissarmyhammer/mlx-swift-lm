// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// A guided turn of a session reads the prefix it shares with the turn before from the prompt
/// cache, and writes what the same turn writes with no cache.
///
/// Each test runs the target of the shared Qwen 3.5 fixture over ``ToolCallAlphabetTokenizer``,
/// in this process or behind a `ClusterLanguageModel` over an in-process worker, with a store of
/// its own. The tokenizer renders a transcript as the text of its entries, one after the other,
/// thus the render of a second turn starts with the whole render of the first.
@Suite(.serialized)
struct GuidedPromptCacheReuseTests {

    /// The session of each test.
    private static let sessionID = "guided-prompt-cache-reuse"

    /// The session a copy of the session's cache makes.
    private static let copySessionID = "guided-prompt-cache-reuse-copy"

    /// The tokens of the instructions.
    private static let instructions = [7, 21, 4, 33, 12, 18]

    /// The tokens of the first prompt.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the second prompt.
    private static let secondPrompt = [11, 40, 8, 27, 19]

    /// The tokens a turn may generate: room for the whole object of the schema.
    private static let responseTokenBudget = 64

    /// The forwards a worker runs before it loses a member: the prompt and a few decode steps,
    /// thus the loss strikes in the middle of the answer.
    private static let forwardsBeforeLoss = 4

    /// The ranks other than 0 each worker stands for in the offset test.
    private static let followerCount = 1

    /// Where the target of a test runs.
    enum Placement: String, CaseIterable, Sendable {
        case local
        case clustered
    }

    /// A model of the fixture, its worker when it is clustered, and a store of its own.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Setup {
        let model: MLXLanguageModel
        let worker: InProcessRemoteModelWorker?
        let store: ExecutorPromptCacheStore
        let tokenizer = ToolCallAlphabetTokenizer()

        /// The folders the setup made, which ``remove()`` deletes.
        private let folders: [URL]

        /// Makes the model of `placement`, and an empty store.
        ///
        /// - Parameters:
        ///   - placement: Where the target runs.
        ///   - speculates: Whether the model speculates with the MTP drafter of the fixture: in
        ///     this process, or on rank 0 of the worker.
        ///   - kvCache: The KV-cache configuration of the model, or `nil` for full precision.
        ///   - followerCount: The ranks other than 0 the worker stands for.
        /// - Throws: The error of the fixture, of the drafter, or of the worker handshake.
        init(
            _ placement: Placement, speculates: Bool = false, kvCache: KVCacheConfiguration? = nil,
            followerCount: Int = 0
        ) throws {
            let weights = try makeScriptedWeightsDirectory()
            let folder = PromptCacheTestStore.folder(named: "GuidedPromptCacheReuseTests")
            folders = [weights, folder]
            let fixture =
                kvCache == nil
                ? try Qwen35MTPFixture()
                : try Qwen35MTPFixture(headDimension: AffineKVCacheConfiguration.eightBit.groupSize)
            let speculativeDecoding =
                speculates ? try Qwen35MTPFixture.makeSpeculativeDecoding() : nil
            let capabilities: [LanguageModelCapabilities.Capability] = [
                .guidedGeneration, .toolCalling,
            ]
            switch placement {
            case .local:
                model = fixture.makeLanguageModel(
                    weights: weights, capabilities: capabilities, tokenizer: tokenizer,
                    speculativeDecoding: speculativeDecoding, kvCache: kvCache)
                worker = nil
            case .clustered:
                let clustered = try fixture.makeClusteredLanguageModel(
                    weights: weights, capabilities: capabilities, tokenizer: tokenizer,
                    speculativeDecoding: speculativeDecoding, drafting: speculates,
                    followerCount: followerCount, kvCache: kvCache)
                model = clustered.model
                worker = clustered.worker
            }
            store = ExecutorPromptCacheStore(
                directory: folder,
                capacityReader: PromptCacheSpoolFixtureVolume.capacity(ofVolumeOf:))
        }

        /// The key of `sessionID` for the model.
        func key(_ sessionID: String = GuidedPromptCacheReuseTests.sessionID)
            -> ExecutorPromptCacheKey
        {
            PromptCacheTestKey.make(modelID: .init(model.modelID), sessionID: .init(sessionID))
        }

        /// Runs `request` inside the store of the setup.
        ///
        /// - Parameter request: The request of the pass.
        /// - Returns: What the pass streamed.
        /// - Throws: The error of the pass or of its channel.
        func respond(to request: LanguageModelExecutorGenerationRequest) async throws -> MTPTurn {
            try await MTPExecutorTurns.respond(to: request, model: model, inside: store)
        }

        /// Runs `request` inside the store of the setup, and records what the pass streamed or
        /// threw.
        ///
        /// - Parameter request: The request of the pass.
        /// - Returns: What the pass streamed, and its failure.
        /// - Throws: The error of an executor's initializer.
        func respondRecordingFailure(
            to request: LanguageModelExecutorGenerationRequest
        ) async throws -> MTPTurn {
            try await ExecutorPromptCacheStore.$current.withValue(store) {
                try await MTPExecutorTurns.respond(to: request, model: model)
            }
        }

        /// Copies the cache of the session to ``copySessionID`` through the public API.
        func copySession() async {
            _ = await ExecutorPromptCacheStore.$current.withValue(store) {
                await model.copyPromptCache(
                    fromSession: GuidedPromptCacheReuseTests.sessionID,
                    toSession: GuidedPromptCacheReuseTests.copySessionID)
            }
        }

        /// The text of `tokens` as the tokenizer spells them.
        func text(_ tokens: [Int]) -> String {
            tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        }

        /// Deletes the folders of the setup.
        func remove() {
            for folder in folders {
                try? FileManager.default.removeItem(at: folder)
            }
        }
    }

    /// What a turn of a request asks for: a schema, a required tool call, or plain text.
    fileprivate enum Kind {
        case schema(includesSchemaInPrompt: Bool)
        case requiredTool
        case plain
    }

    /// A greedy request of `kind` over the transcript of the session: the first prompt, or the
    /// first prompt, the first response and the second prompt.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func request(
        _ kind: Kind, setup: Setup, firstResponse: String? = nil, sessionID: String = sessionID
    ) -> LanguageModelExecutorGenerationRequest {
        let first = [
            TestTranscript.instructions(id: sessionID, setup.text(instructions)),
            TestTranscript.prompt(setup.text(firstPrompt)),
        ]
        let second = firstResponse.map {
            [TestTranscript.response($0), TestTranscript.prompt(setup.text(secondPrompt))]
        }
        let transcript = Transcript(entries: first + (second ?? []))
        switch kind {
        case .schema(let includesSchemaInPrompt):
            return makeExecutorRequest(
                transcript: transcript, schema: FixtureLookupArguments.generationSchema,
                generationOptions: options(),
                contextOptions: ContextOptions(includeSchemaInPrompt: includesSchemaInPrompt))
        case .requiredTool:
            return makeExecutorRequest(
                transcript: transcript,
                enabledTools: [
                    Transcript.ToolDefinition(
                        name: "lookup", description: "Looks up text.",
                        parameters: String.generationSchema)
                ],
                generationOptions: options(toolCallingMode: .required))
        case .plain:
            return makeExecutorRequest(transcript: transcript, generationOptions: options())
        }
    }

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func options(
        toolCallingMode: GenerationOptions.ToolCallingMode? = nil
    ) -> GenerationOptions {
        GenerationOptions(
            samplingMode: .greedy, maximumResponseTokens: responseTokenBudget,
            toolCallingMode: toolCallingMode)
    }

    /// The two turns of one session: `first`, then `second` over the first response.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct TwoTurns {
        let first: MTPTurn
        let second: MTPTurn
    }

    /// Runs `first`, then `second` over the response of `first`, in one session of `setup`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func twoTurns(_ first: Kind, _ second: Kind, setup: Setup) async throws
        -> TwoTurns
    {
        let firstTurn = try await setup.respond(to: request(first, setup: setup))
        let secondTurn = try await setup.respond(
            to: request(second, setup: setup, firstResponse: firstTurn.responseText))
        return TwoTurns(first: firstTurn, second: secondTurn)
    }

    /// The second turn of `twoTurns` run in a new setup of the same arguments, with no cache.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func coldSecondTurn(
        _ second: Kind, after turns: TwoTurns, setup cold: Setup
    ) async throws -> MTPTurn {
        try await cold.respond(
            to: request(second, setup: cold, firstResponse: turns.first.responseText))
    }

    /// The prompt tokens the cache served for `turn`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func cached(_ turn: MTPTurn) -> Int {
        turn.usage.first?.cachedPromptTokenCount ?? 0
    }

    /// Every prompt token of `turn`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(_ turn: MTPTurn) -> Int {
        turn.usage.first?.promptTokenCount ?? 0
    }

    /// The schema turn of each test that carries the schema in its prompt.
    private static let schema = Kind.schema(includesSchemaInPrompt: true)

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aSecondSchemaTurnReadsTheFirstTurnFromTheCache(placement: Placement) async throws {
        let setup = try Setup(placement)
        defer { setup.remove() }

        let turns = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aSecondSchemaTurnWritesTheTextOfAColdTurn(placement: Placement) async throws {
        let setup = try Setup(placement)
        let cold = try Setup(placement)
        defer {
            setup.remove()
            cold.remove()
        }

        let turns = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)
        let coldTurn = try await Self.coldSecondTurn(Self.schema, after: turns, setup: cold)

        #expect(!coldTurn.responseText.isEmpty)
        #expect(turns.second.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aSpeculativeSecondSchemaTurnWritesTheTextOfAColdTurn(placement: Placement) async throws {
        let setup = try Setup(placement, speculates: true)
        let cold = try Setup(placement, speculates: true)
        defer {
            setup.remove()
            cold.remove()
        }

        let turns = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)
        let coldTurn = try await Self.coldSecondTurn(Self.schema, after: turns, setup: cold)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
        #expect(turns.second.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aSpeculativeSecondSchemaTurnSpeculates(placement: Placement) async throws {
        let setup = try Setup(placement, speculates: true)
        defer { setup.remove() }

        let turns = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)

        let drafted =
            turns.second.metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount]
            as? Int
        #expect((drafted ?? 0) > 0)
    }

    /// The entry the session of `setup` keeps.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func entry(of setup: Setup) async throws -> ExecutorPromptCacheEntry {
        try #require(await setup.store.peek(setup.key()))
    }

    /// The MTP drafter continuation `entry` keeps, which must stand at the end of its ledger.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func alignedDrafterState(of entry: ExecutorPromptCacheEntry) throws
        -> MTPDrafterContinuation
    {
        let drafterState = try #require(entry.drafterState, "The entry must keep a drafter state.")
        try #require(
            drafterState.isAligned(
                withMainTokenCount: entry.tokens.count, drafterID: drafterState.drafterID))
        return drafterState
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpeculativeSchemaTurnChecksInADrafterStateAlignedWithItsLedger() async throws {
        let setup = try Setup(.local, speculates: true)
        defer { setup.remove() }

        _ = try await setup.respond(to: Self.request(Self.schema, setup: setup))

        let entry = try await Self.entry(of: setup)
        #expect(
            entry.drafterState.map {
                $0.isAligned(withMainTokenCount: entry.tokens.count, drafterID: $0.drafterID)
            } == true)
    }

    /// The second turn resumes the drafter continuation the first turn checked in, thus its
    /// drafter keeps the entries of the first turn from the start of the prompt. A drafter that
    /// started cold where the cache ends would start its entries there.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpeculativeSecondSchemaTurnResumesTheDrafterStateOfTheFirstTurn() async throws {
        let setup = try Setup(.local, speculates: true)
        defer { setup.remove() }

        _ = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)

        let drafterState = try Self.alignedDrafterState(of: try await Self.entry(of: setup))
        #expect(drafterState.state.cacheStartPosition == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpeculativeSchemaTurnKeepsTheDrafterStateOfItsPromptEndCheckpoint() async throws {
        let setup = try Setup(.local, speculates: true)
        defer { setup.remove() }

        _ = try await setup.respond(to: Self.request(Self.schema, setup: setup))

        let checkpoint = try #require(try await Self.entry(of: setup).checkpoint)
        #expect(
            checkpoint.drafterContinuation.map {
                $0.isAligned(
                    withMainTokenCount: checkpoint.tokens.count, drafterID: $0.drafterID)
            } == true)
    }

    /// The plain turn before the schema turn leaves the MTP drafter continuation of its
    /// generation in the cache, which the schema turn resumes.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aSpeculativeSchemaTurnAfterAPlainTurnWritesTheTextOfAColdTurn(placement: Placement)
        async throws
    {
        let setup = try Setup(placement, speculates: true)
        let cold = try Setup(placement, speculates: true)
        defer {
            setup.remove()
            cold.remove()
        }
        let unprompted = Kind.schema(includesSchemaInPrompt: false)

        let turns = try await Self.twoTurns(.plain, unprompted, setup: setup)
        let coldTurn = try await Self.coldSecondTurn(unprompted, after: turns, setup: cold)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
        #expect(turns.second.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func anEightBitSecondSchemaTurnWritesTheTextOfAColdTurn(placement: Placement) async throws {
        let eightBit = KVCacheConfiguration(strategy: .affine(.eightBit))
        let setup = try Setup(placement, kvCache: eightBit)
        let cold = try Setup(placement, kvCache: eightBit)
        defer {
            setup.remove()
            cold.remove()
        }

        let turns = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)
        let coldTurn = try await Self.coldSecondTurn(Self.schema, after: turns, setup: cold)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
        #expect(turns.second.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aRequiredToolTurnReadsThePlainTurnBeforeItFromTheCache(placement: Placement)
        async throws
    {
        let setup = try Setup(placement)
        defer { setup.remove() }

        let turns = try await Self.twoTurns(.plain, .requiredTool, setup: setup)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func aPlainTurnReadsTheSchemaTurnBeforeItFromTheCache(placement: Placement) async throws {
        let setup = try Setup(placement)
        defer { setup.remove() }

        let turns = try await Self.twoTurns(
            .schema(includesSchemaInPrompt: false), .plain, setup: setup)

        #expect(Self.cached(turns.second) >= Self.prompt(turns.first))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases)
    func theCopyOfASchemaSessionReadsTheCopiedPrefix(placement: Placement) async throws {
        let setup = try Setup(placement)
        defer { setup.remove() }
        let first = try await setup.respond(to: Self.request(Self.schema, setup: setup))
        await setup.copySession()

        let copy = try await setup.respond(
            to: Self.request(
                Self.schema, setup: setup, firstResponse: first.responseText,
                sessionID: Self.copySessionID))

        #expect(Self.cached(copy) >= Self.prompt(first))
    }

    /// Two speculative turns of one session: a guided turn and the turn after it, or a plain
    /// turn and the guided turn after it.
    enum SpeculativeTurnPair: CaseIterable, Sendable {

        /// A schema turn whose prompt carries no schema, then a plain turn.
        case schemaThenPlain

        /// A schema turn, then a schema turn.
        case schemaThenSchema

        /// A plain turn, then a schema turn whose prompt carries no schema.
        case plainThenSchema

        /// The kind of the first turn and of the second.
        fileprivate var kinds: (first: Kind, second: Kind) {
            switch self {
            case .schemaThenPlain: (.schema(includesSchemaInPrompt: false), .plain)
            case .schemaThenSchema:
                (GuidedPromptCacheReuseTests.schema, GuidedPromptCacheReuseTests.schema)
            case .plainThenSchema: (.plain, .schema(includesSchemaInPrompt: false))
            }
        }
    }

    /// The characters at the end of the first response that the render of a second turn writes
    /// in another form, as a chat template writes the end of a message again: none, the last
    /// token, which the first turn never fed, and a tail that reaches into the tokens it fed.
    private static let rewrittenTails = [0, 1, 2, 3]

    /// `text` with each of its last `count` characters written as another character.
    private static func rewritingTail(of text: String, count: Int) -> String {
        String(text.dropLast(count)) + String(text.suffix(count).map { $0 == "_" ? "-" : "_" })
    }

    /// The second turn of `pair` on a clustered setup that speculates, the first response its
    /// render carries, and the start position of the drafter of rank 0 in each round of that
    /// turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ClusteredSecondTurn {
        let turn: MTPTurn
        let renderedFirstResponse: String
        let drafterStarts: [Int]

        /// Runs both turns of `pair` in one session of `setup`.
        ///
        /// - Parameters:
        ///   - pair: The kinds of the two turns.
        ///   - setup: A clustered setup that speculates.
        ///   - rewrittenTail: The characters at the end of the first response that the render
        ///     of the second turn writes in another form.
        /// - Throws: The error of a pass, or the failed requirement when the setup has no
        ///   worker.
        init(of pair: SpeculativeTurnPair, on setup: Setup, rewritingTail rewrittenTail: Int)
            async throws
        {
            let worker = try #require(setup.worker)
            let first = try await setup.respond(
                to: GuidedPromptCacheReuseTests.request(pair.kinds.first, setup: setup))
            let firstRounds = worker.drafterStartPositionsOfRounds.count
            renderedFirstResponse = GuidedPromptCacheReuseTests.rewritingTail(
                of: first.responseText, count: rewrittenTail)
            turn = try await setup.respond(
                to: GuidedPromptCacheReuseTests.request(
                    pair.kinds.second, setup: setup, firstResponse: renderedFirstResponse))
            drafterStarts = Array(worker.drafterStartPositionsOfRounds.dropFirst(firstRounds))
        }
    }

    /// The drafter of rank 0 keeps its entries from the first position of the session's cache
    /// at the first round of the second turn. A drafter that started cold at the second turn
    /// would start its entries where the second turn found the cache.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: SpeculativeTurnPair.allCases, rewrittenTails)
    func aClusteredSecondTurnStartsTheDrafterOfRankZeroWarm(
        pair: SpeculativeTurnPair, rewrittenTail: Int
    ) async throws {
        let setup = try Setup(.clustered, speculates: true)
        defer { setup.remove() }

        let second = try await ClusteredSecondTurn(
            of: pair, on: setup, rewritingTail: rewrittenTail)

        #expect(second.drafterStarts.first == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: SpeculativeTurnPair.allCases, rewrittenTails)
    func aClusteredSecondTurnWritesTheTextOfAColdTurn(
        pair: SpeculativeTurnPair, rewrittenTail: Int
    ) async throws {
        let setup = try Setup(.clustered, speculates: true)
        let cold = try Setup(.clustered, speculates: true)
        defer {
            setup.remove()
            cold.remove()
        }

        let second = try await ClusteredSecondTurn(
            of: pair, on: setup, rewritingTail: rewrittenTail)
        let coldTurn = try await cold.respond(
            to: Self.request(
                pair.kinds.second, setup: cold, firstResponse: second.renderedFirstResponse))

        #expect(!coldTurn.responseText.isEmpty)
        #expect(second.turn.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: SpeculativeTurnPair.allCases)
    func aClusteredSecondTurnReportsTheDraftsTheTargetAccepted(pair: SpeculativeTurnPair)
        async throws
    {
        let setup = try Setup(.clustered, speculates: true)
        defer { setup.remove() }

        let second = try await ClusteredSecondTurn(of: pair, on: setup, rewritingTail: 0)

        let accepted =
            second.turn.metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount]
            as? Int
        #expect((accepted ?? 0) > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func everyRankEndsASpeculativeSchemaTurnAtItsLedger() async throws {
        let setup = try Setup(.clustered, speculates: true, followerCount: Self.followerCount)
        defer { setup.remove() }
        let worker = try #require(setup.worker)

        _ = try await Self.twoTurns(Self.schema, Self.schema, setup: setup)

        let entry = try #require(await setup.store.peek(setup.key()))
        let cache = try #require(entry.caches.first as? RemoteKVCache)
        let offsets = worker.rankStatuses(of: cache.id).map(\.offset)
        #expect(offsets.count == Self.followerCount + 1)
        #expect(offsets.allSatisfy { $0 == entry.tokens.count })
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnThatLosesAMemberChecksNothingIn() async throws {
        let setup = try Setup(.clustered)
        defer { setup.remove() }
        let worker = try #require(setup.worker)
        let first = try await setup.respond(to: Self.request(Self.schema, setup: setup))
        worker.failForwards(
            after: Self.forwardsBeforeLoss, with: InProcessMemberLoss(lostMember: "m5u2.local"))

        let lost = try await setup.respondRecordingFailure(
            to: Self.request(Self.schema, setup: setup, firstResponse: first.responseText))

        #expect(lost.failure != nil)
        #expect(await setup.store.peek(setup.key()) == nil)
    }
}

#endif

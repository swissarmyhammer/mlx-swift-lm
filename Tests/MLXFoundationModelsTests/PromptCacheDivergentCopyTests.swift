// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXFoundationModels

/// A session seeded with a copy of the prompt cache of a template session
/// (``MLXLanguageModel/copyPromptCache(fromSession:toSession:)``) whose first prompt shares only
/// the instructions with the template reads the instructions from the copy, and answers as a
/// cold session does.
///
/// The model is the tiny Inkling 2 target of ``Inkling2TinyModels``: its sliding-window and
/// convolution caches cannot rewind from the end of the ledger of the template to the end of
/// the instructions, so only a checkpoint there serves the seeded session. Each test runs the
/// model in this process and behind a `ClusterLanguageModel` over an in-process worker, with a
/// store of its own. The first entry of each transcript names the session of the pass.
@Suite(.serialized)
struct PromptCacheDivergentCopyTests {

    /// Where the model runs.
    enum Placement: CaseIterable, Sendable {

        /// In this process.
        case local

        /// In an in-process worker, behind a `ClusterLanguageModel`.
        case cluster
    }

    /// The vocabulary of the scalar tokenizer of ``Qwen35MTPFixture``, every row of it a token.
    private static let vocabularySize = Qwen35MTPFixture.vocabularySize

    /// The session whose cache each test copies.
    private static let template = "divergent-copy-template"

    /// The session that each copy seeds.
    private static let seeded = "divergent-copy-seeded"

    /// The session that reads the prompt of the seeded session in full.
    private static let cold = "divergent-copy-cold"

    /// The tokens of the instructions: more than the window of eight of the sliding layers.
    private static let instructions = [7, 21, 4, 33, 12, 18, 25, 30, 9, 41, 16, 22]

    /// The tokens of the first prompt of the template.
    private static let templatePrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the first prompt of the seeded session, which parts from the template right
    /// after the instructions.
    private static let divergentPrompt = [11, 40, 8, 27, 19, 1, 38]

    /// The tokens of the second prompt of the template.
    private static let followUpPrompt = [12, 7, 30, 21]

    /// The positions of the window of the sliding layers of ``TargetWindow/narrow``.
    private static let narrowWindowSize = 8

    /// The positions of the window of the sliding layers of ``TargetWindow/wide``.
    private static let wideWindowSize = 64

    /// The tokens each pass generates.
    private static let generatedTokenCount = 16

    /// A model at one placement, its store, and the folders they use.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct CopySetup {
        let fixture: Qwen35MTPFixture
        let model: MLXLanguageModel
        let store: ExecutorPromptCacheStore

        /// The worker of the model at ``Placement/cluster``, or nil in this process.
        let worker: InProcessRemoteModelWorker?

        /// The folders the setup made, which ``remove()`` deletes.
        private let folders: [URL]

        /// Makes the model at `placement` and an empty store.
        ///
        /// - Parameters:
        ///   - placement: Where the model runs.
        ///   - speculates: Whether rank 0 of the worker holds the DSpark drafter of the target,
        ///     and the model speculates with it. Only a model at ``Placement/cluster`` can.
        ///   - window: The window of the sliding layers of the target.
        /// - Throws: The error of the fixture, of the target, of the drafter, or of the worker
        ///   handshake.
        init(placement: Placement, speculates: Bool = false, window: TargetWindow = .narrow)
            throws
        {
            let weights = try makeScriptedWeightsDirectory()
            let folder = PromptCacheTestStore.folder(named: "PromptCacheDivergentCopyTests")
            folders = [weights, folder]
            fixture = try Qwen35MTPFixture()
            let json = Inkling2TinyModels.draftJSON(
                vocabularySize: PromptCacheDivergentCopyTests.vocabularySize,
                unpaddedVocabularySize: PromptCacheDivergentCopyTests.vocabularySize,
                slidingWindowSize: window.size)
            let target = try Inkling2TinyModels.target(json: json)
            let served: any MLXLMCommon.LanguageModel
            switch placement {
            case .local:
                served = target
                worker = nil
            case .cluster:
                let worker = InProcessRemoteModelWorker(
                    model: target, vocabularySize: PromptCacheDivergentCopyTests.vocabularySize,
                    drafter: speculates ? try Inkling2TinyModels.drafter(json: json) : nil)
                served = try ClusterLanguageModel.connect(over: worker)
                self.worker = worker
            }
            model = fixture.makeLanguageModel(
                weights: weights, target: served,
                speculativeDecoding: speculates
                    ? try Qwen35MTPFixture.makeSpeculativeDecoding() : nil)
            store = ExecutorPromptCacheStore(
                directory: folder,
                capacityReader: PromptCacheSpoolFixtureVolume.capacity(ofVolumeOf:))
        }

        /// Copies the template to the seeded session through the public API.
        ///
        /// - Parameter keptTokenCount: The tokens of the ledger of the template the copy keeps,
        ///   or nil for the whole ledger.
        /// - Returns: What the copy did.
        func copyTemplate(keepingTokens keptTokenCount: Int? = nil) async -> PromptCacheCopyResult {
            await ExecutorPromptCacheStore.$current.withValue(store) {
                if let keptTokenCount {
                    await model.copyPromptCache(
                        fromSession: PromptCacheDivergentCopyTests.template,
                        toSession: PromptCacheDivergentCopyTests.seeded,
                        keepingTokens: keptTokenCount)
                } else {
                    await model.copyPromptCache(
                        fromSession: PromptCacheDivergentCopyTests.template,
                        toSession: PromptCacheDivergentCopyTests.seeded)
                }
            }
        }

        /// Runs the first pass of the template.
        ///
        /// - Returns: What the pass of the template streamed.
        /// - Throws: The error of the pass.
        func respondAsTemplate() async throws -> MTPTurn {
            try await respond(
                as: PromptCacheDivergentCopyTests.template,
                prompt: PromptCacheDivergentCopyTests.templatePrompt)
        }

        /// Runs the first pass of the seeded session over the divergent prompt.
        ///
        /// - Returns: What the pass of the seeded session streamed.
        /// - Throws: The error of the pass.
        func respondAsSeededSession() async throws -> MTPTurn {
            try await respond(
                as: PromptCacheDivergentCopyTests.seeded,
                prompt: PromptCacheDivergentCopyTests.divergentPrompt)
        }

        /// Runs one greedy pass of `sessionID` over the instructions, `history` and `prompt`.
        ///
        /// - Parameters:
        ///   - sessionID: The session of the pass.
        ///   - prompt: The tokens of the prompt after the instructions and `history`.
        ///   - history: The entries of the turns before, after the instructions.
        /// - Returns: What the pass streamed.
        /// - Throws: The error of the pass or of its channel.
        func respond(
            as sessionID: String, prompt: [Int], after history: [Transcript.Entry] = []
        ) async throws -> MTPTurn {
            let instructions = TestTranscript.instructions(
                id: sessionID, fixture.text(for: PromptCacheDivergentCopyTests.instructions))
            let transcript = Transcript(
                entries: [instructions] + history + [
                    TestTranscript.prompt(fixture.text(for: prompt))
                ]
            )
            let request = makeExecutorRequest(
                transcript: transcript,
                generationOptions: GenerationOptions(
                    samplingMode: .greedy,
                    maximumResponseTokens: PromptCacheDivergentCopyTests.generatedTokenCount))
            let turn = try await ExecutorPromptCacheStore.$current.withValue(store) {
                try await MTPExecutorTurns.respond(to: request, model: model)
            }
            try #require(turn.failure == nil, "The pass failed: \(turn.failure ?? "")")
            try #require(
                turn.channelFailure == nil, "The channel failed: \(turn.channelFailure ?? "")")
            return turn
        }

        /// Runs the first pass of the template, copies the template to the seeded session, and
        /// runs the first pass of the seeded session over the divergent prompt.
        ///
        /// - Parameter keptTokenCount: The tokens of the ledger of the template the copy keeps,
        ///   or nil for the whole ledger.
        /// - Returns: What the pass of the seeded session streamed.
        /// - Throws: The error of a pass, or the failed requirement when the copy does not
        ///   copy.
        func respondAsSeededCopy(keepingTokens keptTokenCount: Int? = nil) async throws
            -> MTPTurn
        {
            _ = try await respondAsTemplate()
            let copy = await copyTemplate(keepingTokens: keptTokenCount)
            try #require(copy.copiedTokenCount != nil, "The copy did not copy: \(copy)")
            return try await respondAsSeededSession()
        }

        /// Deletes the folders of the setup.
        func remove() {
            for folder in folders {
                try? FileManager.default.removeItem(at: folder)
            }
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the seeded session reads the instructions from the copy",
        arguments: Placement.allCases)
    func theSeededSessionReadsTheInstructionsFromTheCopy(placement: Placement) async throws {
        let setup = try CopySetup(placement: placement)
        defer { setup.remove() }

        let seededTurn = try await setup.respondAsSeededCopy()

        #expect(seededTurn.usage.first?.cachedPromptTokenCount == Self.instructions.count)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the seeded session answers as a cold session does",
        arguments: Placement.allCases)
    func theSeededSessionAnswersAsAColdSessionDoes(placement: Placement) async throws {
        let setup = try CopySetup(placement: placement)
        defer { setup.remove() }

        let seededTurn = try await setup.respondAsSeededCopy()
        let coldTurn = try await setup.respond(as: Self.cold, prompt: Self.divergentPrompt)

        #expect(seededTurn.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "a copy that keeps the instructions holds the instructions alone",
        arguments: Placement.allCases)
    func aCopyThatKeepsTheInstructionsHoldsTheInstructionsAlone(placement: Placement)
        async throws
    {
        let setup = try CopySetup(placement: placement)
        defer { setup.remove() }
        _ = try await setup.respondAsTemplate()

        let copy = await setup.copyTemplate(keepingTokens: Self.instructions.count)

        #expect(copy == .copied(tokenCount: Self.instructions.count))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the session seeded with the instructions alone reads them from the copy",
        arguments: Placement.allCases)
    func theSessionSeededWithTheInstructionsAloneReadsThem(placement: Placement) async throws {
        let setup = try CopySetup(placement: placement)
        defer { setup.remove() }

        let seededTurn = try await setup.respondAsSeededCopy(
            keepingTokens: Self.instructions.count)

        #expect(seededTurn.usage.first?.cachedPromptTokenCount == Self.instructions.count)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the session seeded with the instructions alone answers as a cold session does",
        arguments: Placement.allCases)
    func theSessionSeededWithTheInstructionsAloneAnswersAsAColdSessionDoes(
        placement: Placement
    ) async throws {
        let setup = try CopySetup(placement: placement)
        defer { setup.remove() }

        let seededTurn = try await setup.respondAsSeededCopy(
            keepingTokens: Self.instructions.count)
        let coldTurn = try await setup.respond(as: Self.cold, prompt: Self.divergentPrompt)

        #expect(seededTurn.responseText == coldTurn.responseText)
    }
}

extension PromptCacheDivergentCopyTests {

    /// The window of the sliding layers of the target.
    enum TargetWindow: CaseIterable, Sendable {

        /// Eight positions, fewer than the instructions: the caches of the template cannot trim
        /// back to the end of the instructions, and only a checkpoint serves there.
        case narrow

        /// More positions than any ledger of a test, as the window of a real Inkling 2 is longer
        /// than a turn: only the convolution layers keep the caches from trimming back.
        case wide

        /// The positions of the window.
        var size: Int {
            switch self {
            case .narrow: PromptCacheDivergentCopyTests.narrowWindowSize
            case .wide: PromptCacheDivergentCopyTests.wideWindowSize
            }
        }
    }

    /// What a copy of the template keeps: its whole ledger, or the instructions alone.
    enum SeededCopy: CaseIterable, Sendable {

        /// The whole ledger of the template.
        case wholeLedger

        /// The tokens of the instructions alone.
        case instructionsAlone

        /// The tokens of the ledger of the template the copy keeps, or nil for the whole ledger.
        var keptTokenCount: Int? {
            switch self {
            case .wholeLedger: nil
            case .instructionsAlone: PromptCacheDivergentCopyTests.instructions.count
            }
        }
    }

    /// The first speculative pass of the seeded session after `copy` of the template, on a
    /// cluster whose rank 0 holds the DSpark drafter, and the start position of the drafter of
    /// rank 0 in each round of that pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct SeededSpeculativePass {
        let turn: MTPTurn
        let drafterStarts: [Int]

        /// Runs the template, `copy`, and the seeded session on `setup`.
        ///
        /// - Parameters:
        ///   - copy: What the copy of the template keeps.
        ///   - setup: A setup at ``Placement/cluster`` that speculates.
        /// - Throws: The error of a pass, or the failed requirement when the setup has no worker
        ///   or the copy does not copy.
        init(after copy: SeededCopy, on setup: CopySetup) async throws {
            let worker = try #require(setup.worker)
            _ = try await setup.respondAsTemplate()
            let copied = await setup.copyTemplate(keepingTokens: copy.keptTokenCount)
            try #require(copied.copiedTokenCount != nil, "The copy did not copy: \(copied)")
            (turn, drafterStarts) = try await worker.drafterStarts {
                try await setup.respondAsSeededSession()
            }
        }
    }

    /// The tokens at the end of the first response that the render of the second turn of the
    /// template writes in another form, as a chat template writes the end of a message again.
    private static let rerenderedTailTokenCount = 2

    /// The second speculative pass of the template over a render that writes the end of the
    /// first response in another form, and the start position of the drafter of rank 0 in each
    /// round of that pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct RerenderedSpeculativePass {
        let turn: MTPTurn
        let history: [Transcript.Entry]
        let drafterStarts: [Int]

        /// Runs both passes of the template on `setup`.
        ///
        /// - Parameter setup: A setup at ``Placement/cluster`` that speculates.
        /// - Throws: The error of a pass, or the failed requirement when the setup has no
        ///   worker.
        init(on setup: CopySetup) async throws {
            let worker = try #require(setup.worker)
            let first = try await setup.respondAsTemplate()
            let rerendered = PromptCacheDivergentCopyTests.rerenderedHistory(
                of: first, setup: setup)
            history = rerendered
            (turn, drafterStarts) = try await worker.drafterStarts {
                try await setup.respond(
                    as: PromptCacheDivergentCopyTests.template,
                    prompt: PromptCacheDivergentCopyTests.followUpPrompt, after: rerendered)
            }
        }
    }

    /// The entries of the first pass of the template, whose response ends in another form:
    /// its last ``rerenderedTailTokenCount`` tokens each one token on.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func rerenderedHistory(of first: MTPTurn, setup: CopySetup)
        -> [Transcript.Entry]
    {
        let tokens = setup.fixture.tokenizer.encode(
            text: first.responseText, addSpecialTokens: false)
        let kept = tokens.dropLast(rerenderedTailTokenCount)
        let rewritten = tokens.suffix(rerenderedTailTokenCount).map { ($0 + 1) % vocabularySize }
        return [
            TestTranscript.prompt(setup.fixture.text(for: templatePrompt)),
            TestTranscript.response(setup.fixture.text(for: Array(kept) + rewritten)),
        ]
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "a speculative turn over a render that rewinds the ledger starts the drafter warm",
        arguments: TargetWindow.allCases)
    func aTurnThatRewindsTheLedgerStartsTheDrafterOfRankZeroWarm(window: TargetWindow)
        async throws
    {
        let setup = try CopySetup(placement: .cluster, speculates: true, window: window)
        defer { setup.remove() }

        let pass = try await RerenderedSpeculativePass(on: setup)

        #expect(pass.drafterStarts.first == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "a speculative turn over a render that rewinds the ledger answers as a cold session does",
        arguments: TargetWindow.allCases)
    func aTurnThatRewindsTheLedgerAnswersAsAColdSessionDoes(window: TargetWindow) async throws {
        let setup = try CopySetup(placement: .cluster, speculates: true, window: window)
        defer { setup.remove() }

        let pass = try await RerenderedSpeculativePass(on: setup)
        let coldTurn = try await setup.respond(
            as: Self.cold, prompt: Self.followUpPrompt, after: pass.history)

        #expect(!coldTurn.responseText.isEmpty)
        #expect(pass.turn.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the seeded session starts the drafter of rank 0 where the template started it",
        arguments: SeededCopy.allCases, TargetWindow.allCases)
    func theSeededSessionStartsTheDrafterOfRankZeroWarm(copy: SeededCopy, window: TargetWindow)
        async throws
    {
        let setup = try CopySetup(placement: .cluster, speculates: true, window: window)
        defer { setup.remove() }

        let pass = try await SeededSpeculativePass(after: copy, on: setup)

        #expect(pass.drafterStarts.first == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the speculative seeded session answers as a cold session does",
        arguments: SeededCopy.allCases, TargetWindow.allCases)
    func theSpeculativeSeededSessionAnswersAsAColdSessionDoes(
        copy: SeededCopy, window: TargetWindow
    ) async throws {
        let setup = try CopySetup(placement: .cluster, speculates: true, window: window)
        defer { setup.remove() }

        let pass = try await SeededSpeculativePass(after: copy, on: setup)
        let coldTurn = try await setup.respond(as: Self.cold, prompt: Self.divergentPrompt)

        #expect(!coldTurn.responseText.isEmpty)
        #expect(pass.turn.responseText == coldTurn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "the speculative seeded session reports the drafts it proposed and the drafts it kept",
        arguments: SeededCopy.allCases, TargetWindow.allCases)
    func theSpeculativeSeededSessionReportsItsDraftCounts(copy: SeededCopy, window: TargetWindow)
        async throws
    {
        let setup = try CopySetup(placement: .cluster, speculates: true, window: window)
        defer { setup.remove() }

        let pass = try await SeededSpeculativePass(after: copy, on: setup)

        let metadata = pass.turn.metadata
        let drafted = metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
        let accepted =
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int
        // The drafter and the target have random weights, so the target can keep none of the
        // drafts: the accepted count is bound by zero and by the drafts the rounds proposed.
        #expect((drafted ?? 0) > 0)
        #expect(accepted != nil)
        #expect((accepted ?? -1) >= 0)
        #expect((accepted ?? Int.max) <= (drafted ?? 0))
    }
}

extension PromptCacheCopyResult {

    /// The length of the token ledger of the copy, or nil when nothing was copied.
    fileprivate var copiedTokenCount: Int? {
        switch self {
        case .copied(let tokenCount): tokenCount
        case .sourceNotFound, .sourceInUse, .refused: nil
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

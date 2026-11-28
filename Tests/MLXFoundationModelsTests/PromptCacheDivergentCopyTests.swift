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

    /// The tokens each pass generates.
    private static let generatedTokenCount = 16

    /// A model at one placement, its store, and the folders they use.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct CopySetup {
        let fixture: Qwen35MTPFixture
        let model: MLXLanguageModel
        let store: ExecutorPromptCacheStore

        /// The folders the setup made, which ``remove()`` deletes.
        private let folders: [URL]

        /// Makes the model at `placement` and an empty store.
        ///
        /// - Parameter placement: Where the model runs.
        /// - Throws: The error of the fixture, of the target, or of the worker handshake.
        init(placement: Placement) throws {
            let weights = try makeScriptedWeightsDirectory()
            let folder = PromptCacheTestStore.folder(named: "PromptCacheDivergentCopyTests")
            folders = [weights, folder]
            fixture = try Qwen35MTPFixture()
            let target = try Inkling2TinyModels.target(
                json: Inkling2TinyModels.draftJSON(
                    vocabularySize: PromptCacheDivergentCopyTests.vocabularySize,
                    unpaddedVocabularySize: PromptCacheDivergentCopyTests.vocabularySize))
            let served: any MLXLMCommon.LanguageModel =
                switch placement {
                case .local: target
                case .cluster:
                    try ClusterLanguageModel.connect(
                        over: InProcessRemoteModelWorker(
                            model: target,
                            vocabularySize: PromptCacheDivergentCopyTests.vocabularySize))
                }
            model = fixture.makeLanguageModel(weights: weights, target: served)
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

        /// Runs one greedy pass of `sessionID` over the instructions and `prompt`.
        ///
        /// - Parameters:
        ///   - sessionID: The session of the pass.
        ///   - prompt: The tokens of the prompt after the instructions.
        /// - Returns: What the pass streamed.
        /// - Throws: The error of the pass or of its channel.
        func respond(as sessionID: String, prompt: [Int]) async throws -> MTPTurn {
            let transcript = Transcript(entries: [
                TestTranscript.instructions(
                    id: sessionID,
                    fixture.text(for: PromptCacheDivergentCopyTests.instructions)),
                TestTranscript.prompt(fixture.text(for: prompt)),
            ])
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

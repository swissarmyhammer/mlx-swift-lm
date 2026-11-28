// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// A copy of the prompt cache of a session of a clustered model
/// (``ExecutorPromptCacheStore/copy(modelID:from:to:)``) makes the worker copy the caches under
/// a second handle, and gives the destination an entry of its own.
///
/// Each test runs the target of the shared Qwen 3.5 fixture behind a `ClusterLanguageModel` over
/// an in-process worker, with a store of its own. The first entry of each transcript names the
/// session of the pass.
@Suite(.serialized)
struct ClusterPromptCacheCopyTests {

    /// The session that each test copies.
    private static let source = "cluster-copy-source"

    /// The session that each copy makes.
    private static let destination = "cluster-copy-destination"

    /// The tokens of the instructions.
    private static let instructions = [7, 21, 4, 33, 12, 18]

    /// The tokens of the first prompt.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the follow-up prompt.
    private static let followUpPrompt = [11, 40, 8, 27, 19]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// A clustered model, its worker, and a store whose source session holds the entry of one
    /// first pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct CopySetup {
        let fixture: Qwen35MTPFixture
        let model: MLXLanguageModel
        let worker: InProcessRemoteModelWorker
        let store: ExecutorPromptCacheStore

        /// The response of the first pass of the source, or empty before that pass.
        private(set) var firstResponse = ""

        /// The folders the setup made, which ``remove()`` deletes.
        private let folders: [URL]

        /// Makes the model and the store, and runs the first pass of the source.
        ///
        /// - Returns: The setup.
        /// - Throws: The error of the fixture, of the worker handshake, or of the pass.
        static func make() async throws -> CopySetup {
            var setup = try CopySetup()
            do {
                setup.firstResponse = try await setup.respond(
                    as: ClusterPromptCacheCopyTests.source
                ).responseText
            } catch {
                setup.remove()
                throw error
            }
            return setup
        }

        /// Makes the model and an empty store.
        ///
        /// - Throws: The error of the fixture or of the worker handshake.
        private init() throws {
            let weights = try makeScriptedWeightsDirectory()
            let folder = PromptCacheTestStore.folder(named: "ClusterPromptCacheCopyTests")
            folders = [weights, folder]
            fixture = try Qwen35MTPFixture()
            (model, worker) = try fixture.makeClusteredLanguageModel(weights: weights)
            store = ExecutorPromptCacheStore(
                directory: folder,
                capacityReader: PromptCacheSpoolFixtureVolume.capacity(ofVolumeOf:))
        }

        /// The key of `sessionID` for the model.
        ///
        /// - Parameter sessionID: The session.
        /// - Returns: The key.
        func key(_ sessionID: String) -> ExecutorPromptCacheKey {
            PromptCacheTestKey.make(modelID: .init(model.modelID), sessionID: .init(sessionID))
        }

        /// Copies the source to the destination through the public API.
        ///
        /// - Returns: What the copy did.
        func copySourceToDestination() async -> PromptCacheCopyResult {
            await ExecutorPromptCacheStore.$current.withValue(store) {
                await model.copyPromptCache(
                    fromSession: ClusterPromptCacheCopyTests.source,
                    toSession: ClusterPromptCacheCopyTests.destination)
            }
        }

        /// Runs one greedy pass of `sessionID`: the first prompt, or the first prompt, the
        /// first response and the follow-up prompt.
        ///
        /// - Parameters:
        ///   - sessionID: The session of the pass.
        ///   - followsUp: Whether the pass renders the follow-up turn.
        /// - Returns: What the pass streamed.
        /// - Throws: The error of the pass or of its channel.
        func respond(as sessionID: String, followsUp: Bool = false) async throws -> MTPTurn {
            let request = makeExecutorRequest(
                transcript: transcript(of: sessionID, followsUp: followsUp),
                generationOptions: GenerationOptions(
                    samplingMode: .greedy,
                    maximumResponseTokens: ClusterPromptCacheCopyTests.generatedTokenCount))
            let turn = try await ExecutorPromptCacheStore.$current.withValue(store) {
                try await MTPExecutorTurns.respond(to: request, model: model)
            }
            try #require(turn.failure == nil, "The pass failed: \(turn.failure ?? "")")
            try #require(
                turn.channelFailure == nil, "The channel failed: \(turn.channelFailure ?? "")")
            return turn
        }

        /// The transcript of one pass of `sessionID`, whose instructions entry names the
        /// session.
        ///
        /// - Parameters:
        ///   - sessionID: The session of the pass.
        ///   - followsUp: Whether the transcript holds the first response and the follow-up
        ///     prompt after the first prompt.
        /// - Returns: The transcript.
        private func transcript(of sessionID: String, followsUp: Bool) -> Transcript {
            let first = [
                TestTranscript.instructions(
                    id: sessionID, fixture.text(for: ClusterPromptCacheCopyTests.instructions)),
                TestTranscript.prompt(fixture.text(for: ClusterPromptCacheCopyTests.firstPrompt)),
            ]
            let followUp = [
                TestTranscript.response(firstResponse),
                TestTranscript.prompt(
                    fixture.text(for: ClusterPromptCacheCopyTests.followUpPrompt)),
            ]
            return Transcript(entries: first + (followsUp ? followUp : []))
        }

        /// Deletes the folders of the setup.
        func remove() {
            for folder in folders {
                try? FileManager.default.removeItem(at: folder)
            }
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the check-out of the copy is .memory and holds the whole ledger of the source")
    func theCheckOutOfTheCopyHoldsTheWholeLedger() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        let sourceLedger = try #require(await setup.store.peek(setup.key(Self.source))).tokens

        _ = await setup.copySourceToDestination()

        #expect(
            await setup.store.checkOut(setup.key(Self.destination)).entry?.tokens == sourceLedger)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the copy makes the worker hold a second handle")
    func theCopyMakesTheWorkerHoldASecondHandle() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        let handlesBefore = setup.worker.liveCacheCount

        _ = await setup.copySourceToDestination()

        #expect(setup.worker.liveCacheCount == handlesBefore + 1)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("after a pass of the source, the copy continues as the source did")
    func afterAPassOfTheSourceTheCopyContinuesAsTheSourceDid() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        _ = await setup.copySourceToDestination()

        let sourceTurn = try await setup.respond(as: Self.source, followsUp: true)
        let copyTurn = try await setup.respond(as: Self.destination, followsUp: true)

        #expect(copyTurn.responseText == sourceTurn.responseText)
        #expect(copyTurn.usage == sourceTurn.usage)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the follow-up of the copy reads its prefix from the copied cache")
    func theFollowUpOfTheCopyReadsItsPrefixFromTheCopiedCache() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        _ = await setup.copySourceToDestination()

        let copyTurn = try await setup.respond(as: Self.destination, followsUp: true)

        #expect((copyTurn.usage.first?.cachedPromptTokenCount ?? 0) > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a worker that fails the copy op makes the copy refuse with copyFailed")
    func aWorkerThatFailsTheCopyOpRefusesWithCopyFailed() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        setup.worker.fail(.copies, after: 0)

        #expect(await setup.copySourceToDestination() == .refused(.copyFailed))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a worker that fails the copy op leaves no entry for the destination")
    func aWorkerThatFailsTheCopyOpLeavesNoDestinationEntry() async throws {
        let setup = try await CopySetup.make()
        defer { setup.remove() }
        setup.worker.fail(.copies, after: 0)

        _ = await setup.copySourceToDestination()

        #expect(await setup.store.checkOut(setup.key(Self.destination)) == .none)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)

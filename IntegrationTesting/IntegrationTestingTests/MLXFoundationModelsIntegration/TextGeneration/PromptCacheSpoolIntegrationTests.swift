// Copyright © 2026 Apple Inc.
//
// Real-weights proof that a prompt cache that went to disk comes back warm. More sessions run
// than the memory budget holds, their turns interleave, and each later turn must reuse the
// prompt of its earlier turn and give the answer of a cold run.
//
// Every measurement line goes to the unified log, in the subsystem
// `com.apple.FoundationModels-MLX` under the category `PromptCacheSpool`, with the
// `PROMPT CACHE SPOOL:` prefix. The executor writes its own plan lines, with `source=disk`
// and the restore time, under the category `ExecutorPromptCache`. Read the lines after a run
// with:
// `log show --info --last 30m --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheSpoolIntegrationTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import Testing
import os

@testable import MLXFoundationModels

/// Proves, with real weights, that sessions whose caches went to disk come back warm.
///
/// The suite binds its own store with `ExecutorPromptCacheStore.$current`, thus it never
/// touches the shared store. The numbers are read from the channel, the way
/// `PromptCacheReuseChannelTests` reads them.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct PromptCacheSpoolIntegrationTests {

    /// A small instruction-tuned model whose chat template lets a later turn extend the render
    /// of an earlier turn. `PromptCacheReuseChannelTests` proves reuse on it from memory.
    private static let modelID = TestFixtures.llamaModelID

    /// Keeps each turn short: this suite measures the prompt, not the answer.
    private static let maximumResponseTokens = 24

    /// The first prompt of each session. There are more sessions than the memory budget holds.
    private static let firstPrompts = [
        "Name one primary color. One word.",
        "Name one planet of the solar system. One word.",
        "Name one metal. One word.",
    ]

    /// The second prompt of each session.
    private static let secondPrompt = "Name one more. One word."

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "PROMPT CACHE SPOOL:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "PromptCacheSpool")

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    /// Greedy sampling, thus the warm answer and the cold answer are comparable.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var greedyOptions: GenerationOptions {
        GenerationOptions(samplingMode: .greedy, maximumResponseTokens: maximumResponseTokens)
    }

    @Test("sessions that went to disk come back warm, with the answers of cold runs")
    func sessionsThatWentToDiskComeBackWarm() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectSessionsThatWentToDiskComeBackWarm()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The check

    /// Runs the first turn of each session, then the second turn of each session in the same
    /// order, thus the turns interleave and each session waits while the others run.
    ///
    /// The memory budget holds one session: the budget is set to the bytes of the first
    /// session, thus the first turn of each later session pushes the session before it to
    /// disk. Each session but the last is thus on disk when its second turn starts. The last
    /// session can stay in memory, or go to disk when a grown entry of an earlier session
    /// pushes it out.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectSessionsThatWentToDiskComeBackWarm() async throws {
        await releaseAllGPUMemory()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PromptCacheSpoolIntegrationTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        let model = makeTestModel(Self.modelID)
        let executor = try makeMLXExecutor(for: model)

        try await ExecutorPromptCacheStore.$current.withValue(store) {
            let sessions = try await runFirstTurns(
                executor: executor, model: model, store: store)
            for (index, session) in sessions.enumerated() {
                try await expectTheSecondTurnIsWarm(
                    of: session, mustBeOnDisk: index < sessions.count - 1,
                    executor: executor, model: model, store: store)
            }
        }
        await releaseAllGPUMemory()
    }

    /// Runs the first turn of each session, and makes the memory budget hold one session.
    ///
    /// - Parameters:
    ///   - executor: the executor of the turns.
    ///   - model: the model of the turns.
    ///   - store: the store the task binds.
    /// - Returns: each session, with the text of its first turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func runFirstTurns(
        executor: MLXLanguageModel.Executor, model: MLXLanguageModel,
        store: ExecutorPromptCacheStore
    ) async throws -> [SpooledSession] {
        var sessions: [SpooledSession] = []
        for prompt in Self.firstPrompts {
            let firstEntry = Transcript.Entry.prompt(
                Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: prompt))]))
            let firstTurn = try await respondReadingTheChannel(
                executor,
                request: makeExecutorRequest(
                    transcript: Transcript(entries: [firstEntry]),
                    generationOptions: Self.greedyOptions),
                model: model)
            let session = SpooledSession(
                firstEntry: firstEntry, firstAnswer: firstTurn.text,
                key: ExecutorPromptCacheKey(modelID: model.modelID, sessionID: firstEntry.id))
            if sessions.isEmpty {
                let entry = try #require(await store.peek(session.key))
                await store.configure(memoryBudgetBytes: entry.byteCount)
            }
            sessions.append(session)
        }
        await store.waitForSpills()
        return sessions
    }

    /// Runs the second turn of `session` and a cold control of the same transcript, and
    /// records an issue unless the second turn is warm and gives the answer of the control.
    ///
    /// - Parameters:
    ///   - session: the session of the turn.
    ///   - mustBeOnDisk: whether the cache of the session must be on disk when the turn starts.
    ///   - executor: the executor of the turns.
    ///   - model: the model of the turns.
    ///   - store: the store the task binds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectTheSecondTurnIsWarm(
        of session: SpooledSession, mustBeOnDisk: Bool, executor: MLXLanguageModel.Executor,
        model: MLXLanguageModel, store: ExecutorPromptCacheStore
    ) async throws {
        let wasOnDisk = await store.peek(session.key) == nil
        let request = makeExecutorRequest(
            transcript: session.secondTranscript(prompt: Self.secondPrompt),
            generationOptions: Self.greedyOptions)
        let warm = try await respondReadingTheChannel(executor, request: request, model: model)
        let cold = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
            try await respondReadingTheChannel(executor, request: request, model: model)
        }
        await store.waitForSpills()

        let line =
            "\(Self.measurementPrefix) session \(session.key.sessionID) onDisk=\(wasOnDisk) "
            + "prompt=\(warm.promptTokenCount) cached=\(warm.cachedTokenCount) "
            + "coldCached=\(cold.cachedTokenCount) sameAnswer=\(warm.text == cold.text)"
        Self.measurementLog.info("\(line, privacy: .public)")

        #expect(
            wasOnDisk || !mustBeOnDisk,
            "The session must be on disk before its second turn. \(line)")
        #expect(warm.cachedTokenCount > 0, "The second turn must reuse the first turn. \(line)")
        #expect(cold.cachedTokenCount == 0, "The cold control must reuse nothing. \(line)")
        #expect(warm.text == cold.text, "A restored cache must not change the answer. \(line)")
    }
}

/// One session of the suite: its first entry, and the answer of its first turn.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct SpooledSession {

    /// The first entry of the session, which names the session.
    let firstEntry: Transcript.Entry

    /// The answer of the first turn.
    let firstAnswer: String

    /// The key of the session in the store.
    let key: ExecutorPromptCacheKey

    /// The transcript of the second turn: the first entry, the first answer, and `prompt`.
    ///
    /// - Parameter prompt: the second prompt.
    /// - Returns: the transcript.
    func secondTranscript(prompt: String) -> Transcript {
        Transcript(entries: [
            firstEntry,
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: firstAnswer))])),
            .prompt(Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: prompt))])),
        ])
    }
}

#endif  // FoundationModelsIntegration

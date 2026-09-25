// Copyright © 2026 Apple Inc.
//
// Real-weights proof that the prompt cache carries across the turns of a
// `LanguageModelSession` -- the framework's own session, not a hand-built
// transcript -- on the Qwen 3.5 hybrid checkpoint. This is the shape
// `FoundationModelsRouter` drives in `secondTurnReusesFirstTurnsKVCache`:
// instructions, one turn, one more turn, and the usage the session reports.
//
// `Qwen35AgenticPromptCacheAssessmentTests` builds every transcript by hand,
// thus it cannot see what the framework does between two turns: which entries
// it appends, in which order, and whether the first entry keeps its identity.
// The executor keys the session's cache on that identity.
//
// Every measurement line goes to the unified log, in the subsystem
// `com.apple.FoundationModels-MLX` under the category
// `Qwen35SessionPromptCache`, with the `QWEN35 SESSION:` prefix. Read the
// numbers after a run with:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/Qwen35SessionPromptCacheTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// Prefix that makes every measurement line greppable in the log.
private let measurementPrefix = "QWEN35 SESSION:"

/// The log every measurement line of this suite goes to.
private let measurementLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "Qwen35SessionPromptCache")

/// The hybrid checkpoint under measurement.
private let hybridModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// The per-test time limit, in minutes.
private let suiteTimeLimitMinutes = 30

/// Tokens each turn may generate. Thinking mode reasons before it answers.
private let generatedTokenBudget = 1_024

/// Divergent tokens decoded from each side of a ledger seam.
private let divergenceReportTokenCount = 12

/// The instructions of the session, the same words the Router test uses.
private let sessionInstructions = "You are a terse, literal assistant."

/// The two turns, the same words the Router test uses.
private let firstPrompt = "My favorite color is teal. Reply with just \"OK\"."
private let secondPrompt = "What is my favorite color? Answer with just the color, lowercase."

/// The model state key under which Qwen 3.5 keeps its M-RoPE anchor. The key in
/// `Libraries/MLXVLM/Models/Qwen35.swift` is private, thus the suite names it again.
private let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

/// What the spilled file of one turn holds, read into fresh caches of the model.
private struct SpilledTurn: Sendable {
    /// Tokens in the token ledger of the file.
    let tokenCount: Int
    /// Tokens in the render ledger of the file.
    let renderTokenCount: Int
    /// The offset of each restored cache.
    let offsets: [Int]
    /// Whether the restored model state holds ``ropeDeltasKey``.
    let holdsRopeDeltas: Bool

    /// Reads the counts of `entry`.
    ///
    /// - Parameter entry: the entry that the file gave.
    init(_ entry: ExecutorPromptCacheEntry) {
        tokenCount = entry.tokens.count
        renderTokenCount = entry.renderTokens.count
        offsets = entry.caches.map(\.offset)
        holdsRopeDeltas = entry.state?[ropeDeltasKey] != nil
    }
}

/// Measures a `LanguageModelSession` across two turns on the hybrid model.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct Qwen35SessionPromptCacheTests {

    /// The second turn of a framework session reuses the whole first turn:
    /// its prompt and the tokens the model wrote.
    @Test func aSecondTurnOfAFrameworkSessionReusesTheFirstTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()

        let model = makeReasoningTestModel(hybridModelID)
        let container = try await model.loadContainer()
        let session = LanguageModelSession(
            model: model, tools: [], instructions: sessionInstructions)
        let options = GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: generatedTokenBudget)

        let first = try await session.respond(to: firstPrompt, options: options)
        let firstEntryID = try #require(session.transcript.first?.id)
        let key = ExecutorPromptCacheKey(modelID: model.modelID, sessionID: firstEntryID)
        let ledger = await ExecutorPromptCacheStore.shared.peek(key)?.tokens ?? []
        report(turn: 1, usage: first.usage, transcript: session.transcript)

        let second = try await session.respond(to: secondPrompt, options: options)
        let render = await ExecutorPromptCacheStore.shared.peek(key)?.renderTokens ?? []
        report(turn: 2, usage: second.usage, transcript: session.transcript)
        let shared = commonPrefixLength(ledger, render)
        let renderTail = await decodeTokens(
            container,
            tokens: divergentTail(of: render, from: shared, limit: divergenceReportTokenCount))
        let ledgerTail = await decodeTokens(
            container,
            tokens: divergentTail(of: ledger, from: shared, limit: divergenceReportTokenCount))
        let seam =
            "\(measurementPrefix) turn 2 render shares \(shared) of the \(ledger.count)-token "
            + "ledger; render <<<\(renderTail)>>> where the ledger holds <<<\(ledgerTail)>>>"
        let secondEntryID = session.transcript.first?.id ?? "-"
        measurementLog.info("\(seam, privacy: .public)")
        measurementLog.info(
            "\(measurementPrefix, privacy: .public) turn 2 first entry id = \(secondEntryID, privacy: .public)"
        )
        measurementLog.info(
            "\(measurementPrefix, privacy: .public) turn 1 first entry id = \(firstEntryID, privacy: .public)"
        )

        #expect(first.usage.input.cachedTokenCount == 0)
        #expect(
            second.usage.input.cachedTokenCount >= first.usage.input.totalTokenCount,
            """
            turn 2 cached \(second.usage.input.cachedTokenCount) tokens, under the \
            \(first.usage.input.totalTokenCount) tokens turn 1 rendered
            """)
        #expect(second.content.lowercased().contains("teal"))
        await releaseAllGPUMemory()
    }

    /// A framework session whose cache went to disk between its turns comes back warm: the
    /// hybrid caches (`MambaCache` and `KVCacheSimple`) read back from the file, and the second
    /// turn reuses the whole first turn.
    ///
    /// The test binds its own store, whose memory budget is zero, thus the check-in of the
    /// first turn goes to disk at once.
    @Test func aSecondTurnOfAFrameworkSessionComesBackWarmFromDisk() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASecondTurnComesBackWarmFromDisk()
        } else {
            Issue.record("The executor needs iOS 27, macOS 27 or visionOS 27.")
        }
    }

    /// Runs two turns of one framework session inside a store that spills each check-in, and
    /// records an issue unless the second turn restores the first turn from disk.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASecondTurnComesBackWarmFromDisk() async throws {
        await releaseAllGPUMemory()
        let model = makeReasoningTestModel(hybridModelID)
        let options = GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: generatedTokenBudget)

        try await withSpillingStore { store, _ in
            let session = LanguageModelSession(
                model: model, tools: [], instructions: sessionInstructions)
            let first = try await session.respond(to: firstPrompt, options: options)
            await store.waitForSpills()
            let firstEntryID = try #require(session.transcript.first?.id)
            let key = ExecutorPromptCacheKey(modelID: model.modelID, sessionID: firstEntryID)
            let wasInMemory = await store.peek(key) != nil
            let diskBytes = await store.diskByteCount
            report(turn: 1, usage: first.usage, transcript: session.transcript)

            let second = try await session.respond(to: secondPrompt, options: options)
            report(turn: 2, usage: second.usage, transcript: session.transcript)
            let line =
                "\(measurementPrefix) disk restore: disk bytes after turn 1 = \(diskBytes), "
                + "turn 2 cached \(second.usage.input.cachedTokenCount) of "
                + "\(second.usage.input.totalTokenCount), turn 1 rendered "
                + "\(first.usage.input.totalTokenCount)"
            measurementLog.info("\(line, privacy: .public)")

            #expect(!wasInMemory, "The cache of turn 1 must leave memory. \(line)")
            #expect(diskBytes > 0, "The cache of turn 1 must be on disk. \(line)")
            #expect(
                second.usage.input.cachedTokenCount >= first.usage.input.totalTokenCount,
                "Turn 2 must reuse the whole render of turn 1. \(line)")
            #expect(second.content.lowercased().contains("teal"))
        }
        await releaseAllGPUMemory()
    }

    /// The file of a spilled turn holds the M-RoPE state of Qwen 3.5, the render ledger, and
    /// caches that stand at the end of the token ledger. The second turn that the store restores
    /// from that file gives the text of an uncached second turn.
    @Test func aTurnRestoredFromDiskHoldsTheRopeStateAndGivesTheUncachedText() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectARestoredTurnMatchesAnUncachedTurn()
        } else {
            Issue.record("The executor needs iOS 27, macOS 27 or visionOS 27.")
        }
    }

    /// Runs turn 1 inside a store that spills each check-in, reads the spilled file, and then
    /// runs turn 2 two times: warm from the file, and uncached from a copy of the transcript of
    /// turn 1. Records an issue unless the file holds the state and the ledgers of the turn, and
    /// the two runs of turn 2 give the same text.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectARestoredTurnMatchesAnUncachedTurn() async throws {
        await releaseAllGPUMemory()
        let model = makeReasoningTestModel(hybridModelID)
        try model.requireLocalWeights()
        let container = try await model.loadContainer()
        let options = GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: generatedTokenBudget)

        try await withSpillingStore { store, directory in
            let session = LanguageModelSession(
                model: model, tools: [], instructions: sessionInstructions)
            _ = try await session.respond(to: firstPrompt, options: options)
            await store.waitForSpills()
            let firstTranscript = session.transcript
            let key = ExecutorPromptCacheKey(
                modelID: model.modelID, sessionID: try #require(firstTranscript.first?.id))
            let file = try await readTheSpilledFile(in: directory, key: key, container: container)
            expectTheFileHoldsTheTurn(file)

            let warm = try await session.respond(to: secondPrompt, options: options)
            let uncached = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
                try await LanguageModelSession(model: model, tools: [], transcript: firstTranscript)
                    .respond(to: secondPrompt, options: options)
            }
            let line =
                "\(measurementPrefix) restored turn 2 cached \(warm.usage.input.cachedTokenCount) "
                + "of \(warm.usage.input.totalTokenCount); uncached turn 2 cached "
                + "\(uncached.usage.input.cachedTokenCount); warm <<<\(warm.content)>>> "
                + "uncached <<<\(uncached.content)>>>"
            measurementLog.info("\(line, privacy: .public)")

            #expect(
                warm.usage.input.cachedTokenCount >= file.renderTokenCount,
                "The restored turn 2 must reuse the whole render of turn 1. \(line)")
            #expect(
                uncached.usage.input.cachedTokenCount == 0,
                "The uncached turn 2 must reuse nothing. \(line)")
            #expect(
                warm.content == uncached.content,
                "A turn restored from disk must give the text of an uncached turn. \(line)")
        }
        await releaseAllGPUMemory()
    }

    /// Binds a new store to the task for `body`. The memory budget of the store is zero, thus
    /// each check-in goes to disk at once. The spill folder is new, and the function removes it
    /// after `body`.
    ///
    /// - Parameter body: the work that runs with the store. It gets the store and the spill
    ///   folder.
    /// - Throws: the error of `body`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func withSpillingStore(
        _ body: (ExecutorPromptCacheStore, URL) async throws -> Void
    ) async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Qwen35SessionPromptCacheTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        await store.configure(memoryBudgetBytes: 0)
        try await ExecutorPromptCacheStore.$current.withValue(store) {
            try await body(store, directory)
        }
    }

    /// Reads the one spilled file of `directory` into fresh caches of the model.
    ///
    /// - Parameters:
    ///   - directory: the folder of the spill files.
    ///   - key: the session that the file must belong to.
    ///   - container: the loaded model.
    /// - Returns: what the file holds.
    /// - Throws: when the folder does not hold exactly one spill file, or the error of the read.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func readTheSpilledFile(
        in directory: URL, key: ExecutorPromptCacheKey, container: ModelContainer
    ) async throws -> SpilledTurn {
        let partialSuffix =
            ".\(ExecutorPromptCacheFile.partialMarker).\(ExecutorPromptCacheFile.fileExtension)"
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { url in
            url.pathExtension == ExecutorPromptCacheFile.fileExtension
                && !url.lastPathComponent.hasSuffix(partialSuffix)
        }
        #expect(files.count == 1, "The store must spill one file for turn 1, not \(files).")
        let url = try #require(files.first)
        return try await container.perform { context in
            SpilledTurn(
                try ExecutorPromptCacheFile.read(
                    from: url, key: key, templates: try context.model.newCache(parameters: nil)))
        }
    }

    /// Records an issue unless `file` holds the M-RoPE state, a render ledger, and caches that
    /// stand at the end of the token ledger.
    ///
    /// - Parameter file: what the spilled file of turn 1 holds.
    private func expectTheFileHoldsTheTurn(_ file: SpilledTurn) {
        let line =
            "\(measurementPrefix) spilled file: ledger \(file.tokenCount) tokens, render "
            + "\(file.renderTokenCount) tokens, offsets \(file.offsets), "
            + "ropeDeltas \(file.holdsRopeDeltas)"
        measurementLog.info("\(line, privacy: .public)")
        #expect(file.holdsRopeDeltas, "The file must hold \(ropeDeltasKey.id). \(line)")
        #expect(file.renderTokenCount > 0, "The file must hold the render ledger. \(line)")
        #expect(!file.offsets.isEmpty, "The file must hold the caches. \(line)")
        #expect(
            file.offsets.allSatisfy { $0 == file.tokenCount },
            "Each cache offset must be the length of the token ledger. \(line)")
    }

    /// Logs the usage of one turn and the entries the transcript holds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func report(turn: Int, usage: LanguageModelSession.Usage, transcript: Transcript) {
        let line = "\(measurementPrefix) turn \(turn)"
        let kinds = transcript.map { entry -> String in
            switch entry {
            case .instructions: return "instructions"
            case .prompt: return "prompt"
            case .reasoning: return "reasoning"
            case .response: return "response"
            case .toolCalls: return "toolCalls"
            case .toolOutput: return "toolOutput"
            default: return "other"
            }
        }
        let lines = [
            "rendered prompt tokens = \(usage.input.totalTokenCount)",
            "cachedTokenCount = \(usage.input.cachedTokenCount)",
            "generated tokens = \(usage.output.totalTokenCount)",
            "transcript = \(kinds.joined(separator: " "))",
        ]
        for measurement in lines {
            measurementLog.info("\(line, privacy: .public) \(measurement, privacy: .public)")
        }
    }
}

#endif  // FoundationModelsIntegration

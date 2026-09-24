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
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Qwen35SessionPromptCacheTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        await store.configure(memoryBudgetBytes: 0)
        let model = makeReasoningTestModel(hybridModelID)
        let options = GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: generatedTokenBudget)

        try await ExecutorPromptCacheStore.$current.withValue(store) {
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

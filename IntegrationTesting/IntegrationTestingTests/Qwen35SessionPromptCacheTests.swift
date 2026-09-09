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
// Every measurement line carries the `QWEN35 SESSION:` prefix.
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

@testable import MLXFoundationModels

/// Prefix that makes every measurement line greppable in a run log.
private let measurementPrefix = "QWEN35 SESSION:"

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
        print(
            "\(measurementPrefix) turn 2 render shares \(shared) of the \(ledger.count)-token "
                + "ledger; render <<<\(renderTail)>>> where the ledger holds <<<\(ledgerTail)>>>")
        print("\(measurementPrefix) turn 2 first entry id = \(session.transcript.first?.id ?? "-")")
        print("\(measurementPrefix) turn 1 first entry id = \(firstEntryID)")

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

    /// Prints the usage of one turn and the entries the transcript holds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func report(turn: Int, usage: LanguageModelSession.Usage, transcript: Transcript) {
        let line = "\(measurementPrefix) turn \(turn)"
        print("\(line) rendered prompt tokens = \(usage.input.totalTokenCount)")
        print("\(line) cachedTokenCount = \(usage.input.cachedTokenCount)")
        print("\(line) generated tokens = \(usage.output.totalTokenCount)")
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
        print("\(line) transcript = \(kinds.joined(separator: " "))")
    }
}

#endif  // FoundationModelsIntegration

// Copyright © 2026 Apple Inc.
//
// Real-weights measurement of the executor prompt cache across TOOL ROUNDS on
// a Qwen 3.5 hybrid checkpoint, `mlx-community/Qwen3.8-27B-mxfp4`, in the
// shape an agent uses: one long user turn, then one tool call and one tool
// result for each round, with thinking on and tools enabled on every round.
//
// The path under measurement is `MLXLanguageModel.Executor.respond`, thus the
// numbers come from the CHANNEL (`updateUsage`) and from the executor's test
// mirror (`GenerationEvent.completion`, which carries the prefill seconds the
// channel has no event for). The session's ledger comes from
// `ExecutorPromptCacheStore.shared.peek`, which lets each round name the first
// token where its render parts from the ledger the round before it left.
//
// A unit test cannot answer this question. Three things stand between a hybrid
// model and a carried cache, and each shows only on real weights:
//
//   (a) Does every recurrent layer report the position the attention layers
//       report? A `MambaCache` that stays at offset 0 leaves the session cold.
//   (b) Does the next render extend the ledger, or does the Qwen committed-turn
//       rule have to splice past the `<|im_end|>` the model wrote? The chat
//       template trims reasoning and reorders tool-call arguments, thus only a
//       real tokenizer render says where the two part.
//   (c) Given a good splice, does the round feed the new tail alone, and does
//       the prefill time stop growing with the transcript?
//
// The control is a pure-attention model, `mlx-community/Qwen3-4B-4bit`, which
// takes the standard prefix rules alone. It shows what the same driver reads on
// a model that has no recurrent layer and no committed-turn rule.
//
// Every measurement line carries the `QWEN35 CACHE:` prefix so a run log can
// be grepped for the numbers alone.
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests`
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`, thus `swift build --build-tests` stays at exit 0 with
// a type error in it. Use `xcodebuild build-for-testing` as the compile
// evidence for any change.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import IntegrationTestHelpers
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

// MARK: - Constants

/// Prefix that makes every measurement line greppable in a run log.
private let measurementPrefix = "QWEN35 CACHE:"

/// The hybrid checkpoint under measurement. It must already stand in the
/// local Hugging Face cache; this suite downloads nothing.
private let hybridModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// The pure-attention control.
private let controlModelID = TestFixtures.qwen3ModelID

/// The per-test time limit, in minutes. One measurement runs five rounds of
/// a 27B model over a prompt of twenty thousand tokens, and then one cold
/// control of the last round.
private let suiteTimeLimitMinutes = 90

/// Rows in the stock report the agent reads. Sized so the first render alone
/// stands near twenty thousand tokens, thus every later round measures a
/// cache that carries a long transcript, not a short one.
private let stockReportRowCount = 800

/// The bays the agent is told to look up, one for each tool call.
private let queriedBays = [3, 7, 11, 15]

/// How many rounds the driver runs. Round 1 is the user turn; every round
/// after it continues from a tool result, thus five rounds give four tool
/// rounds.
private let roundCount = 5

/// Tokens the transcript must reach by the last round.
private let minimumTranscriptTokenCount = 20_000

/// Tokens a round after round 1 may feed. A carried cache feeds the new tail
/// alone: the tool result, its markup and the generation prompt.
private let maximumFedTokensAfterRoundOne = 2_000

/// Tokens the cached count of a round may fall below the rendered prompt of
/// the round before it. The seam between the tokens the model wrote and the
/// tokens the template writes for them is a few tokens wide at most.
private let cacheSeamSlack = 16

/// How much the prefill of round 4 may grow over the prefill of round 2. A
/// prefill that grows with the transcript fails this; one that feeds the new
/// tail alone stays flat.
private let prefillGrowthLimit = 2.0

/// Leading generated tokens a cached round and a cold round must share at
/// temperature 0.
private let comparedTokenCount = 32

/// Tokens each round may generate. Thinking mode reasons before it writes a
/// tool call, thus the budget must hold both.
private let generatedTokenBudget = 2_048

/// Divergent tokens decoded from each side of a ledger seam.
private let divergenceReportTokenCount = 12

/// Greedy decoding keeps a cached round and its cold control comparable.
private let greedyTemperature = 0.0

/// The round whose transcript the cold control runs again. Zero-based: the
/// last round.
private let coldControlRoundIndex = roundCount - 1

/// The pallet count planted on every queried row, thus a tool answer that
/// names it agrees with the report.
private let plantedPalletCount = 4172

/// The system turn of the agent. The tool name is `stockToolName`, which
/// `DeepseekV4IntegrationTests.swift` declares for the whole target.
private let agentInstructions =
    "You are an inventory agent. You read a bay with the get_stock_level tool. "
    + "Look up ONE bay for each tool call, and call the tool again for the next bay "
    + "after each result. Keep your thinking brief."

// MARK: - The tool

/// The one argument of the stock tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct StockArguments {
    @Guide(description: "The bay to read, for example bay 7")
    var bay: String
}

/// The tool definition the transcript carries.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeStockTool() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: stockToolName,
        description: "Read the recorded stock level of one warehouse bay",
        parameters: StockArguments.generationSchema)
}

/// The answer the fake tool gives back for the bay of `roundIndex`.
///
/// - Parameter roundIndex: the zero-based round the call came from.
/// - Returns: the tool result as JSON text.
private func stockToolResult(roundIndex: Int) -> String {
    let bay = queriedBays[min(roundIndex, queriedBays.count - 1)]
    return #"{"bay":"bay \#(bay)","pallets":\#(plantedPalletCount),"status":"sealed"}"#
}

// MARK: - The conversation

/// The stock report the agent reads, which is what makes the first render
/// long enough for a real prefill.
///
/// - Returns: the report rows, one on each line.
private func makeStockReportRows() -> String {
    (1 ... stockReportRowCount).map { index in
        guard queriedBays.contains(index) else {
            return "Row \(index): warehouse bay \(index) was audited last week, its seals "
                + "were intact, and no damage was recorded against its pallets."
        }
        return "Row \(index): warehouse bay \(index) holds \(plantedPalletCount) pallets, "
            + "its seals were intact, and no damage was recorded against them."
    }.joined(separator: "\n")
}

/// The user turn that asks the agent to call the tool for each bay.
private func makeStockReportPrompt() -> String {
    let bays = queriedBays.map(String.init).joined(separator: ", ")
    return "Read the stock report below. Then look up bays \(bays) with the "
        + "get_stock_level tool, one bay for each call, in that order. After the last "
        + "result, list the four stock levels in one sentence.\n"
        + makeStockReportRows()
}

/// The user turn that follows a round which answered in text instead of a
/// tool call, thus the transcript still grows by one round.
///
/// - Parameter roundIndex: the zero-based round that answered.
/// - Returns: the user turn.
private func makeNextBayPrompt(roundIndex: Int) -> String {
    let bay = queriedBays[min(roundIndex, queriedBays.count - 1)]
    return "Now call get_stock_level for bay \(bay)."
}

/// A text segment entry helper.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func textSegment(_ content: String) -> Transcript.Segment {
    .text(Transcript.TextSegment(content: content))
}

/// A prompt entry.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func promptEntry(_ content: String) -> Transcript.Entry {
    .prompt(Transcript.Prompt(segments: [textSegment(content)], responseFormat: nil))
}

/// The instructions entry, which is the first entry of the session and thus
/// the identity the executor keys the session's cache on.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func instructionsEntry() -> Transcript.Entry {
    .instructions(
        Transcript.Instructions(
            segments: [textSegment(agentInstructions)],
            toolDefinitions: [makeStockTool()]))
}

// MARK: - One measured round

/// The tool call one round emitted.
private struct EmittedToolCall {
    /// The call identifier the executor assigned.
    let id: String
    /// The tool the model named.
    let name: String
    /// The arguments as JSON text.
    let arguments: String
}

/// What one round of the executor reported.
private struct RoundMeasurement {
    /// The one-based round number.
    let number: Int
    /// The reasoning the round streamed.
    let reasoning: String
    /// The response text the round streamed.
    let text: String
    /// The first tool call the round emitted, if any.
    let toolCall: EmittedToolCall?
    /// Tokens the whole prompt rendered to.
    let renderedTokenCount: Int
    /// Tokens the round fed to the model after the cache narrowed the input.
    let fedTokenCount: Int
    /// Tokens the channel reported as cached.
    let cachedTokenCount: Int
    /// Seconds the prefill took.
    let prefillSeconds: TimeInterval
    /// Tokens the round generated.
    let generatedTokenCount: Int
    /// Seconds the whole round took.
    let roundSeconds: TimeInterval
    /// Where this round's render parted from the ledger before it.
    let seam: LedgerSeam

    /// The generated stream as one text, for the cold comparison.
    var generatedText: String {
        reasoning + "\n" + text + "\n" + (toolCall?.arguments ?? "")
    }
}

/// Where a round's render parts from the ledger the round before it left.
private struct LedgerSeam {
    /// Tokens the ledger held before the round.
    let ledgerLength: Int
    /// Tokens the round rendered.
    let renderLength: Int
    /// Leading tokens the two share.
    let sharedPrefixLength: Int
    /// The render's text from the first divergent token, or empty when the
    /// render extends the ledger whole.
    let renderTail: String
    /// The ledger's text from the same position.
    let ledgerTail: String

    /// Whether the render extends the ledger whole.
    var isExtension: Bool { sharedPrefixLength == ledgerLength }

    /// One line that names the first divergent token.
    var summary: String {
        guard ledgerLength > 0 else { return "no ledger before this round" }
        guard !isExtension else {
            return "none, the render extends the \(ledgerLength)-token ledger whole"
        }
        return "render <<<\(renderTail)>>> at index \(sharedPrefixLength) of "
            + "\(ledgerLength), where the ledger holds <<<\(ledgerTail)>>>"
    }
}

/// The collected events of one round, before the seam joins them.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RoundEvents {
    var reasoning = ""
    var text = ""
    var toolCall: EmittedToolCall?
    var renderedTokenCount = 0
    var cachedTokenCount = 0
    var generatedTokenCount = 0
    var completion: GenerateCompletionInfo?

    /// Reads one executor event into the collection.
    mutating func consume(_ event: MLXLanguageModel.Executor.GenerationEvent) {
        switch event {
        case .appendText(let chunk, _, .reasoning):
            reasoning += chunk
        case .appendText(let chunk, _, .response):
            text += chunk
        case .toolCall(let id, let name, let arguments):
            if toolCall == nil {
                toolCall = EmittedToolCall(id: id, name: name, arguments: arguments)
            }
        case .updateUsage(let input, let output, _):
            renderedTokenCount = input.totalTokenCount
            cachedTokenCount = input.cachedTokenCount
            generatedTokenCount = output.totalTokenCount
        case .completion(let info):
            completion = info
        case .updateMetadata:
            break
        }
    }
}

// MARK: - The driver

/// Drives one session of the executor round by round and measures each round.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct SessionDriver {
    /// The model under measurement.
    let model: MLXLanguageModel
    /// The executor of the model.
    let executor: MLXLanguageModel.Executor
    /// The loaded container, for token decoding.
    let container: ModelContainer
    /// The greppable label of every line.
    let label: String

    /// Greedy options with the round budget.
    private var options: GenerationOptions {
        GenerationOptions(
            samplingMode: .greedy, temperature: greedyTemperature,
            maximumResponseTokens: generatedTokenBudget)
    }

    /// Runs one round on `entries` and measures it.
    ///
    /// - Parameters:
    ///   - entries: the transcript of the round.
    ///   - number: the one-based round number.
    /// - Returns: the measurement.
    func runRound(entries: [Transcript.Entry], number: Int) async throws -> RoundMeasurement {
        let request = makeExecutorRequest(
            transcript: Transcript(entries: entries),
            enabledTools: [makeStockTool()],
            generationOptions: options,
            contextOptions: ContextOptions(reasoningLevel: .moderate))
        let key = try #require(
            MLXLanguageModel.Executor.sessionCacheKey(for: request, modelID: model.modelID))
        let ledgerBefore = await ExecutorPromptCacheStore.shared.peek(key)?.tokens ?? []

        let start = Date()
        var events = RoundEvents()
        let stream = try await executeResponse(executor, request: request, model: model)
        for try await event in stream {
            events.consume(event)
        }
        let roundSeconds = Date().timeIntervalSince(start)

        let entryAfter = await ExecutorPromptCacheStore.shared.peek(key)
        let seam = try await describeLedgerSeam(
            ledger: ledgerBefore, render: entryAfter?.renderTokens ?? [])
        let completion = try #require(
            events.completion, "round \(number) ended without a completion report")
        return RoundMeasurement(
            number: number,
            reasoning: events.reasoning,
            text: events.text,
            toolCall: events.toolCall,
            renderedTokenCount: events.renderedTokenCount,
            fedTokenCount: events.renderedTokenCount - events.cachedTokenCount,
            cachedTokenCount: events.cachedTokenCount,
            prefillSeconds: completion.promptTime,
            generatedTokenCount: events.generatedTokenCount,
            roundSeconds: roundSeconds,
            seam: seam)
    }

    /// Measures where `render` parts from `ledger`, and decodes both sides.
    private func describeLedgerSeam(ledger: [Int], render: [Int]) async throws -> LedgerSeam {
        let shared = commonPrefixLength(ledger, render)
        let renderTail = await decodeTokens(
            container,
            tokens: divergentTail(of: render, from: shared, limit: divergenceReportTokenCount))
        let ledgerTail = await decodeTokens(
            container,
            tokens: divergentTail(of: ledger, from: shared, limit: divergenceReportTokenCount))
        return LedgerSeam(
            ledgerLength: ledger.count, renderLength: render.count,
            sharedPrefixLength: shared, renderTail: renderTail, ledgerTail: ledgerTail)
    }

    /// The entries the round adds to the transcript: its reasoning, then its
    /// tool call and the tool's answer, or its text and the next user turn.
    ///
    /// - Parameters:
    ///   - round: the finished round.
    ///   - roundIndex: the zero-based index of the round.
    /// - Returns: the entries to append.
    func entriesAdded(by round: RoundMeasurement, roundIndex: Int) throws -> [Transcript.Entry] {
        var added: [Transcript.Entry] = []
        if !round.reasoning.isEmpty {
            added.append(.reasoning(Transcript.Reasoning(segments: [textSegment(round.reasoning)])))
        }
        if let call = round.toolCall {
            let toolCall = Transcript.ToolCall(
                id: call.id, toolName: call.name,
                arguments: try GeneratedContent(json: call.arguments))
            added.append(
                .toolCalls(Transcript.ToolCalls(id: "toolcalls_\(roundIndex)", [toolCall])))
            added.append(
                .toolOutput(
                    Transcript.ToolOutput(
                        id: call.id, toolName: call.name,
                        segments: [textSegment(stockToolResult(roundIndex: roundIndex))])))
            return added
        }
        let text = round.text.isEmpty ? "(no answer)" : round.text
        added.append(.response(Transcript.Response(assetIDs: [], segments: [textSegment(text)])))
        added.append(promptEntry(makeNextBayPrompt(roundIndex: roundIndex)))
        return added
    }

    /// Runs every round of one session.
    ///
    /// - Returns: the measurements, and the transcript of the last round.
    func runSession() async throws -> (rounds: [RoundMeasurement], lastEntries: [Transcript.Entry])
    {
        var entries = [instructionsEntry(), promptEntry(makeStockReportPrompt())]
        var rounds: [RoundMeasurement] = []
        var lastEntries = entries
        for roundIndex in 0 ..< roundCount {
            lastEntries = entries
            let round = try await runRound(entries: entries, number: roundIndex + 1)
            report(round)
            rounds.append(round)
            entries += try entriesAdded(by: round, roundIndex: roundIndex)
        }
        return (rounds, lastEntries)
    }

    /// Runs `entries` once more in a session the store has never seen.
    ///
    /// - Parameter entries: the transcript to run cold. Its first entry is
    ///   replaced by a new instructions entry with the same content, thus the
    ///   executor finds no cache for it.
    /// - Returns: the measurement.
    func runCold(entries: [Transcript.Entry]) async throws -> RoundMeasurement {
        let cold = [instructionsEntry()] + entries.dropFirst()
        let round = try await runRound(entries: cold, number: 0)
        report(round, name: "cold control")
        return round
    }

    /// The first `comparedTokenCount` tokens of `text` through the model's
    /// tokenizer.
    func leadingTokens(of text: String) async -> [Int] {
        await container.perform { context in
            Array(
                context.tokenizer.encode(text: text, addSpecialTokens: false)
                    .prefix(comparedTokenCount))
        }
    }

    /// Prints every number of one round under the measurement prefix.
    private func report(_ round: RoundMeasurement, name: String? = nil) {
        let line = "\(measurementPrefix) \(label) \(name ?? "round \(round.number)")"
        print("\(line) rendered prompt tokens = \(round.renderedTokenCount)")
        print("\(line) fed prompt tokens = \(round.fedTokenCount)")
        print("\(line) cachedTokenCount = \(round.cachedTokenCount)")
        print("\(line) prefill seconds = \(round.prefillSeconds)")
        print("\(line) generated tokens = \(round.generatedTokenCount)")
        print("\(line) round seconds = \(round.roundSeconds)")
        print(
            "\(line) emitted = \(round.toolCall.map { "tool call \($0.name) \($0.arguments)" } ?? "text")"
        )
        print("\(line) first divergent token = \(round.seam.summary)")
    }
}

// MARK: - The suite

/// Measures the executor prompt cache across tool rounds on a Qwen 3.5
/// hybrid checkpoint, with a pure-attention control.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct Qwen35AgenticPromptCacheAssessmentTests {

    /// Five rounds on the hybrid model: the cache carries, the fed tokens
    /// stay small, the prefill stays flat, and a cached round decodes the
    /// same tokens as a cold one.
    @Test func hybridModelCarriesThePromptCacheAcrossToolRounds() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        try await measure(label: "qwen3.8-27b") { makeReasoningTestModel(hybridModelID) }
    }

    /// The same driver on the pure-attention control.
    @Test func controlModelCarriesThePromptCacheAcrossToolRounds() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        try await measure(label: "qwen3-4b") {
            // Built from the registry configuration, the way
            // `MultiTurnToolCallingTests` builds it, thus it carries the
            // family's `extraEOSTokens`. The configuration names
            // `controlModelID`.
            MLXLanguageModel(
                configuration: LLMRegistry.qwen3_4b_4bit,
                capabilities: [.reasoning, .guidedGeneration, .toolCalling],
                weightsLocation: testWeightsLocation(modelID:),
                load: testLoad())
        }
    }

    /// Runs the whole measurement on one model and holds every requirement.
    ///
    /// - Parameters:
    ///   - label: the greppable label of every line.
    ///   - makeModel: builds the model under measurement.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func measure(label: String, makeModel: () -> MLXLanguageModel) async throws {
        await releaseAllGPUMemory()
        let model = makeModel()
        let driver = SessionDriver(
            model: model, executor: try makeMLXExecutor(for: model),
            container: try await model.loadContainer(), label: label)

        let session = try await driver.runSession()
        let cachedRound = session.rounds[coldControlRoundIndex]
        let cold = try await driver.runCold(entries: session.lastEntries)
        let cachedTokens = await driver.leadingTokens(of: cachedRound.generatedText)
        let coldTokens = await driver.leadingTokens(of: cold.generatedText)
        let sharedGenerated = commonPrefixLength(cachedTokens, coldTokens)
        print(
            "\(measurementPrefix) \(label) cached round \(cachedRound.number) and cold control "
                + "share \(sharedGenerated) of the first \(comparedTokenCount) generated tokens")

        expectRounds(session.rounds, label: label)
        #expect(
            cold.cachedTokenCount == 0,
            "\(label): a session the store has never seen must reuse nothing")
        #expect(
            sharedGenerated == comparedTokenCount,
            """
            \(label): the cached round and the cold control part after \(sharedGenerated) of \
            \(comparedTokenCount) generated tokens. Cached: <<<\(cachedRound.generatedText.prefix(400))>>> \
            Cold: <<<\(cold.generatedText.prefix(400))>>>
            """)
        await releaseAllGPUMemory()
    }

    /// Holds the per-round requirements of the card.
    private func expectRounds(_ rounds: [RoundMeasurement], label: String) {
        let last = rounds[rounds.count - 1]
        #expect(
            last.renderedTokenCount >= minimumTranscriptTokenCount,
            """
            \(label): the last round rendered \(last.renderedTokenCount) tokens, under \
            \(minimumTranscriptTokenCount)
            """)
        #expect(
            rounds.filter { $0.toolCall != nil }.count >= queriedBays.count,
            "\(label): fewer than \(queriedBays.count) rounds emitted a tool call")
        for (previous, round) in zip(rounds, rounds.dropFirst()) {
            #expect(
                round.cachedTokenCount >= previous.renderedTokenCount - cacheSeamSlack,
                """
                \(label): round \(round.number) cached \(round.cachedTokenCount) tokens, under the \
                \(previous.renderedTokenCount) tokens round \(previous.number) rendered minus \
                \(cacheSeamSlack). First divergent token: \(round.seam.summary)
                """)
            #expect(
                round.fedTokenCount < maximumFedTokensAfterRoundOne,
                """
                \(label): round \(round.number) fed \(round.fedTokenCount) tokens, not under \
                \(maximumFedTokensAfterRoundOne)
                """)
        }
        let roundTwo = rounds[1]
        let roundFour = rounds[3]
        #expect(
            roundFour.prefillSeconds < roundTwo.prefillSeconds * prefillGrowthLimit,
            """
            \(label): round 4 prefill \(roundFour.prefillSeconds) s is not under \
            \(prefillGrowthLimit) times round 2 prefill \(roundTwo.prefillSeconds) s
            """)
    }
}

#endif  // FoundationModelsIntegration

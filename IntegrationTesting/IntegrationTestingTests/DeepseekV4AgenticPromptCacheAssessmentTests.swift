// Copyright © 2026 Apple Inc.
//
// Real-weights measurement of the OFFICIAL upstream prompt-cache path on
// `mlx-community/DeepSeek-V4-Flash-4bit`, in the shape an agent uses: a user
// turn, a DSML tool call, a tool result, and one more user turn.
//
// Upstream keeps its multi-turn cache-reuse machinery in
// `MLXLMCommon.ChatSession` (`PromptCacheReusePolicy`, `ExtendCachedPrefixRule`,
// `RewindToCommonPrefixRule`). Nothing in this file touches
// `MLXFoundationModels.PromptCache`, thus the numbers it prints are what
// upstream does on its own.
//
// A unit test cannot answer this question. `DeepSeekV4ChatEncoder` renders a
// turn's generation region differently live and as history, and it turns
// `dropsEarlierReasoning` OFF as soon as any turn carries tools. Thus a
// tool round's fed token sequence need not be a prefix of the next render, and
// when it is not the cache silently does nothing while every unit-level
// assertion still passes. The two facts below therefore have to come from a
// real run:
//
//   (a) Is the next render a prefix extension of the one before it?
//       A template that rewrites an already-cached region defeats caching no
//       matter how good the reuse machinery is.
//   (b) Given a valid prefix, does upstream actually skip the reprocessing?
//       `GenerateCompletionInfo.promptTokenCount` reports the tokens really
//       fed this pass, and `promptTime` the seconds spent on them, thus a
//       reusing pass shows both far below the control.
//
// The control is a fresh `ChatSession` seeded with the identical transcript
// through the `history:` initializer, which forces a full cold prefill of the
// same tool-round prompt. It gives the "no reuse" baseline.
//
// The suite drives the tool loop by hand rather than through `toolDispatch`,
// because a hand-driven loop reports one `GenerateCompletionInfo` for each
// generation pass. A dispatched loop restarts inside one `respond` call and
// hides the tool pass's own numbers.
//
// Every measurement line carries the `DSV4 CACHE:` prefix so a run log can be
// grepped for the numbers alone.
//
// This suite is a baseline, not a wish. When a later change moves these
// numbers, the assertions fail on purpose — that is the signal to re-read the
// measurement rather than to trust a memory of it.
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests`
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`, thus `swift build --build-tests` stays at exit 0 with
// a type error in it. Use the `xcodebuild build-for-testing` command of
// `DeepseekV4SharedCheckpoint.swift` as the compile evidence for any change.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

// MARK: - Constants

/// Prefix that makes every measurement line greppable in a run log.
private let measurementPrefix = "DSV4 CACHE:"

/// The per-test time limit of this suite, in minutes. Each measurement runs
/// four generation passes over a prompt of several thousand tokens.
private let suiteTimeLimitMinutes = 120

/// Rows in the stock report the agent reads. Sized so the rendered prompt
/// clears ``minimumPrefillTokenCount`` by a wide margin, which makes prefill —
/// not decoding — dominate each pass's wall clock.
private let stockReportRowCount = 120

/// The bay the agent is told to look up. The tool answer names it again, thus
/// a wrong bay in the reply shows a broken loop rather than a broken cache.
private let queriedBayNumber = 7

/// Tokens the round-1 prompt must reach for the timing comparison to mean
/// anything. Below this, prefill is too cheap to separate reuse from a rebuild.
private let minimumPrefillTokenCount = 1_500

/// Tokens each generation pass may produce. Large enough that thinking mode can
/// reason and still reach its tool call, small enough that decoding stays a
/// minor part of a pass.
private let generatedTokenBudget = 256

/// Divergent tokens printed from each side of a turn seam, enough to show the
/// generation tail without dumping the whole prompt.
private let divergenceReportTokenCount = 16

/// The share of the cold control's prefill time that the tool round must reach
/// for the run to count as "reprocessed everything". Reuse would drop the tool
/// round far below this, thus falling under it means upstream skipped work.
private let noReuseTimeFloorFraction = 0.8

/// Greedy decoding keeps every pass and the control reproducible.
private let greedyTemperature: Float = 0

/// The answer the fake tool gives back, which the agent then reports.
private let stockToolResultJSON = #"{"bay":"bay 7","pallets":42,"status":"sealed"}"#

/// The pallet count planted on the row of ``queriedBayNumber``. The recall
/// test asks the model to read it back, thus the answer proves whether the
/// model read the body of a long prompt at all.
private let plantedPalletCount = "4172"

/// Tokens the recall pass may produce. The answer is one number.
private let recallTokenBudget = 24

/// The system turn of the agent.
///
/// The prompt needs one. `DeepSeekV4ChatEncoder.Message.messages(from:tools:)`
/// attaches the tools to the first system or developer turn, and it makes an
/// EMPTY system turn to hold them when the conversation has none. The published
/// reference always carries a real system prompt there, thus an empty one puts
/// the whole `## Tools` section behind a shape the model never saw in training.
/// Measured on 2026-08-13 without this turn: the model answered with multilingual
/// gibberish and never wrote a tool call.
private let agentInstructions =
    "You are an inventory agent. You answer with the tools you are given."

/// The one tool the prompt offers the model. The encoder renders it into the
/// `## Tools` section, and `DSMLToolCallParser` reads the call back out.
private let stockToolSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_stock_level",
        "description": "Read the recorded stock level of one warehouse bay",
        "parameters": [
            "type": "object",
            "properties": [
                "bay": [
                    "type": "string",
                    "description": "The bay to read, for example bay 7",
                ] as [String: any Sendable]
            ] as [String: any Sendable],
            "required": ["bay"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

// MARK: - One measured pass

/// What one generation pass of `ChatSession` reported.
private struct PassMeasurement {
    /// The text the pass generated.
    let text: String
    /// The tool calls the pass emitted.
    let toolCalls: [ToolCall]
    /// Tokens the same conversation renders to with a cold cache.
    let renderedTokenCount: Int
    /// Tokens actually fed to the model this pass, after any cache reuse
    /// narrowed the input.
    let fedTokenCount: Int
    /// Seconds spent processing those tokens.
    let prefillSeconds: TimeInterval
    /// Tokens the pass generated.
    let generatedTokenCount: Int

    /// Rendered tokens the pass did not feed, which is the work reuse saved.
    var skippedTokenCount: Int { renderedTokenCount - fedTokenCount }
}

/// What one generation pass produced, before the rendered token count joins it.
private struct PassOutcome {
    /// The text the pass generated.
    let text: String
    /// The tool calls the pass emitted.
    let toolCalls: [ToolCall]
    /// What the model reported about the pass.
    let completionInfo: GenerateCompletionInfo
}

/// Collects one generation stream into a ``PassOutcome``.
///
/// - Parameter stream: the stream of one `ChatSession` pass.
/// - Returns: the text, the tool calls, and the completion info.
/// - Throws: ``IntegrationTestFailure`` when the pass reports no completion
///   info, and any error the session raises.
private func collect(
    _ stream: AsyncThrowingStream<Generation, Error>
) async throws -> PassOutcome {
    var text = ""
    var toolCalls: [ToolCall] = []
    var completionInfo: GenerateCompletionInfo?
    for try await generation in stream {
        switch generation {
        case .chunk(let chunk):
            text += chunk
        case .toolCall(let call):
            toolCalls.append(call)
        case .info(let info):
            completionInfo = info
        }
    }
    guard let completionInfo else {
        throw IntegrationTestFailure("a generation pass ended without a GenerateCompletionInfo")
    }
    return PassOutcome(text: text, toolCalls: toolCalls, completionInfo: completionInfo)
}

/// Joins one pass outcome to the cold render of the same conversation.
///
/// - Parameters:
///   - outcome: what the pass produced.
///   - renderedTokenCount: tokens the same conversation renders to cold.
/// - Returns: the measurement.
private func measurement(
    _ outcome: PassOutcome, renderedTokenCount: Int
) -> PassMeasurement {
    PassMeasurement(
        text: outcome.text,
        toolCalls: outcome.toolCalls,
        renderedTokenCount: renderedTokenCount,
        fedTokenCount: outcome.completionInfo.promptTokenCount,
        prefillSeconds: outcome.completionInfo.promptTime,
        generatedTokenCount: outcome.completionInfo.generationTokenCount)
}

// MARK: - The conversation

/// The `additionalContext` that selects one DeepSeek-V4 generation mode.
///
/// - Parameter mode: the mode to select.
/// - Returns: the template variables for that mode.
private func additionalContext(
    for mode: DeepSeekV4ChatEncoder.ThinkingMode
) -> [String: any Sendable] {
    ["thinking": mode == .thinking]
}

/// The stock report the agent reads, which is what makes each prompt long
/// enough for a real prefill rather than a template-only one.
///
/// The row of ``queriedBayNumber`` carries ``plantedPalletCount``, thus a
/// reader can prove the model read the body of the report.
///
/// - Returns: the report rows, one on each line.
private func makeStockReportRows() -> String {
    let rows = (1 ... stockReportRowCount).map { index in
        guard index == queriedBayNumber else {
            return "Row \(index): warehouse bay \(index) was audited last week, its seals "
                + "were intact, and no damage was recorded against its pallets."
        }
        return "Row \(index): warehouse bay \(index) holds \(plantedPalletCount) pallets, "
            + "its seals were intact, and no damage was recorded against them."
    }
    return rows.joined(separator: "\n")
}

/// The user turn that asks the agent to call the tool.
///
/// - Returns: the user turn.
private func makeStockReportPrompt() -> String {
    "Read the stock report below. Then call the get_stock_level tool for bay "
        + "\(queriedBayNumber). Call the tool before you write an answer.\n"
        + makeStockReportRows()
}

/// The user turn that asks the model to read one fact back out of the report,
/// with no tool in the prompt.
///
/// - Returns: the user turn.
private func makeRecallPrompt() -> String {
    "Read the stock report below. Then say how many pallets bay \(queriedBayNumber) "
        + "holds. Reply with just the number.\n"
        + makeStockReportRows()
}

/// The follow-up user turn, which grows the transcript one more time.
private let followUpPrompt = "Now say whether that bay is sealed. Reply in one sentence."

// MARK: - The suite

/// Measures whether upstream's own `ChatSession` path reuses a prompt cache
/// across an agentic tool round on the published DeepSeek-V4-Flash-4bit
/// checkpoint, in each of the two generation modes.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct DeepseekV4AgenticPromptCacheAssessmentTests {

    /// The tool round and the round after it, in chat mode.
    @Test func chatModeToolRoundReusesThePromptCache() async throws {
        try await measureToolRound(mode: .chat)
    }

    /// The tool round and the round after it, in thinking mode.
    @Test func thinkingModeToolRoundReusesThePromptCache() async throws {
        try await measureToolRound(mode: .thinking)
    }

    /// Every layer of a DeepSeek-V4 cache rewinds, which is what
    /// `RewindToCommonPrefixRule` needs. A hybrid stack with recurrent layers
    /// fails this, and that is what disqualified Qwen 3.6.
    @Test func everyCacheLayerRewinds() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(testName: "everyCacheLayerRewinds")
        else { return }

        let (layerCount, trimmableCount, kinds) = try await container.perform { context in
            let caches = try context.model.newCache(parameters: nil)
            let kinds = Set(caches.map { String(describing: type(of: $0)) }).sorted()
            return (caches.count, caches.count { $0.isTrimmable }, kinds)
        }
        print("\(measurementPrefix) cache layers = \(layerCount)")
        print("\(measurementPrefix) trimmable cache layers = \(trimmableCount)")
        print("\(measurementPrefix) cache kinds = \(kinds.joined(separator: ", "))")

        #expect(layerCount > 0, "the model must build one cache for each layer")
        #expect(
            trimmableCount == layerCount,
            """
            \(trimmableCount) of \(layerCount) cache layers rewind, thus \
            RewindToCommonPrefixRule cannot rescue a broken prefix. The kinds are \
            \(kinds.joined(separator: ", ")).
            """)
    }

    /// A long prompt that carries NO tool still answers with a fact planted in
    /// its body.
    ///
    /// This separates the two causes a failed tool round could have: the tools,
    /// and the prompt length. Every other real-weights DeepSeek-V4 test feeds a
    /// prompt of a few dozen tokens, thus none of them reads this range.
    ///
    /// The port has a known reason to fail here. `DeepSeekV4Model` runs plain
    /// dense attention on every layer: `DeepSeekV4Configuration.slidingWindow`
    /// decodes and nothing reads it, and the header of
    /// `Libraries/MLXLLM/Models/DeepSeekV4.swift` records that the compressor
    /// and the indexer both load and neither runs. A prompt under the
    /// 128-token window cannot tell that apart from the real model; a prompt of
    /// several thousand tokens can.
    @Test func longPromptWithoutToolsRecallsAPlantedFact() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "longPromptWithoutToolsRecallsAPlantedFact")
        else { return }

        let context = additionalContext(for: .chat)
        let prompt = makeRecallPrompt()
        let rendered = try await renderPromptTokens(
            container, messages: [.user(prompt)], additionalContext: context)
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(
                maxTokens: recallTokenBudget, temperature: greedyTemperature),
            additionalContext: context)
        let recall = try await collect(session.streamDetails(to: prompt))

        print("\(measurementPrefix) recall rendered prompt tokens = \(rendered.count)")
        print("\(measurementPrefix) recall answer = <<<\(recall.text)>>>")

        #expect(
            rendered.count >= minimumPrefillTokenCount,
            """
            the recall prompt (\(rendered.count) tokens) must reach \
            \(minimumPrefillTokenCount) tokens to read past the short-prompt range every \
            other DeepSeek-V4 test covers
            """)
        #expect(
            recall.text.contains(plantedPalletCount),
            """
            a \(rendered.count)-token prompt must still answer with \
            \(plantedPalletCount), which its own body states. It wrote: \
            <<<\(recall.text)>>>
            """)
    }

    /// Tokens the follow-up round of the long conversation may produce.
    private let conversationFollowUpTokenBudget = 32

    /// Measures what the prompt cache of a long conversation does from one
    /// turn to the next.
    ///
    /// This is the tool round of ``measureToolRound(mode:)`` with the tools
    /// taken out. The cache machinery is the same machinery -- `ChatSession`
    /// renders the whole transcript again for each turn and
    /// `ExtendCachedPrefixRule` feeds only the new tail -- thus this measures
    /// requirement 3 even while the model writes its tool calls in a syntax
    /// `DSMLToolCallParser` does not read.
    ///
    /// The two facts it takes are the two facts
    /// `Qwen36UpstreamPromptCacheAssessmentTests` takes, and DeepSeek-V4
    /// answers them differently:
    ///
    ///   (a) The follow-up render IS a true prefix extension of round 1's.
    ///       Qwen-3.6 fails here, because its template rewrites the priming
    ///       tail. The DeepSeek-V4 chat template does not.
    ///   (b) Upstream reprocesses the whole prompt anyway. A good prefix is
    ///       thus not sufficient, which is the number this suite exists to
    ///       hold.
    ///
    /// This is a baseline, not a wish. Every assertion below records what the
    /// run measured on 2026-08-14. When reuse appears, these assertions fail
    /// on purpose: invert them and record the new numbers.
    @Test func aLongConversationMeasuresPromptCacheReuseAcrossTurns() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "aLongConversationMeasuresPromptCacheReuseAcrossTurns")
        else { return }

        let context = additionalContext(for: .chat)
        let parameters = GenerateParameters(
            maxTokens: conversationFollowUpTokenBudget, temperature: greedyTemperature)
        let prompt = makeRecallPrompt()
        let session = ChatSession(
            container, generateParameters: parameters, additionalContext: context)

        let roundOneRendered = try await renderPromptTokens(
            container, messages: [.user(prompt)], additionalContext: context)
        let roundOne = measurement(
            try await collect(session.streamDetails(to: prompt)),
            renderedTokenCount: roundOneRendered.count)

        let transcript: [Chat.Message] = [.user(prompt), .assistant(roundOne.text)]
        let followUpRendered = try await renderPromptTokens(
            container, messages: transcript + [.user(followUpPrompt)],
            additionalContext: context)
        let followUp = measurement(
            try await collect(session.streamDetails(to: followUpPrompt)),
            renderedTokenCount: followUpRendered.count)

        // The control seeds a cold session with the same transcript, thus it
        // pays the whole prefill the live session skips.
        let controlSession = ChatSession(
            container, history: transcript, generateParameters: parameters,
            additionalContext: context)
        let control = measurement(
            try await collect(controlSession.streamDetails(to: followUpPrompt)),
            renderedTokenCount: followUpRendered.count)

        let seam = try await describeSeam(
            container, earlier: roundOneRendered, later: followUpRendered)
        let label = "\(measurementPrefix) conversation"
        reportPass(label: "\(label) round 1", pass: roundOne)
        reportPass(label: "\(label) follow-up round", pass: followUp)
        reportPass(label: "\(label) control", pass: control)
        print("\(label) follow-up extends round 1 = \(seam.isPrefixExtension)")
        print(
            "\(label) follow-up common prefix with round 1 = \(seam.sharedPrefixLength) "
                + "of \(seam.earlierLength)")
        print("\(label) round 1 divergent tail = <<<\(seam.earlierTail)>>>")
        print("\(label) follow-up divergent tail = <<<\(seam.laterTail)>>>")

        #expect(
            roundOne.renderedTokenCount >= minimumPrefillTokenCount,
            """
            round 1's rendered prompt (\(roundOne.renderedTokenCount) tokens) must reach \
            \(minimumPrefillTokenCount) tokens for prefill to dominate the timing
            """)
        #expect(!followUp.text.isEmpty, "the follow-up round must answer")

        // Fact (a): the DeepSeek-V4 chat template writes round 1's turn again
        // token for token, thus `ExtendCachedPrefixRule` is reachable here. This
        // is where Qwen-3.6 fails and DeepSeek-V4 passes.
        #expect(
            seam.isPrefixExtension,
            """
            (a) the follow-up render does not extend round 1's. The two share \
            \(seam.sharedPrefixLength) of round 1's \(seam.earlierLength) tokens. Round 1 \
            wrote <<<\(seam.earlierTail)>>> where the follow-up writes <<<\(seam.laterTail)>>>.
            """)

        // Fact (b): upstream reprocesses the whole prompt anyway. A good prefix
        // between the two RENDERS is thus not sufficient, because
        // `ExtendCachedPrefixRule` compares the new render against the LEDGER,
        // which is round 1's render PLUS the tokens round 1 generated. The
        // divergent tail printed above shows why the two part: the follow-up
        // render writes `</think>` at position 3626 and the ledger writes the
        // first generated token there. The encoder adds that token when it
        // renders an assistant turn as history, and the live prompt of round 1
        // does not end with it. One token thus breaks the prefix, and the
        // fall-back `RewindToCommonPrefixRule` needs a trimmable cache, which a
        // `RotatingKVCache` past its 128-token window is not.
        //
        // Card ^mscrreq holds the correction; until it lands, this number stays
        // at zero.
        #expect(
            followUp.skippedTokenCount == 0,
            """
            (b) the follow-up round skipped \(followUp.skippedTokenCount) of its \
            \(followUp.renderedTokenCount) rendered tokens; it fed every one of them when this \
            baseline was taken. Reuse appeared -- invert this assertion and close ^mscrreq.
            """)
        #expect(
            followUp.prefillSeconds >= control.prefillSeconds * noReuseTimeFloorFraction,
            """
            (b) the follow-up round spent \(followUp.prefillSeconds) s on prefill against the \
            cold control's \(control.prefillSeconds) s for the same prompt, dropping below \
            \(noReuseTimeFloorFraction) of the control. The follow-up stopped reprocessing the \
            whole prompt -- invert this assertion and close ^mscrreq.
            """)
    }

    // MARK: The measurement

    /// Runs one agentic tool round in `mode` and prints every measured number.
    ///
    /// The passes are: the user turn that emits a tool call, the tool-result
    /// continuation on the live session, the same continuation on a cold
    /// control session, and one more user turn on the live session.
    ///
    /// - Parameter mode: the DeepSeek-V4 generation mode to measure.
    /// - Throws: ``IntegrationTestFailure`` when a pass reports no completion
    ///   info, and any error the session raises.
    private func measureToolRound(mode: DeepSeekV4ChatEncoder.ThinkingMode) async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "\(mode.rawValue)ModeToolRoundReusesThePromptCache")
        else { return }

        let context = additionalContext(for: mode)
        let parameters = GenerateParameters(
            maxTokens: generatedTokenBudget, temperature: greedyTemperature)
        let userPrompt = makeStockReportPrompt()
        let session = ChatSession(
            container, instructions: agentInstructions, generateParameters: parameters,
            additionalContext: context, tools: [stockToolSchema])

        // `ChatSession` puts its `instructions` in front of every render, thus
        // each render below has to carry the same system turn.
        let roundOneMessages: [Chat.Message] = [.system(agentInstructions), .user(userPrompt)]
        let roundOneRendered = try await renderPromptTokens(
            container, messages: roundOneMessages, tools: [stockToolSchema],
            additionalContext: context)
        let roundOne = measurement(
            try await collect(session.streamDetails(to: userPrompt)),
            renderedTokenCount: roundOneRendered.count)
        let call = try #require(
            roundOne.toolCalls.first,
            """
            \(mode.rawValue) mode must emit one DSML tool call for the measurement to \
            reach a tool round. It wrote: \(roundOne.text)
            """)

        let toolMessages: [Chat.Message] = [
            .tool(stockToolResultJSON, id: call.id, name: call.function.name)
        ]
        let transcriptAfterRoundOne =
            roundOneMessages + [.assistant(roundOne.text, toolCalls: roundOne.toolCalls)]
        let toolRoundRendered = try await renderPromptTokens(
            container, messages: transcriptAfterRoundOne + toolMessages,
            tools: [stockToolSchema], additionalContext: context)
        let toolRound = measurement(
            try await collect(session.streamDetails(to: toolMessages)),
            renderedTokenCount: toolRoundRendered.count)

        // The control takes no `instructions:`, because its history already
        // opens with the same system turn. A second one would change its
        // prompt, and the two prefill times would stop comparing.
        let controlSession = ChatSession(
            container, history: transcriptAfterRoundOne, generateParameters: parameters,
            additionalContext: context, tools: [stockToolSchema])
        let control = measurement(
            try await collect(controlSession.streamDetails(to: toolMessages)),
            renderedTokenCount: toolRoundRendered.count)

        let transcriptAfterToolRound =
            transcriptAfterRoundOne + toolMessages
            + [.assistant(toolRound.text, toolCalls: toolRound.toolCalls)]
        let followUpRendered = try await renderPromptTokens(
            container, messages: transcriptAfterToolRound + [.user(followUpPrompt)],
            tools: [stockToolSchema], additionalContext: context)
        let followUp = measurement(
            try await collect(session.streamDetails(to: followUpPrompt)),
            renderedTokenCount: followUpRendered.count)

        let seam = try await describeSeam(
            container, earlier: roundOneRendered, later: toolRoundRendered)
        report(
            mode: mode, roundOne: roundOne, toolRound: toolRound, control: control,
            followUp: followUp, seam: seam)

        expectAgenticRound(
            mode: mode, roundOne: roundOne, toolRound: toolRound, control: control,
            followUp: followUp, seam: seam)
    }

    // MARK: The turn seam

    /// What the two renders share, and what each writes where they part.
    private struct SeamReport {
        /// Whether the later render extends the earlier one whole.
        let isPrefixExtension: Bool
        /// Leading tokens the two renders share.
        let sharedPrefixLength: Int
        /// Tokens the earlier render holds.
        let earlierLength: Int
        /// The earlier render's text from the first divergent token.
        let earlierTail: String
        /// The later render's text from the same position.
        let laterTail: String
    }

    /// Measures where two renders part, and decodes both sides of that point.
    ///
    /// - Parameters:
    ///   - container: the loaded model container.
    ///   - earlier: the render taken first.
    ///   - later: the render taken after one more turn.
    /// - Returns: the seam report.
    private func describeSeam(
        _ container: LLModelContainer, earlier: [Int], later: [Int]
    ) async throws -> SeamReport {
        let sharedPrefixLength = commonPrefixLength(earlier, later)
        let earlierTail = await decodeTokens(
            container,
            tokens: divergentTail(
                of: earlier, from: sharedPrefixLength, limit: divergenceReportTokenCount))
        let laterTail = await decodeTokens(
            container,
            tokens: divergentTail(
                of: later, from: sharedPrefixLength, limit: divergenceReportTokenCount))
        return SeamReport(
            isPrefixExtension: later.starts(with: earlier),
            sharedPrefixLength: sharedPrefixLength,
            earlierLength: earlier.count,
            earlierTail: earlierTail,
            laterTail: laterTail)
    }

    // MARK: Reporting

    /// Prints every measured number under ``measurementPrefix``.
    ///
    /// - Parameters:
    ///   - mode: the generation mode measured.
    ///   - roundOne: the user turn that emitted the tool call.
    ///   - toolRound: the tool-result continuation on the live session.
    ///   - control: the same continuation on a cold session.
    ///   - followUp: the user turn after the tool round.
    ///   - seam: where round 1's render and the tool round's render part.
    private func report(
        mode: DeepSeekV4ChatEncoder.ThinkingMode,
        roundOne: PassMeasurement,
        toolRound: PassMeasurement,
        control: PassMeasurement,
        followUp: PassMeasurement,
        seam: SeamReport
    ) {
        let label = "\(measurementPrefix) \(mode.rawValue)"
        reportPass(label: "\(label) round 1", pass: roundOne)
        print("\(label) round 1 tool calls = \(roundOne.toolCalls.map(\.function.name))")
        reportPass(label: "\(label) tool round", pass: toolRound)
        reportPass(label: "\(label) control", pass: control)
        reportPass(label: "\(label) follow-up round", pass: followUp)
        print("\(label) tool round extends round 1 = \(seam.isPrefixExtension)")
        print(
            "\(label) tool round common prefix with round 1 = \(seam.sharedPrefixLength) "
                + "of \(seam.earlierLength)")
        print("\(label) round 1 divergent tail = <<<\(seam.earlierTail)>>>")
        print("\(label) tool round divergent tail = <<<\(seam.laterTail)>>>")
    }

    /// Prints the four numbers of one pass.
    ///
    /// - Parameters:
    ///   - label: the greppable prefix of each line.
    ///   - pass: the pass to report.
    private func reportPass(label: String, pass: PassMeasurement) {
        print("\(label) rendered prompt tokens = \(pass.renderedTokenCount)")
        print("\(label) fed prompt tokens = \(pass.fedTokenCount)")
        print("\(label) tokens skipped by reuse = \(pass.skippedTokenCount)")
        print("\(label) prefill seconds = \(pass.prefillSeconds)")
        print("\(label) generated tokens = \(pass.generatedTokenCount)")
    }

    // MARK: Assertions

    /// Holds the measured behavior, thus a later change that moves it fails
    /// here rather than passing silently.
    ///
    /// - Parameters:
    ///   - mode: the generation mode measured.
    ///   - roundOne: the user turn that emitted the tool call.
    ///   - toolRound: the tool-result continuation on the live session.
    ///   - control: the same continuation on a cold session.
    ///   - followUp: the user turn after the tool round.
    ///   - seam: where round 1's render and the tool round's render part.
    private func expectAgenticRound(
        mode: DeepSeekV4ChatEncoder.ThinkingMode,
        roundOne: PassMeasurement,
        toolRound: PassMeasurement,
        control: PassMeasurement,
        followUp: PassMeasurement,
        seam: SeamReport
    ) {
        #expect(
            roundOne.renderedTokenCount >= minimumPrefillTokenCount,
            """
            round 1's rendered prompt (\(roundOne.renderedTokenCount) tokens) must reach \
            \(minimumPrefillTokenCount) tokens for prefill to dominate the timing
            """)
        #expect(!toolRound.text.isEmpty, "the tool round must answer from the tool result")
        #expect(!followUp.text.isEmpty, "the follow-up round must answer")

        // Fact (a): the tool round's render extends round 1's render whole, thus
        // `ExtendCachedPrefixRule` is reachable.
        #expect(
            seam.isPrefixExtension,
            """
            (a) \(mode.rawValue) mode: the tool round's render does not extend round 1's. \
            The two share \(seam.sharedPrefixLength) of round 1's \(seam.earlierLength) \
            tokens. Round 1 wrote <<<\(seam.earlierTail)>>> where the tool round writes \
            <<<\(seam.laterTail)>>>.
            """)

        // Fact (b): the tool round feeds only the new tail, and it beats the cold
        // control by a wide margin.
        #expect(
            toolRound.skippedTokenCount > 0,
            """
            (b) \(mode.rawValue) mode: the tool round fed every one of its \
            \(toolRound.renderedTokenCount) rendered tokens and skipped none, thus the \
            live cache saved no work.
            """)
        #expect(
            toolRound.prefillSeconds < control.prefillSeconds * noReuseTimeFloorFraction,
            """
            (b) \(mode.rawValue) mode: the tool round spent \(toolRound.prefillSeconds) s \
            on prefill against the cold control's \(control.prefillSeconds) s for the same \
            prompt, which is not below \(noReuseTimeFloorFraction) of the control.
            """)
        #expect(
            followUp.skippedTokenCount > 0,
            """
            (b) \(mode.rawValue) mode: the round after the tool round fed every one of its \
            \(followUp.renderedTokenCount) rendered tokens and skipped none.
            """)
    }
}

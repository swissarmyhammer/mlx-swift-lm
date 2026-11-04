// MTP speculative decoding in multi-turn chat sessions that reuse their prompt
// cache, beside sessions without speculation over the same turns.

import Foundation
import MLXLMCommon

/// A turn that goes back to the end of an earlier reply of a chat and asks a
/// different question there, as an app does when its user edits a message and
/// regenerates the reply.
public struct MTPChatEdit: Sendable {
    /// The turns the edit keeps: it goes back to the end of the reply of turn
    /// `keptTurnCount - 1` and drops every turn after it.
    public let keptTurnCount: Int
    /// The prompt asked in place of the turns the edit drops.
    public let prompt: String

    /// Creates an edit that keeps the first `keptTurnCount` turns, then asks `prompt`.
    public init(keptTurnCount: Int, prompt: String) {
        self.keptTurnCount = keptTurnCount
        self.prompt = prompt
    }
}

/// A scripted chat that ``Benchmark/mtpSystemPromptCache(container:drafter:scenarios:systemPrompt:systemPromptCharacterCount:blockSize:)``
/// runs in a `ChatSession` with MTP speculative decoding and in one without speculation.
public struct MTPChatScenario: Sendable {
    /// The name of the chat in the result lines.
    public let name: String
    /// The thinking switch of both sessions, `ChatSession/thinkingEnabled`.
    public let thinking: Bool
    /// The user prompts, one for each turn, asked in order in one session.
    public let prompts: [String]
    /// A last turn that goes back to an earlier reply, or `nil` for none.
    public let edit: MTPChatEdit?
    /// Tokens each turn may generate.
    public let maxTokens: Int

    /// Creates a chat of `prompts`, with `edit` as its last turn when given.
    public init(
        name: String, thinking: Bool, prompts: [String], edit: MTPChatEdit? = nil, maxTokens: Int
    ) {
        self.name = name
        self.thinking = thinking
        self.prompts = prompts
        self.edit = edit
        self.maxTokens = maxTokens
    }

    /// Tokens a turn with thinking off may generate.
    public static let answerTokenBudget = 192
    /// Tokens a turn with thinking on may generate: the reasoning comes before the answer.
    public static let reasoningTokenBudget = 1_024
    /// Tokens a warm-up turn may generate.
    static let warmUpTokenBudget = 8

    /// Math follow-up turns with thinking off: each turn extends the turn before it.
    public static let followUps = MTPChatScenario(
        name: "follow-ups", thinking: false,
        prompts: [
            "A train leaves at 9:40 and arrives at 13:05. It stops twice for 12 minutes. "
                + "How long was it moving?",
            "A shop sells 3 pens for $4. How much do 18 pens cost? Show the working.",
            "Add 15% sales tax to that price. What is the total?",
        ],
        maxTokens: answerTokenBudget)

    /// Math follow-up turns with thinking on. The Qwen3.8 template drops the
    /// reasoning of earlier replies when it renders the history again, so each
    /// follow-up differs from the cached tokens of the turn before it.
    public static let thinkingFollowUps = MTPChatScenario(
        name: "thinking", thinking: true,
        prompts: [
            "What is 17 * 23? Show the working briefly.",
            "Divide that result by 7 and round to two decimals.",
            "Is the original product a prime number? Answer in one sentence.",
        ],
        maxTokens: reasoningTokenBudget)

    /// Two math turns, then an edit that goes back to the end of the first reply
    /// and asks a different second question.
    public static let editAndRegenerate = MTPChatScenario(
        name: "edit", thinking: false,
        prompts: [
            "A rectangle is 12 cm by 7 cm. What are its area and perimeter?",
            "Double both sides. What are the new area and perimeter?",
        ],
        edit: MTPChatEdit(
            keptTurnCount: 1,
            prompt: "Halve both sides instead. What are the new area and perimeter?"),
        maxTokens: answerTokenBudget)

    /// Follow-ups, follow-ups with thinking on, and an edit.
    public static let defaults = [followUps, thinkingFollowUps, editAndRegenerate]

    /// A one-turn chat that compiles the kernels of a decode path before the timed chats.
    static let warmUp = MTPChatScenario(
        name: "warm-up", thinking: false, prompts: ["Reply with one word: ready."],
        maxTokens: warmUpTokenBudget)
}

/// What one turn of an ``MTPChatScenario`` reported in its `.info` event.
public struct MTPChatTurn: Sendable {
    /// The position of the turn in its chat. An edit comes after every prompt.
    public let index: Int
    /// Whether the turn is the ``MTPChatScenario/edit`` of its chat.
    public let isEdit: Bool
    /// Prompt tokens the turn prefilled.
    public let promptTokenCount: Int
    /// Prompt tokens the reused cache served, `GenerateCompletionInfo/cachedPromptTokenCount`.
    public let cachedPromptTokenCount: Int
    /// Prefill time, in milliseconds.
    public let promptTimeMs: Double
    /// Tokens the turn generated.
    public let generationTokenCount: Int
    /// Decode throughput, in tokens per second.
    public let decodeTokensPerSecond: Double
    /// Why the session ran the turn without speculation, from the `.info` event.
    public let fallbackReason: SpeculativeDecodingFallbackReason?
    /// Why the MTP iterator stopped drafting, from the `.info` event.
    public let passthroughReason: String?
    /// Tokens the drafter proposed, or `nil` when no MTP iterator ran.
    public let proposedDraftTokenCount: Int?
    /// Proposed tokens the target accepted, or `nil` when no MTP iterator ran.
    public let acceptedDraftTokenCount: Int?
    /// The generated text.
    public let text: String
    /// The generated tokens, with the candidates of each step when the session
    /// reported them.
    public let tokens: [MTPChatToken]

    init(
        index: Int, isEdit: Bool, info: GenerateCompletionInfo, text: String,
        tokens: [MTPChatToken]
    ) {
        self.index = index
        self.isEdit = isEdit
        self.tokens = tokens
        self.promptTokenCount = info.promptTokenCount
        self.cachedPromptTokenCount = info.cachedPromptTokenCount
        self.promptTimeMs = info.promptTime * Benchmark.millisecondsPerSecond
        self.generationTokenCount = info.generationTokenCount
        self.decodeTokensPerSecond = info.tokensPerSecond
        self.fallbackReason = info.speculativeDecodingFallbackReason
        self.passthroughReason = info.passthroughReason
        self.proposedDraftTokenCount = info.proposedDraftTokens
        self.acceptedDraftTokenCount = info.acceptedDraftTokens
        self.text = text
    }

    /// The rendered prompt: the cached prefix and the prefilled tokens.
    public var totalPromptTokenCount: Int { promptTokenCount + cachedPromptTokenCount }

    /// Whether the drafter proposed tokens in this turn.
    public var speculated: Bool { (proposedDraftTokenCount ?? 0) > 0 }

    /// Prompt tokens the stateful MTP drafter prefilled, or `nil` for a session
    /// without speculation.
    ///
    /// No event reports this count. The MTP iterator feeds the drafter the
    /// tokens it feeds the target: the whole prompt on a cold cache, the suffix
    /// past the cached prefix on a warm one. So the drafter prefilled
    /// ``promptTokenCount`` when it drafted, and nothing when the session fell
    /// back or the iterator passed through before its first draft.
    public var drafterPrefillTokenCount: Int? {
        if fallbackReason != nil {
            return 0
        }
        guard proposedDraftTokenCount != nil else { return nil }
        return speculated ? promptTokenCount : 0
    }

    /// Accepted divided by proposed drafts, or `nil` when no draft was proposed.
    public var acceptanceRate: Double? {
        guard let proposed = proposedDraftTokenCount else { return nil }
        return DraftAcceptance.rate(accepted: acceptedDraftTokenCount ?? 0, proposed: proposed)
    }

    /// Whether speculation ran, and why it did not when it was configured.
    public var speculationDescription: String {
        if let fallbackReason {
            return "fell back (\(fallbackReason))"
        }
        guard proposedDraftTokenCount != nil else { return "off" }
        let state = speculated ? "MTP" : "MTP without drafts"
        return passthroughReason.map { "\(state), passthrough: \($0)" } ?? state
    }

    /// One result line for the turn of `scenario` in the `session` named.
    func summaryLine(scenario: String, session: String) -> String {
        let turn = isEdit ? "turn \(index) (edit)" : "turn \(index)"
        let drafter = drafterPrefillTokenCount.map { "\($0) tok" } ?? "-"
        let acceptance = acceptanceRate.map { String(format: "%.1f%%", $0 * 100) } ?? "-"
        return "[\(scenario)] \(session) \(turn): prompt \(totalPromptTokenCount) tok, "
            + "cached \(cachedPromptTokenCount) tok, prefilled \(promptTokenCount) tok in "
            + "\(BenchmarkLog.decimal(promptTimeMs))ms, drafter prefilled \(drafter), "
            + "speculation \(speculationDescription), decode \(generationTokenCount) tok at "
            + "\(BenchmarkLog.decimal(decodeTokensPerSecond)) tok/s, acceptance \(acceptance)"
    }
}

/// The turns of one ``MTPChatScenario`` in the MTP session and in the plain session.
public struct MTPChatScenarioResult: Sendable {
    /// The chat that ran.
    public let scenario: MTPChatScenario
    /// The turns of the session with MTP speculative decoding.
    public let mtp: [MTPChatTurn]
    /// The same turns in the session without speculation.
    public let plain: [MTPChatTurn]

    /// Characters of text a result line quotes.
    static let excerptLength = 100

    /// Writes one line for each turn of each session, the start of the text of
    /// the turn, and where the MTP text leaves the plain text when they differ.
    public func logSummary() {
        for (mtpTurn, plainTurn) in zip(mtp, plain) {
            let speedup = mtpTurn.decodeTokensPerSecond / plainTurn.decodeTokensPerSecond
            let match = MTPChatTokenMatch(mtp: mtpTurn.tokens, plain: plainTurn.tokens)
            BenchmarkLog.line(
                mtpTurn.summaryLine(scenario: scenario.name, session: "MTP  ")
                    + ", text equal to plain: \(mtpTurn.text == plainTurn.text), "
                    + "tokens match plain: \(match.summary), "
                    + "decode \(BenchmarkLog.hundredths(speedup))x plain")
            BenchmarkLog.line(plainTurn.summaryLine(scenario: scenario.name, session: "plain"))
            BenchmarkLog.line(
                "[\(scenario.name)] turn \(mtpTurn.index) text starts "
                    + "<<<\(Self.excerpt(of: plainTurn.text[...]))>>>")
            if mtpTurn.text != plainTurn.text {
                BenchmarkLog.line(Self.divergenceLine(mtpTurn, plainTurn, scenario: scenario.name))
            }
        }
    }

    /// Where the text of `mtpTurn` leaves the text of `plainTurn`, and what each says there.
    private static func divergenceLine(
        _ mtpTurn: MTPChatTurn, _ plainTurn: MTPChatTurn, scenario: String
    ) -> String {
        let shared = zip(mtpTurn.text, plainTurn.text).prefix { $0 == $1 }.count
        return "[\(scenario)] turn \(mtpTurn.index) MTP text leaves plain text after "
            + "\(shared) of \(plainTurn.text.count) characters: MTP <<<"
            + excerpt(of: mtpTurn.text.dropFirst(shared)) + ">>> plain <<<"
            + excerpt(of: plainTurn.text.dropFirst(shared)) + ">>>"
    }

    /// The first ``excerptLength`` characters of `text`, on one line.
    private static func excerpt(of text: Substring) -> String {
        String(text.prefix(excerptLength)).replacingOccurrences(of: "\n", with: "\\n")
    }
}

/// Runs the turns of an ``MTPChatScenario`` in one ``ChatSession``.
private struct MTPChatSessionRunner {
    /// Messages one turn adds to the history: the prompt and the reply.
    static let messagesPerTurn = 2

    let container: ModelContainer
    let instructions: String
    let speculativeDecoding: SpeculativeDecodingConfig?

    /// The candidates each turn reports for each step: the two best for a plain
    /// session (``MTPBlockSweep/reportedCandidateCount``), so a parting from it
    /// can be told a near-tie, and none for a speculative session, which cannot
    /// report them.
    var candidateCount: Int? {
        speculativeDecoding == nil ? MTPBlockSweep.reportedCandidateCount : nil
    }

    /// Runs the prompts of `scenario` in a new session, then its edit in a fork
    /// whose history ends at the reply the edit goes back to.
    func run(_ scenario: MTPChatScenario) async throws -> [MTPChatTurn] {
        let session = ChatSession(
            container, instructions: instructions, speculativeDecoding: speculativeDecoding,
            generateParameters: GenerateParameters(maxTokens: scenario.maxTokens, temperature: 0))
        session.thinkingEnabled = scenario.thinking
        var turns: [MTPChatTurn] = []
        var history: [Chat.Message] = []
        for (index, prompt) in scenario.prompts.enumerated() {
            let turn = try await respond(in: session, to: prompt, index: index, isEdit: false)
            turns.append(turn)
            history += [.user(prompt), .assistant(turn.text)]
        }
        guard let edit = scenario.edit else { return turns }
        let branch = await session.fork(
            history: Array(history.prefix(edit.keptTurnCount * Self.messagesPerTurn)))
        turns.append(
            try await respond(in: branch, to: edit.prompt, index: turns.count, isEdit: true))
        return turns
    }

    /// Asks `prompt` in `session` and collects the text, the tokens and the
    /// `.info` event.
    private func respond(
        in session: ChatSession, to prompt: String, index: Int, isEdit: Bool
    ) async throws -> MTPChatTurn {
        var tokens: [MTPChatToken] = []
        let recorder = MTPChatTokenRecorder(candidateCount: candidateCount)
        let transcript = try await ChatTurnTranscript.collect(
            session.observe(to: prompt, using: recorder)
        ) { event in
            if case .token(let token) = event { tokens.append(token) }
        }
        return MTPChatTurn(
            index: index, isEdit: isEdit, info: transcript.info, text: transcript.text,
            tokens: tokens)
    }
}

extension Benchmark {
    /// Benchmark MTP speculative decoding in a multi-turn `ChatSession` that reuses
    /// its prompt cache, beside a session without speculation over the same turns.
    ///
    /// Both sessions share a system prompt of `systemPromptCharacterCount` benchmark
    /// characters and decode greedily, so their texts can be compared. Each
    /// ``MTPChatScenario`` runs once in a new session of each kind, after an
    /// unrecorded warm-up turn of each kind. Every turn writes one line for each
    /// session: the prompt tokens, the cached and prefilled tokens, the prefill
    /// time, the drafter prefill, whether MTP ran or why it fell back, the decode
    /// throughput and the acceptance.
    ///
    /// - Parameters:
    ///   - container: the target model
    ///   - drafter: the MTP drafter of the target
    ///   - scenarios: the chats to run
    ///   - systemPrompt: the system prompt, or `nil` for a math tutor preamble and
    ///     `systemPromptCharacterCount` characters of benchmark text
    ///   - systemPromptCharacterCount: the characters of benchmark text a `nil`
    ///     system prompt reads
    ///   - blockSize: the tokens of each MTP round
    /// - Returns: one result for each scenario
    public static func mtpSystemPromptCache(
        container: ModelContainer,
        drafter: MTPDrafterContainer,
        scenarios: [MTPChatScenario] = MTPChatScenario.defaults,
        systemPrompt: String? = nil,
        systemPromptCharacterCount: Int = 8_000,
        blockSize: Int = SpeculativeDecodingConfig.defaultMTPBlockSize
    ) async throws -> [MTPChatScenarioResult] {
        let instructions = try await BenchmarkSystemPrompt.load(
            systemPrompt, preamble: BenchmarkSystemPrompt.mathTutorPreamble,
            characterCount: systemPromptCharacterCount)
        let mtp = MTPChatSessionRunner(
            container: container, instructions: instructions,
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: drafter, blockSize: blockSize))
        let plain = MTPChatSessionRunner(
            container: container, instructions: instructions, speculativeDecoding: nil)

        _ = try await plain.run(.warmUp)
        _ = try await mtp.run(.warmUp)

        var results: [MTPChatScenarioResult] = []
        for scenario in scenarios {
            let result = MTPChatScenarioResult(
                scenario: scenario, mtp: try await mtp.run(scenario),
                plain: try await plain.run(scenario))
            result.logSummary()
            results.append(result)
        }
        return results
    }
}

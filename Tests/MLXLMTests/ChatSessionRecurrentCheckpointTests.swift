// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A chat session of a hybrid model goes back to a checkpoint of its recurrent
/// state when the next render parts from the cached tokens.
///
/// The recurrent layers of a hybrid model (the Gated DeltaNet layers of Qwen 3.5)
/// hold one state that sums every token they saw, thus they cannot rewind. The
/// Qwen 3.8 chat template writes `<think>\n` at the end of a prompt, and writes
/// `<think>\n\n</think>` for the same assistant turn in the next render. Card
/// `^95mp4sw` measured that: each turn at 32k prefilled the whole prompt again.
/// The session keeps a checkpoint at the end of the last message and at the end
/// of the prompt, and the next turn goes back to the longest one the new render
/// starts with.
///
/// The model of this suite is a small Qwen 3.5 text model with one linear layer
/// and one attention layer. The tokenizer renders a Qwen-like transcript.
@Suite(.serialized)
struct ChatSessionRecurrentCheckpointTests {

    /// How the template writes a past assistant turn.
    enum HistoryShape: CaseIterable {
        /// The past turn opens with a closed think block where the generation
        /// prompt opened an empty one, as the Qwen 3.8 template does when the
        /// session records the reasoning as content.
        case closesTheThinkBlock

        /// The past turn opens with the generation prompt unchanged, thus the
        /// next render keeps the whole previous render.
        case keepsTheGenerationPrompt

        /// Each render opens with a token that names the message count, thus a
        /// next render agrees with no cached token at all.
        case rewritesTheFirstToken
    }

    /// The tokens of a rendered conversation.
    ///
    /// A content letter is a token from 1 to ``letterCount``. The markers stand
    /// above the letters, and the end-of-sequence token stands outside the
    /// vocabulary of the model, thus the model never ends a turn early.
    enum TemplateToken {
        /// The number of tokens that content text maps to.
        static let letterCount = 16

        /// Opens a user message.
        static let user = letterCount + 1

        /// Opens an assistant message.
        static let assistant = letterCount + 2

        /// Closes a message.
        static let endOfTurn = letterCount + 3

        /// The open think block of the generation prompt, `<think>\n`.
        static let openThink = letterCount + 4

        /// The closed think block of a past turn, `<think>\n\n</think>`.
        static let closedThink = letterCount + 5

        /// Opens a system message.
        static let system = letterCount + 6

        /// The first token of a render that names its message count.
        static let firstMessageCount = letterCount + 7

        /// The vocabulary size of the model.
        static let vocabularySize = 32

        /// The end-of-sequence token, outside the vocabulary of the model.
        static let endOfSequence = vocabularySize + 8

        /// The first Unicode scalar that ``CheckpointTemplateTokenizer`` decodes a
        /// token to.
        static let decodedScalarBase = 0xE000
    }

    /// Renders a Qwen-like transcript, with the shape of a past assistant turn
    /// that ``HistoryShape`` names.
    struct CheckpointTemplateTokenizer: Tokenizer {
        /// How the template writes a past assistant turn.
        let historyShape: HistoryShape

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil
        var eosTokenId: Int? { TemplateToken.endOfSequence }
        var unknownTokenId: Int? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.unicodeScalars.map { Int($0.value) % TemplateToken.letterCount + 1 }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(
                String.UnicodeScalarView(
                    tokenIds.compactMap { UnicodeScalar(TemplateToken.decodedScalarBase + $0) }))
        }

        func convertTokenToId(_ token: String) -> Int? { nil }

        func convertIdToToken(_ id: Int) -> String? {
            UnicodeScalar(TemplateToken.decodedScalarBase + id).map(String.init)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens: [Int] = []
            if historyShape == .rewritesTheFirstToken {
                tokens.append(TemplateToken.firstMessageCount + messages.count)
            }
            for message in messages {
                let content = encode(
                    text: message["content"] as? String ?? "", addSpecialTokens: false)
                tokens += opening(of: message["role"] as? String) + content
                tokens.append(TemplateToken.endOfTurn)
            }
            return tokens + [TemplateToken.assistant, generationThinkBlock(additionalContext)]
        }

        /// The think block of the generation prompt: the closed block when the
        /// template variables turn thinking off, as the Qwen 3.8 template writes
        /// it, and else the open block.
        private func generationThinkBlock(_ additionalContext: [String: any Sendable]?) -> Int {
            additionalContext?[ChatSessionRecurrentCheckpointTests.thinkingFlag] as? Bool == false
                ? TemplateToken.closedThink : TemplateToken.openThink
        }

        /// The tokens that open a message of `role`.
        private func opening(of role: String?) -> [Int] {
            switch role {
            case "user":
                [TemplateToken.user]
            case "assistant":
                [
                    TemplateToken.assistant,
                    historyShape == .closesTheThinkBlock
                        ? TemplateToken.closedThink : TemplateToken.openThink,
                ]
            default:
                [TemplateToken.system]
            }
        }
    }

    /// The seed of the weights of the model, thus every run generates the same
    /// tokens.
    private static let weightSeed: UInt64 = 7

    /// The number of tokens each turn generates.
    private static let generatedTokenCount = 3

    /// The number of layers of the model: one linear layer, then one attention
    /// layer.
    private static let hybridLayerCount = 2

    /// One full-attention layer for each two layers.
    private static let hybridFullAttentionInterval = 2

    /// The tokens the generation prompt writes after the last message.
    private static let generationPromptTokenCount = 2

    /// The text of the first user turn.
    private static let firstTurn = "first"

    /// The text of the second user turn.
    private static let secondTurn = "second"

    /// Greedy generation of ``generatedTokenCount`` tokens.
    private static let parameters = GenerateParameters(
        maxTokens: generatedTokenCount, temperature: 0)

    /// A small hybrid Qwen 3.5 model with a tokenizer that writes past turns in
    /// `historyShape`.
    private func hybridContext(historyShape: HistoryShape) throws -> ModelContext {
        var configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(qwen35TextConfigJSON(mtpLayers: 0).utf8))
        configuration.hiddenLayers = Self.hybridLayerCount
        configuration.fullAttentionInterval = Self.hybridFullAttentionInterval
        configuration.vocabularySize = TemplateToken.vocabularySize
        MLXRandom.seed(Self.weightSeed)
        let model = Qwen35TextModel(configuration)
        eval(model)
        let tokenizer = CheckpointTemplateTokenizer(historyShape: historyShape)
        let modelConfiguration = ModelConfiguration(id: "checkpoint-test")
        return ModelContext(
            configuration: modelConfiguration, model: model,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: modelConfiguration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
    }

    /// The text and the completion information of one turn.
    private struct Turn {
        var text = ""
        var info: GenerateCompletionInfo?
    }

    /// Runs one turn of `session`.
    private func respond(_ session: ChatSession, to prompt: String) async throws -> Turn {
        var turn = Turn()
        for try await item in session.streamDetails(to: prompt) {
            turn.text += item.chunk ?? ""
            turn.info = item.info ?? turn.info
        }
        return turn
    }

    /// Runs the first turn and then the second turn of one session.
    private func twoTurns(historyShape: HistoryShape) async throws -> (
        first: GenerateCompletionInfo, second: GenerateCompletionInfo
    ) {
        let session = ChatSession(
            try hybridContext(historyShape: historyShape), generateParameters: Self.parameters)
        let first = try #require(try await respond(session, to: Self.firstTurn).info)
        let second = try #require(try await respond(session, to: Self.secondTurn).info)
        return (first, second)
    }

    @Test("the premise: the model has a cache that cannot rewind")
    func theModelHasACacheThatCannotRewind() throws {
        let caches = try hybridContext(historyShape: .closesTheThinkBlock).model
            .newCache(parameters: nil)

        #expect(caches.contains { !$0.isTrimmable })
        #expect(caches.contains { $0.isTrimmable })
    }

    @Test("a render that rewrites the generation prompt reuses the prompt up to the last message")
    func aRenderThatRewritesTheGenerationPromptReusesThePromptUpToTheLastMessage()
        async throws
    {
        let turns = try await twoTurns(historyShape: .closesTheThinkBlock)
        let lastMessageEnd = turns.first.totalPromptTokenCount - Self.generationPromptTokenCount

        #expect(turns.second.cachedPromptTokenCount == lastMessageEnd)
        #expect(
            turns.second.promptTokenCount
                == turns.second.totalPromptTokenCount - lastMessageEnd)
    }

    @Test("a render that keeps the generation prompt reuses the whole previous prompt")
    func aRenderThatKeepsTheGenerationPromptReusesTheWholePreviousPrompt() async throws {
        let turns = try await twoTurns(historyShape: .keepsTheGenerationPrompt)

        #expect(turns.second.cachedPromptTokenCount == turns.first.totalPromptTokenCount)
    }

    @Test("the control: a render that rewrites the first token reuses nothing")
    func aRenderThatRewritesTheFirstTokenReusesNothing() async throws {
        let turns = try await twoTurns(historyShape: .rewritesTheFirstToken)

        #expect(turns.second.cachedPromptTokenCount == 0)
        #expect(turns.second.promptTokenCount == turns.second.totalPromptTokenCount)
    }

    @Test("the first turn reports every prompt token as prefilled")
    func theFirstTurnReportsEveryPromptTokenAsPrefilled() async throws {
        let turns = try await twoTurns(historyShape: .closesTheThinkBlock)

        #expect(turns.first.cachedPromptTokenCount == 0)
        #expect(turns.first.promptTokenCount == turns.first.totalPromptTokenCount)
    }

    // MARK: - The state a restore gives

    /// The tokens a checkpoint stands after.
    private static let checkpointTokens = [1, 2, 3, 4, 5]

    /// The tokens fed after the checkpoint and then dropped: the generation
    /// prompt and the generated turn that the next render writes again.
    private static let droppedTokens = [6, 7]

    /// The tokens the next render adds after the checkpoint.
    private static let nextTokens = [8, 9, 10]

    /// The largest difference between two logits that the comparison takes as
    /// equal. A prefill in two forward passes and a prefill in one pass sum in
    /// another order, thus the two are not bit-identical.
    private static let logitTolerance = 1e-4

    /// The logits of ``nextTokens`` fed into `caches`.
    private func logitsOfNextTokens(_ caches: [KVCache], model: any LanguageModel) -> MLXArray {
        let tokens = MLXArray(Self.nextTokens).expandedDimensions(axis: 0)
        return model(LMInput.Text(tokens: tokens), cache: caches, state: nil).logits
    }

    /// The logits of ``nextTokens`` after a cold prefill of
    /// ``checkpointTokens`` and ``nextTokens`` in one forward pass.
    private func coldLogits(model: any LanguageModel) throws -> MLXArray {
        let caches = try model.newCache(parameters: nil)
        let tokens = MLXArray(Self.checkpointTokens + Self.nextTokens).expandedDimensions(axis: 0)
        let logits = model(LMInput.Text(tokens: tokens), cache: caches, state: nil).logits
        return logits[0..., Self.checkpointTokens.count..., 0...]
    }

    /// Caches that hold ``checkpointTokens`` and then ``droppedTokens``, and
    /// the checkpoint they took between the two.
    private func cachesPastACheckpoint(model: any LanguageModel) throws -> (
        caches: [KVCache], checkpoint: PromptCacheCheckpoint
    ) {
        let caches = try model.newCache(parameters: nil)
        _ = try PromptCacheCheckpoint.prefill(
            Self.checkpointTokens, into: caches, state: nil, model: model, prefill: .init())
        let checkpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: Self.checkpointTokens, state: nil))
        _ = try PromptCacheCheckpoint.prefill(
            Self.droppedTokens, into: caches, state: nil, model: model, prefill: .init())
        return (caches, checkpoint)
    }

    @Test("a restored checkpoint continues as a cold prefill of the same tokens")
    func aRestoredCheckpointContinuesAsAColdPrefillOfTheSameTokens() throws {
        let model = try hybridContext(historyShape: .closesTheThinkBlock).model
        let past = try cachesPastACheckpoint(model: model)
        let restored = try #require(past.checkpoint.restored(onto: past.caches))
        #expect(restored.allSatisfy { $0.offset == Self.checkpointTokens.count })

        let warm = logitsOfNextTokens(restored, model: model)

        #expect(allClose(warm, try coldLogits(model: model), atol: Self.logitTolerance).item())
    }

    @Test("the control: caches that keep the dropped tokens do not continue as a cold prefill")
    func cachesThatKeepTheDroppedTokensDoNotContinueAsAColdPrefill() throws {
        let model = try hybridContext(historyShape: .closesTheThinkBlock).model
        let past = try cachesPastACheckpoint(model: model)

        let stale = logitsOfNextTokens(past.caches, model: model)

        #expect(!allClose(stale, try coldLogits(model: model), atol: Self.logitTolerance).item())
    }

    // MARK: - The end of the last message

    /// The template variable that turns thinking on and off, as the Qwen 3.8
    /// template reads it.
    static let thinkingFlag = "enable_thinking"

    /// The tokens of a prompt: the messages, then the generation prompt.
    private static let promptWithGenerationPrompt = [1, 2, 3, 4, 5, 6, 7]

    /// The instructions of a conversation.
    private static let instructions = "terse"

    /// The answer of the assistant to the first user turn.
    private static let firstAnswer = "ok"

    /// The result of a tool call, after the first answer.
    private static let toolResult = "result"

    /// The error a render that refuses its messages throws.
    private struct RefusedRender: Error {}

    /// A conversation whose end of the last message the suite finds: the role
    /// and the text of each message, in order.
    struct ProbedConversation: CustomTestStringConvertible {
        let testDescription: String
        let turns: [(role: Chat.Message.Role, text: String)]

        /// The messages of the conversation.
        var messages: [Chat.Message] {
            turns.map { Chat.Message(role: $0.role, content: $0.text) }
        }
    }

    /// The conversations of the comparison with the full probe.
    static let probedConversations = [
        ProbedConversation(testDescription: "one user turn", turns: [(.user, firstTurn)]),
        ProbedConversation(
            testDescription: "instructions and one user turn",
            turns: [(.system, instructions), (.user, firstTurn)]),
        ProbedConversation(
            testDescription: "a second user turn",
            turns: [(.user, firstTurn), (.assistant, firstAnswer), (.user, secondTurn)]),
        ProbedConversation(
            testDescription: "a tool result",
            turns: [(.user, firstTurn), (.assistant, firstAnswer), (.tool, toolResult)]),
    ]

    /// How the template renders a conversation: the shape of a past assistant
    /// turn, and thinking on or off.
    struct Rendering: CustomTestStringConvertible {
        let historyShape: HistoryShape
        let thinking: Bool

        var testDescription: String { "\(historyShape), thinking \(thinking ? "on" : "off")" }
    }

    /// Each history shape, with thinking on and with thinking off.
    static let renderings = HistoryShape.allCases.flatMap { shape in
        [true, false].map { Rendering(historyShape: shape, thinking: $0) }
    }

    /// A prepared input of `tokens`.
    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens))
    }

    /// The tokens of `messages` as the template renders them in `rendering`.
    private static func render(_ messages: [Chat.Message], _ rendering: Rendering) throws -> [Int] {
        try CheckpointTemplateTokenizer(historyShape: rendering.historyShape).applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: messages), tools: nil,
            additionalContext: [thinkingFlag: rendering.thinking])
    }

    /// The reference: the end of the last message as the full probe finds it,
    /// which is the prefix that the prompt shares with a render of the messages
    /// plus one user message.
    private static func fullProbeEnd(of messages: [Chat.Message], _ rendering: Rendering) throws
        -> Int
    {
        commonPrefixLength(
            of: try render(messages, rendering),
            and: try render(
                messages + [.user(PromptCacheCheckpoint.boundaryProbeText)], rendering))
    }

    /// The end of the last message that the session finds for `messages`, and
    /// each message list the session renders to find it.
    private static func lastMessageEnd(of messages: [Chat.Message], _ rendering: Rendering)
        async throws -> (end: Int?, renders: [[String]])
    {
        var renders: [[String]] = []
        let end = await PromptCacheCheckpoint.lastMessageEnd(
            of: input(try render(messages, rendering)), messages: messages
        ) { probe in
            renders.append(probe.map(\.content))
            return input(try render(probe, rendering))
        }
        return (end, renders)
    }

    @Test(
        "the end of the last message is the end the full probe finds",
        arguments: probedConversations, renderings)
    func theEndOfTheLastMessageIsTheEndTheFullProbeFinds(
        conversation: ProbedConversation, rendering: Rendering
    ) async throws {
        let found = try await Self.lastMessageEnd(of: conversation.messages, rendering)

        #expect(found.end == (try Self.fullProbeEnd(of: conversation.messages, rendering)))
    }

    @Test(
        "the control: a render that rewrites the first token ends the last message away from the generation prompt",
        arguments: probedConversations)
    func aRenderThatRewritesTheFirstTokenEndsTheLastMessageAwayFromTheGenerationPrompt(
        conversation: ProbedConversation
    ) throws {
        let rendering = Rendering(historyShape: .rewritesTheFirstToken, thinking: true)
        let prompt = try Self.render(conversation.messages, rendering)

        #expect(
            try Self.fullProbeEnd(of: conversation.messages, rendering)
                != prompt.count - Self.generationPromptTokenCount)
    }

    @Test("a conversation that ends with a user message renders no full probe")
    func aConversationThatEndsWithAUserMessageRendersNoFullProbe() async throws {
        let messages: [Chat.Message] = [
            .user(Self.firstTurn), .assistant(Self.firstAnswer), .user(Self.secondTurn),
        ]

        let found = try await Self.lastMessageEnd(
            of: messages, Rendering(historyShape: .closesTheThinkBlock, thinking: true))

        #expect(!found.renders.contains { $0.contains(Self.secondTurn) })
    }

    @Test("a conversation that ends with a tool result renders the full probe")
    func aConversationThatEndsWithAToolResultRendersTheFullProbe() async throws {
        let messages: [Chat.Message] = [
            .user(Self.firstTurn), .assistant(Self.firstAnswer), .tool(Self.toolResult),
        ]

        let found = try await Self.lastMessageEnd(
            of: messages, Rendering(historyShape: .closesTheThinkBlock, thinking: true))

        #expect(
            found.renders.contains(
                messages.map(\.content) + [PromptCacheCheckpoint.boundaryProbeText]))
    }

    @Test("a render that refuses the probe message gives no end of the last message")
    func aRenderThatRefusesTheProbeMessageGivesNoEndOfTheLastMessage() async {
        let end = await PromptCacheCheckpoint.lastMessageEnd(
            of: Self.input(Self.promptWithGenerationPrompt),
            messages: [.user(Self.firstTurn)]
        ) { _ in throw RefusedRender() }

        #expect(end == nil)
    }

    // MARK: - The feed up to a checkpoint

    /// The tokens of a prompt that a feed stops inside.
    private static let fedPrompt = checkpointTokens + droppedTokens

    /// A generated token that a protocol rule keeps in the caches, where the
    /// render writes another token.
    private static let keptGeneratedToken = 11

    /// The tokens the caches represent after the whole of ``fedPrompt``: a
    /// protocol rule kept ``keptGeneratedToken`` in place of the first prompt
    /// token.
    private static let fedRepresentedTokens = [keptGeneratedToken] + fedPrompt.dropFirst()

    @Test("a feed to a checkpoint stops at the boundary and takes the checkpoint there")
    func aFeedToACheckpointStopsAtTheBoundaryAndTakesTheCheckpointThere() throws {
        let model = try hybridContext(historyShape: .closesTheThinkBlock).model
        let caches = try model.newCache(parameters: nil)

        let feed = try #require(
            try PromptCacheCheckpoint.prefillToCheckpoint(
                at: Self.checkpointTokens.count, of: Self.fedPrompt,
                representing: Self.fedRepresentedTokens, from: 0, into: caches, state: nil,
                model: model, prefill: .init()))

        #expect(caches.allSatisfy { $0.offset == Self.checkpointTokens.count })
        #expect(
            feed.checkpoint?.tokens
                == Array(Self.fedRepresentedTokens.prefix(Self.checkpointTokens.count)))
    }

    @Test("a feed to a checkpoint the caches hold already feeds nothing")
    func aFeedToACheckpointTheCachesHoldAlreadyFeedsNothing() throws {
        let model = try hybridContext(historyShape: .closesTheThinkBlock).model
        let caches = try model.newCache(parameters: nil)

        let feed = try PromptCacheCheckpoint.prefillToCheckpoint(
            at: Self.checkpointTokens.count, of: Self.fedPrompt,
            representing: Self.fedPrompt, from: Self.checkpointTokens.count, into: caches,
            state: nil, model: model, prefill: .init())

        #expect(feed == nil)
        #expect(caches.allSatisfy { $0.offset == 0 })
    }

    @Test("a feed to a checkpoint at the end of the prompt feeds nothing")
    func aFeedToACheckpointAtTheEndOfThePromptFeedsNothing() throws {
        let model = try hybridContext(historyShape: .closesTheThinkBlock).model
        let caches = try model.newCache(parameters: nil)

        let feed = try PromptCacheCheckpoint.prefillToCheckpoint(
            at: Self.fedPrompt.count, of: Self.fedPrompt, representing: Self.fedPrompt,
            from: 0, into: caches, state: nil, model: model, prefill: .init())

        #expect(feed == nil)
        #expect(caches.allSatisfy { $0.offset == 0 })
    }
}

// Copyright © 2026 Apple Inc.
//
// Renders the real `chat_template.jinja` of `mlx-community/Qwen3.8-27B-mxfp4` with
// thinking on, with the template flag `enable_thinking` false, and with the closed
// block of `QwenReasoningProtocol.qwen35`. The suite reads the tokenizer and the
// template from the local Hugging Face cache and loads no weights.
//
// `Qwen35ThinkingOffRenderTests` in `MLXFoundationModelsTests` holds the executor to
// the same contract with a scripted processor, because the package has no Jinja
// engine.
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/Qwen35ThinkingOffTemplateTests`

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import Testing
import Tokenizers

@testable import MLXLMCommon

/// Downloads nothing new: the files of the pinned revision stand in the local cache.
private let qwen35TemplateDownloader: any Downloader = #hubDownloader()

/// Loads the tokenizer and the chat template of the checkpoint.
private let qwen35TemplateTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

/// The hybrid checkpoint whose template the suite renders.
private let qwen35TemplateModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// The snapshot of the checkpoint in the local cache.
private let qwen35TemplateRevision = "97ab0819817ab1c61d7d39f9169fc71999915641"

/// The chat template flag that turns thinking on and off.
private let thinkingFlag = "enable_thinking"

/// The header of the generation prompt.
private let generationPrompt = "<|im_start|>assistant\n"

/// The open think block that the template writes after the generation prompt when
/// thinking is on.
private let openThinkBlock = "<think>\n"

/// The closed, empty think block that the template writes after the generation prompt
/// when the flag is false.
private let closedThinkBlock = "<think>\n\n</think>\n\n"

/// The text that the template writes between the reasoning and the answer of a past
/// assistant turn, which is also the text the model writes to close its reasoning.
private let reasoningClose = "\n</think>\n\n"

/// The marker that ends a turn, and thus the system block.
private let endOfTurn = "<|im_end|>"

/// Renders the real Qwen 3.5 chat template with thinking on and off.
@Suite(.serialized)
struct Qwen35ThinkingOffTemplateTests {

    /// The instructions of the conversation.
    private static let instructions = "You are a terse, literal assistant."

    /// The first prompt of the conversation.
    private static let firstPrompt = "My favorite color is teal. Reply with just \"OK\"."

    /// The second prompt of the conversation.
    private static let secondPrompt = "What is my favorite color?"

    /// The answer of the assistant to the first prompt.
    private static let firstAnswer = "OK"

    /// The reasoning of the assistant before its first answer, when thinking is on.
    private static let firstReasoning = "The user states a fact and asks for OK."

    /// One tool, in the shape of a tool specification.
    private static let tool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "lookup",
            "description": "Looks a word up.",
            "parameters": [
                "type": "object",
                "properties": ["word": ["type": "string"]],
                "required": ["word"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// The render with thinking off keeps the system block of the render with thinking
    /// on, ends with the generation prompt and the closed block, and holds the tokens of
    /// a joint render of that text. The render with the flag false changes the system
    /// block, which is the defect the closed block avoids.
    @Test(arguments: [false, true])
    func aThinkingOffRenderKeepsTheSystemBlockOfThinkingOn(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer()
        let tools = withTools ? [Self.tool] : nil
        let messages = [system(), user(Self.firstPrompt)]

        let thinkingOn = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: [thinkingFlag: true])
        let flagOff = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: [thinkingFlag: false])
        let thinkingOff = try thinkingOffRender(
            messages: messages, tools: tools, tokenizer: tokenizer)

        let onText = tokenizer.decode(tokenIds: thinkingOn)
        let flagOffText = tokenizer.decode(tokenIds: flagOff)
        let offText = tokenizer.decode(tokenIds: thinkingOff)
        #expect(systemBlock(of: offText) == systemBlock(of: onText), "off <<<\(offText)>>>")
        #expect(
            systemBlock(of: flagOffText) != systemBlock(of: onText),
            "The flag false must change the system block, or the closed block is not needed.")
        #expect(offText.hasSuffix(generationPrompt + closedThinkBlock), "off <<<\(offText)>>>")
        #expect(flagOffText.hasSuffix(generationPrompt + closedThinkBlock))
        #expect(onText.hasSuffix(generationPrompt + openThinkBlock))
        #expect(thinkingOff == tokenizer.encode(text: offText, addSpecialTokens: false))
    }

    /// A turn with thinking off, after a turn with thinking on, starts with the whole
    /// conversation that the turn with thinking on rendered before its generation prompt.
    @Test(arguments: [false, true])
    func aThinkingOffTurnStartsWithTheThinkingOnConversation(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer()
        let tools = withTools ? [Self.tool] : nil
        let firstTurn = [system(), user(Self.firstPrompt)]
        let secondTurn =
            firstTurn + [
                assistant(Self.firstAnswer, reasoning: Self.firstReasoning),
                user(Self.secondPrompt),
            ]

        let thinkingOn = try tokenizer.applyChatTemplate(
            messages: firstTurn, tools: tools, additionalContext: [thinkingFlag: true])
        let thinkingOff = try thinkingOffRender(
            messages: secondTurn, tools: tools, tokenizer: tokenizer)

        let onText = tokenizer.decode(tokenIds: thinkingOn)
        let conversation = String(onText.dropLast((generationPrompt + openThinkBlock).count))
        let offText = tokenizer.decode(tokenIds: thinkingOff)
        #expect(offText.hasPrefix(conversation), "off <<<\(offText)>>>")
    }

    /// A turn with thinking on, after a turn with thinking off, extends the tokens of the
    /// turn with thinking off: the history render of the empty think block is the closed
    /// block the turn with thinking off ended with.
    @Test(arguments: [false, true])
    func aThinkingOnTurnExtendsTheThinkingOffTurn(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer()
        let tools = withTools ? [Self.tool] : nil
        let firstTurn = [system(), user(Self.firstPrompt)]
        let secondTurn =
            firstTurn + [assistant(Self.firstAnswer, reasoning: ""), user(Self.secondPrompt)]

        let thinkingOff = try thinkingOffRender(
            messages: firstTurn, tools: tools, tokenizer: tokenizer)
        let thinkingOn = try tokenizer.applyChatTemplate(
            messages: secondTurn, tools: tools, additionalContext: [thinkingFlag: true])

        #expect(
            Array(thinkingOn.prefix(thinkingOff.count)) == thinkingOff,
            "on <<<\(tokenizer.decode(tokenIds: thinkingOn))>>>")
    }

    /// A turn with thinking on, after a turn with thinking on, extends the tokens of the
    /// turn before it and the text the model wrote in that turn: the template keeps the
    /// `reasoning_content` of a past turn inside its think block (card `^z4aax33`). Thus a
    /// session that records the reasoning there keeps the generated region in the render
    /// of the next turn, and the prompt cache can hold that region.
    @Test(arguments: [false, true])
    func aThinkingOnTurnKeepsTheReasoningOfThePastTurn(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer()
        let tools = withTools ? [Self.tool] : nil
        let firstTurn = [system(), user(Self.firstPrompt)]
        let secondTurn =
            firstTurn + [
                assistant(Self.firstAnswer, reasoning: Self.firstReasoning),
                user(Self.secondPrompt),
            ]

        let first = try tokenizer.applyChatTemplate(
            messages: firstTurn, tools: tools, additionalContext: [thinkingFlag: true])
        let second = try tokenizer.applyChatTemplate(
            messages: secondTurn, tools: tools, additionalContext: [thinkingFlag: true])

        let generated = Self.firstReasoning + reasoningClose + Self.firstAnswer + endOfTurn
        let secondText = tokenizer.decode(tokenIds: second)
        #expect(
            secondText.hasPrefix(tokenizer.decode(tokenIds: first) + generated),
            "second <<<\(secondText)>>>")
        #expect(Array(second.prefix(first.count)) == first)
    }

    // MARK: - The end of the last message

    /// The result of a tool call, after the first answer.
    private static let toolResult = "teal is a blue-green color"

    /// The template variable that keeps or drops the reasoning of a past assistant turn.
    private static let preserveThinkingFlag = "preserve_thinking"

    /// How a render states thinking: on or off as the template flag, or off as the closed
    /// block of the Qwen 3.5 protocol.
    enum ThinkingState {
        case flag(Bool)
        case closedBlock

        /// Thinking on, thinking off as the flag, and thinking off as the closed block.
        static let all: [ThinkingState] = [.flag(true), .flag(false), .closedBlock]
    }

    /// The end of the last message that `PromptCacheCheckpoint.lastMessageEnd` finds is the
    /// end that the full probe finds: the prefix that the prompt shares with a render of the
    /// messages plus one user message (card `^phckrdd`).
    @Test(arguments: ThinkingState.all, [false, true])
    func theEndOfTheLastMessageIsTheEndTheFullProbeFinds(
        thinking: ThinkingState, withTools: Bool
    ) async throws {
        let tokenizer = try await loadTokenizer()
        let render = probeRender(
            thinking: thinking, tools: withTools ? [Self.tool] : nil, tokenizer: tokenizer)
        let firstTurn: [Chat.Message] = [.system(Self.instructions), .user(Self.firstPrompt)]
        let secondTurn =
            firstTurn + [
                .assistant(Self.firstAnswer, reasoning: Self.firstReasoning),
                .user(Self.secondPrompt),
            ]

        for messages in [firstTurn, secondTurn] {
            let end = await PromptCacheCheckpoint.lastMessageEnd(
                of: try render(messages), messages: messages, render: render)

            #expect(end == (try fullProbeEnd(of: messages, render: render)))
        }
    }

    /// The control: after a tool result, with `preserve_thinking` false, the full probe parts
    /// from the prompt inside the past assistant turn, because one more user message drops
    /// its reasoning. The end of the last message is then not the start of the generation
    /// prompt, and `PromptCacheCheckpoint.lastMessageEnd` still finds the end of the full probe.
    @Test func afterAToolResultTheEndOfTheLastMessageIsTheEndTheFullProbeFinds() async throws {
        let tokenizer = try await loadTokenizer()
        let render = templateRender(
            context: [thinkingFlag: true, Self.preserveThinkingFlag: false], tools: nil,
            tokenizer: tokenizer)
        let messages: [Chat.Message] = [
            .system(Self.instructions), .user(Self.firstPrompt),
            .assistant(Self.firstAnswer, reasoning: Self.firstReasoning), .tool(Self.toolResult),
        ]
        let prompt = try render(messages)

        let reference = try fullProbeEnd(of: messages, render: render)
        let end = await PromptCacheCheckpoint.lastMessageEnd(
            of: prompt, messages: messages, render: render)

        let shared = tokenizer.decode(
            tokenIds: Array(prompt.text.tokens.asArray(Int.self).prefix(reference)))
        #expect(!shared.contains(Self.firstReasoning), "shared <<<\(shared)>>>")
        #expect(end == reference)
    }

    /// The reference: the prefix that the prompt of `messages` shares with a render of
    /// `messages` plus one user message.
    private func fullProbeEnd(
        of messages: [Chat.Message], render: ([Chat.Message]) throws -> LMInput
    ) throws -> Int {
        commonPrefixLength(
            of: try render(messages).text.tokens.asArray(Int.self),
            and: try render(messages + [.user(PromptCacheCheckpoint.boundaryProbeText)]).text
                .tokens.asArray(Int.self))
    }

    /// Renders messages with the real template, in the thinking state `thinking`.
    private func probeRender(
        thinking: ThinkingState, tools: [[String: any Sendable]]?,
        tokenizer: any MLXLMCommon.Tokenizer
    ) -> ([Chat.Message]) throws -> LMInput {
        switch thinking {
        case .flag(let enabled):
            return templateRender(
                context: [thinkingFlag: enabled], tools: tools, tokenizer: tokenizer)
        case .closedBlock:
            return { messages in
                Self.input(
                    try thinkingOffRender(
                        messages: DefaultMessageGenerator().generate(messages: messages),
                        tools: tools, tokenizer: tokenizer))
            }
        }
    }

    /// Renders messages with the real template and the template variables `context`.
    private func templateRender(
        context: [String: any Sendable], tools: [[String: any Sendable]]?,
        tokenizer: any MLXLMCommon.Tokenizer
    ) -> ([Chat.Message]) throws -> LMInput {
        { messages in
            Self.input(
                try tokenizer.applyChatTemplate(
                    messages: DefaultMessageGenerator().generate(messages: messages), tools: tools,
                    additionalContext: context))
        }
    }

    /// A prepared input of `tokens`.
    private static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens))
    }

    // MARK: - Fixtures

    /// Loads the tokenizer and the chat template of the pinned snapshot.
    private func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        let directory = try await qwen35TemplateDownloader.download(
            id: qwen35TemplateModelID, revision: qwen35TemplateRevision,
            matching: ["*.json", "*.jinja"], useLatest: false, progressHandler: { _ in })
        return try await qwen35TemplateTokenizerLoader.load(from: directory)
    }

    /// The render with thinking off that the Qwen 3.5 protocol asks for: the template
    /// variables of the strategy, and the closed block in place of the open block.
    private func thinkingOffRender(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> [Int] {
        let config = QwenReasoningProtocol.qwen35
        let block = try #require(config.promptStrategy.closedBlock(forThinkingEnabled: false))
        let rendered = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools,
            additionalContext: try config.promptStrategy.additionalContext(
                forThinkingEnabled: false))
        return config.closingReasoning(in: rendered, with: block, tokenizer: tokenizer)
    }

    /// The text of `render` up to and with the marker that ends its first turn.
    private func systemBlock(of render: String) -> String {
        guard let end = render.range(of: endOfTurn) else { return render }
        return String(render[..<end.upperBound])
    }

    /// The system message of the conversation.
    private func system() -> [String: any Sendable] {
        ["role": "system", "content": Self.instructions]
    }

    /// A user message.
    private func user(_ content: String) -> [String: any Sendable] {
        ["role": "user", "content": content]
    }

    /// An assistant message with its reasoning.
    private func assistant(_ content: String, reasoning: String) -> [String: any Sendable] {
        ["role": "assistant", "content": content, "reasoning_content": reasoning]
    }
}

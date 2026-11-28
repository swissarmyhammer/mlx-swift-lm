// Copyright © 2026 Apple Inc.
//
// Renders the real chat templates of `mlx-community/Qwen3.8-27B-mxfp4` and
// `mlx-community/Qwen3.5-9B-4bit` through `ThinkingRender` with a named reasoning effort
// level (decision D218). The suite reads the tokenizers and the templates from the local
// Hugging Face cache and loads no weights.
//
// `ReasoningEffortLevelTests` in `MLXFoundationModelsTests` holds the executor to the same
// contract with a scripted processor, because the package has no Jinja engine.
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/ReasoningEffortTemplateTests`

import Foundation
import HuggingFace
import MLXHuggingFace
import Testing
import Tokenizers

@testable import MLXLMCommon

/// Downloads the template files of the pinned revisions, which stand in the local cache.
private let effortTemplateDownloader: any Downloader = #hubDownloader()

/// Loads the tokenizer and the chat template of a checkpoint.
private let effortTemplateTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

/// Renders the real Qwen3.8 and Qwen 3.5 templates with and without a named effort level.
@Suite(.serialized)
struct ReasoningEffortTemplateTests {

    /// A checkpoint whose template the suite renders.
    struct Checkpoint: Sendable, CustomTestStringConvertible {
        let id: String
        let revision: String
        var testDescription: String { id }

        /// The Qwen3.8 checkpoint, whose template reads `reasoning_effort`.
        static let qwen38 = Checkpoint(
            id: "mlx-community/Qwen3.8-27B-mxfp4",
            revision: "97ab0819817ab1c61d7d39f9169fc71999915641")

        /// The Qwen 3.5 checkpoint, whose template does not read `reasoning_effort`.
        static let qwen35 = Checkpoint(
            id: "mlx-community/Qwen3.5-9B-4bit",
            revision: "8b2b98c00a6b4d291155e4890773ca8f769aee53")
    }

    /// The instructions of the conversation.
    private static let instructions = "You are a terse, literal assistant."

    /// The prompt of the conversation.
    private static let prompt = "What is 2 + 2?"

    /// The instructions that the Qwen3.8 template writes at the level `low`.
    private static let lowInstructions = "Reasoning effort is set to low."

    /// The instructions that the Qwen3.8 template writes at its default level.
    private static let defaultInstructions = "Reasoning effort is set to xhigh."

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

    @Test(arguments: [false, true])
    func `the Qwen3.8 template writes the named level`(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer(.qwen38)

        let text = try render(effort: "low", withTools: withTools, tokenizer: tokenizer)

        #expect(text.contains(Self.lowInstructions), "<<<\(text)>>>")
        #expect(!text.contains(Self.defaultInstructions), "<<<\(text)>>>")
    }

    @Test(arguments: [false, true])
    func `the Qwen3.8 template keeps its default with no level`(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer(.qwen38)

        let text = try render(effort: nil, withTools: withTools, tokenizer: tokenizer)

        #expect(text.contains(Self.defaultInstructions), "<<<\(text)>>>")
    }

    @Test(arguments: [false, true])
    func `the Qwen3.8 template with thinking off writes no level`(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer(.qwen38)

        let withLevel = try render(
            effort: "low", thinking: false, withTools: withTools, tokenizer: tokenizer)
        let withoutLevel = try render(
            effort: nil, thinking: false, withTools: withTools, tokenizer: tokenizer)

        #expect(withLevel == withoutLevel, "<<<\(withLevel)>>>")
        #expect(!withLevel.contains(Self.lowInstructions))
    }

    @Test(arguments: [false, true])
    func `the Qwen 3.5 template does not change with a level`(withTools: Bool) async throws {
        let tokenizer = try await loadTokenizer(.qwen35)

        let withLevel = try render(effort: "low", withTools: withTools, tokenizer: tokenizer)
        let withoutLevel = try render(effort: nil, withTools: withTools, tokenizer: tokenizer)

        #expect(withLevel == withoutLevel, "<<<\(withLevel)>>>")
    }

    // MARK: - Fixtures

    /// Renders the conversation with the variables of a `QwenReasoningProtocol.qwen35`
    /// render, and gives the text of the tokens.
    private func render(
        effort: String?, thinking: Bool = true, withTools: Bool,
        tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> String {
        let config = QwenReasoningProtocol.qwen35
        let thinkingRender = try ThinkingRender(
            config: config, thinkingEnabled: thinking, effort: effort)
        var tokens = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": Self.instructions],
                ["role": "user", "content": Self.prompt],
            ],
            tools: withTools ? [Self.tool] : nil,
            additionalContext: thinkingRender.additionalContext)
        if let block = config.promptStrategy.closedBlock(forThinkingEnabled: thinking) {
            tokens = config.closingReasoning(in: tokens, with: block, tokenizer: tokenizer)
        }
        return tokenizer.decode(tokenIds: tokens)
    }

    /// Loads the tokenizer and the chat template of the pinned snapshot of `checkpoint`.
    private func loadTokenizer(_ checkpoint: Checkpoint) async throws
        -> any MLXLMCommon.Tokenizer
    {
        let directory = try await effortTemplateDownloader.download(
            id: checkpoint.id, revision: checkpoint.revision,
            matching: ["*.json", "*.jinja"], useLatest: false, progressHandler: { _ in })
        return try await effortTemplateTokenizerLoader.load(from: directory)
    }
}

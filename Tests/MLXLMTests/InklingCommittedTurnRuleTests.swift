// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests of the cache-reuse rule of the Inkling 2 protocol. The rule is pure, thus the
/// splice is examined without a model.
///
/// Each fixture writes the frames the real weights write. The LEDGER holds the tokens the model
/// wrote, ending at the `<|content_model_end_sampling|>` commit that closes its turn. The RENDER
/// holds the same conversation as history, and its version of a tool call differs: the
/// published template writes compact JSON with sorted keys, the model writes its own spacing.
/// The token 70 below stands for the call the model wrote, and 71 for the call the render
/// writes in its place.
struct InklingCommittedTurnRuleTests {

    /// Stands in for `<|content_model_end_sampling|>`.
    private static let commit = 1

    /// Stands in for `<|content_thinking|>`.
    private static let thinking = 2

    /// Stands in for `<|end_message|>`.
    private static let endMessage = 3

    /// Stands in for `<|message_model|>`.
    private static let messageModel = 4

    /// Stands in for `<|content_invoke_tool_json|>`.
    private static let toolJSON = 5

    /// Stands in for `<|message_tool|>`.
    private static let messageTool = 6

    /// Stands in for `<|content_text|>`.
    private static let text = 7

    /// Stands in for the tokens of the conversation before the turn.
    private static let conversation = [10, 11, 12]

    /// Stands in for the reasoning the model wrote.
    private static let reasoning = 13

    /// Stands in for the tool name in a frame header.
    private static let toolName = 14

    /// Stands in for the text of a tool result.
    private static let toolResult = 15

    /// Stands in for the call the model wrote.
    private static let generatedCall = 70

    /// Stands in for the call the render writes where the model wrote ``generatedCall``.
    private static let renderedCall = 71

    /// The thinking frame of the turn.
    private static let thinkingFrame = [messageModel, thinking, reasoning, endMessage]

    /// The tool result frame that answers one call.
    private static let toolFrame = [messageTool, toolName, text, toolResult, endMessage]

    /// A tokenizer that names the three markers of the rule.
    private static let tokenizer = MarkerTokenizer(identifierOfMarker: [
        InklingCommittedTurnRule.endOfTurnMarker: commit,
        InklingCommittedTurnRule.openThinkMarker: thinking,
        InklingCommittedTurnRule.closeThinkMarker: endMessage,
    ])

    /// A tool frame of the model that holds `call`.
    private static func callFrame(_ call: Int) -> [Int] {
        [messageModel, toolName, toolJSON, call, endMessage]
    }

    /// A cache whose timeline agrees with its ledger.
    private static func alignedCache(_ cached: [Int], previousRender: [Int]) -> PromptCacheState {
        PromptCacheState(
            cachedTokens: cached, previousRenderTokens: previousRender,
            processedTokenCount: cached.count, mainCacheIsAligned: true)
    }

    /// The rule the Inkling format contributes.
    private static func rule() throws -> any PromptCacheReuseRule {
        try #require(ToolCallFormat.inkling.promptCacheReuseRules(tokenizer: tokenizer).first)
    }

    // MARK: - Splicing

    @Test func `splices the tool results onto the tool call the model wrote`() throws {
        let written = Self.thinkingFrame + Self.callFrame(Self.generatedCall) + [Self.commit]
        let prompt =
            Self.conversation + Self.thinkingFrame + Self.callFrame(Self.renderedCall)
            + [Self.commit] + Self.toolFrame

        let decision = try Self.rule().reuse(
            turn: PromptCacheTurn(promptTokens: prompt, isToolResultContinuation: true),
            cache: Self.alignedCache(
                Self.conversation + written, previousRender: Self.conversation))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: prompt.count - Self.toolFrame.count,
                    representedTokens: Self.conversation + written + Self.toolFrame))
    }

    @Test func `splices the results of parallel calls at the one commit of the turn`() throws {
        let written =
            Self.thinkingFrame + Self.callFrame(Self.generatedCall)
            + Self.callFrame(Self.generatedCall) + [Self.commit]
        let results = Self.toolFrame + Self.toolFrame
        let prompt =
            Self.conversation + Self.thinkingFrame + Self.callFrame(Self.renderedCall)
            + Self.callFrame(Self.renderedCall) + [Self.commit] + results

        let decision = try Self.rule().reuse(
            turn: PromptCacheTurn(promptTokens: prompt, isToolResultContinuation: true),
            cache: Self.alignedCache(
                Self.conversation + written, previousRender: Self.conversation))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: prompt.count - results.count,
                    representedTokens: Self.conversation + written + results))
    }

    // MARK: - Selection

    @Test func `the inkling format contributes the rule`() {
        let rules = ToolCallFormat.inkling.promptCacheReuseRules(tokenizer: Self.tokenizer)

        #expect(rules.count == 1)
        #expect(rules.first is InklingCommittedTurnRule)
    }

    @Test func `a tokenizer without the end-of-sampling marker contributes no rule`() {
        let tokenizer = MarkerTokenizer(identifierOfMarker: [:])

        #expect(ToolCallFormat.inkling.promptCacheReuseRules(tokenizer: tokenizer).isEmpty)
    }

    @Test func `no other format contributes the Inkling rule`() {
        for format in ToolCallFormat.allCases where format != .inkling {
            let rules = format.promptCacheReuseRules(tokenizer: Self.tokenizer)
            #expect(rules.allSatisfy { !($0 is InklingCommittedTurnRule) })
        }
    }
}

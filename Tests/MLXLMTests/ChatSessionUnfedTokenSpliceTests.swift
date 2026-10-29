// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The follow-up of an MTP chat turn that stopped on its token budget reads every token that
/// turn generated, through the splice of the Qwen protocol (card `^g2t7caa`).
///
/// An MTP round returns its final verified token before any forward pass feeds it, thus the
/// caches of a turn that stops on its budget there miss that token, and the session keeps it as
/// an uncommitted token. The splice of the follow-up feeds it ahead of the commit that closes the
/// turn. The follow-up then writes the text of a session without speculation, and it prefills
/// that token and the new message alone.
@Suite(.serialized)
struct ChatSessionUnfedTokenSpliceTests {

    /// The tokens of the instructions.
    private static let instructions = [7, 21, 4, 33, 12, 18]

    /// The tokens of the first user message.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the follow-up user message.
    private static let followUpPrompt = [11, 40, 8, 27, 19]

    /// The tokens each turn generates. The MTP round that returns the last of them leaves it
    /// unfed on the fixture.
    private static let generatedTokenCount = 16

    /// The token that opens a think block. The template writes it after the last message, thus
    /// each turn starts inside its reasoning.
    private static let reasoningStartToken = 0

    /// The position in the greedy decode of the first message of the token that closes the
    /// reasoning, thus the turn answers after its reasoning.
    private static let reasoningEndIndex = 5

    /// The token the MTP follow-up prefills beyond what the plain follow-up prefills: the last
    /// token of the first answer, which no forward pass of the MTP stream fed.
    private static let unfedTokenCount = 1

    /// A greedy chat over the fixture's target whose template writes the Qwen transcript with
    /// the reasoning of each past turn, with the fixture's MTP drafter or without it.
    ///
    /// - Parameters:
    ///   - fixture: The fixture.
    ///   - tokenizer: The tokenizer whose markers close the turn and its reasoning.
    ///   - speculates: Whether the chat drafts with the fixture's drafter.
    /// - Returns: The chat.
    /// - Throws: The error of the MTP configuration.
    private func chat(
        _ fixture: Qwen35MTPFixture, tokenizer: QwenThinkBlockScalarTokenizer, speculates: Bool
    ) throws -> ChatSession {
        let configuration = ModelConfiguration(
            id: fixture.modelConfiguration.name, toolCallFormat: .qwen35,
            reasoningConfig: ReasoningConfig(
                startDelimiter: tokenizer.decode(
                    tokenIds: [tokenizer.open], skipSpecialTokens: false),
                endDelimiter: tokenizer.decode(
                    tokenIds: [tokenizer.close], skipSpecialTokens: false),
                promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
                replaysReasoningIntoHistory: true))
        let context = ModelContext(
            configuration: configuration, model: fixture.target,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        return ChatSession(
            context, instructions: fixture.text(for: Self.instructions),
            speculativeDecoding: speculates ? try fixture.makeSpeculativeDecodingConfig() : nil,
            generateParameters: GenerateParameters(
                maxTokens: Self.generatedTokenCount, temperature: 0))
    }

    /// The tokenizer whose end-of-turn token no turn writes, and whose close of the think block
    /// is the token at ``reasoningEndIndex`` of the greedy decode of the first message.
    ///
    /// - Parameter fixture: The fixture.
    /// - Returns: The tokenizer.
    /// - Throws: The error of the decode, or an issue when the decode writes its closing token
    ///   earlier.
    private func tokenizer(_ fixture: Qwen35MTPFixture) async throws
        -> QwenThinkBlockScalarTokenizer
    {
        let decode = try await fixture.markerFreeDecode(
            of: [Self.instructions, Self.firstPrompt], reserving: Self.followUpPrompt,
            open: Self.reasoningStartToken, maxTokens: Self.generatedTokenCount)
        let close = decode.tokens[Self.reasoningEndIndex]
        try #require(decode.tokens.firstIndex(of: close) == Self.reasoningEndIndex)
        return QwenThinkBlockScalarTokenizer(
            open: Self.reasoningStartToken, close: close, endOfTurn: decode.endOfTurn)
    }

    @Test("the follow-up of an MTP turn that stopped on its budget reads the whole answer")
    func aFollowUpReadsTheUnfedLastTokenOfTheTurn() async throws {
        let fixture = try Qwen35MTPFixture()
        let tokenizer = try await tokenizer(fixture)
        try #require(
            QwenThinkBlockScalarTokenizer.endOfTurnMarker == QwenCommittedTurnRule.endOfTurnMarker)
        let speculative = try chat(fixture, tokenizer: tokenizer, speculates: true)
        let plain = try chat(fixture, tokenizer: tokenizer, speculates: false)
        let prompt = fixture.text(for: Self.firstPrompt)

        let speculativeFirst = try await collectChatTurn(speculative.streamDetails(to: prompt))
        let plainFirst = try await collectChatTurn(plain.streamDetails(to: prompt))
        try #require(speculativeFirst.text == plainFirst.text)
        let speculativeHeld = try #require(try await speculative.cacheStatus().processedTokenCount)
        let plainHeld = try #require(try await plain.cacheStatus().processedTokenCount)
        try #require(
            speculativeHeld == plainHeld - Self.unfedTokenCount,
            "The MTP turn must leave its last token out of the caches.")

        let followUp = fixture.text(for: Self.followUpPrompt)
        let speculativeSecond = try await collectChatTurn(speculative.streamDetails(to: followUp))
        let plainSecond = try await collectChatTurn(plain.streamDetails(to: followUp))

        #expect(speculativeSecond.text == plainSecond.text)
        #expect(
            speculativeSecond.info.promptTokenCount
                == plainSecond.info.promptTokenCount + Self.unfedTokenCount)
        #expect(
            speculativeSecond.info.cachedPromptTokenCount + speculativeSecond.info.promptTokenCount
                == plainSecond.info.cachedPromptTokenCount + plainSecond.info.promptTokenCount)
        #expect((speculativeSecond.info.proposedDraftTokens ?? 0) > 0)
        // The drafter resumed its continuation instead of starting again at the cached position.
        let continuation = try #require(await speculative.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == 0)
    }
}

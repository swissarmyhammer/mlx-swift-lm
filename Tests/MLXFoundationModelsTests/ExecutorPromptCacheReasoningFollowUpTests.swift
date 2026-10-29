// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// A follow-up of a reasoning turn of the Qwen protocol reuses the whole prompt cache of that
/// turn, with the MTP drafter and without it, and resumes the drafter state of that turn.
///
/// The history render of a model whose protocol replays reasoning writes the reasoning of the
/// turn inside its think block, as `ChatSession` records it. A turn that stopped on its token
/// budget inside its reasoning leaves an empty response entry beside its reasoning entry, and
/// its follow-up extends the tokens the caches hold. A turn that closed its reasoning and then
/// answered gets its follow-up spliced at the commit that closes the turn.
@Suite(.serialized)
struct ExecutorPromptCacheReasoningFollowUpTests {

    /// The identifier of the instructions entry, which names the session of every pass.
    private static let sessionID = "executor-prompt-cache-reasoning-follow-up"

    /// The tokens of the instructions.
    private static let instructions = [7, 21, 4, 33, 12, 18]

    /// The tokens of the first prompt.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the follow-up prompt.
    private static let followUpPrompt = [11, 40, 8, 27, 19]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The token that opens a think block. The template writes it after the last message, thus
    /// each turn starts inside its reasoning.
    private static let reasoningStartToken = 0

    /// The position in the greedy decode of the first prompt of the token that closes the
    /// reasoning of a turn that answers after its reasoning.
    private static let reasoningEndIndex = 5

    /// How the first turn ends.
    private enum FirstTurn {

        /// The token budget runs out inside the reasoning: the think block never closes.
        case stopsInsideReasoning

        /// The reasoning closes, and the token budget runs out inside the answer.
        case answersAfterReasoning
    }

    /// The fixture's target, the tokenizer that renders its prompts, and the reasoning protocol
    /// of each call.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Setup {
        let fixture: Qwen35MTPFixture
        let tokenizer: ThinkBlockScalarTokenizer
        let resolver: FixtureReasoningResolver
    }

    /// What the two passes of one conversation left.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Conversation {

        /// What the first pass streamed.
        let first: MTPTurn

        /// What the follow-up pass streamed.
        let second: MTPTurn

        /// The ledger the first pass checked in.
        let firstLedger: [Int]

        /// The entry the follow-up pass checked in, or `nil` when it checked in none.
        let entry: ExecutorPromptCacheEntry?
    }

    /// The end-of-turn token, and the greedy decode of the first prompt that token closes the
    /// messages of: the first token of the vocabulary outside the prompts whose decode writes
    /// neither it nor ``reasoningStartToken``, thus no turn writes a marker.
    ///
    /// - Parameter fixture: The fixture whose target decodes.
    /// - Returns: The token, as `<|im_end|>` closes a Qwen message, and the decoded tokens.
    /// - Throws: The error of the decode, or an issue when every decode writes a marker.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func endOfTurnAndDecode(of fixture: Qwen35MTPFixture) async throws
        -> (endOfTurn: Int, tokens: [Int])
    {
        let prompts = Self.instructions + Self.firstPrompt + Self.followUpPrompt
        for endOfTurn in 0 ..< Qwen35MTPFixture.vocabularySize
        where endOfTurn != Self.reasoningStartToken && !prompts.contains(endOfTurn) {
            let decode = try await fixture.decodeGreedily(
                prompt: Self.instructions + [endOfTurn] + Self.firstPrompt
                    + [endOfTurn, Self.reasoningStartToken],
                maxTokens: Self.generatedTokenCount, speculative: false)
            if !decode.tokens.contains(endOfTurn),
                !decode.tokens.contains(Self.reasoningStartToken)
            {
                return (endOfTurn, decode.tokens)
            }
        }
        throw NoMarkerFreeDecode()
    }

    /// The error of a fixture whose every decode writes a marker token.
    private struct NoMarkerFreeDecode: Error {}

    /// The setup whose think block closes where `firstTurn` asks, in the greedy decode of the
    /// first prompt.
    ///
    /// - Parameter firstTurn: How the first turn ends.
    /// - Returns: The setup.
    /// - Throws: The error of the fixture, or an issue when the decode writes a marker token or
    ///   has no token to close the reasoning with.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeSetup(_ firstTurn: FirstTurn) async throws -> Setup {
        let fixture = try Qwen35MTPFixture()
        let decode = try await endOfTurnAndDecode(of: fixture)
        let markers = [Self.reasoningStartToken, decode.endOfTurn]
        let endToken: Int
        switch firstTurn {
        case .stopsInsideReasoning:
            endToken = try #require(
                (0 ..< Qwen35MTPFixture.vocabularySize).first {
                    !markers.contains($0) && !decode.tokens.contains($0)
                })
        case .answersAfterReasoning:
            endToken = decode.tokens[Self.reasoningEndIndex]
            try #require(decode.tokens.firstIndex(of: endToken) == Self.reasoningEndIndex)
        }
        let tokenizer = ThinkBlockScalarTokenizer(
            open: Self.reasoningStartToken, close: endToken, endOfTurn: decode.endOfTurn)
        return Setup(
            fixture: fixture, tokenizer: tokenizer,
            resolver: FixtureReasoningResolver(
                start: Self.reasoningStartToken, end: endToken, tokenizer: tokenizer,
                replaysReasoningIntoHistory: true))
    }

    /// The transcript of the instructions and the first prompt, and, after `first`, the
    /// entries the framework appends for that turn, and the follow-up prompt.
    ///
    /// - Parameters:
    ///   - fixture: The fixture whose tokens the prompts spell.
    ///   - first: What the first pass streamed, or `nil` for the transcript of the first pass.
    /// - Returns: The transcript.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func transcript(_ fixture: Qwen35MTPFixture, after first: MTPTurn?) -> Transcript {
        var entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    id: Self.sessionID,
                    segments: Self.textSegments(fixture.text(for: Self.instructions)),
                    toolDefinitions: [])),
            .prompt(
                Transcript.Prompt(segments: Self.textSegments(fixture.text(for: Self.firstPrompt)))),
        ]
        if let first {
            entries += [
                // The response entry opens with the turn, and the reasoning entry opens when
                // the first reasoning fragment arrives.
                .response(
                    Transcript.Response(
                        assetIDs: [],
                        segments: first.responseText.isEmpty
                            ? [] : Self.textSegments(first.responseText))),
                .reasoning(Transcript.Reasoning(segments: Self.textSegments(first.reasoningText))),
                .prompt(
                    Transcript.Prompt(
                        segments: Self.textSegments(fixture.text(for: Self.followUpPrompt)))),
            ]
        }
        return Transcript(entries: entries)
    }

    /// One text segment of `text`.
    ///
    /// - Parameter text: The text of the segment.
    /// - Returns: The segments.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func textSegments(_ text: String) -> [Transcript.Segment] {
        [.text(Transcript.TextSegment(content: text))]
    }

    /// Runs the first pass and its follow-up of the reasoning model of `setup` in a store of
    /// its own.
    ///
    /// - Parameters:
    ///   - setup: The setup.
    ///   - speculates: Whether the model drafts with the fixture's drafter.
    /// - Returns: What the passes left.
    /// - Throws: The error of a pass, or an issue when the first pass checked in nothing.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func converse(_ setup: Setup, speculates: Bool) async throws -> Conversation {
        let weights = try makeScriptedWeightsDirectory()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ExecutorPromptCacheReasoningFollowUpTests-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: weights)
            try? FileManager.default.removeItem(at: folder)
        }
        let model = setup.fixture.makeLanguageModel(
            weights: weights, capabilities: [.reasoning], resolver: setup.resolver,
            tokenizer: setup.tokenizer, toolCallFormat: .qwen35,
            speculativeDecoding: speculates ? try Qwen35MTPFixture.makeSpeculativeDecoding() : nil)
        let store = ExecutorPromptCacheStore(directory: folder, writer: { _, _ in })
        let key = ExecutorPromptCacheKey(modelID: model.modelID, sessionID: Self.sessionID)
        let options = GenerationOptions(
            samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount)

        let first = try await MTPExecutorTurns.respond(
            to: makeExecutorRequest(
                transcript: transcript(setup.fixture, after: nil), generationOptions: options),
            model: model, inside: store)
        let firstEntry = try #require(await store.peek(key), "The first pass must check in.")
        let second = try await MTPExecutorTurns.respond(
            to: makeExecutorRequest(
                transcript: transcript(setup.fixture, after: first), generationOptions: options),
            model: model, inside: store)
        return Conversation(
            first: first, second: second, firstLedger: firstEntry.tokens,
            entry: await store.peek(key))
    }

    /// The drafter state the follow-up of `conversation` checked in, which stands where the
    /// main caches of its entry end.
    ///
    /// - Parameter conversation: The conversation.
    /// - Returns: The drafter state.
    /// - Throws: An issue when the entry or its drafter state is missing.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func drafterState(of conversation: Conversation) throws -> MTPDrafterContinuation {
        let entry = try #require(conversation.entry, "The follow-up pass must check in.")
        let drafterState = try #require(entry.drafterState, "The entry must keep a drafter state.")
        #expect(
            drafterState.isAligned(
                withMainTokenCount: entry.tokens.count, drafterID: drafterState.drafterID))
        return drafterState
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [false, true])
    func aFollowUpOfATurnThatStoppedInsideItsReasoningExtendsItsLedger(speculates: Bool)
        async throws
    {
        let conversation = try await converse(
            makeSetup(.stopsInsideReasoning), speculates: speculates)

        try #require(conversation.first.responseText.isEmpty)
        try #require(!conversation.first.reasoningText.isEmpty)
        let usage = try #require(conversation.second.usage.first)
        #expect(usage.cachedPromptTokenCount == conversation.firstLedger.count)
        #expect(conversation.second.speculated == speculates)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpOfATurnThatStoppedInsideItsReasoningResumesTheDrafterState() async throws {
        let conversation = try await converse(
            makeSetup(.stopsInsideReasoning), speculates: true)

        // A drafter state that the follow-up resumed still begins where the first turn
        // started it.
        #expect(try drafterState(of: conversation).state.cacheStartPosition == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [false, true])
    func aFollowUpOfATurnThatAnsweredReusesItsWholeLedger(speculates: Bool) async throws {
        let conversation = try await converse(
            makeSetup(.answersAfterReasoning), speculates: speculates)

        try #require(!conversation.first.responseText.isEmpty)
        try #require(!conversation.first.reasoningText.isEmpty)
        let usage = try #require(conversation.second.usage.first)
        #expect(usage.cachedPromptTokenCount >= conversation.firstLedger.count)
        #expect(conversation.second.speculated == speculates)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpOfATurnThatAnsweredResumesTheDrafterState() async throws {
        let conversation = try await converse(
            makeSetup(.answersAfterReasoning), speculates: true)

        // The splice feeds the follow-up onto the caches where the first turn left them, thus
        // the drafter state that stands there still begins where the first turn started it.
        #expect(try drafterState(of: conversation).state.cacheStartPosition == 0)
    }
}

/// A ``ScalarTokenizer`` whose chat template closes each message with an end-of-turn token,
/// writes the reasoning of a past assistant turn inside a think block before its content, and
/// opens a think block after the last message, as the Qwen 3.8 template does with
/// `<|im_end|>`, `reasoning_content` and `<think>`. It names the three markers as the Qwen
/// protocol does, thus the cache-reuse rule of that protocol finds them.
private struct ThinkBlockScalarTokenizer: Tokenizer {

    /// The token that opens a think block.
    let open: Int

    /// The token that closes a think block.
    let close: Int

    /// The token that closes each message.
    let endOfTurn: Int

    /// The tokenizer that encodes, decodes and spells the tokens.
    private let base = ScalarTokenizer(vocabularySize: Qwen35MTPFixture.vocabularySize)

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    /// The marker token `token` names in the Qwen protocol, or the token of `base`.
    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case QwenCommittedTurnRule.endOfTurnMarker: endOfTurn
        case QwenCommittedTurnRule.openThinkMarker: open
        case QwenCommittedTurnRule.closeThinkMarker: close
        default: base.convertTokenToId(token)
        }
    }

    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    /// The tokens of each message, then the open think block of the generation prompt.
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap(render) + [open]
    }

    /// The tokens of `message`, then the end-of-turn token: its content, after its reasoning
    /// inside a think block when it is an assistant message.
    ///
    /// - Parameter message: The message dictionary the message generator made.
    /// - Returns: The tokens.
    private func render(_ message: [String: any Sendable]) -> [Int] {
        let content = tokens(of: message, key: "content") + [endOfTurn]
        guard message["role"] as? String == Chat.Message.Role.assistant.rawValue else {
            return content
        }
        return [open] + tokens(of: message, key: "reasoning_content") + [close] + content
    }

    /// The tokens of the text `message` holds under `key`, or none when it holds no text there.
    ///
    /// - Parameters:
    ///   - message: The message dictionary.
    ///   - key: The key of the text.
    /// - Returns: The tokens.
    private func tokens(of message: [String: any Sendable], key: String) -> [Int] {
        encode(text: message[key] as? String ?? "", addSpecialTokens: false)
    }
}

#endif

// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// A first pass and its follow-up on the fixture's target with the reasoning protocol of Qwen,
/// with the MTP drafter or without it, in a prompt cache store of its own.
///
/// The history render of a model whose protocol replays reasoning writes the reasoning of the
/// turn inside its think block, as `ChatSession` records it. The suites of the executor's prompt
/// cache after a reasoning turn share this conversation.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ExecutorReasoningConversation {

    /// The identifier of the instructions entry, which names the session of every pass.
    static let sessionID = "executor-prompt-cache-reasoning-follow-up"

    /// The token IDs the prompts spell. Each is an arbitrary word of the fixture's vocabulary
    /// with no role beyond being distinct, thus each is named after its value. No two prompts
    /// share one.
    private enum PromptToken {
        static let two = 2
        static let three = 3
        static let four = 4
        static let five = 5
        static let six = 6
        static let seven = 7
        static let eight = 8
        static let nine = 9
        static let eleven = 11
        static let twelve = 12
        static let fourteen = 14
        static let fifteen = 15
        static let eighteen = 18
        static let nineteen = 19
        static let twentyOne = 21
        static let twentySeven = 27
        static let thirtyThree = 33
        static let thirtyFive = 35
        static let forty = 40
    }

    /// The tokens of the instructions.
    static let instructions = [
        PromptToken.seven, PromptToken.twentyOne, PromptToken.four, PromptToken.thirtyThree,
        PromptToken.twelve, PromptToken.eighteen,
    ]

    /// The tokens of the first prompt.
    static let firstPrompt = [
        PromptToken.three, PromptToken.fourteen, PromptToken.fifteen, PromptToken.nine,
        PromptToken.two, PromptToken.six, PromptToken.five, PromptToken.thirtyFive,
    ]

    /// The tokens of the follow-up prompt.
    static let followUpPrompt = [
        PromptToken.eleven, PromptToken.forty, PromptToken.eight, PromptToken.twentySeven,
        PromptToken.nineteen,
    ]

    /// The token that opens a think block. The template writes it after the last message, thus
    /// each turn starts inside its reasoning.
    static let reasoningStartToken = 0

    /// The position in the greedy decode of the first prompt of the token that closes the
    /// reasoning of a turn that answers after its reasoning.
    static let reasoningEndIndex = 5

    /// How the first turn ends.
    enum FirstTurn {

        /// The token budget runs out inside the reasoning: the think block never closes.
        case stopsInsideReasoning

        /// The reasoning closes, and the token budget runs out inside the answer.
        case answersAfterReasoning
    }

    /// The fixture's target, the tokenizer that renders its prompts, the reasoning protocol of
    /// each call, the token budget of each pass, and the plain greedy decode of the first pass.
    struct Setup {
        let fixture: Qwen35MTPFixture
        let tokenizer: QwenThinkBlockScalarTokenizer
        let resolver: FixtureReasoningResolver
        let generatedTokenCount: Int
        let firstDecode: [Int]
    }

    /// What the two passes of one conversation left.
    struct Passes {

        /// What the first pass streamed.
        let first: MTPTurn

        /// What the follow-up pass streamed.
        let second: MTPTurn

        /// The entry the first pass checked in.
        let firstEntry: ExecutorPromptCacheEntry

        /// The entry the follow-up pass checked in, or `nil` when it checked in none.
        let entry: ExecutorPromptCacheEntry?

        /// The ledger the first pass checked in.
        var firstLedger: [Int] { firstEntry.tokens }

        /// The drafter state the follow-up checked in, which stands where the main caches of its
        /// entry end.
        ///
        /// - Returns: The drafter state.
        /// - Throws: An issue when the entry or its drafter state is missing.
        func drafterState() throws -> MTPDrafterContinuation {
            let entry = try #require(entry, "The follow-up pass must check in.")
            let drafterState = try #require(
                entry.drafterState, "The entry must keep a drafter state.")
            #expect(
                drafterState.isAligned(
                    withMainTokenCount: entry.tokens.count, drafterID: drafterState.drafterID))
            return drafterState
        }
    }

    /// The setup whose think block closes where `firstTurn` asks, in the greedy decode of the
    /// first prompt.
    ///
    /// - Parameters:
    ///   - firstTurn: How the first turn ends.
    ///   - generatedTokenCount: The tokens each pass generates.
    /// - Returns: The setup.
    /// - Throws: The error of the fixture, or an issue when the decode writes a marker token or
    ///   has no token to close the reasoning with.
    static func makeSetup(_ firstTurn: FirstTurn, generatedTokenCount: Int) async throws -> Setup {
        let fixture = try Qwen35MTPFixture()
        let decode = try await fixture.markerFreeDecode(
            of: [instructions, firstPrompt], reserving: followUpPrompt, open: reasoningStartToken,
            maxTokens: generatedTokenCount)
        let markers = [reasoningStartToken, decode.endOfTurn]
        let endToken: Int
        switch firstTurn {
        case .stopsInsideReasoning:
            endToken = try #require(
                (0 ..< Qwen35MTPFixture.vocabularySize).first {
                    !markers.contains($0) && !decode.tokens.contains($0)
                })
        case .answersAfterReasoning:
            endToken = decode.tokens[reasoningEndIndex]
            try #require(decode.tokens.firstIndex(of: endToken) == reasoningEndIndex)
        }
        let tokenizer = QwenThinkBlockScalarTokenizer(
            open: reasoningStartToken, close: endToken, endOfTurn: decode.endOfTurn)
        return Setup(
            fixture: fixture, tokenizer: tokenizer,
            resolver: FixtureReasoningResolver(
                start: reasoningStartToken, end: endToken, tokenizer: tokenizer,
                replaysReasoningIntoHistory: true),
            generatedTokenCount: generatedTokenCount, firstDecode: decode.tokens)
    }

    /// The transcript of the instructions and the first prompt, and, after `first`, the
    /// entries the framework appends for that turn, and the follow-up prompt.
    ///
    /// - Parameters:
    ///   - fixture: The fixture whose tokens the prompts spell.
    ///   - first: What the first pass streamed, or `nil` for the transcript of the first pass.
    /// - Returns: The transcript.
    static func transcript(_ fixture: Qwen35MTPFixture, after first: MTPTurn?) -> Transcript {
        var entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    id: sessionID, segments: textSegments(fixture.text(for: instructions)),
                    toolDefinitions: [])),
            .prompt(Transcript.Prompt(segments: textSegments(fixture.text(for: firstPrompt)))),
        ]
        if let first {
            entries += [
                // The response entry opens with the turn, and the reasoning entry opens when
                // the first reasoning fragment arrives.
                .response(
                    Transcript.Response(
                        assetIDs: [],
                        segments: first.responseText.isEmpty
                            ? [] : textSegments(first.responseText))),
                .reasoning(Transcript.Reasoning(segments: textSegments(first.reasoningText))),
                .prompt(
                    Transcript.Prompt(segments: textSegments(fixture.text(for: followUpPrompt)))),
            ]
        }
        return Transcript(entries: entries)
    }

    /// One text segment of `text`.
    ///
    /// - Parameter text: The text of the segment.
    /// - Returns: The segments.
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
    static func converse(_ setup: Setup, speculates: Bool) async throws -> Passes {
        let weights = try makeScriptedWeightsDirectory()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ExecutorReasoningConversation-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: weights)
            try? FileManager.default.removeItem(at: folder)
        }
        let model = setup.fixture.makeLanguageModel(
            weights: weights, capabilities: [.reasoning], resolver: setup.resolver,
            tokenizer: setup.tokenizer, toolCallFormat: .qwen35,
            speculativeDecoding: speculates ? try Qwen35MTPFixture.makeSpeculativeDecoding() : nil)
        let store = ExecutorPromptCacheStore(directory: folder, writer: { _, _ in })
        let key = ExecutorPromptCacheKey(modelID: model.modelID, sessionID: sessionID)
        let options = GenerationOptions(
            samplingMode: .greedy, maximumResponseTokens: setup.generatedTokenCount)

        let first = try await MTPExecutorTurns.respond(
            to: makeExecutorRequest(
                transcript: transcript(setup.fixture, after: nil), generationOptions: options),
            model: model, inside: store)
        let firstEntry = try #require(await store.peek(key), "The first pass must check in.")
        let second = try await MTPExecutorTurns.respond(
            to: makeExecutorRequest(
                transcript: transcript(setup.fixture, after: first), generationOptions: options),
            model: model, inside: store)
        return Passes(
            first: first, second: second, firstEntry: firstEntry, entry: await store.peek(key))
    }
}

#endif

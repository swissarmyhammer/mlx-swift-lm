// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Synchronization
import Testing

@testable import MLXFoundationModels

/// The usage counts one usage event of the executor reported.
struct MTPTurnUsage: Sendable, Equatable {

    /// Every prompt token of the pass: the cached part and the fed part.
    let promptTokenCount: Int

    /// The prompt tokens the cache served.
    let cachedPromptTokenCount: Int

    /// Every token the pass generated.
    let generatedTokenCount: Int

    /// The generated tokens inside the reasoning.
    let reasoningTokenCount: Int
}

/// What the executor streamed for one session turn.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct MTPTurn: Sendable {

    /// The response text, in the order the turn streamed it.
    var responseText = ""

    /// The reasoning text, in the order the turn streamed it.
    var reasoningText = ""

    /// Each usage event, in order.
    var usage: [MTPTurnUsage] = []

    /// The completion information of each generation pass, in order.
    var completions: [GenerateCompletionInfo] = []

    /// The keys of each metadata update, in order.
    var metadataKeys: [String] = []

    /// The metadata of the response entry: each update merged over the ones before it.
    var metadata: [String: any ConvertibleToGeneratedContent & Sendable] = [:]

    /// The keys of the MTP outcome of a call, which only a model with a drafter writes.
    static let speculativeDecodingOutcomeKeys: Set = [
        SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding,
        SpeculativeDecodingOutcome.MetadataKey.fallbackReason,
        SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount,
        SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount,
    ]

    /// The keys of each metadata update other than the MTP outcome keys, in order: the keys a
    /// model writes whether it speculates or not.
    var outputMetadataKeys: [String] {
        metadataKeys.filter { !Self.speculativeDecodingOutcomeKeys.contains($0) }
    }

    /// The name and the arguments of each tool call, in order.
    var toolCalls: [String] = []

    /// The description of the error the turn threw, or `nil` when it returned.
    var failure: String?

    /// The description of the error the consumer of the executor's channel met while it
    /// drained the channel, or `nil` when it drained cleanly.
    var channelFailure: String?

    /// Whether some pass of the turn speculated: it proposed drafts and did not fall back to
    /// plain decode.
    var speculated: Bool {
        completions.contains { ($0.proposedDraftTokens ?? 0) > 0 && $0.passthroughReason == nil }
    }
}

/// Records the events the executor streams while a test runs one turn.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class MTPTurnLog: Sendable {
    private let turn = Mutex(MTPTurn())

    /// What the events so far give.
    var recorded: MTPTurn { turn.withLock { $0 } }

    /// The observer that records each event into this log.
    var observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void {
        { self.record($0) }
    }

    /// Adds `event` to the turn.
    ///
    /// - Parameter event: One event the executor streamed.
    func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
        turn.withLock { turn in
            switch event {
            case .appendText(let text, _, .response): turn.responseText += text
            case .appendText(let text, _, .reasoning): turn.reasoningText += text
            case .updateUsage(let input, let output, _):
                turn.usage.append(
                    MTPTurnUsage(
                        promptTokenCount: input.totalTokenCount,
                        cachedPromptTokenCount: input.cachedTokenCount,
                        generatedTokenCount: output.totalTokenCount,
                        reasoningTokenCount: output.reasoningTokenCount))
            case .completion(let info): turn.completions.append(info)
            case .updateMetadata(let values, _):
                turn.metadataKeys += values.keys.sorted()
                turn.metadata.merge(values) { _, update in update }
            case .toolCall(_, let name, let arguments): turn.toolCalls.append(name + arguments)
            }
        }
    }
}

/// Runs session turns on the fixture models of the MTP executor tests.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum MTPExecutorTurns {

    /// Whether a pass is cancelled at an event it streamed.
    typealias Cancellation = @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Bool

    /// Runs one turn of `session` and records what the executor streamed.
    ///
    /// - Parameters:
    ///   - session: The session.
    ///   - prompt: The prompt of the turn.
    ///   - options: The generation options of the turn.
    /// - Returns: What the executor streamed.
    /// - Throws: The error of the turn.
    static func respond(
        _ session: LanguageModelSession, to prompt: Prompt, options: GenerationOptions
    ) async throws -> MTPTurn {
        let log = MTPTurnLog()
        try await MLXLanguageModel.Executor.$generationObserver.withValue(log.observer) {
            _ = try await session.respond(to: prompt, options: options)
        }
        return log.recorded
    }

    /// Runs one turn of a new session of `model` for each of the two models, the plain model
    /// first.
    ///
    /// - Parameters:
    ///   - plain: The model that decodes plainly.
    ///   - speculative: The model that speculates.
    ///   - prompt: The prompt of the turn.
    ///   - options: The generation options of the turn.
    ///   - tools: The tools of each session.
    /// - Returns: What each turn streamed.
    /// - Throws: The error of a turn.
    static func respond(
        plain: MLXLanguageModel, speculative: MLXLanguageModel, to prompt: Prompt,
        options: GenerationOptions, tools: [any FoundationModels.Tool] = []
    ) async throws -> (plain: MTPTurn, speculative: MTPTurn) {
        let plainTurn = try await respond(
            LanguageModelSession(model: plain, tools: tools, instructions: nil), to: prompt,
            options: options)
        let speculativeTurn = try await respond(
            LanguageModelSession(model: speculative, tools: tools, instructions: nil),
            to: prompt, options: options)
        return (plainTurn, speculativeTurn)
    }

    /// Runs one executor pass of `model` for `request` and records what it streamed, with the
    /// error it threw.
    ///
    /// The pass bypasses the session, thus a turn that a session would continue -- a tool
    /// call -- or a request a session cannot build -- an image prompt -- runs one pass.
    ///
    /// The pass runs on a task of its own, which `cancels` may cancel: the observer cancels
    /// that task right after it recorded the first event `cancels` accepts, from inside the
    /// executor, at that point of the pass. The task of the caller stays uncancelled.
    ///
    /// - Parameters:
    ///   - request: The request of the pass.
    ///   - model: The model of the pass.
    ///   - cancels: Whether the pass is cancelled at an event it streamed; by default no event
    ///     cancels it.
    /// - Returns: What the pass streamed, the error it threw in ``MTPTurn/failure``, and the
    ///   error the channel's consumer met in ``MTPTurn/channelFailure``.
    /// - Throws: The error of the executor's initializer.
    static func respond(
        to request: LanguageModelExecutorGenerationRequest, model: MLXLanguageModel,
        cancellingAt cancels: @escaping Cancellation = { _ in false }
    ) async throws -> MTPTurn {
        let executor = try makeMLXExecutor(for: model)
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside the executor or every
        // send parks it.
        let consumer = Task<Void, any Error> {
            for try await _ in channel {}
        }
        let log = MTPTurnLog()
        let observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void = { event in
            log.record(event)
            if cancels(event) {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        // An unstructured task inherits the task-local values of the caller: the prompt cache
        // store and the guided diagnostic sink a test bound.
        let pass = Task {
            try await MLXLanguageModel.Executor.$generationObserver.withValue(observer) {
                try await executor.respond(to: request, model: model, streamingInto: channel)
            }
        }
        let failure: String?
        do {
            try await pass.value
            failure = nil
        } catch {
            failure = String(describing: error)
        }
        consumer.cancel()
        var turn = log.recorded
        turn.failure = failure
        turn.channelFailure = await channelFailure(of: consumer)
        return turn
    }

    /// Runs one executor pass of `model` for `request` with `store` as its prompt cache store,
    /// and requires that the pass and its channel succeeded.
    ///
    /// - Parameters:
    ///   - request: The request of the pass.
    ///   - model: The model of the pass.
    ///   - store: The prompt cache store the pass binds.
    /// - Returns: What the pass streamed.
    /// - Throws: The error of the executor's initializer, or an issue when the pass or its
    ///   channel failed.
    static func respond(
        to request: LanguageModelExecutorGenerationRequest, model: MLXLanguageModel,
        inside store: ExecutorPromptCacheStore
    ) async throws -> MTPTurn {
        let turn = try await ExecutorPromptCacheStore.$current.withValue(store) {
            try await respond(to: request, model: model)
        }
        try #require(turn.failure == nil, "The pass failed: \(turn.failure ?? "")")
        try #require(turn.channelFailure == nil, "The channel failed: \(turn.channelFailure ?? "")")
        return turn
    }

    /// The error a cancelled channel consumer met, other than the `CancellationError` that
    /// stopping it gives.
    ///
    /// - Parameter consumer: The consumer, already cancelled.
    /// - Returns: The description of the error, or `nil` when the consumer ended cleanly.
    private static func channelFailure(of consumer: Task<Void, any Error>) async -> String? {
        do {
            try await consumer.value
            return nil
        } catch is CancellationError {
            return nil
        } catch {
            return String(describing: error)
        }
    }
}

/// A resolver that gives each call a reasoning protocol whose delimiters are tokens of the
/// fixture's vocabulary, toggled by a template flag.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct FixtureReasoningResolver: ModelConfigurationResolver {

    /// The reasoning protocol each call gets.
    let reasoningConfig: ReasoningConfig

    /// Makes a resolver whose delimiters are the texts of two fixture tokens.
    ///
    /// - Parameters:
    ///   - start: The token that opens the reasoning.
    ///   - end: The token that closes the reasoning.
    ///   - tokenizer: The tokenizer that spells the tokens.
    ///   - replaysReasoningIntoHistory: Whether the reasoning of a past turn goes back into its
    ///     history render, as ``ReasoningConfig/replaysReasoningIntoHistory`` states.
    init(
        start: Int, end: Int, tokenizer: any Tokenizer, replaysReasoningIntoHistory: Bool = false
    ) {
        reasoningConfig = ReasoningConfig(
            startDelimiter: tokenizer.decode(tokenIds: [start], skipSpecialTokens: false),
            endDelimiter: tokenizer.decode(tokenIds: [end], skipSpecialTokens: false),
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
            replaysReasoningIntoHistory: replaysReasoningIntoHistory)
    }

    /// Makes a resolver that gives each call `reasoningConfig`.
    ///
    /// - Parameter reasoningConfig: The reasoning protocol each call gets.
    init(reasoningConfig: ReasoningConfig) {
        self.reasoningConfig = reasoningConfig
    }

    /// A resolver whose reasoning opens with `start` and closes with the token at `endIndex` of
    /// the greedy decode of `prompt` on the fixture's target, so a turn on that prompt reads the
    /// tokens before `endIndex` as reasoning.
    ///
    /// - Parameters:
    ///   - endIndex: The position in the greedy decode of the token that closes the reasoning.
    ///   - prompt: The prompt tokens.
    ///   - maxTokens: The tokens the greedy decode generates.
    ///   - start: The token that opens the reasoning; the decode must not hold it.
    ///   - tokenizer: The tokenizer that spells the delimiters.
    /// - Returns: The resolver.
    /// - Throws: An issue when the closing token stands earlier in the decode, or the decode
    ///   holds `start`; the error of the fixture.
    static func closingAtGreedyToken(
        _ endIndex: Int, of prompt: [Int], maxTokens: Int, start: Int, tokenizer: any Tokenizer
    ) async throws -> FixtureReasoningResolver {
        let decode = try await Qwen35MTPFixture().decodeGreedily(
            prompt: prompt, maxTokens: maxTokens, speculative: false)
        let endToken = decode.tokens[endIndex]
        try #require(decode.tokens.firstIndex(of: endToken) == endIndex)
        try #require(!decode.tokens.contains(start))
        return FixtureReasoningResolver(start: start, end: endToken, tokenizer: tokenizer)
    }

    func resolve(
        _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
    ) -> ModelConfiguration {
        var resolved = configuration
        resolved.reasoningConfig = reasoningConfig
        return resolved
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct FixtureLookupArguments {
    @Guide(description: "The text to look up.")
    var query: String
}

/// A tool the fixture model can see but never calls: its random weights write no tool call.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct FixtureLookupTool: FoundationModels.Tool {
    let name = "lookup"
    let description = "Looks up text."

    func call(arguments: FixtureLookupArguments) async throws -> String {
        arguments.query
    }
}

/// A tokenizer whose tokens are single characters, among them each character of a JSON tool
/// call, so that the grammar pass of a required tool turn on the fixture's target can write a
/// tool call. It has one token for each token of ``Qwen35MTPFixture/vocabularySize``.
struct ToolCallAlphabetTokenizer: Tokenizer {

    /// The character of each token, in token order.
    static let alphabet = Array(
        "{}[]\":, abcdefghijklmnopqrstuvwxyz0123456789<>/_-.\nABCDEFGHIJKLM")

    var bosToken: String? { nil }

    /// The last character: a grammar takes the end-of-sequence token for the end of its
    /// output, thus that token is none of the characters a tool call needs.
    var eosToken: String? { Self.alphabet.last.map(String.init) }

    var unknownToken: String? { nil }

    /// The token of each character of `text`; a character outside the alphabet takes the token
    /// of its scalar value modulo the alphabet size.
    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.map { character in
            Self.alphabet.firstIndex(of: character)
                ?? Int(character.unicodeScalars.first?.value ?? 0) % Self.alphabet.count
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(tokenIds.compactMap(convertIdToToken).joined())
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.count == 1 ? token.first.flatMap(Self.alphabet.firstIndex(of:)) : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        Self.alphabet.indices.contains(id) ? String(Self.alphabet[id]) : nil
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
    }
}

/// A tokenizer that decodes as a ``ScalarTokenizer`` does, slowly, and counts each decode.
///
/// The text handler of a generation decodes once for each token it handles, thus the count
/// shows whether a generation still handles tokens, and the delay keeps a generation running
/// long enough for a test to cancel it mid-decode.
///
/// The counter is a class shared by every copy of the tokenizer, and a `Mutex` holds its count.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class DecodeCounter: Sendable {
    private let count = Mutex(0)

    /// Counts one decode.
    func record() {
        count.withLock { $0 += 1 }
    }

    /// The decodes so far.
    var decodeCount: Int {
        count.withLock { $0 }
    }
}

/// See ``DecodeCounter``.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct SlowCountingTokenizer: Tokenizer {

    /// The tokenizer the decode copies.
    let base: ScalarTokenizer

    /// Counts each decode.
    let counter: DecodeCounter

    /// How long each decode sleeps.
    let decodeDelay: TimeInterval

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        counter.record()
        Thread.sleep(forTimeInterval: decodeDelay)
        return base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }

    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

#endif

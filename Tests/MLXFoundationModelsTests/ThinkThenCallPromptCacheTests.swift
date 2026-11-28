// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXGuidedGeneration
import MLXLMCommon
import MLXLMTestSupport
import Synchronization
import Testing

@testable import MLXFoundationModels

/// A required tool turn that thinks before it calls reads the turn before it from the prompt
/// cache in its reasoning phase, its grammar phase reads what the reasoning phase left, and the
/// turn writes what the same turn writes with no cache.
///
/// Each test runs a plain turn, then a required tool turn over the plain turn's response whose
/// prompt ends with the opening of the reasoning, thus the reasoning phase starts inside its
/// reasoning. The target is a hybrid model, the tiny Inkling 2 target whose sliding-window and
/// convolution caches cannot rewind, or a Qwen 3.5 target whose layers are all full attention,
/// in this process or behind a `ClusterLanguageModel` over an in-process worker. The tokenizer
/// is ``ToolCallAlphabetTokenizer``, thus the grammar phase can write a tool call, and its
/// render has no generation prompt.
///
/// The random weights close no reasoning on their own. A probe turn, with no cache, reads the
/// reasoning the target writes for the tool turn, and the reasoning then closes at a token of
/// that reasoning that stands there first.
@Suite(.serialized)
struct ThinkThenCallPromptCacheTests {

    /// Where the target runs.
    enum Placement: CaseIterable, Sendable {
        case local
        case clustered
    }

    /// The model each turn runs.
    enum Target: CaseIterable, Sendable {

        /// The tiny Inkling 2 target, whose caches cannot rewind.
        case hybrid

        /// A Qwen 3.5 target whose layers are all full attention, whose caches all rewind.
        case attentionOnly

        /// Makes the target.
        ///
        /// - Returns: The target.
        /// - Throws: The error of the target.
        func make() throws -> any MLXLMCommon.LanguageModel {
            switch self {
            case .hybrid:
                try Inkling2TinyModels.target(
                    json: Inkling2TinyModels.draftJSON(
                        vocabularySize: Qwen35MTPFixture.vocabularySize,
                        unpaddedVocabularySize: Qwen35MTPFixture.vocabularySize))
            case .attentionOnly:
                try Qwen35AttentionOnlyTarget.make()
            }
        }
    }

    /// The tokens of the instructions: more than the window of eight of the sliding layers.
    private static let instructions = [7, 21, 4, 33, 12, 18, 25, 30, 9, 41, 16, 22]

    /// The tokens of the prompt of the plain turn.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the prompt of the tool turn, before the opening of its reasoning.
    private static let secondPrompt = [11, 40, 8, 27, 19]

    /// The text that opens the reasoning. The prompt of the tool turn ends with it.
    private static let reasoningStart = "<think>"

    /// The text that closes the reasoning of the probe turn, which the random weights never
    /// write, thus the probe reasons to the end of its budget.
    private static let probeReasoningEnd = "</think>"

    /// The fewest reasoning tokens before the token that closes the reasoning.
    private static let minimumReasoningTokenCount = 2

    /// The tokens each turn may generate: room for the reasoning and the whole tool call.
    private static let responseTokenBudget = 64

    /// The session of the plain turn and of the tool turn after it.
    private static let warmSession = "think-then-call-warm"

    /// The session of the probe turn.
    private static let probeSession = "think-then-call-probe"

    /// The session of the tool turn with no cache.
    private static let coldSession = "think-then-call-cold"

    /// A resolver whose reasoning opens with ``reasoningStart`` behind a template flag, and
    /// closes with a text a test sets once the probe turn read the reasoning.
    ///
    /// A `Mutex` holds the closing text: the executor resolves on its own task.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class ClosingReasoningResolver: ModelConfigurationResolver {

        /// The text that closes the reasoning.
        private let reasoningEnd = Mutex(ThinkThenCallPromptCacheTests.probeReasoningEnd)

        /// Closes the reasoning of each later turn with `text`.
        ///
        /// - Parameter text: The text that closes the reasoning.
        func close(with text: String) {
            reasoningEnd.withLock { $0 = text }
        }

        func resolve(
            _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
        ) -> ModelConfiguration {
            var resolved = configuration
            resolved.reasoningConfig = ReasoningConfig(
                startDelimiter: ThinkThenCallPromptCacheTests.reasoningStart,
                endDelimiter: reasoningEnd.withLock { $0 },
                promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
            return resolved
        }
    }

    /// A model of one target at one placement, its resolver, and a store of its own.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Setup {
        let model: MLXLanguageModel
        let store: ExecutorPromptCacheStore
        let resolver = ClosingReasoningResolver()
        let tokenizer = ToolCallAlphabetTokenizer()

        /// The folders the setup made, which ``remove()`` deletes.
        private let folders: [URL]

        /// Makes the model of `target` at `placement`, and an empty store.
        ///
        /// - Parameters:
        ///   - placement: Where the target runs.
        ///   - target: The model each turn runs.
        /// - Throws: The error of the fixture, of the target, or of the worker handshake.
        init(_ placement: Placement, _ target: Target) throws {
            let weights = try makeScriptedWeightsDirectory()
            let folder = PromptCacheTestStore.folder(named: "ThinkThenCallPromptCacheTests")
            folders = [weights, folder]
            let local = try target.make()
            let served: any MLXLMCommon.LanguageModel =
                switch placement {
                case .local: local
                case .clustered:
                    try ClusterLanguageModel.connect(
                        over: InProcessRemoteModelWorker(
                            model: local, vocabularySize: Qwen35MTPFixture.vocabularySize))
                }
            model = try Qwen35MTPFixture().makeLanguageModel(
                weights: weights, target: served,
                capabilities: [.reasoning, .toolCalling, .guidedGeneration],
                resolver: resolver, tokenizer: tokenizer)
            store = ExecutorPromptCacheStore(
                directory: folder,
                capacityReader: PromptCacheSpoolFixtureVolume.capacity(ofVolumeOf:))
        }

        /// The text of `tokens` as the tokenizer spells them.
        func text(_ tokens: [Int]) -> String {
            tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        }

        /// Deletes the folders of the setup.
        func remove() {
            for folder in folders {
                try? FileManager.default.removeItem(at: folder)
            }
        }
    }

    /// What one tool turn streamed, and what its guided pass recorded.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ToolTurn {
        let turn: MTPTurn
        let sink: GuidedGenerationDiagnosticSink
    }

    /// The turns of one test: the plain turn, the tool turn after it, and the same tool turn
    /// with no cache.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Turns {
        let plain: MTPTurn
        let warm: ToolTurn
        let cold: ToolTurn
    }

    /// The instructions and the prompt of the plain turn, with the instructions naming
    /// `sessionID`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func firstEntries(sessionID: String, setup: Setup) -> [Transcript.Entry] {
        [
            TestTranscript.instructions(id: sessionID, setup.text(instructions)),
            TestTranscript.prompt(setup.text(firstPrompt)),
        ]
    }

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func options(
        toolCallingMode: GenerationOptions.ToolCallingMode? = nil
    ) -> GenerationOptions {
        GenerationOptions(
            samplingMode: .greedy, maximumResponseTokens: responseTokenBudget,
            toolCallingMode: toolCallingMode)
    }

    /// The plain turn of the warm session.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func plainRequest(setup: Setup) -> LanguageModelExecutorGenerationRequest {
        makeExecutorRequest(
            transcript: Transcript(entries: firstEntries(sessionID: warmSession, setup: setup)),
            generationOptions: options())
    }

    /// The tool turn of `sessionID` over `plainResponse`: a required call of a `lookup` tool,
    /// whose prompt ends with the opening of the reasoning.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func toolRequest(
        sessionID: String, plainResponse: String, setup: Setup
    ) -> LanguageModelExecutorGenerationRequest {
        let entries =
            firstEntries(sessionID: sessionID, setup: setup) + [
                TestTranscript.response(plainResponse),
                TestTranscript.prompt(setup.text(secondPrompt) + reasoningStart),
            ]
        return makeExecutorRequest(
            transcript: Transcript(entries: entries),
            enabledTools: [
                Transcript.ToolDefinition(
                    name: "lookup", description: "Looks up text.",
                    parameters: String.generationSchema)
            ],
            generationOptions: options(toolCallingMode: .required))
    }

    /// Runs `request` inside the store of `setup`, with a diagnostic sink bound.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respondGuided(
        to request: LanguageModelExecutorGenerationRequest, setup: Setup
    ) async throws -> ToolTurn {
        let sink = GuidedGenerationDiagnosticSink()
        let turn = try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            try await MTPExecutorTurns.respond(to: request, model: setup.model, inside: setup.store)
        }
        return ToolTurn(turn: turn, sink: sink)
    }

    /// The position of the token that closes the reasoning: the first position past
    /// ``minimumReasoningTokenCount`` whose token stands nowhere before it.
    ///
    /// - Parameter reasoning: The reasoning tokens the probe turn wrote.
    /// - Returns: The position.
    /// - Throws: An issue when no token of `reasoning` qualifies.
    private static func closingPosition(of reasoning: [Int]) throws -> Int {
        try #require(
            reasoning.indices.dropFirst(minimumReasoningTokenCount).first {
                !reasoning[..<$0].contains(reasoning[$0])
            })
    }

    /// Runs the plain turn, the probe turn, the tool turn after the plain turn, and the same
    /// tool turn with no cache, at `placement` on `target`.
    ///
    /// The reasoning closes at the token of ``closingPosition(of:)`` of the probe's reasoning
    /// from the tool turn after the plain turn on.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func turns(_ placement: Placement, _ target: Target) async throws -> Turns {
        let setup = try Setup(placement, target)
        defer { setup.remove() }
        let plain = try await MTPExecutorTurns.respond(
            to: plainRequest(setup: setup), model: setup.model, inside: setup.store)
        let probe = try await MTPExecutorTurns.respond(
            to: toolRequest(
                sessionID: probeSession, plainResponse: plain.responseText, setup: setup),
            model: setup.model, inside: setup.store)
        let reasoning = setup.tokenizer.encode(text: probe.reasoningText, addSpecialTokens: false)
        let closing = try closingPosition(of: reasoning)
        setup.resolver.close(with: setup.text([reasoning[closing]]))

        let warm = try await respondGuided(
            to: toolRequest(
                sessionID: warmSession, plainResponse: plain.responseText, setup: setup),
            setup: setup)
        let cold = try await respondGuided(
            to: toolRequest(
                sessionID: coldSession, plainResponse: plain.responseText, setup: setup),
            setup: setup)
        return Turns(plain: plain, warm: warm, cold: cold)
    }

    /// The prompt tokens the cache served for `turn`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func cached(_ turn: MTPTurn) -> Int {
        turn.usage.first?.cachedPromptTokenCount ?? 0
    }

    /// Every prompt token of `turn`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(_ turn: MTPTurn) -> Int {
        turn.usage.first?.promptTokenCount ?? 0
    }

    /// The prompt tokens the reasoning phase of `turn` fed to the model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func reasoningPhaseFed(_ turn: MTPTurn) throws -> Int {
        try #require(turn.completions.first).promptTokenCount
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases, Target.allCases)
    func theReasoningPhaseReadsThePlainTurnFromTheCache(placement: Placement, target: Target)
        async throws
    {
        let turns = try await Self.turns(placement, target)
        let fed = try Self.reasoningPhaseFed(turns.warm.turn)

        #expect(fed <= Self.prompt(turns.warm.turn) - Self.prompt(turns.plain))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases, Target.allCases)
    func theToolTurnReportsThePlainTurnAsCached(placement: Placement, target: Target)
        async throws
    {
        let turns = try await Self.turns(placement, target)

        #expect(Self.cached(turns.warm.turn) >= Self.prompt(turns.plain))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases, Target.allCases)
    func theToolTurnReportsThePrefixItsReasoningPhaseRead(placement: Placement, target: Target)
        async throws
    {
        let turns = try await Self.turns(placement, target)
        let fed = try Self.reasoningPhaseFed(turns.warm.turn)

        #expect(Self.cached(turns.warm.turn) == Self.prompt(turns.warm.turn) - fed)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases, Target.allCases)
    func theGrammarPhaseFeedsNoMoreThanTheReasoning(placement: Placement, target: Target)
        async throws
    {
        let turns = try await Self.turns(placement, target)
        let reasoningTokenCount = try #require(turns.warm.turn.usage.first).reasoningTokenCount
        let fed = try #require(turns.warm.sink.fedPromptTokenCounts.last)

        #expect(fed <= reasoningTokenCount)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Placement.allCases, Target.allCases)
    func theToolTurnWritesTheToolCallOfAColdTurn(placement: Placement, target: Target)
        async throws
    {
        let turns = try await Self.turns(placement, target)

        #expect(!turns.cold.sink.sampledTokenIDs.isEmpty)
        #expect(turns.warm.turn.reasoningText == turns.cold.turn.reasoningText)
        #expect(turns.warm.sink.sampledTokenIDs == turns.cold.sink.sampledTokenIDs)
        #expect(turns.warm.turn.toolCalls == turns.cold.turn.toolCalls)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

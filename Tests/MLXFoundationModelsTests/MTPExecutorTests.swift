// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import CoreGraphics
import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// The executor runs its text, reasoning and tool turns with the MTP drafter of the model, and
/// writes the text plain decode writes.
///
/// Each test compares a model with the drafter of the shared Qwen 3.5 + MTP fixture against
/// the same target without it. The support types live in `MTPExecutorTestSupport.swift` and
/// `Qwen35MTPFixture+LanguageModel.swift`.
@Suite(.serialized)
struct MTPExecutorTests {

    /// The tokens of the prompt of each turn.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static let greedy = GenerationOptions(
        samplingMode: .greedy, maximumResponseTokens: generatedTokenCount)

    /// The token that opens the reasoning of the reasoning tests. The prompt ends with it, so
    /// each reasoning turn starts inside its reasoning.
    private static let reasoningStartToken = 0

    /// The position in the greedy decode of the reasoning prompt of the token that closes the
    /// reasoning: the reasoning tests read the tokens before it as reasoning.
    private static let reasoningEndIndex = 5

    /// The position in the greedy decode of ``prompt`` of a token that MTP returns as the
    /// final token of a round, which no forward pass has fed, when it ends the generation.
    private static let bonusSlotStopIndex = 3

    /// The seed of the sampled turns.
    private static let samplingSeed: UInt64 = 7

    /// A nucleus of 1, which keeps every token: the default sampling of the executor, with a
    /// seed.
    private static let fullNucleus = 1.0

    /// The token budget of the cancelled turn: more than the turn decodes before the cancel.
    private static let cancelledTurnTokenBudget = 64

    /// The time each decode of the cancelled turn's tokenizer takes, so the cancel lands
    /// while the generation runs.
    private static let cancelledTurnDecodeDelay: TimeInterval = 0.02

    /// Decodes of the cancelled turn's tokenizer that show the generation runs.
    private static let decodesBeforeCancel = 4

    /// How long the cancellation test watches for decodes after the turn returned.
    private static let quiescenceWindow = Duration.milliseconds(300)

    /// Upper bound on the wait for the cancelled turn to start generating.
    private static let generationStartTimeout = Duration.seconds(30)

    /// Pause between two reads of the decode count while polling.
    private static let pollInterval = Duration.milliseconds(5)

    /// The text and the model of one fixture target, with and without the MTP drafter.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Models {
        let fixture: Qwen35MTPFixture
        let plain: MLXLanguageModel
        let speculative: MLXLanguageModel
    }

    /// The fixture's target, once plain and once with its drafter.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModels(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        resolver: any ModelConfigurationResolver = DefaultConfigurationResolver(),
        eosTokenIds: Set<Int> = [], tokenizer: (any Tokenizer)? = nil
    ) throws -> Models {
        let fixture = try Qwen35MTPFixture()
        return Models(
            fixture: fixture,
            plain: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, resolver: resolver,
                eosTokenIds: eosTokenIds, tokenizer: tokenizer),
            speculative: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, resolver: resolver,
                eosTokenIds: eosTokenIds, tokenizer: tokenizer,
                speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding()))
    }

    /// The reasoning resolver of the reasoning tests: its end delimiter is the token at
    /// ``reasoningEndIndex`` of the greedy decode of `prompt`, as `tokenizer` spells it.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func reasoningResolver(
        prompt: [Int],
        tokenizer: any Tokenizer = ScalarTokenizer(
            vocabularySize: Qwen35MTPFixture.vocabularySize)
    ) async throws -> FixtureReasoningResolver {
        let fixture = try Qwen35MTPFixture()
        let decode = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let endToken = decode.tokens[Self.reasoningEndIndex]
        try #require(decode.tokens.firstIndex(of: endToken) == Self.reasoningEndIndex)
        try #require(!decode.tokens.contains(Self.reasoningStartToken))
        return FixtureReasoningResolver(
            start: Self.reasoningStartToken, end: endToken, tokenizer: tokenizer)
    }

    // MARK: - Paths

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aPlainTextTurnWritesThePlainText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)

        let turns = try await MTPExecutorTurns.respond(
            plain: models.plain, speculative: models.speculative,
            to: Prompt(models.fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(!turns.plain.responseText.isEmpty)
        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(!turns.plain.speculated)
        #expect(turns.speculative.speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aReasoningTurnWritesThePlainReasoningAndResponse() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let prompt = Self.prompt + [Self.reasoningStartToken]
        let models = try makeModels(
            weights: weights, capabilities: [.reasoning],
            resolver: try await reasoningResolver(prompt: prompt))

        let turns = try await MTPExecutorTurns.respond(
            plain: models.plain, speculative: models.speculative,
            to: Prompt(models.fixture.text(for: prompt)), options: Self.greedy)

        #expect(!turns.plain.reasoningText.isEmpty)
        #expect(!turns.plain.responseText.isEmpty)
        #expect(turns.speculative.reasoningText == turns.plain.reasoningText)
        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(turns.speculative.speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anAllowedToolTurnWritesThePlainText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights, capabilities: [.toolCalling])

        let turns = try await MTPExecutorTurns.respond(
            plain: models.plain, speculative: models.speculative,
            to: Prompt(models.fixture.text(for: Self.prompt)), options: Self.greedy,
            tools: [FixtureLookupTool()])

        #expect(!turns.plain.responseText.isEmpty)
        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(turns.speculative.speculated)
    }

    /// Phase 1 of think-then-call speculates; phase 2, the grammar pass, starts from the
    /// prompt and the reasoning tokens alone, so the drafter state of phase 1 does not reach
    /// it, and the grammar pass runs as it runs after plain decode.
    ///
    /// The tokens spell characters, so the grammar pass on the fixture's target writes JSON.
    /// The random weights do not close a tool call within the budget, thus both passes end
    /// with incomplete output, the same output.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aThinkThenCallTurnSpeculatesInItsReasoningPhase() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let tokenizer = ToolCallAlphabetTokenizer()
        let prompt = Self.prompt + [Self.reasoningStartToken]
        let models = try makeModels(
            weights: weights, capabilities: [.reasoning, .toolCalling],
            resolver: try await reasoningResolver(prompt: prompt, tokenizer: tokenizer),
            tokenizer: tokenizer)
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(
                                Transcript.TextSegment(
                                    content: tokenizer.decode(
                                        tokenIds: prompt, skipSpecialTokens: false)))
                        ],
                        responseFormat: nil))
            ]),
            enabledTools: [
                Transcript.ToolDefinition(
                    name: "lookup", description: "Looks up text.",
                    parameters: String.generationSchema)
            ],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount,
                toolCallingMode: .required))

        let plain = try await MTPExecutorTurns.respond(to: request, model: models.plain)
        let speculative = try await MTPExecutorTurns.respond(
            to: request, model: models.speculative)

        #expect(plain.failure == nil)
        #expect(plain.channelFailure == nil)
        #expect(!plain.reasoningText.isEmpty)
        #expect(speculative.failure == nil)
        #expect(speculative.channelFailure == nil)
        #expect(speculative.reasoningText == plain.reasoningText)
        #expect(speculative.toolCalls == plain.toolCalls)
        #expect(speculative.metadataKeys == plain.metadataKeys)
        #expect(speculative.usage == plain.usage)
        #expect(speculative.speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anImageTurnOnAVisionLanguageTargetWritesThePlainText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35VLMMTPFixture(readsImages: true)
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(
                                Transcript.TextSegment(
                                    content: fixture.tokenizer.decode(
                                        tokenIds: Self.prompt, skipSpecialTokens: false))),
                            .attachment(
                                Transcript.AttachmentSegment(
                                    content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
                                    label: "photo")),
                        ],
                        responseFormat: nil))
            ]),
            generationOptions: Self.greedy)

        let plain = try await MTPExecutorTurns.respond(
            to: request, model: try fixture.makeLanguageModel(weights: weights, speculates: false))
        let speculative = try await MTPExecutorTurns.respond(
            to: request, model: try fixture.makeLanguageModel(weights: weights, speculates: true))

        #expect(plain.failure == nil)
        #expect(plain.channelFailure == nil)
        #expect(speculative.failure == nil)
        #expect(speculative.channelFailure == nil)
        #expect(!plain.responseText.isEmpty)
        #expect(speculative.responseText == plain.responseText)
        #expect(speculative.speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anUncachedTurnWritesThePlainText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)

        let turns = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
            try await MTPExecutorTurns.respond(
                plain: models.plain, speculative: models.speculative,
                to: Prompt(models.fixture.text(for: Self.prompt)), options: Self.greedy)
        }

        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(turns.speculative.speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSampledTurnSpeculatesAndRepeatsWithItsSeed() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)
        let sampled = GenerationOptions(
            samplingMode: .random(probabilityThreshold: Self.fullNucleus, seed: Self.samplingSeed),
            maximumResponseTokens: Self.generatedTokenCount)
        let prompt = Prompt(models.fixture.text(for: Self.prompt))

        let first = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: models.speculative, tools: [], instructions: nil),
            to: prompt, options: sampled)
        let second = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: models.speculative, tools: [], instructions: nil),
            to: prompt, options: sampled)

        #expect(first.speculated)
        #expect(!first.responseText.isEmpty)
        #expect(second.responseText == first.responseText)
    }

    // MARK: - Usage

    /// A stop token that MTP returns as the final token of a round is not in the cache, and
    /// the turn still reports it as generated, as plain decode does.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnReportsThePlainUsage() async throws {
        let fixture = try Qwen35MTPFixture()
        let decode = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let stopToken = decode.tokens[Self.bonusSlotStopIndex]
        try #require(decode.tokens.firstIndex(of: stopToken) == Self.bonusSlotStopIndex)
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights, eosTokenIds: [stopToken])

        let turns = try await MTPExecutorTurns.respond(
            plain: models.plain, speculative: models.speculative,
            to: Prompt(models.fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(turns.speculative.speculated)
        #expect(turns.speculative.completions.map(\.stopTokenFedToCache) == [nil])
        #expect(turns.plain.usage.map(\.generatedTokenCount) == [Self.bonusSlotStopIndex + 1])
        #expect(turns.speculative.usage == turns.plain.usage)
    }

    // MARK: - Drafter lifecycle

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theDrafterLoadsOnceAcrossWarmUpAndTwoSessions() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let model = fixture.makeLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(log: log))

        try await model.warmUp()
        let first = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: model, tools: [], instructions: nil),
            to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)
        let second = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: model, tools: [], instructions: nil),
            to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(log.loadCount == 1)
        #expect(first.speculated)
        #expect(second.speculated)
        await model.evict()
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func evictingTheModelFreesItsDrafter() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let model = fixture.makeLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(log: log))
        let prompt = Prompt(fixture.text(for: Self.prompt))

        _ = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: model, tools: [], instructions: nil), to: prompt,
            options: Self.greedy)
        #expect(log.liveDrafterCount == 1)

        await model.evict()
        #expect(log.liveDrafterCount == 0)

        let afterEviction = try await MTPExecutorTurns.respond(
            LanguageModelSession(model: model, tools: [], instructions: nil), to: prompt,
            options: Self.greedy)
        #expect(log.loadCount == 2)
        #expect(afterEviction.speculated)
        await model.evict()
    }

    // MARK: - Cancellation

    /// One `next()` of the MTP iterator runs a whole round, and the cancelled turn still
    /// returns only after its generation stopped handling tokens.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aCancelledTurnReturnsAfterItsMTPGenerationDrained() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let counter = DecodeCounter()
        let log = DrafterLoadLog()
        let model = fixture.makeLanguageModel(
            weights: weights,
            tokenizer: SlowCountingTokenizer(
                base: fixture.tokenizer, counter: counter,
                decodeDelay: Self.cancelledTurnDecodeDelay),
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(log: log))
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)
        let text = fixture.text(for: Self.prompt)

        let turn = Task {
            _ = try await session.respond(
                to: text,
                options: GenerationOptions(
                    samplingMode: .greedy, maximumResponseTokens: Self.cancelledTurnTokenBudget))
        }
        let started = await Self.waitForDecodes(Self.decodesBeforeCancel, of: counter)
        turn.cancel()
        let result = await turn.result
        let decodesAtReturn = counter.decodeCount
        try await Task.sleep(for: Self.quiescenceWindow)

        #expect(started)
        #expect(log.loadCount == 1)
        #expect(throws: CancellationError.self) { try result.get() }
        #expect(decodesAtReturn < Self.cancelledTurnTokenBudget)
        #expect(counter.decodeCount == decodesAtReturn)
        await model.evict()
    }

    /// Waits until `counter` counted `decodes` decodes.
    ///
    /// - Returns: true when the count was reached, false when ``generationStartTimeout`` won.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func waitForDecodes(_ decodes: Int, of counter: DecodeCounter) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: generationStartTimeout)
        while counter.decodeCount < decodes {
            guard ContinuousClock.now < deadline else { return false }
            try? await Task.sleep(for: pollInterval)
        }
        return true
    }
}

#endif

// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import IntegrationTestHelpers
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Each call of an `MLXLanguageModel` with an MTP drafter says in its response metadata whether
/// MTP drafted, why it fell back, and how many drafts the call proposed and the target accepted.
///
/// The session tests read the metadata of the response entry of the session's own transcript.
/// The executor tests run one pass with no session -- a required tool call, a schema -- and read
/// the metadata the executor sent. The expected counts come from the completion reports of the
/// passes and from the guided diagnostic sink, the same sources the executor reads. The support
/// types live in `MTPExecutorTestSupport.swift`, `GuidedGenerationMTPTestSupport.swift` and
/// `Qwen35MTPFixture+LanguageModel.swift`.
@Suite(.serialized)
struct SpeculativeDecodingOutcomeMetadataTests {

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
    /// reasoning.
    private static let reasoningEndIndex = 5

    /// The tool of the required tool passes.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static let lookupTool = Transcript.ToolDefinition(
        name: "lookup", description: "Looks up text.", parameters: String.generationSchema)

    /// The metadata keys of the MTP outcome, spelled as the consumers of the metadata read them.
    private enum Key {
        static let speculativeDecoding = "speculativeDecoding"
        static let fallbackReason = "speculativeDecodingFallbackReason"
        static let acceptedDraftTokenCount = "acceptedDraftTokenCount"
        static let draftedTokenCount = "draftedTokenCount"

        /// Every key of the outcome.
        static let all: Set = [
            speculativeDecoding, fallbackReason, acceptedDraftTokenCount, draftedTokenCount,
        ]
    }

    /// The MTP outcome one response entry reported.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ReportedOutcome {
        var speculativeDecoding: Bool?
        var fallbackReason: String?
        var acceptedDraftTokenCount: Int?
        var draftedTokenCount: Int?

        /// The outcome in the metadata of a transcript entry.
        init(_ metadata: [String: GeneratedContent]) {
            speculativeDecoding = try? metadata[Key.speculativeDecoding]?.value(Bool.self)
            fallbackReason = try? metadata[Key.fallbackReason]?.value(String.self)
            acceptedDraftTokenCount = try? metadata[Key.acceptedDraftTokenCount]?.value(Int.self)
            draftedTokenCount = try? metadata[Key.draftedTokenCount]?.value(Int.self)
        }

        /// The outcome in the metadata the executor sent.
        init(_ metadata: [String: any ConvertibleToGeneratedContent & Sendable]) {
            speculativeDecoding = metadata[Key.speculativeDecoding] as? Bool
            fallbackReason = metadata[Key.fallbackReason] as? String
            acceptedDraftTokenCount = metadata[Key.acceptedDraftTokenCount] as? Int
            draftedTokenCount = metadata[Key.draftedTokenCount] as? Int
        }
    }

    /// The drafts the passes of a call proposed and the drafts the target accepted.
    private struct DraftCounts: Equatable {
        var drafted = 0
        var accepted = 0

        /// The counts of the speculative telemetry of `completions`.
        init(_ completions: [GenerateCompletionInfo]) {
            for telemetry in completions.compactMap(\.speculativeDecodingTelemetry) {
                drafted += telemetry.draftTokenCount
                accepted += telemetry.acceptedDraftTokenCount
            }
        }

        /// The counts of the MTP rounds `sink` recorded.
        init(_ sink: GuidedGenerationDiagnosticSink) {
            drafted = sink.draftedTokenCount
            accepted = sink.acceptedDraftTokenCount
        }

        /// The sum of two counts.
        static func + (lhs: DraftCounts, rhs: DraftCounts) -> DraftCounts {
            var sum = lhs
            sum.drafted += rhs.drafted
            sum.accepted += rhs.accepted
            return sum
        }
    }

    /// Expects that `reported` says MTP drafted the `counts` drafts and did not fall back.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectSpeculated(_ reported: ReportedOutcome, counts: DraftCounts) {
        #expect(counts.drafted > 0)
        #expect(reported.speculativeDecoding == true)
        #expect(reported.draftedTokenCount == counts.drafted)
        #expect(reported.acceptedDraftTokenCount == counts.accepted)
        #expect(reported.fallbackReason == nil)
    }

    /// Expects that `reported` says MTP never drafted because the drafter cannot draft for the
    /// target.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectIncompatibleFallback(_ reported: ReportedOutcome) {
        expectNoDrafts(reported, fallbackReason: "incompatibleDrafter")
    }

    /// Expects that `reported` says MTP never drafted, with `fallbackReason` as the reason.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectNoDrafts(_ reported: ReportedOutcome, fallbackReason: String?) {
        #expect(reported.speculativeDecoding == false)
        #expect(reported.fallbackReason == fallbackReason)
        #expect(reported.draftedTokenCount == 0)
        #expect(reported.acceptedDraftTokenCount == 0)
    }

    /// Whether `event` is the completion report of a pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func isCompletion(_ event: MLXLanguageModel.Executor.GenerationEvent) -> Bool {
        guard case .completion = event else { return false }
        return true
    }

    /// Runs one turn of `session` and returns what the executor streamed with the metadata of
    /// the response entry the turn left in the session's transcript.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func respond(
        _ session: LanguageModelSession, to prompt: Prompt, streaming: Bool = false
    ) async throws -> (turn: MTPTurn, metadata: [String: GeneratedContent]) {
        let log = MTPTurnLog()
        try await MLXLanguageModel.Executor.$generationObserver.withValue(log.observer) {
            if streaming {
                for try await _ in session.streamResponse(to: prompt, options: Self.greedy) {}
            } else {
                _ = try await session.respond(to: prompt, options: Self.greedy)
            }
        }
        let response = try #require(session.transcript.lastResponse)
        return (log.recorded, response.metadata)
    }

    /// The fixture's target with its MTP drafter, under `capabilities` and `resolver`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func speculativeModel(
        weights: URL, fixture: Qwen35MTPFixture,
        capabilities: [LanguageModelCapabilities.Capability] = [],
        resolver: any ModelConfigurationResolver = DefaultConfigurationResolver()
    ) throws -> MLXLanguageModel {
        fixture.makeLanguageModel(
            weights: weights, capabilities: capabilities, resolver: resolver,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())
    }

    // MARK: - Session turns

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a text turn reports in its transcript entry that MTP drafted, and how much")
    func aTextTurnReportsItsDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let session = LanguageModelSession(
            model: try speculativeModel(weights: weights, fixture: fixture), tools: [],
            instructions: nil)

        let (turn, metadata) = try await respond(
            session, to: Prompt(fixture.text(for: Self.prompt)))

        expectSpeculated(ReportedOutcome(metadata), counts: DraftCounts(turn.completions))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a streamed text turn reports the same outcome in its transcript entry")
    func aStreamedTextTurnReportsItsDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let session = LanguageModelSession(
            model: try speculativeModel(weights: weights, fixture: fixture), tools: [],
            instructions: nil)

        let (turn, metadata) = try await respond(
            session, to: Prompt(fixture.text(for: Self.prompt)), streaming: true)

        expectSpeculated(ReportedOutcome(metadata), counts: DraftCounts(turn.completions))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a reasoning turn reports the drafts of its reasoning and its response")
    func aReasoningTurnReportsItsDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let prompt = Self.prompt + [Self.reasoningStartToken]
        let resolver = try await FixtureReasoningResolver.closingAtGreedyToken(
            Self.reasoningEndIndex, of: prompt, maxTokens: Self.generatedTokenCount,
            start: Self.reasoningStartToken, tokenizer: fixture.tokenizer)
        let session = LanguageModelSession(
            model: try speculativeModel(
                weights: weights, fixture: fixture, capabilities: [.reasoning], resolver: resolver),
            tools: [], instructions: nil)

        let (turn, metadata) = try await respond(session, to: Prompt(fixture.text(for: prompt)))

        #expect(!turn.reasoningText.isEmpty)
        expectSpeculated(ReportedOutcome(metadata), counts: DraftCounts(turn.completions))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("an allowed tool turn reports its drafts")
    func anAllowedToolTurnReportsItsDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let session = LanguageModelSession(
            model: try speculativeModel(
                weights: weights, fixture: fixture, capabilities: [.toolCalling]),
            tools: [FixtureLookupTool()], instructions: nil)

        let (turn, metadata) = try await respond(
            session, to: Prompt(fixture.text(for: Self.prompt)))

        expectSpeculated(ReportedOutcome(metadata), counts: DraftCounts(turn.completions))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a text turn whose drafter cannot draft for the target reports why, and no drafts")
    func anIncompatibleDrafterReportsWhy() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let session = LanguageModelSession(
            model: fixture.makeLanguageModel(
                weights: weights,
                speculativeDecoding: try Qwen35MTPFixture.makeIncompatibleSpeculativeDecoding()),
            tools: [], instructions: nil)

        let (_, metadata) = try await respond(session, to: Prompt(fixture.text(for: Self.prompt)))

        expectIncompatibleFallback(ReportedOutcome(metadata))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a text turn of a model with no MTP drafter reports none of the MTP keys")
    func aModelWithoutADrafterReportsNoKeys() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let session = LanguageModelSession(
            model: fixture.makeLanguageModel(weights: weights), tools: [], instructions: nil)

        let (_, metadata) = try await respond(session, to: Prompt(fixture.text(for: Self.prompt)))

        #expect(metadata["modelID"] != nil)
        #expect(Key.all.isDisjoint(with: metadata.keys))
    }

    // MARK: - Executor passes

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a schema pass reports the drafts of its guided generation; a plain one reports none")
    func aSchemaPassReportsItsGuidedDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(weights: weights, capabilities: [.guidedGeneration])
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            schema: FixtureLookupArguments.generationSchema,
            generationOptions: GuidedExecutorModels.greedy)

        let turns = try await models.respond(to: request)

        #expect(turns.speculative.turn.failure == nil)
        expectSpeculated(
            ReportedOutcome(turns.speculative.turn.metadata),
            counts: DraftCounts(turns.speculative.sink))
        #expect(Key.all.isDisjoint(with: turns.plain.turn.metadata.keys))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a required tool pass reports the drafts of its guided generation")
    func aRequiredToolPassReportsItsGuidedDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(weights: weights, capabilities: [.toolCalling])
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            enabledTools: [Self.lookupTool],
            generationOptions: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: GuidedExecutorModels.responseTokenBudget,
                toolCallingMode: .required))

        let turns = try await models.respond(to: request)

        #expect(turns.speculative.turn.failure == nil)
        expectSpeculated(
            ReportedOutcome(turns.speculative.turn.metadata),
            counts: DraftCounts(turns.speculative.sink))
    }

    /// The fixture models of a think-then-call pass -- a reasoning phase that closes inside the
    /// budget, then the grammar phase of a required tool call -- and the request of the pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func thinkThenCallPass(weights: URL) async throws -> (
        models: GuidedExecutorModels, request: LanguageModelExecutorGenerationRequest
    ) {
        let prompt = Self.prompt + [Self.reasoningStartToken]
        let models = try GuidedExecutorModels(
            weights: weights, capabilities: [.reasoning, .toolCalling],
            resolver: try await FixtureReasoningResolver.closingAtGreedyToken(
                Self.reasoningEndIndex, of: prompt, maxTokens: Self.generatedTokenCount,
                start: Self.reasoningStartToken, tokenizer: ToolCallAlphabetTokenizer()))
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: prompt),
            enabledTools: [Self.lookupTool],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount,
                toolCallingMode: .required))
        return (models, request)
    }

    /// Think-then-call runs two passes: the reasoning phase, which reports its drafts in its
    /// completion report, and the grammar phase, which reports them to the guided sink. The
    /// response reports the sum.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a think-then-call pass reports the drafts of both of its phases, summed")
    func aThinkThenCallPassSumsBothPhases() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let (models, request) = try await thinkThenCallPass(weights: weights)

        let turns = try await models.respond(to: request)

        let speculative = turns.speculative
        #expect(speculative.turn.failure == nil)
        #expect(DraftCounts(speculative.turn.completions).drafted > 0)
        #expect(speculative.sink.draftedTokenCount > 0)
        expectSpeculated(
            ReportedOutcome(speculative.turn.metadata),
            counts: DraftCounts(speculative.turn.completions) + DraftCounts(speculative.sink))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a schema pass whose drafter cannot draft for the target reports why, and no drafts")
    func aSchemaPassWithAnIncompatibleDrafterReportsWhy() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(
            weights: weights, capabilities: [.guidedGeneration],
            speculativeDecoding: try Qwen35MTPFixture.makeIncompatibleSpeculativeDecoding())
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            schema: FixtureLookupArguments.generationSchema,
            generationOptions: GuidedExecutorModels.greedy)

        let turns = try await models.respond(to: request)

        #expect(turns.speculative.turn.failure == nil)
        expectIncompatibleFallback(ReportedOutcome(turns.speculative.turn.metadata))
    }

    // MARK: - Passes that throw

    /// The pass is cancelled at the completion report of its reasoning phase: that phase drafted,
    /// and the grammar phase never starts. The response reports the drafts of the reasoning
    /// phase.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a cancelled pass reports the drafts its passes reported before the cancellation")
    func aCancelledPassReportsItsDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let (models, request) = try await thinkThenCallPass(weights: weights)

        let speculative = try await models.respondSpeculatively(
            to: request, cancellingAt: Self.isCompletion)

        #expect(speculative.turn.failure == String(describing: CancellationError()))
        #expect(speculative.sink.draftedTokenCount == 0)
        expectSpeculated(
            ReportedOutcome(speculative.turn.metadata),
            counts: DraftCounts(speculative.turn.completions) + DraftCounts(speculative.sink))
    }

    /// A model that always reasons, called without `.reasoning`, fails the call inside the
    /// response, before any pass generates.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a pass that throws reports in its response entry that MTP never drafted")
    func aFailedPassReportsItsOutcome() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(
            weights: weights, capabilities: [],
            resolver: FixtureReasoningResolver(reasoningConfig: .alwaysOnThinking))
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            generationOptions: GuidedExecutorModels.greedy)

        let turns = try await models.respond(to: request)

        let failure = turns.speculative.turn.failure
        #expect(failure?.contains("always reasons") == true, "\(failure ?? "no failure")")
        expectNoDrafts(ReportedOutcome(turns.speculative.turn.metadata), fallbackReason: nil)
        #expect(Key.all.isDisjoint(with: turns.plain.turn.metadata.keys))
    }
}

#endif

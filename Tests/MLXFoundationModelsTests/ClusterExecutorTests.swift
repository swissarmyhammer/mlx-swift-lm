// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The executor runs unchanged on a model whose forward pass and caches live in a worker, and
/// writes what it writes on the local model.
///
/// Each test runs the same turns on the target of the shared Qwen 3.5 fixture twice: once
/// local, and once behind a `ClusterLanguageModel` over an in-process worker that runs the same
/// target.
@Suite(.serialized)
struct ClusterExecutorTests {

    /// The tokens of the prompt of each turn.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the prompt of a second turn.
    private static let followUp = [12, 7, 30, 21]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static let greedy = GenerationOptions(
        samplingMode: .greedy, maximumResponseTokens: generatedTokenCount)

    /// The instructions of each session, which name the session of the prompt cache.
    private static let instructions = "Answer."

    /// The local and the clustered model of one fixture, with the same tokenizer.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Models {
        let fixture: Qwen35MTPFixture
        let local: MLXLanguageModel
        let clustered: MLXLanguageModel
        let worker: InProcessRemoteModelWorker

        init(
            weights: URL, capabilities: [LanguageModelCapabilities.Capability],
            tokenizer: (any Tokenizer)? = nil
        ) throws {
            fixture = try Qwen35MTPFixture()
            local = fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, tokenizer: tokenizer)
            (clustered, worker) = try fixture.makeClusteredLanguageModel(
                weights: weights, capabilities: capabilities, tokenizer: tokenizer)
        }
    }

    /// The turns of one new session of `model`, one for each of `prompts`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respond(
        _ model: MLXLanguageModel, to prompts: [String],
        tools: [any FoundationModels.Tool] = []
    ) async throws -> [MTPTurn] {
        let session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
        var turns: [MTPTurn] = []
        for prompt in prompts {
            turns.append(
                try await MTPExecutorTurns.respond(session, to: Prompt(prompt), options: greedy))
        }
        return turns
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aPlainTurnWritesTheLocalText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try Models(weights: weights, capabilities: [])
        let prompts = [models.fixture.text(for: Self.prompt)]

        let local = try await Self.respond(models.local, to: prompts)
        let clustered = try await Self.respond(models.clustered, to: prompts)

        #expect(!local[0].responseText.isEmpty)
        #expect(clustered.map(\.responseText) == local.map(\.responseText))
        #expect(clustered.map(\.usage) == local.map(\.usage))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSecondTurnReusesThePromptCacheAsTheLocalTurnDoes() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try Models(weights: weights, capabilities: [])
        let prompts = [Self.prompt, Self.followUp].map(models.fixture.text(for:))

        let local = try await Self.respond(models.local, to: prompts)
        let clustered = try await Self.respond(models.clustered, to: prompts)

        #expect(clustered.map(\.responseText) == local.map(\.responseText))
        #expect(clustered.map(\.usage) == local.map(\.usage))
        #expect((clustered[1].usage.first?.cachedPromptTokenCount ?? 0) > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anAllowedToolTurnWritesTheLocalText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try Models(weights: weights, capabilities: [.toolCalling])
        let prompts = [models.fixture.text(for: Self.prompt)]

        let local = try await Self.respond(models.local, to: prompts, tools: [FixtureLookupTool()])
        let clustered = try await Self.respond(
            models.clustered, to: prompts, tools: [FixtureLookupTool()])

        #expect(!local[0].responseText.isEmpty)
        #expect(clustered.map(\.responseText) == local.map(\.responseText))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aRequiredToolTurnWritesTheLocalToolCall() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let tokenizer = ToolCallAlphabetTokenizer()
        let models = try Models(
            weights: weights, capabilities: [.toolCalling], tokenizer: tokenizer)
        let request = makeExecutorRequest(
            transcript: Self.transcript(prompt: Self.prompt, tokenizer: tokenizer),
            enabledTools: [
                Transcript.ToolDefinition(
                    name: "lookup", description: "Looks up text.",
                    parameters: String.generationSchema)
            ],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount,
                toolCallingMode: .required))

        let local = try await Self.respondGuided(to: request, model: models.local)
        let clustered = try await Self.respondGuided(to: request, model: models.clustered)

        #expect(local.turn.failure == nil)
        #expect(clustered.turn.failure == nil)
        #expect(!local.sampledTokens.isEmpty)
        #expect(clustered.sampledTokens == local.sampledTokens)
        #expect(clustered.turn.toolCalls == local.turn.toolCalls)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnWritesTheLocalResponse() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let tokenizer = ToolCallAlphabetTokenizer()
        let models = try Models(
            weights: weights, capabilities: [.guidedGeneration], tokenizer: tokenizer)
        let request = makeExecutorRequest(
            transcript: Self.transcript(prompt: Self.prompt, tokenizer: tokenizer),
            schema: FixtureLookupArguments.generationSchema, generationOptions: Self.greedy)

        let local = try await Self.respondGuided(to: request, model: models.local)
        let clustered = try await Self.respondGuided(to: request, model: models.clustered)

        #expect(local.turn.failure == nil)
        #expect(!local.turn.responseText.isEmpty)
        #expect(clustered.sampledTokens == local.sampledTokens)
        #expect(clustered.turn.responseText == local.turn.responseText)
        #expect(clustered.turn.usage == local.turn.usage)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithADrafterIsRefused() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let (model, _) = try fixture.makeClusteredLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())
        let request = makeExecutorRequest(
            transcript: Self.transcript(prompt: Self.prompt, tokenizer: fixture.tokenizer),
            generationOptions: Self.greedy)

        let turn = try await MTPExecutorTurns.respond(to: request, model: model)

        #expect(
            turn.failure
                == String(describing: ClusterModelError.unsupported(.speculativeDecoding)))
    }

    /// The forwards a worker runs before it loses a member: the prompt and a few decode steps,
    /// thus the loss strikes in the middle of the answer.
    private static let forwardsBeforeLoss = 8

    /// The member the worker loses.
    private static let lostMember = "m5u2.local"

    /// A session of a clustered model whose worker loses ``lostMember`` after
    /// ``forwardsBeforeLoss`` forwards, and the text of ``prompt``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeSessionThatLosesAMember(weights: URL) throws -> (
        session: LanguageModelSession, prompt: String
    ) {
        let fixture = try Qwen35MTPFixture()
        let (model, worker) = try fixture.makeClusteredLanguageModel(weights: weights)
        worker.failForwards(
            after: forwardsBeforeLoss, with: InProcessMemberLoss(lostMember: lostMember))
        let session = LanguageModelSession(model: model, instructions: instructions)
        return (session, fixture.text(for: prompt))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aResponseWhoseForwardLosesAMemberThrowsTheLoss() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let (session, prompt) = try Self.makeSessionThatLosesAMember(weights: weights)

        await #expect(throws: ClusterModelError.memberLost(Self.lostMember)) {
            _ = try await session.respond(to: prompt, options: Self.greedy)
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aStreamWhoseForwardLosesAMemberThrowsTheLossAfterItsSnapshots() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let (session, prompt) = try Self.makeSessionThatLosesAMember(weights: weights)
        var snapshotCount = 0

        await #expect(throws: ClusterModelError.memberLost(Self.lostMember)) {
            for try await _ in session.streamResponse(to: prompt, options: Self.greedy) {
                snapshotCount += 1
            }
        }
        #expect(snapshotCount > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aGenerationCancelledByALostMemberThrowsTheLoss() async throws {
        let worker = try Qwen35MTPFixture().makeWorker()
        let model = try ClusterLanguageModel.connect(over: worker)
        worker.failOps(after: 0, with: InProcessMemberLoss(lostMember: Self.lostMember))
        _ = try? model.newCache(parameters: nil)

        await #expect(throws: ClusterModelError.memberLost(Self.lostMember)) {
            try await MLXLanguageModel.Executor.reportingClusterFailure(of: model) {
                throw CancellationError()
            }
        }
    }

    /// What one guided pass streamed, and the tokens its decoder sampled.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct GuidedTurn {
        let turn: MTPTurn
        let sampledTokens: [Int]
    }

    /// Runs `request` on `model` with a diagnostic sink bound.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respondGuided(
        to request: LanguageModelExecutorGenerationRequest, model: MLXLanguageModel
    ) async throws -> GuidedTurn {
        let sink = GuidedGenerationDiagnosticSink()
        let turn = try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            try await MTPExecutorTurns.respond(to: request, model: model)
        }
        return GuidedTurn(turn: turn, sampledTokens: sink.sampledTokenIDs)
    }

    /// A transcript of one prompt of `prompt` tokens, as `tokenizer` spells them.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: [Int], tokenizer: any Tokenizer) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(
                            Transcript.TextSegment(
                                content: tokenizer.decode(
                                    tokenIds: prompt, skipSpecialTokens: false)))
                    ],
                    responseFormat: nil))
        ])
    }
}

#endif

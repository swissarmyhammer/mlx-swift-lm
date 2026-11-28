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
            tokenizer: (any Tokenizer)? = nil, fixture: Qwen35MTPFixture? = nil,
            kvCache: KVCacheConfiguration? = nil
        ) throws {
            self.fixture = try fixture ?? Qwen35MTPFixture()
            local = self.fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, tokenizer: tokenizer,
                kvCache: kvCache)
            (clustered, worker) = try self.fixture.makeClusteredLanguageModel(
                weights: weights, capabilities: capabilities, tokenizer: tokenizer,
                kvCache: kvCache)
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

    /// The models of a fixture whose attention caches quantize, under affine 8-bit KV-cache
    /// compression.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func eightBitModels(weights: URL) throws -> Models {
        try Models(
            weights: weights, capabilities: [],
            fixture: Qwen35MTPFixture(
                headDimension: AffineKVCacheConfiguration.eightBit.groupSize),
            kvCache: KVCacheConfiguration(strategy: .affine(.eightBit)))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSecondTurnUnderAnEightBitKVCacheWritesTheLocalText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try Self.eightBitModels(weights: weights)
        let prompts = [Self.prompt, Self.followUp].map(models.fixture.text(for:))

        let local = try await Self.respond(models.local, to: prompts)
        let clustered = try await Self.respond(models.clustered, to: prompts)

        #expect(clustered.map(\.responseText) == local.map(\.responseText))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSecondTurnUnderAnEightBitKVCacheReusesThePromptCacheAsTheLocalTurnDoes() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try Self.eightBitModels(weights: weights)
        let prompts = [Self.prompt, Self.followUp].map(models.fixture.text(for:))

        let local = try await Self.respond(models.local, to: prompts)
        let clustered = try await Self.respond(models.clustered, to: prompts)

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
        let request = Self.requiredToolRequest(tokenizer: tokenizer)

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
        let request = Self.schemaRequest(tokenizer: tokenizer)

        let local = try await Self.respondGuided(to: request, model: models.local)
        let clustered = try await Self.respondGuided(to: request, model: models.clustered)

        #expect(local.turn.failure == nil)
        #expect(!local.turn.responseText.isEmpty)
        #expect(clustered.sampledTokens == local.sampledTokens)
        #expect(clustered.turn.responseText == local.turn.responseText)
        #expect(clustered.turn.usage == local.turn.usage)
    }

    /// The turns of a clustered model whose rank 0 holds the drafter of the fixture, and of the
    /// local model that decodes plainly.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func speculativeTurns(weights: URL) async throws -> (
        clustered: [MTPTurn], local: [MTPTurn]
    ) {
        let fixture = try Qwen35MTPFixture()
        let (clustered, _) = try fixture.makeClusteredLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(), drafting: true)
        let prompts = [Self.prompt, Self.followUp].map(fixture.text(for:))
        return (
            try await Self.respond(clustered, to: prompts),
            try await Self.respond(fixture.makeLanguageModel(weights: weights), to: prompts)
        )
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithADrafterWritesTheLocalText() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.speculativeTurns(weights: weights)

        #expect(turns.clustered.map(\.responseText) == turns.local.map(\.responseText))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithADrafterSpeculatesEachTurn() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.speculativeTurns(weights: weights)

        #expect(turns.clustered.allSatisfy { $0.speculated })
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithADrafterReportsItsAcceptedDrafts() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.speculativeTurns(weights: weights)

        #expect(
            (turns.clustered[0].metadata[
                SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int ?? 0) > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithoutADrafterReportsAnIncompatibleDrafter() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let (model, _) = try fixture.makeClusteredLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())

        let turns = try await Self.respond(model, to: [fixture.text(for: Self.prompt)])

        #expect(
            turns[0].metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] as? String
                == "incompatibleDrafter")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aClusteredModelWithoutADrafterStatesWhyInItsMetadata() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let (model, _) = try fixture.makeClusteredLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())

        let turns = try await Self.respond(model, to: [fixture.text(for: Self.prompt)])

        let detail =
            turns[0].metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackDetail] as? String
        #expect(detail?.contains("holds no MTP drafter") == true, "\(detail ?? "no detail")")
    }

    /// A schema turn, then a session of a prompt and a follow-up, of one clustered model whose
    /// rank 0 holds the drafter of the fixture. The schema pass speculates with the drafter of
    /// rank 0, which drafts without the grammar mask.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaThenPlainTurns(weights: URL) async throws -> (
        schema: GuidedTurn, plain: [MTPTurn]
    ) {
        let tokenizer = ToolCallAlphabetTokenizer()
        let (model, _) = try Qwen35MTPFixture().makeClusteredLanguageModel(
            weights: weights, capabilities: [.guidedGeneration], tokenizer: tokenizer,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(), drafting: true)
        let schema = try await respondGuided(to: schemaRequest(tokenizer: tokenizer), model: model)
        let plain = try await respond(
            model,
            to: [prompt, followUp].map {
                tokenizer.decode(tokenIds: $0, skipSpecialTokens: false)
            })
        return (schema, plain)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnOfAClusteredModelWithADrafterSpeculates() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.schemaThenPlainTurns(weights: weights)

        #expect(
            turns.schema.turn.metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding]
                as? Bool == true)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnOfAClusteredModelWithADrafterReportsTheDraftsOfItsRounds() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.schemaThenPlainTurns(weights: weights)

        let metadata = turns.schema.turn.metadata
        #expect(turns.schema.draftedTokenCount > 0)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
                == turns.schema.draftedTokenCount)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int
                == turns.schema.acceptedDraftTokenCount)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnOfAClusteredModelWithADrafterWritesTheLocalResponse() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let tokenizer = ToolCallAlphabetTokenizer()
        let local = try Qwen35MTPFixture().makeLanguageModel(
            weights: weights, capabilities: [.guidedGeneration], tokenizer: tokenizer)

        let turns = try await Self.schemaThenPlainTurns(weights: weights)
        let plain = try await Self.respondGuided(
            to: Self.schemaRequest(tokenizer: tokenizer), model: local)

        #expect(!plain.turn.responseText.isEmpty)
        #expect(turns.schema.sampledTokens == plain.sampledTokens)
        #expect(turns.schema.turn.responseText == plain.turn.responseText)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aPlainTurnAfterASchemaTurnOfAClusteredModelStillSpeculates() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.schemaThenPlainTurns(weights: weights)

        #expect(turns.plain[0].speculated)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpAfterASpeculativeSchemaTurnOfAClusteredModelReusesThePromptCache() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let turns = try await Self.schemaThenPlainTurns(weights: weights)

        #expect(
            turns.schema.turn.metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding]
                as? Bool == true)
        #expect((turns.plain[1].usage.first?.cachedPromptTokenCount ?? 0) > 0)
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

    /// The longest a guided pass whose forward loses a member may take to throw.
    ///
    /// It stands far under the 60 s quiescence timeout of the drain coordinator, which a pass
    /// that waits on its own guided section runs into, and far above the second the pass takes.
    private static let guidedLossDeadline: Duration = .seconds(10)

    /// What a guided pass whose forward loses a member threw, and how long it took.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct GuidedLoss {
        let failure: String?
        let elapsed: Duration
    }

    /// The forwards a worker runs before it loses a member in a required-tool turn: none, as
    /// the required-tool pass of the fixture runs one forward, its prompt, where the loss strikes.
    private static let requiredToolForwardsBeforeLoss = 0

    /// Runs `request` on a clustered model of `capabilities` and the tool-call alphabet, whose
    /// worker loses ``lostMember`` after `forwards` forwards, and times the pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respondGuidedLosingAMember(
        to request: LanguageModelExecutorGenerationRequest,
        capabilities: [LanguageModelCapabilities.Capability], afterForwards forwards: Int,
        weights: URL
    ) async throws -> GuidedLoss {
        let (model, worker) = try Qwen35MTPFixture().makeClusteredLanguageModel(
            weights: weights, capabilities: capabilities, tokenizer: ToolCallAlphabetTokenizer())
        worker.failForwards(after: forwards, with: InProcessMemberLoss(lostMember: lostMember))
        let clock = ContinuousClock()
        let start = clock.now
        let turn = try await respondGuided(to: request, model: model)
        return GuidedLoss(failure: turn.turn.failure, elapsed: clock.now - start)
    }

    /// The rounds a worker runs before it loses a member in a speculative schema turn: a few,
    /// thus the loss strikes in the middle of the answer.
    private static let roundsBeforeLoss = 2

    /// Runs the schema request on a clustered model whose rank 0 holds the drafter of the
    /// fixture, whose worker loses ``lostMember`` after ``roundsBeforeLoss`` speculative rounds,
    /// and times the pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respondSpeculativelyLosingAMember(weights: URL) async throws
        -> GuidedLoss
    {
        let tokenizer = ToolCallAlphabetTokenizer()
        let (model, worker) = try Qwen35MTPFixture().makeClusteredLanguageModel(
            weights: weights, capabilities: [.guidedGeneration], tokenizer: tokenizer,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(), drafting: true)
        worker.failRounds(
            after: roundsBeforeLoss, with: InProcessMemberLoss(lostMember: lostMember))
        let clock = ContinuousClock()
        let start = clock.now
        let turn = try await respondGuided(to: schemaRequest(tokenizer: tokenizer), model: model)
        return GuidedLoss(failure: turn.turn.failure, elapsed: clock.now - start)
    }

    /// The failure a pass that lost ``lostMember`` reports: the loss itself, unwrapped.
    private static let lossFailure = String(describing: ClusterModelError.memberLost(lostMember))

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnWhoseForwardLosesAMemberThrowsTheLossPromptly() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let request = Self.schemaRequest(tokenizer: ToolCallAlphabetTokenizer())

        let loss = try await Self.respondGuidedLosingAMember(
            to: request, capabilities: [.guidedGeneration],
            afterForwards: Self.forwardsBeforeLoss, weights: weights)

        #expect(loss.failure == Self.lossFailure)
        #expect(loss.elapsed < Self.guidedLossDeadline)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpeculativeSchemaTurnWhoseRoundLosesAMemberThrowsTheLoss() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let loss = try await Self.respondSpeculativelyLosingAMember(weights: weights)

        #expect(loss.failure == Self.lossFailure)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpeculativeSchemaTurnWhoseRoundLosesAMemberThrowsPromptly() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let loss = try await Self.respondSpeculativelyLosingAMember(weights: weights)

        #expect(loss.failure != nil)
        #expect(loss.elapsed < Self.guidedLossDeadline)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aRequiredToolTurnWhoseForwardLosesAMemberThrowsTheLossPromptly() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let request = Self.requiredToolRequest(tokenizer: ToolCallAlphabetTokenizer())

        let loss = try await Self.respondGuidedLosingAMember(
            to: request, capabilities: [.toolCalling],
            afterForwards: Self.requiredToolForwardsBeforeLoss, weights: weights)

        #expect(loss.failure == Self.lossFailure)
        #expect(loss.elapsed < Self.guidedLossDeadline)
    }

    /// What one guided pass streamed, the tokens its decoder sampled, and the drafts its MTP
    /// rounds proposed and kept.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct GuidedTurn {
        let turn: MTPTurn
        let sampledTokens: [Int]
        let draftedTokenCount: Int
        let acceptedDraftTokenCount: Int
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
        return GuidedTurn(
            turn: turn, sampledTokens: sink.sampledTokenIDs,
            draftedTokenCount: sink.draftedTokenCount,
            acceptedDraftTokenCount: sink.acceptedDraftTokenCount)
    }

    /// A greedy request of ``prompt`` that requires a call of a `lookup` tool, as `tokenizer`
    /// spells the prompt.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func requiredToolRequest(
        tokenizer: any Tokenizer
    ) -> LanguageModelExecutorGenerationRequest {
        makeExecutorRequest(
            transcript: transcript(prompt: prompt, tokenizer: tokenizer),
            enabledTools: [
                Transcript.ToolDefinition(
                    name: "lookup", description: "Looks up text.",
                    parameters: String.generationSchema)
            ],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: generatedTokenCount,
                toolCallingMode: .required))
    }

    /// A greedy request of ``prompt`` whose response follows the schema of
    /// `FixtureLookupArguments`, as `tokenizer` spells the prompt.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaRequest(
        tokenizer: any Tokenizer
    ) -> LanguageModelExecutorGenerationRequest {
        makeExecutorRequest(
            transcript: transcript(prompt: prompt, tokenizer: tokenizer),
            schema: FixtureLookupArguments.generationSchema, generationOptions: greedy)
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

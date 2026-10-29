// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Guided generation -- schema output and required tool calls -- speculates with the MTP drafter
/// of the target and writes what plain guided decoding writes.
///
/// The loop tests run `GuidedGenerationLoop` on the shared Qwen 3.5 + MTP fixture, whose tokens
/// spell characters (``ToolCallAlphabetTokenizer``), so its grammar writes JSON. The executor
/// tests run the same comparison through `MLXLanguageModel`. The support types live in
/// `MTPExecutorTestSupport.swift`, `GuidedGenerationMTPTestSupport.swift` and
/// `Qwen35MTPFixture+LanguageModel.swift`.
@Suite(.serialized)
struct GuidedGenerationMTPTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the continuation the verify-forward test feeds after the prompt.
    private static let continuation: [Int32] = [12, 7, 30, 21]

    /// The widest gap between a verify row and the one-token logits of its position, in float32.
    private static let verifyLogitTolerance: Float = 1e-4

    /// The block sizes of the loop tests: the shipped single draft, and chains of three drafts.
    private static let blockSizes = [2, 4]

    // MARK: - The loop writes the plain output

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Qwen35MTPFixture.DrafterMode.allCases)
    func aSchemaWritesThePlainOutputWithTheDrafter(
        schema: GuidedFixtureSchema, mode: Qwen35MTPFixture.DrafterMode
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil)

        for blockSize in Self.blockSizes {
            let speculative = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize)

            #expect(!plain.text.isEmpty)
            #expect(speculative.text == plain.text)
            #expect(speculative.tokenCount == plain.tokenCount)
            #expect(speculative.incomplete == plain.incomplete)
            #expect(speculative.sink.draftedTokenCount > 0)
        }
        #expect(plain.sink.draftedTokenCount == 0)
        #expect(plain.sink.decisionMargins.count == plain.sink.sampledTokenIDs.count)
        #expect(plain.sink.decisionMargins.allSatisfy { $0 >= 0 })
    }

    /// The independent drafter makes the target reject most drafts. After each rejection the
    /// matcher rolls back, with the tokens a rejected draft forced, and every decision reads the
    /// mask and the biases plain decoding read at its position.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func aRejectedDraftLeavesThePlainMaskAndBiases(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let plain = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil)
        let speculative = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize)

        #expect(!speculative.sink.rollbacks.isEmpty)
        #expect(!plain.sink.decisions.isEmpty)
        #expect(speculative.sink.decisions == plain.sink.decisions)
    }

    // MARK: - Matcher rollback

    /// A rollback of drafts that forced no tokens restores the mask of the position before them.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func rollingBackDraftsWithoutFastForwardRestoresTheMask() throws {
        let grammar = try GuidedFixtureGrammar(schema: GuidedFixtureSchema.unbounded.json)
        let start = try grammar.rules.startPosition()
        let chain = GuidedDraftChain(rules: grammar.rules, start: start)

        try grammar.draft(["{", " ", " "], into: chain)
        let afterBrace = try #require(chain.drafts.first)
        try chain.rollBack(draftsFrom: 1)
        let restoredAfterBrace = try grammar.rules.constraint.computeMask()
        try chain.rollBack(draftsFrom: 0)
        let restoredStart = try grammar.rules.constraint.computeMask()

        #expect(chain.drafts.allSatisfy { $0.commit.tokens.isEmpty })
        #expect(chain.drafts.map(\.acceptanceCount) == [1, 1, 1])
        #expect(restoredAfterBrace.mask == chain.drafts[1].position.mask.mask)
        #expect(afterBrace.position.mask.mask == start.mask.mask)
        #expect(restoredStart.mask == start.mask.mask)
        #expect(restoredStart.isTerminated == start.mask.isTerminated)
    }

    /// The draft that opens a property name forces the rest of the name. Its rollback counts the
    /// forced tokens too, and restores the mask before the draft.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func rollingBackADraftWithFastForwardRestoresTheMask() throws {
        let grammar = try GuidedFixtureGrammar(schema: GuidedFixtureSchema.unbounded.json)
        let start = try grammar.rules.startPosition()
        let chain = GuidedDraftChain(rules: grammar.rules, start: start)

        try grammar.draft(["{", "\""], into: chain)
        let quote = try #require(chain.drafts.last)
        try chain.rollBack(draftsFrom: 1)
        let restored = try grammar.rules.constraint.computeMask()

        #expect(chain.drafts.count == 2)
        #expect(!quote.commit.tokens.isEmpty)
        #expect(quote.acceptanceCount == 1 + quote.commit.tokens.count)
        #expect(restored.mask == quote.position.mask.mask)
        #expect(restored.needsApply == quote.position.mask.needsApply)
    }

    /// The chain stops at a draft that forces tokens: the drafts after it need target hidden rows
    /// of the forced tokens, which only a verify gives.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theChainEndsAtADraftThatForcesTokens() throws {
        let grammar = try GuidedFixtureGrammar(schema: GuidedFixtureSchema.unbounded.json)
        let chain = GuidedDraftChain(
            rules: grammar.rules, start: try grammar.rules.startPosition())

        try grammar.draft(["{", "\"", "x"], into: chain)

        #expect(chain.drafts.count == 2)
        #expect(chain.hasEnded)
    }

    // MARK: - Batched forced tokens

    /// The MTP path puts the forced tokens of a draft in its verify batch: on a Qwen 3.5 target, a
    /// verify forward of several tokens on a populated cache gives each row the logits a forward
    /// of that one token gives.
    @Test
    func aVerifyForwardOnAPopulatedCacheGivesTheOneTokenLogits() throws {
        let fixture = try Qwen35MTPFixture()
        let prompt = LMInput.Text(tokens: MLXArray(Self.prompt.map(Int32.init)))

        let stepCache = try fixture.target.newCache(parameters: nil)
        _ = fixture.target(prompt[text: .newAxis], cache: stepCache, state: nil)
        let stepped = Self.continuation.map { token in
            fixture.target(
                LMInput.Text(tokens: MLXArray([token]))[text: .newAxis], cache: stepCache,
                state: nil
            ).logits[0, -1]
        }

        let verifyCache = try fixture.target.newCache(parameters: nil)
        _ = fixture.target(prompt[text: .newAxis], cache: verifyCache, state: nil)
        var verifyState = LMOutput.State()
        verifyState[mtpEmitFlagKey] = true
        verifyState[mtpVerifyKeepsSnapshotsKey] = true
        let verified = fixture.target(
            LMInput.Text(tokens: MLXArray(Self.continuation))[text: .newAxis],
            cache: verifyCache, state: verifyState
        ).logits[0]

        for (row, logits) in stepped.enumerated() {
            #expect(argMax(verified[row]).item(Int.self) == argMax(logits).item(Int.self))
            #expect(abs(verified[row] - logits).max().item(Float.self) < Self.verifyLogitTolerance)
        }
    }

    // MARK: - The executor writes the plain output

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSchemaTurnWritesThePlainResponse() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(weights: weights, capabilities: [.guidedGeneration])
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            schema: FixtureLookupArguments.generationSchema,
            generationOptions: GuidedExecutorModels.greedy)

        let turns = try await models.respond(to: request)

        #expect(turns.plain.turn.failure == nil)
        #expect(turns.speculative.turn.failure == nil)
        #expect(!turns.plain.turn.responseText.isEmpty)
        #expect(turns.speculative.turn.responseText == turns.plain.turn.responseText)
        #expect(turns.speculative.turn.outputMetadataKeys == turns.plain.turn.outputMetadataKeys)
        #expect(turns.speculative.turn.usage == turns.plain.turn.usage)
        #expect(turns.speculative.sink.draftedTokenCount > 0)
        #expect(turns.plain.sink.draftedTokenCount == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aRequiredToolTurnWritesThePlainToolCall() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try GuidedExecutorModels(weights: weights, capabilities: [.toolCalling])
        let request = makeExecutorRequest(
            transcript: models.transcript(prompt: Self.prompt),
            enabledTools: [
                Transcript.ToolDefinition(
                    name: "lookup", description: "Looks up text.",
                    parameters: String.generationSchema)
            ],
            generationOptions: GenerationOptions(
                samplingMode: .greedy,
                maximumResponseTokens: GuidedExecutorModels.responseTokenBudget,
                toolCallingMode: .required))

        let turns = try await models.respond(to: request)

        #expect(turns.plain.turn.failure == nil)
        #expect(turns.speculative.turn.failure == nil)
        #expect(turns.speculative.turn.toolCalls == turns.plain.turn.toolCalls)
        #expect(turns.speculative.turn.outputMetadataKeys == turns.plain.turn.outputMetadataKeys)
        #expect(turns.speculative.turn.usage == turns.plain.turn.usage)
        #expect(turns.speculative.sink.draftedTokenCount > 0)
    }
}

#endif

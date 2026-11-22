// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXGuidedGeneration
@testable import MLXLMCommon

/// Guided generation on an Inkling 2 target with its DSpark drafter: each draft takes the grammar
/// mask, the target verifies the round in place, and the output is the plain output.
///
/// The fixtures are a tiny Inkling 2 target and drafter with seeded random weights, over the
/// 64 tokens of ``ToolCallAlphabetTokenizer``. A real model's calls are
/// `Inkling2FoundationModelsMTPTests` in the integration project.
@Suite(.serialized)
struct GuidedGenerationInklingMTPTests {

    /// The vocabulary of ``ToolCallAlphabetTokenizer``, every row of it a token.
    private static let vocabularySize = 64

    /// The tiny Inkling 2 target of ``Inkling2TinyModels`` over the vocabulary of
    /// ``ToolCallAlphabetTokenizer``, with its DSpark head: two draft layers, a window of six,
    /// and blocks of four.
    private static let json = Inkling2TinyModels.draftJSON(
        vocabularySize: vocabularySize, unpaddedVocabularySize: vocabularySize)

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The block size of a generation with one draft per round.
    private static let singleDraftBlockSize = 2

    /// The block size of a generation with chains of three drafts.
    private static let chainBlockSize = 4

    /// The block sizes of the generations: one draft, and chains of three drafts.
    private static let blockSizes = [singleDraftBlockSize, chainBlockSize]

    /// A prefill step shorter than ``prompt``, whose chunks feed the drafter.
    private static let shortPrefillStep = 3

    /// The prefills of the generations: the whole prompt in one forward, and in chunks.
    private static let prefillSteps: [Int?] = [nil, shortPrefillStep]

    /// Every schema of the fixtures speculates to the plain output, with the whole
    /// prompt in one forward and in chunks, and reports no fallback.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Self.blockSizes, Self.prefillSteps)
    func aGuidedGenerationSpeculatesToThePlainOutput(blockSize: Int, prefillStep: Int?) throws {
        let fixture = try Qwen35MTPFixture()
        let target = try Inkling2TinyModels.target(json: Self.json)
        let drafter = try Inkling2TinyModels.drafter(json: Self.json)
        let prefill = PrefillParameters(stepSize: prefillStep)
        for schema in GuidedFixtureSchema.allCases {
            let plain = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil,
                target: target, prefill: prefill)
            let speculative = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture,
                blockSize: blockSize, drafter: drafter, target: target, prefill: prefill)

            #expect(!plain.text.isEmpty, "\(schema)")
            #expect(speculative.text == plain.text, "\(schema)")
            #expect(speculative.tokenCount == plain.tokenCount, "\(schema)")
            #expect(speculative.incomplete == plain.incomplete, "\(schema)")
            #expect(speculative.sink.draftedTokenCount > 0, "\(schema)")
            #expect(speculative.sink.speculationFallbacks.isEmpty, "\(schema)")
        }
    }

    /// The guided drafter of an Inkling 2 speculation is the DSpark head, and the target's
    /// forwards emit the hidden states of the taps the head reads.
    @Test
    func theDSparkHeadDraftsUnderAGrammar() throws {
        let target = try Inkling2TinyModels.target(json: Self.json)
        let drafter = try Inkling2TinyModels.drafter(json: Self.json)
        let speculation = MTPSpeculation(
            drafter: drafter, blockSize: Self.chainBlockSize, minimumDraftProbability: 0)
        var guidedTarget = try GuidedTarget(model: target, parameters: GenerateParameters())
        #expect(guidedTarget.drafterTargetLayerIds == nil)

        #expect(
            GuidedSpeculation.drafter(of: speculation, for: &guidedTarget) is Inkling2DraftModel)
        #expect(guidedTarget.drafterTargetLayerIds == drafter.targetLayerIds)
        #expect(guidedTarget.drafterTargetLayerIds != nil)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

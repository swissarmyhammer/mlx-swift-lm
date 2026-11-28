// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The prompt of the grammar phase of a required tool turn after its reasoning phase
/// (`MLXLanguageModel.Executor.grammarPhasePrompt(continuing:opener:reasoning:splicedPrompt:checkpointBoundaries:)`).
///
/// The render of each fixture is `[1, 2, 3]`, the opener of the frame `[4]`, and the reasoning
/// `[7, 8]`. The splice stands for a reasoning phase whose caches hold `[1, 2, 9, 3, 4]`: the
/// model wrote a token the render does not hold.
@Suite
struct GrammarPhasePromptTests {

    /// The render of the tool-aware prompt.
    private static let render = [1, 2, 3]

    /// The opener of the frame.
    private static let opener = [4]

    /// The reasoning the reasoning phase wrote.
    private static let reasoning = [7, 8]

    /// The splice of a reasoning phase whose prompt was the render and the opener.
    private static let splice = ExecutorSplicedPrompt(
        renderedTokenCount: render.count + opener.count, representedTokens: [1, 2, 9, 3, 4])

    /// The checkpoint boundaries of the render.
    private static let boundaries = ExecutorCheckpointBoundaries(
        instructionsEnd: 1, transcriptEnd: render.count)

    /// The grammar-phase prompt over `tokens`, after `splicedPrompt`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(
        _ tokens: MLXArray, after splicedPrompt: ExecutorSplicedPrompt?
    ) -> (input: LMInput, checkpointBoundaries: ExecutorCheckpointBoundaries) {
        MLXLanguageModel.Executor.grammarPhasePrompt(
            continuing: LMInput(tokens: tokens), opener: opener, reasoning: reasoning,
            splicedPrompt: splicedPrompt, checkpointBoundaries: boundaries)
    }

    /// The render as a plain token vector.
    private static var renderTokens: MLXArray { MLXArray(render.map { Int32($0) }) }

    @Test @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func withoutASpliceThePromptContinuesTheRender() {
        let prompt = Self.prompt(Self.renderTokens, after: nil)

        #expect(prompt.input.text.tokens.asArray(Int.self) == [1, 2, 3, 4, 7, 8])
    }

    @Test @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func withoutASpliceTheBoundariesStay() {
        let prompt = Self.prompt(Self.renderTokens, after: nil)

        #expect(prompt.checkpointBoundaries == Self.boundaries)
    }

    @Test @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func afterASpliceThePromptContinuesTheTokensTheCachesRepresent() {
        let prompt = Self.prompt(Self.renderTokens, after: Self.splice)

        #expect(prompt.input.text.tokens.asArray(Int.self) == [1, 2, 9, 3, 4, 7, 8])
    }

    @Test @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func afterASpliceABatchedRenderKeepsItsRank() {
        let prompt = Self.prompt(Self.renderTokens.expandedDimensions(axis: 0), after: Self.splice)

        #expect(prompt.input.text.tokens.shape == [1, 7])
    }

    @Test @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func afterASpliceTheEndOfTheTranscriptMovesOntoTheRepresentedTokens() {
        let prompt = Self.prompt(Self.renderTokens, after: Self.splice)

        #expect(prompt.checkpointBoundaries.transcriptEnd == 4)
    }
}

#endif

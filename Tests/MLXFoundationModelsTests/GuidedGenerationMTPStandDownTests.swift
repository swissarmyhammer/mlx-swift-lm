// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXGuidedGeneration
@testable import MLXLLM

/// A guided generation whose target stops giving drafter state stands down from speculation,
/// writes the plain output, and records the stand-down once on its diagnostic sink, which the
/// response metadata reads as its fallback reason.
@Suite(.serialized)
struct GuidedGenerationMTPStandDownTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The block sizes of the generations: the shipped single draft, and chains of three drafts.
    private static let blockSizes = [2, 4]

    /// The first verify of a round gives no hidden row: the generation drafts that round, stands
    /// down once, writes the plain output, and the stand-down reaches the sink as an
    /// `unavailableContinuation` fallback with its stated reason.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func aForwardWithoutHiddenStateRecordsOneStandDown(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture()
        let target = try HiddenWithholdingTarget(copying: fixture.target)
        let plain = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil)
        let speculative = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize,
            target: target)

        #expect(speculative.text == plain.text)
        #expect(speculative.tokenCount == plain.tokenCount)
        #expect(speculative.incomplete == plain.incomplete)
        #expect(speculative.sink.draftedTokenCount > 0)
        #expect(speculative.sink.speculationFallbacks == [.unavailableContinuation])
        #expect(
            speculative.sink.speculationFallbackReason
                == GuidedSpeculation.missingDrafterStateReason(of: target))
        #expect(plain.sink.speculationFallbacks.isEmpty)
    }
}

/// The fixture's target, whose verify forwards of several tokens give no drafter hidden state.
///
/// The prefill gives its state, so the speculation starts and drafts its first round; the verify
/// of that round asks to keep its snapshots, and gives no hidden row. A subclass of the target,
/// and not a wrapper, because the MTP drafter drafts for a Qwen 3.5 target only.
private final class HiddenWithholdingTarget: Qwen35TextModel {

    /// Makes a target with the configuration and the weights of `target`.
    ///
    /// - Parameter target: The target whose configuration and weights the copy takes.
    /// - Throws: The error of the weight update.
    init(copying target: Qwen35TextModel) throws {
        super.init(target.configuration)
        try update(parameters: target.parameters(), verify: .noUnusedKeys)
        eval(self)
    }

    /// Runs the target's forward, and drops its hidden state when the forward is a verify that
    /// keeps its snapshots.
    ///
    /// - Parameters:
    ///   - input: The tokens of the forward.
    ///   - cache: The cache of each layer.
    ///   - state: The model state of the forward.
    /// - Returns: The output of the target's forward.
    override func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let output = super.callAsFunction(input, cache: cache, state: state)
        guard state?[mtpVerifyKeepsSnapshotsKey] ?? false else { return output }
        var withheld = output.state
        withheld?[mtpLastHiddenStatesKey] = nil
        return LMOutput(logits: output.logits, state: withheld)
    }
}

#endif

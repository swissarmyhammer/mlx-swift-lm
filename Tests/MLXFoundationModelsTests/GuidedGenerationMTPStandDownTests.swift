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

    /// The schemas whose first round verifies several tokens, so the first one-token forward of
    /// a guarded generation is the probe after it.
    private static let schemasProbingFirst: [GuidedFixtureSchema] = [.activity, .threeActivities]

    /// Under a guard that probes plain decode after the first round, the probe's one-token
    /// forwards give no hidden row: the paused decision stands down, and the stand-down reaches
    /// the sink as an `unavailableContinuation` fallback with its stated reason, once.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Self.schemasProbingFirst, Self.blockSizes)
    func aPausedForwardWithoutHiddenStateRecordsTheStandDown(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let (plain, guarded, target) = try Self.generate(schema: schema, blockSize: blockSize)

        #expect(guarded.sink.draftedTokenCount > 0)
        #expect(guarded.sink.adaptiveProbeCount > 0)
        Self.expectOneStandDown(guarded, after: plain, target: target)
    }

    /// Whichever forward gives no hidden row first, a round's or a paused decision's, the
    /// generation stands down once and writes the plain output.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func aForwardWithoutHiddenStateRecordsOneStandDown(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let (plain, guarded, target) = try Self.generate(schema: schema, blockSize: blockSize)

        Self.expectOneStandDown(guarded, after: plain, target: target)
    }

    /// Generates `schema` plainly on the fixture's target, and with MTP under a guard that
    /// loses every round on a target whose one-token forwards give no hidden row.
    private static func generate(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws -> (plain: GuidedFixtureRun, guarded: GuidedFixtureRun, target: Qwen35TextModel) {
        let fixture = try Qwen35MTPFixture()
        let target = try HiddenWithholdingTarget(copying: fixture.target)
        let plain = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: prompt, fixture: fixture, blockSize: nil)
        let guarded = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: prompt, fixture: fixture, blockSize: blockSize,
            target: target, adaptivePolicy: .losingEveryRound)
        return (plain, guarded, target)
    }

    /// Expects `guarded` to write the output of `plain`, and to record one stand-down of
    /// `target` on its sink.
    private static func expectOneStandDown(
        _ guarded: GuidedFixtureRun, after plain: GuidedFixtureRun, target: Qwen35TextModel
    ) {
        #expect(guarded.text == plain.text)
        #expect(guarded.tokenCount == plain.tokenCount)
        #expect(guarded.incomplete == plain.incomplete)
        #expect(guarded.sink.speculationFallbacks == [.unavailableContinuation])
        #expect(
            guarded.sink.speculationFallbackReason
                == GuidedSpeculation.missingDrafterStateReason(of: target))
        #expect(plain.sink.speculationFallbacks.isEmpty)
    }
}

/// The fixture's target, whose one-token forwards give no drafter hidden state.
///
/// A paused decision feeds one token per forward, and so does a round whose drafts force more
/// tokens than one verify holds; the prefill and the other verifies feed several tokens and keep
/// their state. A subclass of the target, and not a wrapper, because the MTP drafter drafts for
/// a Qwen 3.5 target only.
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

    /// Runs the target's forward, and drops its hidden state when `input` holds one token.
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
        guard input.tokens.size == 1 else { return output }
        var withheld = output.state
        withheld?[mtpLastHiddenStatesKey] = nil
        return LMOutput(logits: output.logits, state: withheld)
    }
}

#endif

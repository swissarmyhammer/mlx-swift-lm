// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXGuidedGeneration
@testable import MLXLLM
@testable import MLXLMCommon

/// Guided generation on the target the FoundationModels router loads: the vision-language Qwen 3.5,
/// whose `prepare` evaluates the whole prompt itself and returns logits. Each prompt speculates
/// and writes the plain output.
///
/// The fixtures are tiny models with seeded random weights. A real model's turn at 8k tokens of
/// context is `MTPGuidedRouterPathTests` in the integration project.
@Suite(.serialized)
struct GuidedGenerationMTPRouterPathTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The block sizes of the generations: the shipped single draft, and chains of three drafts.
    private static let blockSizes = [2, 4]

    /// A prefill step shorter than ``prompt``, which takes the target's windowed forward.
    private static let shortPrefillStep = 3

    /// One generation without speculation and one with it, of one prompt on one target.
    private struct Comparison {
        let plain: GuidedFixtureRun
        let speculative: GuidedFixtureRun

        /// Runs `schema` after `prompt` on `vision`, plainly and with its drafter.
        init(
            schema: String, blockSize: Int, vision: Qwen35VLMMTPFixture,
            prompt: [Int] = GuidedGenerationMTPRouterPathTests.prompt,
            prefill: PrefillParameters = .init(stepSize: PrefillParameters.defaultStepSize),
            image: LMInput.ProcessedImage? = nil
        ) throws {
            let fixture = try Qwen35MTPFixture()
            plain = try GuidedFixtureRun.generate(
                schema: schema, prompt: prompt, fixture: fixture, blockSize: nil,
                target: vision.target, prefill: prefill, image: image)
            speculative = try GuidedFixtureRun.generate(
                schema: schema, prompt: prompt, fixture: fixture, blockSize: blockSize,
                drafter: vision.drafter, target: vision.target, prefill: prefill, image: image)
        }

        /// Whether the speculative generation drafted, wrote the plain output and fell back
        /// from nothing.
        var speculatedToThePlainOutput: Bool {
            !plain.text.isEmpty && speculative.text == plain.text
                && speculative.tokenCount == plain.tokenCount
                && speculative.incomplete == plain.incomplete
                && speculative.sink.draftedTokenCount > 0
                && speculative.sink.speculationFallbacks.isEmpty
        }
    }

    /// A prompt within one prefill step takes the target's single forward.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func aPromptOfOneForwardSpeculatesToThePlainOutput(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let comparison = try Comparison(
            schema: schema.json, blockSize: blockSize, vision: try Qwen35VLMMTPFixture())

        #expect(comparison.speculatedToThePlainOutput)
    }

    /// A prompt over the prefill step takes the target's windowed forward, whose chunks feed the
    /// drafter and whose last forward leaves the final segment.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func aChunkedPromptSpeculatesToThePlainOutput(
        schema: GuidedFixtureSchema, blockSize: Int
    ) throws {
        let comparison = try Comparison(
            schema: schema.json, blockSize: blockSize, vision: try Qwen35VLMMTPFixture(),
            prefill: PrefillParameters(stepSize: Self.shortPrefillStep))

        #expect(comparison.speculatedToThePlainOutput)
    }

    /// An image in the prompt shifts the target's positions, which the drafter follows.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Self.blockSizes)
    func anImagePromptSpeculatesToThePlainOutput(blockSize: Int) throws {
        let grid = Qwen35VLMMTPFixture.imageGrid
        let imageTokens = Qwen35VLMMTPFixture.imageTokens
        let placeholders = [Int](repeating: imageTokens.image, count: grid.t * grid.h * grid.w)

        let comparison = try Comparison(
            schema: GuidedFixtureSchema.activity.json, blockSize: blockSize,
            vision: try Qwen35VLMMTPFixture(readsImages: true),
            prompt: Self.prompt + [imageTokens.visionStart] + placeholders,
            image: Qwen35VLMMTPFixture.makeImage())

        #expect(comparison.speculatedToThePlainOutput)
    }

    // MARK: - A turn that cannot speculate says why

    /// One generation of the fixture's schema with `drafter` on `target`, and the plain output.
    private static func fallbackRun(
        drafter: (any MTPDrafterModel)? = nil, target: Qwen35TextModel? = nil
    ) throws -> (plain: GuidedFixtureRun, run: GuidedFixtureRun) {
        let fixture = try Qwen35MTPFixture()
        let schema = GuidedFixtureSchema.activity.json
        return (
            try GuidedFixtureRun.generate(
                schema: schema, prompt: Self.prompt, fixture: fixture, blockSize: nil),
            try GuidedFixtureRun.generate(
                schema: schema, prompt: Self.prompt, fixture: fixture,
                blockSize: Qwen35MTPFixture.blockSize, drafter: drafter, target: target)
        )
    }

    /// A drafter that cannot draft through a logit processor cannot draft under a grammar.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aDrafterWithoutGuidedDraftsRecordsItsFallback() throws {
        let (plain, run) = try Self.fallbackRun(drafter: StatelessMTPDrafter())

        #expect(run.text == plain.text)
        #expect(run.sink.draftedTokenCount == 0)
        #expect(run.sink.speculationFallbacks == [.incompatibleDrafter])
        #expect(run.sink.speculationFallbackReason?.contains("StatelessMTPDrafter") == true)
    }

    /// A target whose cache cannot rewind a verify from the start decodes plainly.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTargetThatCannotVerifyInPlaceRecordsItsFallback() throws {
        let fixture = try Qwen35MTPFixture()
        let target = try RefusingTarget(copying: fixture.target, refusal: .rewind(fromOffset: 0))

        let (plain, run) = try Self.fallbackRun(target: target)

        #expect(run.text == plain.text)
        #expect(run.sink.draftedTokenCount == 0)
        #expect(run.sink.speculationFallbacks == [.unsupportedCache])
        #expect(run.sink.speculationFallbackReason?.contains("RefusingTarget") == true)
    }

    /// A prefill that gives no hidden rows leaves the drafter nothing to start from.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aPrefillWithoutDrafterStateRecordsItsFallback() throws {
        let fixture = try Qwen35MTPFixture()
        let target = try RefusingTarget(copying: fixture.target, refusal: .drafterState)

        let (plain, run) = try Self.fallbackRun(target: target)

        #expect(run.text == plain.text)
        #expect(run.sink.draftedTokenCount == 0)
        #expect(run.sink.speculationFallbacks == [.unavailableContinuation])
        #expect(run.sink.speculationFallbackReason?.contains("RefusingTarget") == true)
    }

    /// A cache that stops rewinding in the middle of a generation hands the rest of it to plain
    /// decoding, and says so once.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aCacheThatStopsRewindingRecordsOneFallback() throws {
        let fixture = try Qwen35MTPFixture()
        let target = try RefusingTarget(
            copying: fixture.target, refusal: .rewind(fromOffset: Self.prompt.count + 1))

        let (plain, run) = try Self.fallbackRun(target: target)

        #expect(run.text == plain.text)
        #expect(run.sink.draftedTokenCount > 0)
        #expect(run.sink.speculationFallbacks == [.unavailableContinuation])
        #expect(run.sink.speculationFallbackReason?.contains("RefusingTarget") == true)
    }
}

/// An attention cache that stops trimming once it holds `refusedOffset` tokens.
private final class RewindRefusingCache: KVCacheSimple {

    /// The offset from which the cache refuses a trim.
    private let refusedOffset: Int

    /// Makes an empty cache.
    ///
    /// - Parameter refusedOffset: The offset from which the cache refuses a trim.
    init(refusedOffset: Int) {
        self.refusedOffset = refusedOffset
        super.init()
    }

    override var isTrimmable: Bool { offset < refusedOffset }
}

/// The fixture's target with one thing refused. A subclass of the target, and not a wrapper,
/// because the MTP drafter drafts for a Qwen 3.5 target only.
private final class RefusingTarget: Qwen35TextModel {

    /// What the target refuses a generation.
    enum Refusal {

        /// Every forward gives no drafter hidden state.
        case drafterState

        /// The attention cache stops rewinding at `fromOffset`.
        case rewind(fromOffset: Int)
    }

    /// What the target refuses.
    private let refusal: Refusal

    /// Makes a target with the configuration and the weights of `target`.
    ///
    /// - Parameters:
    ///   - target: The target whose configuration and weights the copy takes.
    ///   - refusal: What the target refuses.
    /// - Throws: The error of the weight update.
    init(copying target: Qwen35TextModel, refusal: Refusal) throws {
        self.refusal = refusal
        super.init(target.configuration)
        try update(parameters: target.parameters(), verify: .noUnusedKeys)
        eval(self)
    }

    /// The cache of the target, with its attention layers refusing a trim as ``refusal`` says.
    ///
    /// - Parameter parameters: The generation parameters.
    /// - Returns: The cache of each layer.
    /// - Throws: The error of the target's cache.
    override func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let cache = try super.newCache(parameters: parameters)
        guard case .rewind(let refusedOffset) = refusal else { return cache }
        return cache.map {
            $0 is MambaCache ? $0 : RewindRefusingCache(refusedOffset: refusedOffset)
        }
    }

    /// Runs the target's forward, and drops its hidden state when the target refuses it.
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
        guard case .drafterState = refusal else { return output }
        var withheld = output.state
        withheld?[mtpLastHiddenStatesKey] = nil
        return LMOutput(logits: output.logits, state: withheld)
    }
}

#endif

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

/// A guided generation hands over the caches of its target when its decoding ends, and those
/// caches hold the prompt and the tokens plain guided decoding fed: no draft a speculative round
/// did not keep.
///
/// Each test runs `GuidedGenerationLoop` on the target of the shared Qwen 3.5 fixture.
@Suite(.serialized)
struct GuidedGenerationCacheHandOffTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The block sizes of the generations: plain decoding, one draft, and chains of three
    /// drafts.
    private static let blockSizes: [Int?] = [nil, 2, 4]

    /// The block sizes of the speculative generations: one draft, and chains of three drafts.
    private static let speculativeBlockSizes = blockSizes.compactMap { $0 }

    /// The token budgets of the generations: room for the whole output, and budgets that end the
    /// outputs inside a speculative round, some of them inside the tokens the grammar forced
    /// after a draft.
    private static let tokenBudgets = [GuidedFixtureRun.maxTokens, 6, 10, 14]

    /// The largest difference a drafter continuation may show from the one a cold prefill of
    /// the same tokens builds: a verify forward and a prefill forward of the same positions round
    /// their float32 sums in different orders.
    private static let coldPrefillTolerance: Float = 1e-3

    /// The MTP drafter continuation a cold prefill of `tokens` feeds the drafter of `fixture`,
    /// at the end of `tokens`.
    ///
    /// - Parameters:
    ///   - tokens: The tokens the caches of a generation hold.
    ///   - fixture: The fixture whose target prefills, and whose drafter the prefill feeds.
    /// - Returns: The continuation.
    /// - Throws: The error of the target, or of a missing feed or continuation.
    private static func coldContinuation(
        over tokens: [Int], fixture: Qwen35MTPFixture
    ) throws -> MTPDrafterContinuation {
        let parameters = GenerateParameters()
        let speculation = MTPSpeculation(
            drafter: fixture.drafter, blockSize: speculativeBlockSizes[0],
            minimumDraftProbability: 0)
        var target = try GuidedTarget(model: fixture.target, parameters: parameters)
        _ = GuidedSpeculation.drafter(of: speculation, for: &target)
        let feed = try #require(
            speculation.prefillFeed(
                target: fixture.target, parameters: parameters, cachePosition: 0,
                continuation: nil))
        let prefilled = try target.prefill(
            LMInput(tokens: MLXArray(tokens.map(Int32.init))), parameters: PrefillParameters(),
            drafterFeed: feed)
        let segmentTokens = try #require(prefilled.segmentTokens)
        let segmentHidden = try #require(prefilled.segmentHidden)
        return try #require(
            feed.promptEndContinuation(
                finalSegment: segmentTokens, segmentHidden: segmentHidden,
                positionDeltas: prefilled.positionDeltas))
    }

    /// The largest difference between the arrays of two drafter continuations: their drafter
    /// caches and their boundary hidden rows.
    ///
    /// - Returns: The difference, or infinity when the two hold different positions or arrays
    ///   of different shapes.
    private static func largestDifference(
        _ lhs: MTPDrafterContinuation, _ rhs: MTPDrafterContinuation
    ) -> Float {
        let lhsArrays = lhs.state.cache.flatMap(\.state) + [lhs.targetBoundaryHidden]
        let rhsArrays = rhs.state.cache.flatMap(\.state) + [rhs.targetBoundaryHidden]
        guard lhs.state.cacheStartPosition == rhs.state.cacheStartPosition,
            lhs.state.nextPosition == rhs.state.nextPosition,
            lhsArrays.map(\.shape) == rhsArrays.map(\.shape)
        else { return .infinity }
        return zip(lhsArrays, rhsArrays).map { abs($0 - $1).max().item(Float.self) }.max() ?? 0
    }

    /// The positions the first cache of `run` holds when its decoding ended.
    private static func offset(of run: GuidedFixtureRun) throws -> Int {
        let generated = try #require(run.generated)
        return try #require(generated.cache.first).offset
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.tokenBudgets)
    func aSchemaHandsOverCachesThatHoldThePromptAndACountedPrefix(
        schema: GuidedFixtureSchema, tokenBudget: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)

        let excess = try Self.blockSizes.map { blockSize in
            let run = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize,
                tokenBudget: tokenBudget)
            let generated = try #require(run.generated)
            return try Self.offset(of: run) - Self.prompt.count - generated.countedTokens.count
        }

        #expect(excess.allSatisfy { $0 <= 0 })
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.tokenBudgets)
    func aSpeculativeSchemaHandsOverADrafterContinuationAlignedWithItsCaches(
        schema: GuidedFixtureSchema, tokenBudget: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)

        let aligned = try Self.speculativeBlockSizes.map { blockSize in
            try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize,
                tokenBudget: tokenBudget
            ).handsOverAnAlignedDrafterContinuation()
        }

        #expect(aligned.allSatisfy { $0 })
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.tokenBudgets)
    func aSpeculativeSchemaHandsOverTheDrafterContinuationAColdPrefillOfItsCachesBuilds(
        schema: GuidedFixtureSchema, tokenBudget: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)

        let differences = try Self.speculativeBlockSizes.map { blockSize in
            let run = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize,
                tokenBudget: tokenBudget)
            let generated = try #require(run.generated)
            let handedOver = try #require(generated.drafterContinuation)
            let cachedTokens =
                Self.prompt
                + generated.countedTokens.prefix(try Self.offset(of: run) - Self.prompt.count)
            return Self.largestDifference(
                handedOver, try Self.coldContinuation(over: cachedTokens, fixture: fixture))
        }

        #expect(differences.allSatisfy { $0 <= Self.coldPrefillTolerance })
    }
}

extension GuidedFixtureRun {

    /// Whether the generation handed over an MTP drafter continuation that stands where the
    /// caches it handed over end, so a next generation over those caches resumes it.
    ///
    /// - Returns: `true` when the continuation is aligned with the first cache.
    /// - Throws: The error of a missing hand-off or cache.
    func handsOverAnAlignedDrafterContinuation() throws -> Bool {
        let generated = try #require(generated)
        let offset = try #require(generated.cache.first).offset
        let continuation = try #require(generated.drafterContinuation)
        return continuation.isAligned(
            withMainTokenCount: offset, drafterID: continuation.drafterID)
    }
}

#endif

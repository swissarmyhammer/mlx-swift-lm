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

    /// The token budgets of the generations: room for the whole output, and budgets that end the
    /// outputs inside a speculative round, some of them inside the tokens the grammar forced
    /// after a draft.
    private static let tokenBudgets = [GuidedFixtureRun.maxTokens, 6, 10, 14]

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
}

#endif

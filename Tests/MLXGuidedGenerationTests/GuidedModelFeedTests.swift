// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXGuidedGeneration

/// The tokens a guided decision feeds the model: the decided token, then the tokens the grammar
/// forced after it, so the cache holds every token the generation emits.
@Suite
struct GuidedModelFeedTests {

    /// A decided token.
    private static let decided: Int32 = 5

    /// Tokens the grammar forced after ``decided``.
    private static let forced: [Int32] = [6, 7, 8]

    @Test
    func aDecisionWithoutForcedTokensFeedsTheDecidedToken() {
        #expect(GuidedGenerationLoop.modelFeed(sampled: Self.decided, fastForward: []) == [5])
    }

    @Test
    func aDecisionWithForcedTokensFeedsTheDecidedTokenBeforeThem() {
        #expect(
            GuidedGenerationLoop.modelFeed(sampled: Self.decided, fastForward: Self.forced)
                == [5, 6, 7, 8])
    }
}

// Copyright © 2026 Apple Inc.
//
// What one speculative verify of a Qwen 3.5 model leaves, on its compiled schedule or on the
// general path of its layers, shared by the verify tests of the text and the vision-language
// models, so both compare the same things.

import MLX
import MLXLMCommon
import Testing

/// What one verify leaves: its logits and hidden rows, the cache after its rollback, and the
/// snapshot slots each GDN cache kept before the rollback.
struct CompiledVerifyOutcome {

    /// The logits of each token of the verify.
    let logits: MLXArray

    /// The hidden rows of each token of the verify.
    let hidden: MLXArray

    /// The states of the caches after the rollback.
    let states: [MLXArray]

    /// The offset of each cache after the rollback.
    let offsets: [Int]

    /// The snapshot slots each GDN cache held before the rollback.
    let slotCounts: [Int]

    /// Rolls `cache` back by `rollback` tokens, as a rejected round does, and evaluates the
    /// outcome.
    ///
    /// - Parameters:
    ///   - logits: The logits of the verify.
    ///   - hidden: The hidden rows of the verify.
    ///   - cache: The cache of each layer after the verify.
    ///   - rollback: The tokens of the verify to roll back.
    init(logits: MLXArray, hidden: MLXArray, cache: [KVCache], rollingBack rollback: Int) {
        self.logits = logits
        self.hidden = hidden
        self.slotCounts = cache.compactMap { ($0 as? MambaCache)?.speculativeSlotCount }

        if rollback > 0 {
            for entry in cache {
                if let mamba = entry as? MambaCache {
                    #expect(mamba.selectSpeculativeSlot(rollingBack: rollback))
                } else {
                    #expect(entry.trim(rollback) == rollback)
                }
            }
        }
        self.states = cache.flatMap(\.state)
        self.offsets = cache.map(\.offset)
        eval(logits, hidden, states)
    }

    /// Holds that this compiled verify keeps a slot for each of its tokens and stays within
    /// `tolerance` of the `general` path.
    ///
    /// - Parameters:
    ///   - general: The outcome of the same verify on the general path.
    ///   - tolerance: The largest relative gap of any array.
    ///   - tokenCount: The tokens of the verify.
    ///   - rollback: The tokens the verify rolled back.
    ///   - sourceLocation: Where a failed expectation is reported.
    /// - Throws: When the two caches hold other numbers of states.
    func expectMatches(
        _ general: Self, tolerance: Float, tokenCount: Int, rollback: Int,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let round = "\(tokenCount) tokens, rollback \(rollback)"
        let expectWithin = { (got: MLXArray, expected: MLXArray, what: String) in
            #expect(got.shape == expected.shape, "\(what)", sourceLocation: sourceLocation)
            let gap = AttentionKernelTolerance.worstGap(got, expected)
            #expect(gap <= tolerance, "\(what): worst gap \(gap)", sourceLocation: sourceLocation)
        }

        #expect(
            slotCounts.allSatisfy { $0 == tokenCount }, "\(round)", sourceLocation: sourceLocation)
        #expect(slotCounts == general.slotCounts, "\(round)", sourceLocation: sourceLocation)
        #expect(offsets == general.offsets, "\(round)", sourceLocation: sourceLocation)
        expectWithin(logits, general.logits, "\(round) logits")
        expectWithin(hidden, general.hidden, "\(round) hidden rows")
        try #require(states.count == general.states.count, sourceLocation: sourceLocation)
        for (index, (got, expected)) in zip(states, general.states).enumerated() {
            expectWithin(got, expected, "\(round) cache state \(index)")
        }
    }
}

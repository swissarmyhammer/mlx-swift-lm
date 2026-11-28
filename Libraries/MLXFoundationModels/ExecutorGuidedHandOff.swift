// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXGuidedGeneration
import MLXLMCommon

/// Holds what the guided loop of one executor pass leaves -- the model state of its prefill, and
/// the tokens it counted with the caches of its target and the MTP drafter continuation where
/// they end -- until the pass commits its cache.
///
/// The loop runs on the respond task, which then commits, thus one task at a time reads and
/// writes this box.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class ExecutorGuidedHandOff {

    /// The slot that takes the checkpoint at the end of the prompt, and the cache of the pass.
    private let promptCache: ExecutorPromptCacheSlot

    /// Records the model state the prefill left, and reports the end of the prefill to the slot.
    private let prefill: ExecutorPrefillRecorder

    /// The tokens the loop counted, the caches of its target and the drafter continuation, or
    /// nil until its decoding ended.
    private var generated: GuidedGeneratedCache?

    /// Creates a hand-off that holds nothing yet, for a pass of `promptCache`.
    ///
    /// - Parameter promptCache: The slot of the pass.
    init(promptCache: ExecutorPromptCacheSlot) {
        self.promptCache = promptCache
        self.prefill = ExecutorPrefillRecorder(promptCache: promptCache)
    }

    /// The hand-off the guided loop gives what it leaves to.
    ///
    /// The end of the prefill goes to the slot at once, which takes a checkpoint there when the
    /// plan of the pass asked for one at the end of the prompt. That checkpoint keeps the MTP
    /// drafter continuation the loop reads at the end of the prompt, when it speculates with a
    /// drafter of this process.
    var cacheHandOff: GuidedCacheHandOff {
        GuidedCacheHandOff(
            prefillDidEnd: { [prefill] state, drafterState in
                prefill.record(state: state, drafterState: drafterState)
            },
            generationDidEnd: { [self] generated in
                self.generated = generated
            })
    }

    /// Commits the cache of the pass to the slot: the caches of the target, whose ledger is the
    /// tokens `plan` represents and the tokens the loop counted that those caches hold.
    ///
    /// The entry keeps the MTP drafter continuation the loop finalized when it stands at the end
    /// of that ledger, thus the drafter of the next pass resumes it there. A loop that leaves
    /// none -- a plain or cluster pass, or one whose speculation stood down -- leaves the drafter
    /// of the next pass to start cold where the caches end, and its output does not change.
    ///
    /// - Parameter plan: The plan the pass ran, or nil when the pass carried none.
    func commit(_ plan: ExecutorPromptCachePlan?) {
        let state: LMOutput.State? =
            switch prefill.outcome {
            // A loop that stopped before its prefill ended left no model state.
            case .pending: nil
            case .ended(let state): state
            }
        promptCache.commit(
            plan?.holding(generationCache: generated?.cache),
            generatedTokens: generated?.countedTokens ?? [], state: state,
            drafterState: generated?.drafterContinuation)
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension ExecutorPromptCachePlan {

    /// The cache a guided loop continues for this plan: its caches, the model state that goes
    /// with them, and the MTP drafter continuation that stands where they end.
    var guidedCarriedCache: GuidedCarriedCache {
        GuidedCarriedCache(cache: caches, state: state, drafterContinuation: drafterState)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration

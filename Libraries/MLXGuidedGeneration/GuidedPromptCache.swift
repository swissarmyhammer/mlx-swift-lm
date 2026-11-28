// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// The cache an earlier generation left, which a guided generation continues: the caches of the
/// target, the model state that goes with them, and the MTP drafter continuation that stands
/// where they end.
///
/// A prompt cache hands it over when the prompt of the generation starts with tokens the caches
/// already hold. The generation then feeds only the rest of the prompt.
package struct GuidedCarriedCache {

    /// The caches of the target, one for each layer.
    package let cache: [KVCache]

    /// The model state the next forward of the target carries, or `nil` for a model that keeps
    /// none.
    package let state: LMOutput.State?

    /// The MTP drafter continuation aligned with ``cache``, or `nil` for a drafter that starts
    /// cold where the caches end.
    package let drafterContinuation: MTPDrafterContinuation?

    /// Creates the carried cache of `cache`.
    ///
    /// - Parameters:
    ///   - cache: The caches of the target.
    ///   - state: The model state that goes with them.
    ///   - drafterContinuation: The MTP drafter continuation aligned with them, or `nil`.
    package init(
        cache: [KVCache], state: LMOutput.State?, drafterContinuation: MTPDrafterContinuation?
    ) {
        self.cache = cache
        self.state = state
        self.drafterContinuation = drafterContinuation
    }
}

/// What a guided generation leaves when its decoding ends: the tokens it counted, the caches of
/// its target, and the MTP drafter continuation that stands where those caches end.
///
/// The caches hold the prompt and a prefix of ``countedTokens``: every token a forward fed, and
/// none of the drafts a speculative round did not keep. A token the generation counted but did
/// not feed -- the last decision, which no forward needed -- is past the end of the caches. A
/// prompt cache thus names what the caches hold with the prompt and the first
/// `offset - prompt length` of ``countedTokens``.
package struct GuidedGeneratedCache {

    /// Each token the generation counted, sampled and fast-forwarded, in order.
    package let countedTokens: [Int]

    /// The caches of the target, which a KV-cache configuration can have replaced with
    /// compressed caches while the generation ran.
    package let cache: [KVCache]

    /// The MTP drafter continuation that stands where ``cache`` ends, finalized as the plain MTP
    /// iterator finalizes its own, or `nil` when the generation did not speculate with a drafter
    /// of this process to its end. Its arrays are evaluated.
    package let drafterContinuation: MTPDrafterContinuation?
}

/// Receives what a guided generation leaves for the prompt cache of its caller.
///
/// A generation that throws before its decoding ends -- a cancellation, the failure of a cluster
/// forward, an error of the matcher -- calls ``generationDidEnd`` never: its caches can hold
/// positions no counted token names.
package struct GuidedCacheHandOff {

    /// Receives the model state the prefill left, and a reader of the MTP drafter continuation
    /// at the end of the prompt, once the prefill ended and before the first decision feeds the
    /// target. The caches then stand at the end of the prompt.
    ///
    /// The reader gives `nil` when the generation does not feed a drafter of this process. It
    /// copies the drafter caches, thus a receiver reads it only when it keeps the continuation,
    /// and only before it returns.
    package let prefillDidEnd: (LMOutput.State?, () -> MTPDrafterContinuation?) -> Void

    /// Receives the counted tokens, the caches and the MTP drafter continuation once the
    /// decoding ended, before the generation reports how it ended. A generation that runs out of
    /// budget hands them over too.
    package let generationDidEnd: (GuidedGeneratedCache) -> Void

    /// Creates a hand-off that gives the end of the prefill to `prefillDidEnd`, and the end of
    /// the decoding to `generationDidEnd`.
    ///
    /// - Parameters:
    ///   - prefillDidEnd: Receives the model state of the prefill and the reader of the drafter
    ///     continuation at the end of the prompt; the default drops both.
    ///   - generationDidEnd: Receives the counted tokens, the caches and the drafter
    ///     continuation; the default drops them.
    package init(
        prefillDidEnd: @escaping (LMOutput.State?, () -> MTPDrafterContinuation?) -> Void = {
            _, _ in
        },
        generationDidEnd: @escaping (GuidedGeneratedCache) -> Void = { _ in }
    ) {
        self.prefillDidEnd = prefillDidEnd
        self.generationDidEnd = generationDidEnd
    }
}

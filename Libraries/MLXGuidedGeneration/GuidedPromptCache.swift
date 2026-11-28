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

/// What a guided generation leaves when its decoding ends: the tokens it counted, and the caches
/// of its target.
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
}

/// Receives what a guided generation leaves for the prompt cache of its caller.
///
/// A generation that throws before its decoding ends -- a cancellation, the failure of a cluster
/// forward, an error of the matcher -- calls ``generationDidEnd`` never: its caches can hold
/// positions no counted token names.
package struct GuidedCacheHandOff {

    /// Receives the model state the prefill left, once the prefill ended and before the first
    /// decision feeds the target. The caches then stand at the end of the prompt.
    package let prefillDidEnd: (LMOutput.State?) -> Void

    /// Receives the counted tokens and the caches once the decoding ended, before the generation
    /// reports how it ended. A generation that runs out of budget hands them over too.
    package let generationDidEnd: (GuidedGeneratedCache) -> Void

    /// Creates a hand-off that gives the end of the prefill to `prefillDidEnd`, and the end of
    /// the decoding to `generationDidEnd`.
    ///
    /// - Parameters:
    ///   - prefillDidEnd: Receives the model state of the prefill; the default drops it.
    ///   - generationDidEnd: Receives the counted tokens and the caches; the default drops them.
    package init(
        prefillDidEnd: @escaping (LMOutput.State?) -> Void = { _ in },
        generationDidEnd: @escaping (GuidedGeneratedCache) -> Void = { _ in }
    ) {
        self.prefillDidEnd = prefillDidEnd
        self.generationDidEnd = generationDidEnd
    }
}

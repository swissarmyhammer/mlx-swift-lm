// Copyright © 2026 Apple Inc.

import Foundation

/// The MTP drafter a package generation function speculates with, and the shape of its rounds.
///
/// The FoundationModels executor takes the drafter model out of its ``MTPDrafterContainer`` one
/// time for a response and hands it to each generation of that response inside the target's
/// `ModelContainer.perform`, thus no generation nests a second container lock.
package struct MTPSpeculation {

    /// The drafter. Its instances hold no per-stream state, so generations may share one.
    package let drafter: any MTPDrafterModel

    /// The tokens of each round: up to `blockSize - 1` drafts and the bonus token.
    package let blockSize: Int

    /// The drafter top-1 probability below which a chaining drafter stops a round's chain.
    package let minimumDraftProbability: Float

    /// How each generation guards against decoding slower than plain decode.
    package let adaptivePolicy: MTPAdaptivePolicy

    /// Creates the speculation of `drafter`.
    ///
    /// - Parameters:
    ///   - drafter: the drafter.
    ///   - blockSize: the tokens of each round.
    ///   - minimumDraftProbability: the confidence floor of a chaining drafter.
    ///   - adaptivePolicy: how each generation guards against decoding slower than plain decode.
    package init(
        drafter: any MTPDrafterModel, blockSize: Int, minimumDraftProbability: Float,
        adaptivePolicy: MTPAdaptivePolicy = .default
    ) {
        self.drafter = drafter
        self.blockSize = blockSize
        self.minimumDraftProbability = minimumDraftProbability
        self.adaptivePolicy = adaptivePolicy
    }

    /// The drafter when it can resume a drafter state across generations, or `nil`.
    package var resumableDrafter: (any ResumableMTPDrafterModel)? {
        drafter as? any ResumableMTPDrafterModel
    }

    /// A feed that takes the prompt tokens a caller prefills into `target` before the
    /// generation into the drafter too, so the drafter state stands where the main cache stands.
    ///
    /// - Parameters:
    ///   - target: the target model.
    ///   - parameters: the generation parameters, which give a new drafter state.
    ///   - cachePosition: the position of the main cache before the prefill.
    ///   - continuation: the drafter continuation carried with the main cache, or `nil`. The
    ///     feed extends it in place when it is aligned with `cachePosition`, and else starts a
    ///     new drafter state there.
    /// - Returns: the feed, or `nil` when the drafter cannot draft for `target` or cannot take a
    ///   prompt chunk by chunk.
    package func prefillFeed(
        target: any LanguageModel, parameters: GenerateParameters, cachePosition: Int,
        continuation: MTPDrafterContinuation?
    ) -> MTPDrafterPrefillFeed? {
        guard drafter.isCompatible(with: target) else { return nil }
        return MTPDrafterPrefillFeed.starting(
            drafter: drafter, target: target, parameters: parameters,
            cachePosition: cachePosition, continuation: continuation)
    }

    /// The drafter continuation at the end of the prompt of `iterator`, which a prompt cache
    /// checkpoint there keeps.
    ///
    /// - Parameter iterator: an iterator that ``makeTokenIterator(speculation:input:cache:state:mtpContinuation:parameters:model:components:)``
    ///   made, before its first token is read.
    /// - Returns: the continuation, or `nil` when `iterator` does not speculate with a resumable
    ///   drafter.
    package static func promptBoundaryContinuation(
        of iterator: any TokenIteratorProtocol
    ) -> MTPDrafterContinuation? {
        (iterator as? MTPSpeculativeTokenIterator)?.promptBoundaryContinuation
    }

    /// The token iterator of one generation: an ``MTPSpeculativeTokenIterator`` when
    /// `speculation` names a drafter that can draft for `model` and whose rounds the cache can
    /// stage, else a ``TokenIterator``.
    ///
    /// The MTP iterator decides before the prefill whether the stream speculates, thus an
    /// incompatible drafter or a cache that cannot stage a round leaves the cache untouched for
    /// the plain iterator. The plain iterator of an incompatible drafter carries its
    /// ``MTPDrafterFallback``, which the completion information reports and the process log
    /// states once.
    ///
    /// - Parameters:
    ///   - speculation: the speculation of the generation, or `nil` for plain decode.
    ///   - input: the prompt, or the suffix of the prompt that `cache` does not hold.
    ///   - cache: the cache of `model`, or `nil` for a new cache.
    ///   - state: the model state saved with `cache`.
    ///   - mtpContinuation: the drafter continuation saved with `cache`, or `nil`. The MTP
    ///     iterator resumes it when it is aligned with `cache` and was made by the drafter of
    ///     `speculation`, and else starts the drafter at the cached position.
    ///   - parameters: the generation parameters.
    ///   - model: the target model.
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``.
    /// - Returns: the iterator, after its prefill.
    /// - Throws: the error of the iterator.
    package static func makeTokenIterator(
        speculation: MTPSpeculation?, input: LMInput, cache: [KVCache]?, state: LMOutput.State?,
        mtpContinuation: MTPDrafterContinuation? = nil,
        parameters: GenerateParameters, model: any LanguageModel,
        components: GenerationComponents
    ) throws -> any TokenIteratorProtocol {
        let fallback = speculation.flatMap {
            MTPDrafterFallback.incompatibility(of: $0.drafter, with: model)
        }
        if let speculation, fallback == nil {
            do {
                return try MTPSpeculativeTokenIterator(
                    input: input, mainModel: model, drafter: speculation.drafter,
                    mainCache: cache, mainState: state, mtpContinuation: mtpContinuation,
                    parameters: parameters, blockSize: speculation.blockSize,
                    minimumDraftProbability: speculation.minimumDraftProbability,
                    adaptivePolicy: speculation.adaptivePolicy, components: components)
            } catch MTPInitializationError.unsupportedSpeculativeCache {
                // The probe throws before the prefill, thus the plain iterator below starts
                // from the same cache.
            }
        }
        var iterator = try TokenIterator(
            input: input, model: model, cache: cache, state: state, parameters: parameters,
            components: components)
        iterator.speculationFallback = fallback
        return iterator
    }
}

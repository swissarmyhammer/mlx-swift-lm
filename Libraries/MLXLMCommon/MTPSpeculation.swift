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

    /// Creates the speculation of `drafter`.
    ///
    /// - Parameters:
    ///   - drafter: the drafter.
    ///   - blockSize: the tokens of each round.
    ///   - minimumDraftProbability: the confidence floor of a chaining drafter.
    package init(drafter: any MTPDrafterModel, blockSize: Int, minimumDraftProbability: Float) {
        self.drafter = drafter
        self.blockSize = blockSize
        self.minimumDraftProbability = minimumDraftProbability
    }

    /// The token iterator of one generation: an ``MTPSpeculativeTokenIterator`` when
    /// `speculation` names a drafter that can draft for `model` and whose rounds the cache can
    /// stage, else a ``TokenIterator``.
    ///
    /// The MTP iterator decides before the prefill whether the stream speculates, thus an
    /// incompatible drafter or a cache that cannot stage a round leaves the cache untouched for
    /// the plain iterator.
    ///
    /// - Parameters:
    ///   - speculation: the speculation of the generation, or `nil` for plain decode.
    ///   - input: the prompt, or the suffix of the prompt that `cache` does not hold.
    ///   - cache: the cache of `model`, or `nil` for a new cache.
    ///   - state: the model state saved with `cache`.
    ///   - parameters: the generation parameters.
    ///   - model: the target model.
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``.
    /// - Returns: the iterator, after its prefill.
    /// - Throws: the error of the iterator.
    package static func makeTokenIterator(
        speculation: MTPSpeculation?, input: LMInput, cache: [KVCache]?, state: LMOutput.State?,
        parameters: GenerateParameters, model: any LanguageModel,
        components: GenerationComponents
    ) throws -> any TokenIteratorProtocol {
        if let speculation, speculation.drafter.isCompatible(with: model) {
            do {
                return try MTPSpeculativeTokenIterator(
                    input: input, mainModel: model, drafter: speculation.drafter,
                    mainCache: cache, mainState: state, parameters: parameters,
                    blockSize: speculation.blockSize,
                    minimumDraftProbability: speculation.minimumDraftProbability,
                    components: components)
            } catch MTPInitializationError.unsupportedSpeculativeCache {
                // The probe throws before the prefill, thus the plain iterator below starts
                // from the same cache.
            }
        }
        return try TokenIterator(
            input: input, model: model, cache: cache, state: state, parameters: parameters,
            components: components)
    }
}

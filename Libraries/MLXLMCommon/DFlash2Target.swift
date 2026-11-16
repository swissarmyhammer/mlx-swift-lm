// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// DFlash2 (https://inco.ai/blog/dflash2/, reference https://github.com/z-lab/dflash)
// drafts a block of tokens in one pass from the hidden states of some target
// layers, and the target verifies the block in one forward pass. The drafter
// runs as an MTP drafter: ``MTPSpeculativeTokenIterator`` drives its rounds,
// and the target writes the outputs of the layers the drafter names in
// ``MTPDrafterModel/targetLayerIds`` into ``mtpLastHiddenStatesKey``, side by
// side, when ``mtpTargetLayerIdsKey`` asks for them.

/// The target shape a DFlash2 drafter was distilled for, or the shape a target has.
public struct DFlash2Geometry: Equatable, Sendable, CustomStringConvertible {

    /// The decoder layers, or `nil` when a drafter checkpoint does not record them.
    public var layerCount: Int?

    /// The width of the hidden state.
    public var hiddenSize: Int

    /// The tokens of the vocabulary.
    public var vocabularySize: Int

    /// Makes a geometry.
    ///
    /// - Parameters:
    ///   - layerCount: The decoder layers, or `nil` when unknown.
    ///   - hiddenSize: The width of the hidden state.
    ///   - vocabularySize: The tokens of the vocabulary.
    public init(layerCount: Int?, hiddenSize: Int, vocabularySize: Int) {
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
    }

    /// The layers, the hidden width and the vocabulary, for an error message.
    public var description: String {
        let layers = layerCount.map(String.init) ?? "?"
        return "\(layers) layers, hidden \(hiddenSize), vocabulary \(vocabularySize)"
    }

    /// Whether a drafter distilled for this geometry, reading the outputs of `layerIds`, can
    /// draft for a target of `target` geometry.
    ///
    /// - Parameters:
    ///   - target: The geometry of the target.
    ///   - layerIds: The target layers the drafter reads.
    /// - Returns: `true` when the widths and the vocabularies agree, the depths agree where the
    ///   drafter records one, and the target has every layer the drafter reads.
    package func accepts(_ target: DFlash2Geometry, layerIds: [Int]) -> Bool {
        guard let targetLayers = target.layerCount else { return false }
        return hiddenSize == target.hiddenSize && vocabularySize == target.vocabularySize
            && (layerCount ?? targetLayers) == targetLayers
            && layerIds.allSatisfy { (0 ..< targetLayers).contains($0) }
    }
}

/// A target a DFlash2 drafter can speculate for.
///
/// The drafter borrows the target's token embedding and LM head on each proposal, and reads
/// the outputs of some target layers, which the target writes into ``mtpLastHiddenStatesKey``
/// side by side when the state of a forward carries ``mtpTargetLayerIdsKey``.
public protocol DFlash2TargetModel: LanguageModel {

    /// The shape of the target.
    var dflash2Geometry: DFlash2Geometry { get }

    /// The token embedding the drafter borrows.
    var dflash2Embedding: Embedding { get }

    /// The LM head the drafter borrows, or `nil` when the head is tied to the embedding.
    var dflash2Head: Linear? { get }
}

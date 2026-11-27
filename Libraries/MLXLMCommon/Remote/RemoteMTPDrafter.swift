// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The MTP drafter that rank 0 of a ``ClusterLanguageModel`` holds, as this process names it.
///
/// It holds no weights. Rank 0 runs the drafter beside its shard of the target, and a
/// generation of the cluster model drives each round with one ``RemoteModelOp/speculate(_:)``.
/// Pass ``ClusterLanguageModel/mtpDrafter`` as the MTP drafter of a generation of that model:
/// the drafter drafts for that model alone, and a generation of any other model refuses it.
///
/// The drafter keeps no state in this process: rank 0 keeps the drafter state beside each cache
/// on the worker, thus a generation hands back no drafter continuation.
public final class RemoteMTPDrafter: Module, MTPDrafterModel {

    /// What the worker reported of the drafter.
    public let capabilities: RemoteDrafterCapabilities

    /// The stream of requests of the model whose rank 0 holds the drafter.
    private let session: RemoteModelSession

    /// Makes the drafter the worker of `session` reported.
    ///
    /// - Parameters:
    ///   - capabilities: What the worker reported of the drafter.
    ///   - session: The stream of requests of the model.
    init(capabilities: RemoteDrafterCapabilities, session: RemoteModelSession) {
        self.capabilities = capabilities
        self.session = session
        super.init()
    }

    /// The largest verification block the drafter of rank 0 produces.
    public var maximumBlockSize: Int? { capabilities.maximumBlockSize }

    /// The block the drafter of rank 0 decodes fastest with.
    public var preferredBlockSize: Int? { capabilities.preferredBlockSize }

    /// False: rank 0 hands the drafter the hidden rows it reads.
    public var requiresSharedTargetKV: Bool { false }

    /// False: rank 0 feeds the drafter every prompt forward itself.
    public var requiresPromptPrefill: Bool { false }

    /// False: a sampled stream verifies the argmax drafts of rank 0 by sample-and-match.
    public var requiresGreedySampling: Bool { false }

    /// Never called: rank 0 drafts, and a generation of the cluster model drives its rounds.
    ///
    /// - Parameters:
    ///   - target: Unused.
    ///   - lastToken: Unused.
    ///   - lastHidden: Unused.
    ///   - sharedKV: Unused.
    ///   - positionDeltas: Unused.
    ///   - queryOffset: Unused.
    ///   - blockSize: Unused.
    ///   - sampler: Unused.
    /// - Returns: Never returns.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        preconditionFailure("A remote MTP drafter drafts on rank 0 of the worker of its model")
    }

    /// Throws unless `target` is the cluster model whose rank 0 holds the drafter.
    ///
    /// - Parameter target: The model to draft for.
    /// - Throws: ``MTPDrafterCompatibilityError/incompatibleTarget(drafter:expected:actual:)``
    ///   for any other model.
    public func validateCompatibility(with target: any LanguageModel) throws {
        try requireCompatibleTarget(
            target, expected: "the cluster model whose rank 0 holds the drafter")
    }

    /// Whether `target` is the cluster model whose rank 0 holds the drafter.
    ///
    /// - Parameter target: The model to draft for.
    /// - Returns: The answer.
    public func isCompatible(with target: any LanguageModel) -> Bool {
        (target as? ClusterLanguageModel)?.session === session
    }
}

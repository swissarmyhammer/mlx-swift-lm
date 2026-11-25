// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// Where the forward pass of an ``MLXLanguageModel`` runs. It is part of the identity of the
/// model in the shared caches, thus a local load and a clustered load of one checkpoint never
/// share a container, an executor or a prompt cache entry.
public enum ModelLoadIdentity: Hashable, Sendable {

    /// The forward pass runs in this process.
    case local

    /// The forward pass runs in the workers of a cluster, under the plan `planID`.
    case clustered(planID: String)

    /// The text between the checkpoint identity and the plan of a clustered load.
    private static let clusterSeparator = "#cluster="

    /// The identity of a load of `checkpoint`: `checkpoint` itself for a local load, thus the
    /// keys of local models do not change, else `checkpoint` with the plan after it.
    ///
    /// - Parameter checkpoint: The cache identity of the checkpoint.
    /// - Returns: The identity of the load.
    func cacheIdentity(of checkpoint: String) -> String {
        switch self {
        case .local:
            checkpoint
        case .clustered(let planID):
            checkpoint + Self.clusterSeparator + planID
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration

// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

// MARK: - Identity

/// Names the one session whose turns share a prompt cache.
struct ExecutorPromptCacheKey: Hashable, Sendable {

    /// The identifier of the model a cache belongs to.
    ///
    /// It is a type of its own, not a `String`, thus a session identifier
    /// passed where a model identifier belongs does not compile.
    struct ModelID: Hashable, Sendable {

        /// The identifier as the model configuration names it.
        let rawValue: String

        /// Creates the identifier of the model that `rawValue` names.
        ///
        /// - Parameter rawValue: the identifier as the model configuration
        ///   names it.
        init(_ rawValue: String) {
            self.rawValue = rawValue
        }
    }

    /// The identifier of the session a cache belongs to.
    ///
    /// It is a type of its own, not a `String`, thus a model identifier
    /// passed where a session identifier belongs does not compile.
    struct SessionID: Hashable, Sendable {

        /// The identifier as the request names it: the session of the request,
        /// or the identifier of the first entry of its transcript.
        let rawValue: String

        /// Creates the identifier of the session that `rawValue` names.
        ///
        /// - Parameter rawValue: the identifier as the request names it.
        init(_ rawValue: String) {
            self.rawValue = rawValue
        }
    }

    /// The model the cache belongs to, thus two models never share one cache.
    let modelID: ModelID

    /// The session the cache belongs to.
    let sessionID: SessionID

    /// The KV-cache configuration the caches were built under, or nil for full
    /// precision.
    ///
    /// Generation converts the caches it is handed to the configuration of the
    /// pass, thus a full-precision cache would be served to an 8-bit turn, or a
    /// compressed cache to a full-precision turn, if the key did not name the
    /// configuration. Two configurations of one session therefore name two
    /// caches, and neither serves the other.
    ///
    /// Read by the synthesized `Hashable` conformance; periphery sees no caller.
    // periphery:ignore
    let kvCache: KVCacheConfiguration?

    /// Creates the key of `sessionID` of `modelID` under `kvCache`.
    ///
    /// - Parameters:
    ///   - modelID: the model the cache belongs to.
    ///   - sessionID: the session the cache belongs to.
    ///   - kvCache: the KV-cache configuration the caches were built under, or
    ///     nil for full precision.
    init(modelID: ModelID, sessionID: SessionID, kvCache: KVCacheConfiguration? = nil) {
        self.modelID = modelID
        self.sessionID = sessionID
        self.kvCache = kvCache
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration

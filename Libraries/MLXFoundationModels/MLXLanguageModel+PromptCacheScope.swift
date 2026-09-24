// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel {

    /// The prompt cache that one executor pass uses, as the host sets it.
    ///
    /// Without a scope, the executor names the session of a pass by the
    /// identifier of the first transcript entry. A host that knows its
    /// sessions can set a better name: a fork copies the transcript of its
    /// parent, thus it also copies the first entry, and a compaction that
    /// removes the first entry changes the name.
    public enum PromptCacheScope: Sendable, Hashable {
        /// The pass belongs to this session. The host owns the identifier.
        case session(String)
        /// The pass takes no carried cache and leaves none.
        case uncached
    }

    /// The prompt cache scope of the executor passes that the current task
    /// runs, or nil when the host sets no scope.
    ///
    /// Bind it around the call to the executor:
    ///
    /// ```swift
    /// try await MLXLanguageModel.$promptCacheScope.withValue(.session(id)) {
    ///     try await executor.respond(to: request, model: model, streamingInto: channel)
    /// }
    /// ```
    ///
    /// - `.session(id)`: the pass uses the cache of session `id`.
    /// - `.uncached`: the pass checks out no cache and checks in no cache.
    /// - nil: the identifier of the first transcript entry names the session.
    ///
    /// The binding is guaranteed to reach the executor only when the host calls
    /// `Executor.respond` directly, on the same task that binds it.
    @TaskLocal public static var promptCacheScope: PromptCacheScope?
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration

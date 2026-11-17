// Copyright © 2026 Apple Inc.

import MLXLMCommon
import os

/// Records each progress report of a prefill. The lock serializes every read
/// and write of the counts, so the `@Sendable` progress closure of
/// `PrefillParameters` can record from any thread.
public struct PrefillProgressLog: Sendable {
    private let reports = OSAllocatedUnfairLock<[Int]>(initialState: [])

    /// Makes an empty log.
    public init() {}

    /// The `processed` count of each report, in order.
    public var processed: [Int] { reports.withLock { $0 } }

    /// Records the `processed` count of one report.
    ///
    /// - Parameter processed: The positions the prefill has processed so far.
    public func record(_ processed: Int) {
        reports.withLock { $0.append(processed) }
    }

    /// Prefill parameters with `stepSize` whose progress reports go to this
    /// log.
    ///
    /// - Parameter stepSize: The step size of the caller, or `nil` for none.
    /// - Returns: The parameters, with the default chunking.
    public func recordingPrefill(stepSize: Int? = nil) -> PrefillParameters {
        PrefillParameters(stepSize: stepSize, progress: { processed, _ in record(processed) })
    }
}

extension PrefillProgressLog {
    /// The token every checkpoint feed reads. Token 0 is in the vocabulary of
    /// every model, and the chunks of a feed do not depend on the token ids.
    private static let checkpointFeedToken = 0

    /// Runs one checkpoint feed of `tokenCount` prompt tokens into a fresh
    /// cache of `model`, the feed that `ChatSession` and the
    /// `MLXFoundationModels` executor run up to each message boundary.
    ///
    /// - Parameters:
    ///   - model: The model that reads the feed.
    ///   - tokenCount: The prompt tokens the feed reads.
    ///   - stepSize: The step size of the caller, or `nil` for none.
    /// - Returns: The `processed` count of each progress report of the feed,
    ///   in order.
    /// - Throws: An error of the cache or of the feed.
    public static func progressOfCheckpointFeed(
        _ model: any LanguageModel, tokenCount: Int, stepSize: Int? = nil
    ) throws -> [Int] {
        let log = PrefillProgressLog()
        _ = try PromptCacheCheckpoint.prefill(
            Array(repeating: checkpointFeedToken, count: tokenCount),
            into: try model.newCache(parameters: nil), state: nil, model: model,
            prefill: log.recordingPrefill(stepSize: stepSize))
        return log.processed
    }
}

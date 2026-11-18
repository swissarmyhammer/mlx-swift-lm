// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

// MARK: - The spool writer

/// Writes a prepared entry to a URL. The spool calls it off the store actor,
/// one write at a time.
typealias ExecutorPromptCacheFileWriter = @Sendable (PromptCacheSaveInput, URL) throws -> Void

/// One write of the spool, and what to do when it ends.
struct ExecutorPromptCacheSpillJob: Sendable {

    /// The prepared entry.
    let input: PromptCacheSaveInput

    /// The URL of the file.
    let url: URL

    /// Receives the result of the write and the time it took.
    let didEnd: @Sendable (Result<Void, any Error>, Duration) async -> Void
}

/// Runs the writes of one store in order, one at a time, on a task of its own.
///
/// A write holds the process-wide evaluation lock of MLX for its whole run,
/// because the safetensors save evaluates the arrays. Two writes at the same
/// time thus gain nothing, and one write at a time keeps the store actor free:
/// the actor only puts a job in the queue.
final class ExecutorPromptCacheSpoolWriter: Sendable {

    /// The queue of the writes.
    private let jobs: AsyncStream<ExecutorPromptCacheSpillJob>.Continuation

    /// The task that runs the writes.
    private let consumer: Task<Void, Never>

    /// Creates a writer and starts the task that runs its writes.
    ///
    /// - Parameter write: writes one prepared entry to one URL.
    init(write: @escaping ExecutorPromptCacheFileWriter) {
        let (stream, jobs) = AsyncStream.makeStream(of: ExecutorPromptCacheSpillJob.self)
        self.jobs = jobs
        self.consumer = Task.detached {
            for await job in stream {
                await Self.run(job, write: write)
            }
        }
    }

    deinit {
        jobs.finish()
        consumer.cancel()
    }

    /// Puts `job` at the end of the queue.
    ///
    /// - Parameter job: the write to run.
    func enqueue(_ job: ExecutorPromptCacheSpillJob) {
        jobs.yield(job)
    }

    /// Writes the file of `job`, and gives the result and the duration to the
    /// job.
    ///
    /// - Parameters:
    ///   - job: the write to run.
    ///   - write: writes one prepared entry to one URL.
    private static func run(
        _ job: ExecutorPromptCacheSpillJob, write: ExecutorPromptCacheFileWriter
    ) async {
        let measured = ExecutorPromptCacheTiming.measure {
            Result<Void, any Error> { try write(job.input, job.url) }
        }
        await job.didEnd(measured.value, measured.duration)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration

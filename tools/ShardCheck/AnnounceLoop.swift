// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// The transport of rank 0 in the announce check: it announces each request to every rank with
/// one all-sum, then runs it on the executor of rank 0, as a cluster worker does.
///
/// It announces one park after the first forward of the run, so the other ranks see a park
/// between two ops of one request.
// swiftlint:disable:next no_unchecked_sendable every access to a stored var holds `lock`
final class AnnouncingTransport: RemoteModelTransport, @unchecked Sendable {

    /// The executor over the shard of rank 0.
    private let executor: RemoteModelExecutor

    /// The group of every rank.
    private let group: MLXDistributed.Group

    /// Guards the stored vars and keeps one request in flight.
    private let lock = NSLock()

    /// The last request announced, or nil before the first.
    private var lastRequest: RemoteModelRequest?

    /// Whether the park of the run is still to announce.
    private var parkIsPending = true

    /// Whether ``close()`` ran, so no rank listens any more.
    private var isClosed = false

    /// Makes the transport of rank 0.
    ///
    /// - Parameters:
    ///   - executor: The executor over the shard of rank 0.
    ///   - group: The group of every rank.
    init(executor: RemoteModelExecutor, group: MLXDistributed.Group) {
        self.executor = executor
        self.group = group
    }

    /// Announces `request` to every rank, then runs it on rank 0.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply of rank 0.
    /// - Throws: ``AnnounceLoopError/closed`` after ``close()``, or the error of the exchange.
    func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        try lock.withLock {
            guard !isClosed else { throw AnnounceLoopError.closed }
            if parkIsPending, case .forward = lastRequest?.op {
                parkIsPending = false
                _ = try RemoteOpAnnouncement.exchange(
                    RemoteOpAnnouncement.park(requestID: request.requestID), group: group)
            }
            _ = try RemoteOpAnnouncement.exchange(
                RemoteOpAnnouncement.encode(request), group: group)
            lastRequest = request
            return try executor.handle(request)
        }
    }

    /// Ends the open request, which stops the loop of every other rank, and refuses every later
    /// request: a cache released after this goes to no rank.
    ///
    /// - Throws: ``AnnounceLoopError/noOpenRequest`` before any request, or the error of the
    ///   exchange.
    func close() throws {
        try lock.withLock {
            guard let lastRequest else { throw AnnounceLoopError.noOpenRequest }
            let end = RemoteModelRequest(
                requestID: lastRequest.requestID, sequence: lastRequest.sequence + 1, op: .end)
            _ = try RemoteOpAnnouncement.exchange(RemoteOpAnnouncement.encode(end), group: group)
            _ = try executor.handle(end)
            isClosed = true
        }
    }
}

/// The loop of a rank other than 0 in the announce check.
enum AnnounceFollower {

    /// What one rank other than 0 applied.
    struct Summary {
        /// The ops applied.
        var opCount = 0
        /// The parks seen.
        var parkCount = 0
        /// The ops whose reply failed.
        var failedOpCount = 0
    }

    /// Applies each op rank 0 announces, until an announce raises the stop flag.
    ///
    /// - Parameters:
    ///   - executor: The executor over the shard of this rank.
    ///   - group: The group of every rank.
    /// - Returns: What the rank applied.
    /// - Throws: The error of an exchange, or of words that name no op.
    static func follow(
        _ executor: RemoteModelExecutor, group: MLXDistributed.Group
    ) throws -> Summary {
        var summary = Summary()
        while true {
            let words = try RemoteOpAnnouncement.exchange(nil, group: group)
            guard let reply = try executor.apply(announcement: words) else {
                summary.parkCount += 1
                continue
            }
            summary.opCount += 1
            if case .failed(let reason) = reply {
                summary.failedOpCount += 1
                FileHandle.standardError.write(Data("ShardCheck: an op failed: \(reason)\n".utf8))
            }
            if RemoteOpAnnouncement.stopsLoop(words) {
                return summary
            }
        }
    }
}

/// A failure of the announce check.
enum AnnounceLoopError: Error, CustomStringConvertible {
    /// A request arrived after ``AnnouncingTransport/close()``.
    case closed
    /// ``AnnouncingTransport/close()`` ran before any request.
    case noOpenRequest

    var description: String {
        switch self {
        case .closed: "the announce loop is closed"
        case .noOpenRequest: "no request was announced"
        }
    }
}

/// One greedy generation through ``TokenIterator``, as an app runs it.
struct GenerationRun {

    /// The tokens the generation wrote.
    let tokens: [Int]

    /// The time to the first token: the prefill.
    let prefillTime: Duration

    /// The time of every later token together.
    let decodeTime: Duration

    /// Generates `decodeSteps + 1` tokens greedily after `prompt`.
    init(model: any LanguageModel, prompt: [Int], decodeSteps: Int) throws {
        let clock = ContinuousClock()
        let start = clock.now
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
            parameters: GenerateParameters(maxTokens: decodeSteps + 1, temperature: 0))
        var tokens: [Int] = []
        var prefillTime = Duration.zero
        while let token = iterator.next() {
            tokens.append(token)
            if tokens.count == 1 {
                prefillTime = clock.now - start
            }
        }
        self.tokens = tokens
        self.prefillTime = prefillTime
        self.decodeTime = clock.now - start - prefillTime
    }
}

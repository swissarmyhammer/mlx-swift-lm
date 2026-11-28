// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// The failure an ``InProcessRemoteModelWorker`` throws for the ops a test told it to fail.
public struct InProcessWorkerFault: Error, Equatable, CustomStringConvertible {

    /// Makes the fault.
    public init() {}

    /// The fault in words.
    public var description: String { "the in-process worker is down" }
}

/// A failure an ``InProcessRemoteModelWorker`` throws that conforms to ``ClusterMemberLoss``, as
/// a cluster package throws its own errors: one that lost members of the cluster, or one that
/// lost no member.
public struct InProcessMemberLoss: ClusterMemberLoss, Equatable, CustomStringConvertible {

    /// The members that are gone, in words, or nil when the failure lost no member.
    public let lostMember: String?

    /// Makes the failure that lost `lostMember`.
    ///
    /// - Parameter lostMember: The members that are gone, in words, or nil for a failure that
    ///   lost no member.
    public init(lostMember: String?) {
        self.lostMember = lostMember
    }

    /// The failure in words.
    public var description: String {
        lostMember.map { "the cluster lost \($0)" } ?? "the cluster failed and lost no member"
    }
}

// The directive sits above the documentation, which must stay next to the declaration.
// swiftlint:disable no_unchecked_sendable every access to a stored var holds `lock`
/// A worker that runs a real local model in this process, behind the ``RemoteModelTransport``
/// a ``ClusterLanguageModel`` drives.
///
/// Each request travels as its ``RemoteOpAnnouncement`` words, as rank 0 announces it to the
/// other ranks, so a value the words cannot carry fails the test. A ``RemoteModelExecutor``
/// runs the decoded request, as a cluster worker runs it. The worker records every request it
/// received.
///
/// A worker can also stand for the other ranks of a cluster: each follower is an executor of its
/// own over the same model, with no drafter, that applies the announce words of every op after
/// rank 0 ran it, and takes the drafts of rank 0 from its draft exchange, as a rank other than 0
/// does. The worker records the cache status each follower last reported.
public final class InProcessRemoteModelWorker: RemoteModelTransport, @unchecked Sendable {
    // swiftlint:enable no_unchecked_sendable

    /// The executor that runs each op: rank 0.
    private let executor: RemoteModelExecutor

    /// The executors of the other ranks, which apply the announce words of each op.
    private let followers: [RemoteModelExecutor]

    /// The cache status each follower last reported for each handle, one map per follower.
    private var followerStatuses: [[RemoteCacheID: RemoteCacheStatus]]

    /// The failed replies of the followers, in order.
    private var followerFailureReasons: [String] = []

    /// The protocol version the worker reports in place of the executor's.
    private let protocolVersion: Int32

    /// Guards the stored vars and runs one op at a time.
    private let lock = NSLock()

    /// Every request received, in order.
    private var received: [RemoteModelRequest] = []

    /// The requests the worker still runs before each op fails, or nil for no limit.
    private var remainingSuccesses: Int?

    /// The forwards the worker still runs before each forward fails, or nil for no limit.
    private var remainingForwards: Int?

    /// The error each op throws after ``remainingSuccesses`` runs out, and each forward after
    /// ``remainingForwards`` runs out.
    private var fault: any Error = InProcessWorkerFault()

    /// The number of times the transport was closed.
    private var closes = 0

    /// Makes a worker over `model`.
    ///
    /// - Parameters:
    ///   - model: The local model the worker runs.
    ///   - vocabularySize: The last dimension of the logits of `model`.
    ///   - protocolVersion: The protocol version the worker reports.
    ///   - modelType: The `model_type` the worker reports, or nil for none.
    ///   - drafter: The MTP drafter rank 0 holds, or nil for none.
    ///   - followerCount: The ranks other than 0 the worker stands for.
    public init(
        model: any LanguageModel, vocabularySize: Int,
        protocolVersion: Int32 = RemoteModelProtocol.version, modelType: String? = nil,
        drafter: (any ResumableMTPDrafterModel)? = nil, followerCount: Int = 0
    ) {
        let configuration = RemoteModelExecutor.Configuration(
            vocabularySize: vocabularySize, rankCount: followerCount + 1, modelType: modelType)
        // The draft words rank 0 exchanged in the op that runs, which each follower takes.
        let draftWords = DraftWordsRelay()
        self.executor = RemoteModelExecutor(
            model: model, configuration: configuration, drafter: drafter,
            draftExchange: RemoteDraftExchange { words, _ in
                let words = words ?? []
                draftWords.words = words
                return words
            })
        self.followers = (0 ..< followerCount).map { _ in
            RemoteModelExecutor(
                model: model, configuration: configuration,
                draftExchange: RemoteDraftExchange { _, _ in draftWords.words })
        }
        self.followerStatuses = Array(repeating: [:], count: followerCount)
        self.protocolVersion = protocolVersion
    }

    /// The status each follower last reported for the caches of `cache`, one per follower.
    ///
    /// - Parameter cache: The handle of the caches.
    /// - Returns: The statuses, in the order of the followers; a follower that reported none
    ///   for `cache` is missing.
    public func followerStatuses(of cache: RemoteCacheID) -> [RemoteCacheStatus] {
        lock.withLock { followerStatuses.compactMap { $0[cache] } }
    }

    /// The reasons of the failed replies of the followers, in order.
    public var followerFailures: [String] {
        lock.withLock { followerFailureReasons }
    }

    /// Every request the worker received, in order.
    public var requests: [RemoteModelRequest] {
        lock.withLock { received }
    }

    /// The handles whose caches the worker holds.
    public var liveCacheCount: Int {
        lock.withLock { executor.cacheCount }
    }

    /// The number of times ``close()`` ran.
    public var closeCount: Int {
        lock.withLock { closes }
    }

    /// Records the teardown of the transport.
    public func close() {
        lock.withLock { closes += 1 }
    }

    /// Makes every op after the next `count` throw `fault`.
    ///
    /// - Parameters:
    ///   - count: The ops that still succeed.
    ///   - fault: The error each later op throws.
    public func failOps(after count: Int, with fault: any Error = InProcessWorkerFault()) {
        lock.withLock {
            remainingSuccesses = count
            self.fault = fault
        }
    }

    /// Makes every forward after the next `count` forwards throw `fault`, as a member lost in
    /// the middle of a generation fails it. The other ops still succeed.
    ///
    /// - Parameters:
    ///   - count: The forwards that still succeed.
    ///   - fault: The error each later forward throws.
    public func failForwards(after count: Int, with fault: any Error = InProcessWorkerFault()) {
        lock.withLock {
            remainingForwards = count
            self.fault = fault
        }
    }

    /// Runs `request` on the executor.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply.
    /// - Throws: The fault of ``failOps(after:with:)`` after its ops, the fault of
    ///   ``failForwards(after:with:)`` after its forwards, or the error of the announce words.
    public func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        try lock.withLock {
            let announced = try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.encode(request))
            received.append(announced)
            try spend(&remainingSuccesses)
            if case .forward = announced.op {
                try spend(&remainingForwards)
            }
            let reply = try executor.handle(announced)
            try follow(RemoteOpAnnouncement.encode(announced))
            return reported(reply)
        }
    }

    /// Applies announce `words` on each follower, and records the status or the failure of its
    /// reply. The caller holds ``lock``.
    ///
    /// - Parameter words: The announce words of the op rank 0 ran.
    /// - Throws: The error of a follower's executor.
    private func follow(_ words: [Int32]) throws {
        for (index, follower) in followers.enumerated() {
            switch try follower.apply(announcement: words) {
            case .failed(let reason): followerFailureReasons.append(reason)
            case .cache(let status), .trimmed(_, let status), .forwarded(let status, _),
                .drafted(let status, _, _):
                let cache = try RemoteOpAnnouncement.decode(words).op.cache
                followerStatuses[index][cache] = status
            case .capabilities, .done, nil: break
            }
        }
    }

    /// Counts one request against `budget`, and throws ``fault`` once `budget` is spent. A nil
    /// budget has no limit. The caller holds ``lock``.
    ///
    /// - Parameter budget: The requests that still succeed, or nil for no limit.
    /// - Throws: ``fault`` when `budget` is zero.
    private func spend(_ budget: inout Int?) throws {
        guard let remaining = budget else { return }
        guard remaining > 0 else { throw fault }
        budget = remaining - 1
    }

    /// `reply` with ``protocolVersion`` in place of the version of a describe reply.
    private func reported(_ reply: RemoteModelReply) -> RemoteModelReply {
        guard case .capabilities(let capabilities) = reply else { return reply }
        return .capabilities(
            RemoteModelCapabilities(
                protocolVersion: protocolVersion, modelFamilies: capabilities.modelFamilies,
                supportsSharding: capabilities.supportsSharding,
                rankCount: capabilities.rankCount, vocabularySize: capabilities.vocabularySize,
                defaultPrefillStepSize: capabilities.defaultPrefillStepSize,
                maximumPrefillStepSize: capabilities.maximumPrefillStepSize,
                preparesEveryPrompt: capabilities.preparesEveryPrompt,
                modelType: capabilities.modelType, quantization: capabilities.quantization,
                drafter: capabilities.drafter))
    }
}

/// The draft words rank 0 exchanged in the op that runs, which each follower of an
/// ``InProcessRemoteModelWorker`` takes as its exchange gives them to a rank other than 0.
// swiftlint:disable:next no_unchecked_sendable the worker's lock guards every op, thus every access
private final class DraftWordsRelay: @unchecked Sendable {

    /// The words of the last exchange.
    var words: [Int32] = []
}

extension RemoteModelOp {

    /// The cache handle the op names, which a reply's status describes.
    fileprivate var cache: RemoteCacheID {
        switch self {
        case .newCache(let cache, _), .forward(_, let cache, _), .trim(let cache, _),
            .copy(_, into: let cache), .reset(let cache), .release(let cache),
            .quantize(let cache):
            cache
        case .speculate(let round): round.cache
        case .describe, .begin, .end: RemoteCacheID(rawValue: 0)
        }
    }
}

extension Qwen35MTPFixture {

    /// A worker that runs the target of the fixture.
    ///
    /// - Parameters:
    ///   - drafting: Whether rank 0 of the worker holds the drafter of the fixture.
    ///   - followerCount: The ranks other than 0 the worker stands for.
    /// - Returns: The worker.
    public func makeWorker(drafting: Bool = false, followerCount: Int = 0)
        -> InProcessRemoteModelWorker
    {
        InProcessRemoteModelWorker(
            model: target, vocabularySize: Self.vocabularySize,
            drafter: drafting ? drafter : nil, followerCount: followerCount)
    }

    /// The context of ``makeModelContext(eosTokenIds:)`` with a cluster model over `worker` in
    /// the place of the target.
    ///
    /// - Parameters:
    ///   - worker: The worker that runs the target.
    ///   - eosTokenIds: The tokens that end a generation.
    /// - Returns: The context.
    /// - Throws: The error of the worker handshake.
    public func makeClusteredModelContext(
        worker: InProcessRemoteModelWorker, eosTokenIds: Set<Int> = []
    ) throws -> ModelContext {
        var context = makeModelContext(eosTokenIds: eosTokenIds)
        context.model = try ClusterLanguageModel.connect(over: worker)
        return context
    }
}

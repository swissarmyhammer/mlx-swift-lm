// Copyright © 2026 Apple Inc.

import Foundation

/// Carries each ``RemoteModelRequest`` to the worker that holds a remote model, and its reply
/// back.
///
/// This package holds no networking. A cluster package implements the transport, for example
/// with a control channel to the rank-0 worker and shared memory for the logits.
///
/// ``ClusterLanguageModel`` calls ``send(_:)`` from the thread that runs the model: the call
/// blocks until the reply arrives. It sends one request at a time, in the order of
/// ``RemoteModelRequest/sequence``.
public protocol RemoteModelTransport: Sendable {

    /// Sends `request` to the worker and waits for its reply.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply of the worker.
    /// - Throws: The error of the transport when the request or the reply did not arrive.
    func send(_ request: RemoteModelRequest) throws -> RemoteModelReply

    /// Tears down the workers behind the transport. ``ClusterLanguageModel/close()`` calls it
    /// one time, and sends nothing after it.
    func close()
}

extension RemoteModelTransport {

    /// Does nothing: a transport with no workers to tear down needs no teardown.
    public func close() {}
}

/// A feature a ``ClusterLanguageModel`` does not run.
public enum ClusterModelFeature: String, Sendable, Hashable {

    /// Image, video or audio input: the workers would need the embeddings.
    case mediaInput

    /// A KV-cache configuration or `maxKVSize`: the worker makes its caches itself.
    case cacheConfiguration

    /// A batch of more than one sequence.
    case batchedInput
}

/// An error of a ``RemoteModelTransport`` that can say that members of the cluster are gone, for
/// example a Mac that shut down or dropped off the network.
///
/// A cluster package conforms the error type it throws, even when most of its errors lose no
/// member. When ``lostMember`` is not nil, a ``ClusterLanguageModel`` fails with
/// ``ClusterModelError/memberLost(_:)`` and not with ``ClusterModelError/transportFailed(_:)``.
/// A host can then drop the model and load a local one. When ``lostMember`` is nil, the model
/// fails with ``ClusterModelError/transportFailed(_:)``, as for any other error.
public protocol ClusterMemberLoss: Error {

    /// The members that are gone, in words for a log or a message, or nil when the error lost no
    /// member.
    ///
    /// The words name one member or several, for example a comma-separated list of the `.local`
    /// names of every member when the cluster cannot tell which one of them is gone. A host shows
    /// or logs them, and does not parse them for a single host.
    var lostMember: String? { get }
}

/// The errors of a ``ClusterLanguageModel``.
public enum ClusterModelError: Error, Hashable, LocalizedError {

    /// The worker speaks another version of the op set.
    case protocolVersionMismatch(expected: Int32, found: Int32)

    /// The worker runs a model of another `model_type` than the checkpoint the app loaded.
    case modelTypeMismatch(expected: String, found: String)

    /// The model does not run `feature`.
    case unsupported(ClusterModelFeature)

    /// The transport failed. Every later op of the model fails with this error too.
    case transportFailed(String)

    /// The transport lost members of the cluster, which the words of
    /// ``ClusterMemberLoss/lostMember`` describe for a log or a message. Every later op of the
    /// model fails with this error too, and the model must load again.
    case memberLost(String)

    /// The worker answered an op with ``RemoteModelReply/failed(_:)``.
    case workerFailed(String)

    /// The worker answered an op with a reply of another op.
    case unexpectedReply(op: String, reply: String)

    /// The caches given to the model are not the one ``RemoteKVCache`` it made.
    case foreignCache

    /// ``ClusterLanguageModel/close()`` tore down the workers. Every later op fails with this
    /// error.
    case closed

    /// A description of the error.
    public var errorDescription: String? {
        switch self {
        case .protocolVersionMismatch(let expected, let found):
            "The cluster worker speaks remote model protocol version \(found), "
                + "but this app speaks version \(expected)."
        case .modelTypeMismatch(let expected, let found):
            "The cluster worker runs a \(found) model, but the checkpoint is a \(expected) model."
        case .unsupported(let feature):
            "A cluster model does not run \(Self.describe(feature))."
        case .transportFailed(let reason):
            "The connection to the cluster worker failed: \(reason)"
        case .memberLost(let member):
            "The cluster lost \(member). Load the model again to use it."
        case .workerFailed(let reason):
            "The cluster worker failed: \(reason)"
        case .unexpectedReply(let op, let reply):
            "The cluster worker answered \(op) with \(reply)."
        case .foreignCache:
            "A cluster model runs only on the one remote cache its newCache made."
        case .closed:
            "The cluster model was closed, and its workers are gone."
        }
    }

    /// The feature in words.
    private static func describe(_ feature: ClusterModelFeature) -> String {
        switch feature {
        case .mediaInput: "image, video or audio input"
        case .cacheConfiguration:
            "a KV-cache configuration or maxKVSize; the worker sets its caches"
        case .batchedInput: "a batch of more than one sequence"
        }
    }
}

/// The stream of requests of one ``ClusterLanguageModel``: it numbers each request, allocates
/// the cache handles, and remembers the first failure.
///
/// A failed transport leaves the worker in a state the app cannot know, thus every op after a
/// failure fails at once with the same error, and sends nothing.
// swiftlint:disable:next no_unchecked_sendable every access to a stored var holds `lock`
final class RemoteModelSession: @unchecked Sendable {

    /// The transport to the worker.
    private let transport: any RemoteModelTransport

    /// Guards the stored vars, and keeps one request in flight at a time.
    private let lock = NSLock()

    /// The sequence number of the last request sent.
    private var sequence: UInt64 = 0

    /// The last cache handle allocated.
    private var lastCacheID: UInt64 = 0

    /// The last request identifier allocated.
    private var lastRequestID: UInt64 = 0

    /// The request the ops belong to.
    private var requestID = RemoteRequestID.none

    /// Whether ``requestID`` is open: begun and not ended.
    private var requestIsOpen = false

    /// The first failure, which every later op throws.
    private var failure: ClusterModelError?

    /// Makes the session of `transport`.
    ///
    /// - Parameter transport: The transport to the worker.
    init(transport: any RemoteModelTransport) {
        self.transport = transport
    }

    /// A new cache handle, which no other cache of this session has.
    func makeCacheID() -> RemoteCacheID {
        lock.withLock {
            lastCacheID += 1
            return RemoteCacheID(rawValue: lastCacheID)
        }
    }

    /// Sends `op` and returns the reply.
    ///
    /// - Parameter op: The op.
    /// - Returns: The reply of the worker, never ``RemoteModelReply/failed(_:)``.
    /// - Throws: The failure of an earlier op, or ``ClusterModelError/closed`` after
    ///   ``close()``, without sending; ``ClusterModelError/memberLost(_:)`` when the transport
    ///   lost members of the cluster; ``ClusterModelError/transportFailed(_:)`` for another
    ///   failure of the transport; ``ClusterModelError/workerFailed(_:)`` for a failed reply.
    func perform(_ op: RemoteModelOp) throws -> RemoteModelReply {
        try lock.withLock { try send(op) }
    }

    /// Ends the open request, if any, and begins a new one.
    ///
    /// - Throws: The error of ``perform(_:)``.
    func beginRequest() throws {
        try lock.withLock {
            if requestIsOpen {
                requestIsOpen = false
                try expectDone(.end)
            }
            lastRequestID += 1
            requestID = RemoteRequestID(rawValue: lastRequestID)
            try expectDone(.begin)
            requestIsOpen = true
        }
    }

    /// Records `error` as the failure of the session when it is the first, for an op that cannot
    /// throw.
    ///
    /// - Parameter error: The error of an op.
    func recordFailure(_ error: any Error) {
        lock.withLock {
            failure = failure ?? Self.clusterError(error)
        }
    }

    /// Tears down the transport one time, and fails every later op with
    /// ``ClusterModelError/closed``. A second call does nothing.
    func close() {
        lock.withLock {
            guard failure != .closed else { return }
            failure = .closed
            transport.close()
        }
    }

    /// Whether an op failed, thus the session sends nothing more.
    var hasFailed: Bool {
        lock.withLock { failure != nil }
    }

    /// Throws the failure of the session, if any: the first failed op, or
    /// ``ClusterModelError/closed`` after ``close()``.
    ///
    /// - Throws: The failure, which every later op throws too.
    func checkFailure() throws {
        if let failure = lock.withLock({ failure }) { throw failure }
    }

    /// Sends `op` and expects ``RemoteModelReply/done``. The caller holds ``lock``.
    private func expectDone(_ op: RemoteModelOp) throws {
        let reply = try send(op)
        guard reply == .done else {
            throw ClusterModelError.unexpectedReply(
                op: String(describing: op), reply: String(describing: reply))
        }
    }

    /// Sends `op` as the next request. The caller holds ``lock``.
    private func send(_ op: RemoteModelOp) throws -> RemoteModelReply {
        if let failure { throw failure }
        sequence += 1
        let reply: RemoteModelReply
        do {
            reply = try transport.send(
                RemoteModelRequest(requestID: requestID, sequence: sequence, op: op))
        } catch {
            let failure = Self.clusterError(error)
            self.failure = failure
            throw failure
        }
        if case .failed(let reason) = reply {
            failure = .workerFailed(reason)
            throw ClusterModelError.workerFailed(reason)
        }
        return reply
    }

    /// `error` as a ``ClusterModelError``: itself when it is one, a lost member for a
    /// ``ClusterMemberLoss`` whose ``ClusterMemberLoss/lostMember`` is not nil, else a transport
    /// failure.
    private static func clusterError(_ error: any Error) -> ClusterModelError {
        if let error = error as? ClusterModelError { return error }
        if let member = (error as? any ClusterMemberLoss)?.lostMember { return .memberLost(member) }
        return .transportFailed(String(describing: error))
    }
}

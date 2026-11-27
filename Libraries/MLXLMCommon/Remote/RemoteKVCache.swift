// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The caches of every layer of one sequence of a ``ClusterLanguageModel``, which live in the
/// worker. This object holds their handle, and mirrors their ``RemoteCacheStatus``.
///
/// The prompt-cache code reads and changes it as it does a local cache: ``offset``,
/// ``maxTrimCount`` and ``isTrimmable(after:)`` read the mirrored status, ``trim(_:)`` and
/// ``copy()`` run as ops on the worker, and the cache releases its handle when it goes away.
///
/// The cache holds no arrays in the app. A layer never reads it, thus ``update(keys:values:)``
/// and ``makeMask(n:windowSize:returnArray:)`` are programmer errors. It has no
/// ``KVCache/serializedClassName``, thus a prompt cache file cannot hold it.
public final class RemoteKVCache: KVCache {

    /// The handle of the caches on the worker.
    public let id: RemoteCacheID

    /// The stream of requests of the model that made the cache.
    let session: RemoteModelSession

    /// The status the worker reported after the last op on the cache.
    public private(set) var status: RemoteCacheStatus

    /// Makes the mirror of the caches of `id`.
    ///
    /// - Parameters:
    ///   - id: The handle of the caches on the worker.
    ///   - session: The stream of requests of the model.
    ///   - status: The status the worker reported for them.
    init(id: RemoteCacheID, session: RemoteModelSession, status: RemoteCacheStatus) {
        self.id = id
        self.session = session
        self.status = status
    }

    /// Frees the caches on the worker. A failed release leaves the session failed, which the
    /// next throwing op of the model reports.
    deinit {
        guard !session.hasFailed else { return }
        _ = try? session.perform(.release(id))
    }

    // MARK: - The mirrored status

    /// The positions the caches hold.
    public var offset: Int { status.offset }

    /// The smallest window of the layers, or nil when none is bounded.
    public var maxSize: Int? { status.maxSize }

    /// The most positions a layer evicted.
    public var evictedTokenCount: Int { status.evictedTokenCount }

    /// Whether every layer can trim.
    public var isTrimmable: Bool { status.isTrimmable }

    /// The most positions every layer can rewind exactly.
    public var maxTrimCount: Int { status.maxTrimCount }

    /// The bytes the caches hold on the worker.
    public var residentByteCount: Int { status.residentByteCount }

    /// Whether every layer could still rewind after `positions` more positions.
    ///
    /// - Parameter positions: The positions the caches would append.
    /// - Returns: The answer the worker reported.
    public func isTrimmable(after positions: Int) -> Bool {
        status.rewindLimit.allows(positions)
    }

    // MARK: - Ops on the worker

    /// Drops the newest `n` positions on the worker.
    ///
    /// - Parameter n: The positions to drop.
    /// - Returns: The positions the worker dropped, or 0 when the op failed. The failure stays
    ///   on the model, and its next throwing op reports it.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        do {
            let reply = try session.perform(.trim(id, count: n))
            guard case .trimmed(let trimmed, let status) = reply else {
                throw ClusterModelError.unexpectedReply(
                    op: "trim", reply: String(describing: reply))
            }
            self.status = status
            return trimmed
        } catch {
            session.recordFailure(error)
            return 0
        }
    }

    /// A copy of the caches under a new handle, which the worker makes.
    ///
    /// - Returns: The copy. When the op fails, the copy mirrors this status, and the failure
    ///   stays on the model: each later op fails, and the next throwing op reports it.
    public func copy() -> any KVCache {
        let copyID = session.makeCacheID()
        do {
            let status = try session.perform(.copy(id, into: copyID)).cacheStatus(of: "copy")
            return RemoteKVCache(id: copyID, session: session, status: status)
        } catch {
            session.recordFailure(error)
            return RemoteKVCache(id: copyID, session: session, status: status)
        }
    }

    /// Empties the caches on the worker, and keeps the handle.
    ///
    /// - Throws: The ``ClusterModelError`` of the op.
    public func reset() throws {
        status = try session.perform(.reset(id)).cacheStatus(of: "reset")
    }

    /// Runs `tokens` through the model into the caches, a chunk of at most
    /// ``RemoteModelProtocol/maximumChunkTokenCount`` tokens at a time.
    ///
    /// - Parameters:
    ///   - tokens: The tokens.
    ///   - scope: The logits to return.
    /// - Returns: The logits of `scope`, or nil for ``RemoteLogitsScope/none``.
    /// - Throws: The ``ClusterModelError`` of an op, or the error of the logits.
    func forward(_ tokens: [Int32], logits scope: RemoteLogitsScope) throws -> MLXArray? {
        let chunkLength = RemoteModelProtocol.maximumChunkTokenCount
        let starts = Array(stride(from: 0, to: tokens.count, by: chunkLength))
        var rows: [MLXArray] = []
        for start in starts {
            let end = min(start + chunkLength, tokens.count)
            let isLast = end == tokens.count
            let chunkScope: RemoteLogitsScope =
                scope == .allPositions || isLast ? scope : .none
            let reply = try session.perform(
                .forward(tokens: Array(tokens[start ..< end]), cache: id, logits: chunkScope))
            guard case .forwarded(let status, let logits) = reply else {
                throw ClusterModelError.unexpectedReply(
                    op: "forward", reply: String(describing: reply))
            }
            self.status = status
            if let logits {
                rows.append(try logits.array())
            }
        }
        guard scope != .none, !rows.isEmpty else { return nil }
        return rows.count == 1 ? rows[0] : concatenated(rows, axis: 1)
    }

    // MARK: - What a layer would read

    /// Never called: the worker runs the attention.
    ///
    /// - Parameters:
    ///   - keys: Unused.
    ///   - values: Unused.
    /// - Returns: Never returns.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("A remote cache holds no keys and values; the worker runs attention")
    }

    /// Never called: the worker runs the attention.
    ///
    /// - Parameters:
    ///   - n: Unused.
    ///   - windowSize: Unused.
    ///   - returnArray: Unused.
    /// - Returns: Never returns.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        preconditionFailure("A remote cache makes no mask; the worker runs attention")
    }

    /// No arrays: the arrays of the caches live on the worker. Only an empty state can be set.
    public var state: [MLXArray] {
        get { [] }
        set { precondition(newValue.isEmpty, "A remote cache cannot take local arrays") }
    }

    /// No meta state. Only an empty meta state can be set.
    public var metaState: [String] {
        get { [] }
        set { precondition(newValue.isEmpty, "A remote cache cannot take a local meta state") }
    }

    /// No arrays to evaluate in the app.
    ///
    /// - Returns: An empty array.
    public func innerState() -> [MLXArray] { [] }
}

extension KVCache {

    /// The bytes the buffers of this cache hold in one place: on its worker, or in the memory
    /// of this process.
    ///
    /// It is ``KVCache/residentByteCount`` when the buffers live in that place, and zero
    /// otherwise. The buffers of a ``RemoteKVCache`` live on its worker, and those of a local
    /// cache in this process: a budget of one place must count only the bytes there.
    ///
    /// - Parameter onWorker: true for the bytes on the worker, false for the bytes of this
    ///   process.
    /// - Returns: the bytes this cache holds in that place.
    package func residentByteCount(onWorker: Bool) -> Int {
        isRemote == onWorker ? residentByteCount : 0
    }

    /// Whether this cache is a ``RemoteKVCache``, whose buffers live on its worker. A prompt
    /// cache file cannot hold it.
    package var isRemote: Bool {
        self is RemoteKVCache
    }
}

extension RemoteModelReply {

    /// The status of a ``RemoteModelReply/cache(_:)`` reply.
    ///
    /// - Parameter op: The name of the op, for the error.
    /// - Returns: The status.
    /// - Throws: ``ClusterModelError/unexpectedReply(op:reply:)`` for any other reply.
    func cacheStatus(of op: String) throws -> RemoteCacheStatus {
        guard case .cache(let status) = self else {
            throw ClusterModelError.unexpectedReply(op: op, reply: String(describing: self))
        }
        return status
    }
}

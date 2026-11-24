// Copyright © 2026 Apple Inc.

import Foundation

/// The constants of the remote model protocol, which ``ClusterLanguageModel`` speaks to a worker.
///
/// A distributed model runs in worker processes, one for each Mac. The app process keeps
/// sampling, logit processors, grammar, stop strings, tool and reasoning parsing, and the
/// prompt-cache policy. Only the forward pass and the cache state live in the worker. The app
/// drives each step with one ``RemoteModelRequest``, and the worker answers with one
/// ``RemoteModelReply``.
///
/// ## The ops
///
/// | Op | Reply | What the worker does |
/// |---|---|---|
/// | ``RemoteModelOp/describe`` | ``RemoteModelReply/capabilities(_:)`` | Names its protocol version, model families and sharding. |
/// | ``RemoteModelOp/begin`` | ``RemoteModelReply/done`` | Starts the request that ``RemoteModelRequest/requestID`` names. |
/// | ``RemoteModelOp/end`` | ``RemoteModelReply/done`` | Ends that request. Its announce raises the stop flag. |
/// | ``RemoteModelOp/newCache(_:)`` | ``RemoteModelReply/cache(_:)`` | Makes an empty cache under the handle the app chose. |
/// | ``RemoteModelOp/forward(tokens:cache:logits:)`` | ``RemoteModelReply/forwarded(_:logits:)`` | Runs the tokens through the model into the cache. |
/// | ``RemoteModelOp/trim(_:count:)`` | ``RemoteModelReply/trimmed(_:status:)`` | Drops the newest positions of the cache. |
/// | ``RemoteModelOp/copy(_:into:)`` | ``RemoteModelReply/cache(_:)`` | Copies a cache under a second handle the app chose. |
/// | ``RemoteModelOp/reset(_:)`` | ``RemoteModelReply/cache(_:)`` | Empties a cache and keeps its handle. |
/// | ``RemoteModelOp/release(_:)`` | ``RemoteModelReply/done`` | Frees a cache and forgets its handle. |
///
/// Any op can get ``RemoteModelReply/failed(_:)`` instead.
///
/// ## The rules of the stream
///
/// - Every request carries ``RemoteModelRequest/sequence``, which starts at 1 and grows by one
///   for each request the app sends. A rank that sees a gap must fail, not skip.
/// - The app allocates every cache handle. A worker only maps a handle to its caches.
/// - A forward carries at most ``maximumChunkTokenCount`` tokens.
/// - No op carries a path, a command or any other text the worker would run. Every field is a
///   typed value.
/// - A cache handle names the caches of every layer of the model, so one op moves the whole
///   cache. The worker keeps the model state (``LMOutput/State``) of each cache beside it.
///
/// ## The cache status
///
/// Each op that changes a cache answers with a ``RemoteCacheStatus``. The app mirrors it, so a
/// read of ``KVCache/offset`` or ``KVCache/maxTrimCount`` costs no round trip. A worker builds
/// it with ``RemoteCacheStatus/init(describing:)``.
public enum RemoteModelProtocol {

    /// The version of the op set. A worker that names another version is refused.
    public static let version: Int32 = 1

    /// The most tokens one forward carries. A prefill longer than this goes in several forwards.
    public static let maximumChunkTokenCount = 1024
}

/// The handle of the caches of every layer of one sequence, which the app allocates.
public struct RemoteCacheID: Hashable, Codable, Sendable, CustomStringConvertible {

    /// The value of the handle.
    public let rawValue: UInt64

    /// Makes a handle of `rawValue`.
    ///
    /// - Parameter rawValue: The value of the handle.
    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    /// The handle in words, for a log or an error.
    public var description: String { "cache \(rawValue)" }
}

/// The identifier of one generation request, which the app allocates.
///
/// An op outside a ``RemoteModelOp/begin`` and ``RemoteModelOp/end`` pair carries the
/// identifier of the last request, or ``none`` before the first.
public struct RemoteRequestID: Hashable, Codable, Sendable {

    /// The value of the identifier.
    public let rawValue: UInt64

    /// Makes an identifier of `rawValue`.
    ///
    /// - Parameter rawValue: The value of the identifier.
    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    /// The identifier of the ops before the first request.
    public static let none = RemoteRequestID(rawValue: 0)
}

/// Which logits a forward returns.
public enum RemoteLogitsScope: Int32, Codable, Sendable, CaseIterable {

    /// No logits, for a prefill chunk whose logits nobody reads.
    case none = 0

    /// The logits of the last position, shape `[1, 1, vocabulary]`. The default of a decode step.
    case lastPosition = 1

    /// The logits of every position, shape `[1, tokens, vocabulary]`.
    case allPositions = 2
}

/// One operation of a remote model.
public enum RemoteModelOp: Codable, Sendable, Hashable {

    /// Asks for the ``RemoteModelCapabilities`` of the worker.
    case describe

    /// Starts a generation request.
    case begin

    /// Ends the current generation request.
    case end

    /// Makes an empty cache under `cache`, a handle the worker does not hold yet.
    case newCache(RemoteCacheID)

    /// Runs `tokens` through the model into `cache`, and returns the logits of `logits`.
    case forward(tokens: [Int32], cache: RemoteCacheID, logits: RemoteLogitsScope)

    /// Drops the newest `count` positions of the cache.
    case trim(RemoteCacheID, count: Int)

    /// Copies the first cache under the second handle, which the worker does not hold yet.
    case copy(RemoteCacheID, into: RemoteCacheID)

    /// Empties the cache, and keeps its handle.
    case reset(RemoteCacheID)

    /// Frees the cache, and forgets its handle.
    case release(RemoteCacheID)
}

/// One op with the identifiers that order it.
public struct RemoteModelRequest: Codable, Sendable, Hashable {

    /// The generation request the op belongs to.
    public let requestID: RemoteRequestID

    /// The position of the op in the stream: 1 for the first op, one more for each next op.
    public let sequence: UInt64

    /// The op.
    public let op: RemoteModelOp

    /// Makes a request.
    ///
    /// - Parameters:
    ///   - requestID: The generation request the op belongs to.
    ///   - sequence: The position of the op in the stream.
    ///   - op: The op.
    public init(requestID: RemoteRequestID, sequence: UInt64, op: RemoteModelOp) {
        self.requestID = requestID
        self.sequence = sequence
        self.op = op
    }
}

/// What a worker can run.
public struct RemoteModelCapabilities: Codable, Sendable, Hashable {

    /// The ``RemoteModelProtocol/version`` the worker speaks.
    public let protocolVersion: Int32

    /// The `model_type` values of the model families the worker can load.
    public let modelFamilies: [String]

    /// Whether the worker shards the model over several ranks.
    public let supportsSharding: Bool

    /// The number of ranks the model runs on.
    public let rankCount: Int

    /// The rows of the vocabulary projection: the last dimension of the logits.
    public let vocabularySize: Int

    /// The prefill step of the model when the caller sets none.
    public let defaultPrefillStepSize: Int

    /// The largest prefill step of the model, or nil when it has none.
    public let maximumPrefillStepSize: Int?

    /// Makes the capabilities of a worker.
    ///
    /// - Parameters:
    ///   - protocolVersion: The protocol version the worker speaks.
    ///   - modelFamilies: The `model_type` values the worker can load.
    ///   - supportsSharding: Whether the worker shards the model.
    ///   - rankCount: The number of ranks the model runs on.
    ///   - vocabularySize: The last dimension of the logits.
    ///   - defaultPrefillStepSize: The prefill step of the model when the caller sets none.
    ///   - maximumPrefillStepSize: The largest prefill step of the model, or nil.
    public init(
        protocolVersion: Int32, modelFamilies: [String], supportsSharding: Bool, rankCount: Int,
        vocabularySize: Int, defaultPrefillStepSize: Int, maximumPrefillStepSize: Int?
    ) {
        self.protocolVersion = protocolVersion
        self.modelFamilies = modelFamilies
        self.supportsSharding = supportsSharding
        self.rankCount = rankCount
        self.vocabularySize = vocabularySize
        self.defaultPrefillStepSize = defaultPrefillStepSize
        self.maximumPrefillStepSize = maximumPrefillStepSize
    }
}

/// The answer of a worker to one ``RemoteModelRequest``.
public enum RemoteModelReply: Codable, Sendable, Hashable {

    /// The answer to ``RemoteModelOp/describe``.
    case capabilities(RemoteModelCapabilities)

    /// The answer to an op that changes no cache.
    case done

    /// The status of the cache that the op made, copied or reset.
    case cache(RemoteCacheStatus)

    /// The positions a trim dropped, and the status after it.
    case trimmed(Int, status: RemoteCacheStatus)

    /// The status of the cache after a forward, and the logits the forward asked for.
    case forwarded(RemoteCacheStatus, logits: RemoteLogits?)

    /// The worker could not run the op. The text says why.
    case failed(String)
}

/// The errors of the remote model protocol itself: a value that does not fit the op set.
public enum RemoteModelProtocolError: Error, Hashable, LocalizedError {

    /// A forward of more tokens than ``RemoteModelProtocol/maximumChunkTokenCount``.
    case chunkTooLong(tokenCount: Int)

    /// Announce words of another length than ``RemoteOpAnnouncement/wordCount``.
    case announceLength(Int)

    /// An announce whose opcode names no op.
    case unknownOpcode(Int32)

    /// An announce whose logits field names no ``RemoteLogitsScope``.
    case unknownLogitsScope(Int32)

    /// An announce whose count does not fit its op.
    case invalidCount(Int32)

    /// Logits whose bytes do not fill their shape.
    case logitsByteCount(expected: Int, found: Int)

    /// A description of the error.
    public var errorDescription: String? {
        switch self {
        case .chunkTooLong(let tokenCount):
            "A remote forward carries at most \(RemoteModelProtocol.maximumChunkTokenCount) "
                + "tokens, but this one carries \(tokenCount)."
        case .announceLength(let count):
            "A remote op announce holds \(RemoteOpAnnouncement.wordCount) words, not \(count)."
        case .unknownOpcode(let opcode):
            "No remote op has the opcode \(opcode)."
        case .unknownLogitsScope(let scope):
            "No remote logits scope has the value \(scope)."
        case .invalidCount(let count):
            "A remote op announce cannot carry the count \(count)."
        case .logitsByteCount(let expected, let found):
            "Remote logits need \(expected) bytes for their shape, but carry \(found)."
        }
    }
}

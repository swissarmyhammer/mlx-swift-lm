// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The worker side of the remote model protocol: runs each ``RemoteModelRequest`` on a loaded
/// model and answers with its ``RemoteModelReply``.
///
/// The executor holds no networking. A worker adds the transport to the app, the distributed
/// group, and the announce loop. Each rank of a sharded model holds its own executor over its
/// own shard.
///
/// ## Rank 0 and the other ranks
///
/// Rank 0 gets each request from the app. It announces the request to every rank with
/// ``RemoteOpAnnouncement/exchange(_:group:)``, then calls ``handle(_:)`` and sends the reply
/// to the app. Every other rank gets the same words from the exchange and calls
/// ``apply(announcement:)``, which runs the same code and whose reply it drops.
///
/// Every rank must apply the same ops in the same order. A sharded layer sums its partial
/// results across the group inside the forward pass, so a forward that runs on one rank and
/// not on another blocks both, or mixes their buffers. Thus a rank other than 0 runs each
/// forward too, although nobody reads its logits.
///
/// ## The rules of the stream
///
/// Each request must carry the sequence number after the last one, starting at 1. A request
/// with another number gets ``RemoteModelReply/failed(_:)`` naming the expected and the found
/// number, and the executor still waits for the expected one. A ``RemoteModelOp/begin`` names
/// a new request id, larger than the last one, while no request is open. Every other op
/// carries the id of the open request, or of the last request when none is open
/// (``RemoteRequestID/none`` before the first). A request that breaks a rule, or names a cache
/// handle the executor does not hold (or, for a new cache, does hold), gets a failed reply and
/// changes nothing.
///
/// A park (see ``RemoteOpAnnouncement/park(requestID:)``) changes nothing either: the op after
/// it carries the same request id and the next sequence number.
///
/// ## Threads
///
/// The executor is not thread-safe. Call it from one thread, one request at a time. The model
/// runs on the default stream, as a local model does.
public final class RemoteModelExecutor {

    /// The `model_type` values whose model conforms to ``TensorParallelShardable`` and shards
    /// with no error, as the LLM factory loads them, or the VLM factory for a type only it
    /// knows.
    ///
    /// `qwen3_5_moe` is not here: its mixture-of-experts layers do not shard.
    /// `prism_hadamard_qwen35` is not either: its rotated projections are not stock projections.
    public static let shardableModelFamilies = [
        "inkling2", "inkling2_loop", "qwen3_5", "qwen3_5_text", "qwen3_8", "qwen3_8_text",
    ]

    /// What a worker runs before it loads a model, for a planner.
    public static let capabilities = RemoteWorkerCapabilities(
        protocolVersion: RemoteModelProtocol.version, modelFamilies: shardableModelFamilies,
        supportsSharding: true)

    /// The model this rank runs: the whole model, or this rank's shard of it.
    private let model: any LanguageModel

    /// What ``RemoteModelOp/describe`` answers.
    private let modelCapabilities: RemoteModelCapabilities

    /// The sequence and request rules.
    private var stream = RemoteRequestStream()

    /// The caches and model state of each handle.
    private var handles: [RemoteCacheID: RemoteCacheHandle] = [:]

    /// Makes an executor over `model`.
    ///
    /// - Parameters:
    ///   - model: The loaded model, sharded already when it runs on several ranks.
    ///   - configuration: The vocabulary and the rank count of the model.
    public init(model: any LanguageModel, configuration: Configuration) {
        self.model = model
        self.modelCapabilities = RemoteModelCapabilities(
            protocolVersion: Self.capabilities.protocolVersion,
            modelFamilies: Self.capabilities.modelFamilies,
            supportsSharding: Self.capabilities.supportsSharding
                && model is TensorParallelShardable,
            rankCount: configuration.rankCount, vocabularySize: configuration.vocabularySize,
            defaultPrefillStepSize: model.defaultPrefillStepSize,
            maximumPrefillStepSize: model.maximumPrefillStepSize)
    }

    /// The cache handles the executor holds.
    public var cacheCount: Int { handles.count }

    /// Runs `request` on the model.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply. A request that breaks a rule of the stream, or names a handle the
    ///   executor cannot use, gets ``RemoteModelReply/failed(_:)``.
    /// - Throws: The error of the model when it cannot make a cache.
    public func handle(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        do {
            try stream.admit(request)
            return try perform(request.op)
        } catch let error as RemoteModelExecutorError {
            return .failed(error.localizedDescription)
        } catch let error as RemoteModelProtocolError {
            return .failed(error.localizedDescription)
        }
    }

    /// Runs the op of announce `words` on this rank, as ``handle(_:)`` runs it on rank 0.
    ///
    /// A rank other than 0 calls this with the words of each exchange, and drops the reply.
    ///
    /// - Parameter words: The words of one announce, from
    ///   ``RemoteOpAnnouncement/exchange(_:group:)``.
    /// - Returns: The reply of ``handle(_:)``, or nil for a park.
    /// - Throws: ``RemoteModelProtocolError`` when the words name no op, or the error of
    ///   ``handle(_:)``.
    public func apply(announcement words: [Int32]) throws -> RemoteModelReply? {
        guard !RemoteOpAnnouncement.isPark(words) else { return nil }
        return try handle(RemoteOpAnnouncement.decode(words))
    }

    /// Runs `op`, which the stream admitted.
    private func perform(_ op: RemoteModelOp) throws -> RemoteModelReply {
        switch op {
        case .describe:
            return .capabilities(modelCapabilities)
        case .begin, .end:
            return .done
        case .newCache(let id):
            guard handles[id] == nil else { throw RemoteModelExecutorError.cacheExists(id) }
            handles[id] = RemoteCacheHandle(layers: try model.newCache(parameters: nil))
            return .cache(try status(of: id))
        case .forward(let tokens, let id, let scope):
            return try forward(tokens, cache: id, scope: scope)
        case .trim(let id, let count):
            return try trim(id, count: count)
        case .copy(let id, let target):
            return try copy(id, into: target)
        case .reset(let id):
            _ = try handle(of: id)
            handles[id] = RemoteCacheHandle(layers: try model.newCache(parameters: nil))
            return .cache(try status(of: id))
        case .release(let id):
            _ = try handle(of: id)
            handles[id] = nil
            return .done
        }
    }

    /// Runs `tokens` into the caches of `id`, as a local generation runs them.
    private func forward(
        _ tokens: [Int32], cache id: RemoteCacheID, scope: RemoteLogitsScope
    ) throws -> RemoteModelReply {
        guard !tokens.isEmpty else { throw RemoteModelExecutorError.emptyForward }
        guard tokens.count <= RemoteModelProtocol.maximumChunkTokenCount else {
            throw RemoteModelProtocolError.chunkTooLong(tokenCount: tokens.count)
        }
        let handle = try handle(of: id)
        let input = LMInput.Text(tokens: MLXArray(tokens)[.newAxis])
        let output =
            scope == .allPositions
            ? model(input, cache: handle.layers, state: handle.state)
            : model.nextTokenLogits(input, cache: handle.layers, state: handle.state)
        handles[id]?.state = output.state
        eval([output.logits] + handle.layers.flatMap(\.evaluationState))
        return .forwarded(
            RemoteCacheStatus(describing: handle.layers),
            logits: scope == .none ? nil : RemoteLogits(output.logits))
    }

    /// Drops the newest `count` positions of the caches of `id`.
    private func trim(_ id: RemoteCacheID, count: Int) throws -> RemoteModelReply {
        guard count >= 0 else {
            throw RemoteModelProtocolError.invalidCount(Int32(clamping: count))
        }
        let layers = try handle(of: id).layers
        let trimmed = trimPromptCache(layers, numTokens: count)
        return .trimmed(trimmed, status: RemoteCacheStatus(describing: layers))
    }

    /// Copies the caches and the model state of `id` under `target`.
    private func copy(_ id: RemoteCacheID, into target: RemoteCacheID) throws -> RemoteModelReply {
        let source = try handle(of: id)
        guard handles[target] == nil else { throw RemoteModelExecutorError.cacheExists(target) }
        handles[target] = RemoteCacheHandle(
            layers: source.layers.map { $0.copy() }, state: source.state)
        return .cache(try status(of: target))
    }

    /// The caches and state of `id`.
    private func handle(of id: RemoteCacheID) throws -> RemoteCacheHandle {
        guard let handle = handles[id] else { throw RemoteModelExecutorError.unknownCache(id) }
        return handle
    }

    /// The status of the caches of `id`.
    private func status(of id: RemoteCacheID) throws -> RemoteCacheStatus {
        RemoteCacheStatus(describing: try handle(of: id).layers)
    }
}

extension RemoteModelExecutor {

    /// What the executor reports of its model beside the model itself.
    public struct Configuration: Sendable, Hashable {

        /// The rows of the vocabulary projection: the last dimension of the logits.
        public var vocabularySize: Int

        /// The number of ranks the model runs on.
        public var rankCount: Int

        /// Makes a configuration.
        ///
        /// - Parameters:
        ///   - vocabularySize: The last dimension of the logits.
        ///   - rankCount: The number of ranks the model runs on.
        public init(vocabularySize: Int, rankCount: Int = 1) {
            self.vocabularySize = vocabularySize
            self.rankCount = rankCount
        }

        /// Reads the vocabulary from the `config.json` of `modelDirectory`: its `vocab_size`,
        /// or the `vocab_size` of its `text_config`.
        ///
        /// - Parameters:
        ///   - modelDirectory: The checkpoint directory the model loaded from.
        ///   - rankCount: The number of ranks the model runs on.
        /// - Throws: ``RemoteModelExecutorError/missingVocabularySize`` when neither field is
        ///   there, or the error of the read.
        public init(modelDirectory: URL, rankCount: Int = 1) throws {
            let data = try Data(contentsOf: modelDirectory.appending(component: "config.json"))
            let fields = try JSONDecoder.json5().decode(VocabularyFields.self, from: data)
            guard let vocabularySize = fields.vocabularySize ?? fields.textConfig?.vocabularySize
            else {
                throw RemoteModelExecutorError.missingVocabularySize
            }
            self.init(vocabularySize: vocabularySize, rankCount: rankCount)
        }
    }
}

/// The fields of a `config.json` that name the vocabulary.
private struct VocabularyFields: Decodable {

    /// The `vocab_size` of the configuration.
    let vocabularySize: Int?

    /// The text configuration of a vision-language model.
    let textConfig: TextFields?

    /// The `vocab_size` of a text configuration.
    struct TextFields: Decodable {

        /// The `vocab_size`.
        let vocabularySize: Int?

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case textConfig = "text_config"
    }
}

/// The caches of every layer of one handle, and the model state the last forward left.
private struct RemoteCacheHandle {

    /// The cache of each layer.
    let layers: [any KVCache]

    /// The model state the last forward left, or nil before one.
    var state: LMOutput.State?

    /// Makes a handle of `layers`.
    init(layers: [any KVCache], state: LMOutput.State? = nil) {
        self.layers = layers
        self.state = state
    }
}

/// The sequence and request rules of ``RemoteModelExecutor``.
private struct RemoteRequestStream {

    /// Whether a request is open, and the id of the open or the last request.
    private enum RequestState {
        /// The request is begun and not ended.
        case open(RemoteRequestID)
        /// No request is open. The id is the one of the last request.
        case closed(RemoteRequestID)
    }

    /// The sequence number the next request must carry.
    private var expectedSequence: UInt64 = 1

    /// The request state.
    private var state = RequestState.closed(.none)

    /// Takes `request` when it follows the rules, and moves the state on.
    ///
    /// A request with the wrong sequence number moves nothing. A request with the right one
    /// takes its sequence number even when it breaks a request rule.
    ///
    /// - Parameter request: The request.
    /// - Throws: The ``RemoteModelExecutorError`` of the broken rule.
    mutating func admit(_ request: RemoteModelRequest) throws {
        guard request.sequence == expectedSequence else {
            throw RemoteModelExecutorError.sequenceGap(
                expected: expectedSequence, found: request.sequence)
        }
        expectedSequence += 1
        switch (request.op, state) {
        case (.begin, .open(let open)):
            throw RemoteModelExecutorError.requestStillOpen(open)
        case (.begin, .closed(let last)):
            guard request.requestID.rawValue > last.rawValue else {
                throw RemoteModelExecutorError.staleRequest(request.requestID)
            }
            state = .open(request.requestID)
        case (.end, .closed):
            throw RemoteModelExecutorError.noOpenRequest
        case (.end, .open(let open)):
            try Self.expect(open, in: request)
            state = .closed(open)
        case (_, .open(let current)), (_, .closed(let current)):
            try Self.expect(current, in: request)
        }
    }

    /// Throws unless `request` carries `id`.
    private static func expect(_ id: RemoteRequestID, in request: RemoteModelRequest) throws {
        guard request.requestID == id else {
            throw RemoteModelExecutorError.requestMismatch(expected: id, found: request.requestID)
        }
    }
}

/// Why a ``RemoteModelExecutor`` refused a request, or a configuration.
public enum RemoteModelExecutorError: Error, Hashable, LocalizedError {

    /// The request carries another sequence number than the one after the last request.
    case sequenceGap(expected: UInt64, found: UInt64)

    /// The request carries another id than the open or the last request.
    case requestMismatch(expected: RemoteRequestID, found: RemoteRequestID)

    /// A begin arrived while this request is still open.
    case requestStillOpen(RemoteRequestID)

    /// A begin names an id that is not larger than the id of the last request.
    case staleRequest(RemoteRequestID)

    /// An end arrived while no request is open.
    case noOpenRequest

    /// The op names a cache handle the executor does not hold.
    case unknownCache(RemoteCacheID)

    /// The op makes a cache under a handle the executor holds already.
    case cacheExists(RemoteCacheID)

    /// A forward of no tokens.
    case emptyForward

    /// The `config.json` names no vocabulary size.
    case missingVocabularySize

    /// A description of the error.
    public var errorDescription: String? {
        switch self {
        case .sequenceGap(let expected, let found):
            "The worker expected request sequence \(expected), but got \(found)."
        case .requestMismatch(let expected, let found):
            "The worker expected request \(expected.rawValue), but got \(found.rawValue)."
        case .requestStillOpen(let open):
            "The worker cannot begin a request while request \(open.rawValue) is open."
        case .staleRequest(let id):
            "Request \(id.rawValue) does not follow the last request of the worker."
        case .noOpenRequest:
            "The worker cannot end a request when none is open."
        case .unknownCache(let id):
            "The worker holds no \(id)."
        case .cacheExists(let id):
            "The worker holds \(id) already."
        case .emptyForward:
            "A remote forward needs at least one token."
        case .missingVocabularySize:
            "The config.json of the model names no vocab_size."
        }
    }
}

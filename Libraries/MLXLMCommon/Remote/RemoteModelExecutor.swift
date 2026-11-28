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
/// ## The drafter of rank 0
///
/// Rank 0 can hold the MTP drafter of the model. Its executor then keeps a drafter state beside
/// each handle and feeds it every forward of the handle, so the state stands where the cache
/// stands across the turns of a prompt cache. A ``RemoteModelOp/speculate(_:)`` round drafts on
/// rank 0 alone, and hands the drafts to every rank through the ``RemoteDraftExchange`` each
/// executor was given, the one collective an op runs outside its forward pass. Every rank runs
/// the exchange of a round whose draft limit is above zero, a rank with no drafter too, and no
/// rank runs it for a round of no drafts.
///
/// ## A quantized cache
///
/// A ``RemoteModelOp/newCache(_:quantization:)`` that carries a ``RemoteKVCacheQuantization``
/// makes the caches of the handle under that affine configuration, on every rank. A handle keeps
/// its quantization across a copy and a reset. Its layers convert only at a
/// ``RemoteModelOp/quantize(_:)``, which checks the compatibility policy of the quantization, and
/// its status names whether a layer still waits to convert.
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
/// A park (see ``RemoteOpAnnouncement/park(requestID:)``) takes no sequence number and changes
/// nothing: the op after it carries the same request id and the next sequence number. It must
/// carry the id of the open request, or of the last request when none is open, as an op does.
/// A park of another request means the ranks are out of step: rank 0 gets a failed reply from
/// ``handlePark(requestID:)``, and ``apply(announcement:)`` throws on every other rank.
///
/// ## Threads
///
/// The executor is not thread-safe. Call it from one thread, one request at a time. The model
/// runs on the default stream, as a local model does.
public final class RemoteModelExecutor {

    /// The `model_type` values whose model conforms to ``TensorParallelShardable`` and shards
    /// with no error, as the LLM factory loads them, or the VLM factory for a type only it
    /// knows.
    public static let shardableModelFamilies = [
        "inkling2", "inkling2_loop", "prism_hadamard_qwen35", "qwen3_5", "qwen3_5_moe",
        "qwen3_5_text", "qwen3_8", "qwen3_8_moe", "qwen3_8_text",
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

    /// The drafting of rank 0, or nil on a rank that holds no drafter.
    private let drafting: RemoteDrafting?

    /// Hands the drafts of rank 0 to every rank inside a round.
    private let draftExchange: RemoteDraftExchange

    /// Makes an executor over `model`.
    ///
    /// Rank 0 passes the MTP drafter of the model, when it holds one, and every rank passes the
    /// same kind of draft exchange: ``RemoteDraftExchange/group(_:)`` over the group of the
    /// ranks, or ``RemoteDraftExchange/alone`` for a model on one rank.
    ///
    /// - Parameters:
    ///   - model: The loaded model, sharded already when it runs on several ranks.
    ///   - configuration: The vocabulary and the rank count of the model.
    ///   - drafter: The MTP drafter of rank 0, or nil on every other rank and for a model that
    ///     does not speculate.
    ///   - draftExchange: Hands the drafts of rank 0 to every rank.
    public init(
        model: any LanguageModel, configuration: Configuration,
        drafter: (any ResumableMTPDrafterModel)? = nil, draftExchange: RemoteDraftExchange = .alone
    ) {
        self.model = model
        self.drafting = drafter.map { RemoteDrafting(drafter: $0, target: model) }
        self.draftExchange = draftExchange
        self.modelCapabilities = RemoteModelCapabilities(
            protocolVersion: Self.capabilities.protocolVersion,
            modelFamilies: Self.capabilities.modelFamilies,
            supportsSharding: Self.capabilities.supportsSharding
                && model is TensorParallelShardable,
            rankCount: configuration.rankCount, vocabularySize: configuration.vocabularySize,
            defaultPrefillStepSize: model.defaultPrefillStepSize,
            maximumPrefillStepSize: model.maximumPrefillStepSize,
            preparesEveryPrompt: model.preparesEveryPrompt,
            modelType: configuration.modelType, quantization: configuration.quantization,
            drafter: drafter.map(RemoteDrafterCapabilities.init))
    }

    /// The cache handles the executor holds.
    public var cacheCount: Int { handles.count }

    /// The cache position of the first entry of the drafter state rank 0 keeps beside the caches
    /// of `id`: zero while the drafter has followed the caches since their first position, and
    /// the position of its last cold start otherwise.
    ///
    /// - Parameter id: The handle of the caches.
    /// - Returns: The position, or nil on a rank that holds no drafter, or for a handle the
    ///   executor does not hold.
    package func drafterStartPosition(of id: RemoteCacheID) -> Int? {
        handles[id]?.drafter?.startPosition
    }

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

    /// Checks a park of `requestID` on rank 0, before rank 0 announces it.
    ///
    /// - Parameter requestID: The request the park names.
    /// - Returns: ``RemoteModelReply/done``, or ``RemoteModelReply/failed(_:)`` when the park
    ///   names another request than the open one (or the last one, when none is open).
    public func handlePark(requestID: RemoteRequestID) -> RemoteModelReply {
        do {
            try stream.admitPark(requestID)
            return .done
        } catch {
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
    /// - Throws: ``RemoteModelExecutorError/requestMismatch(expected:found:)`` for a park of
    ///   another request, ``RemoteModelProtocolError`` when the words name no op, or the error of
    ///   ``handle(_:)``.
    public func apply(announcement words: [Int32]) throws -> RemoteModelReply? {
        if let parked = RemoteOpAnnouncement.parkedRequestID(words) {
            try stream.admitPark(parked)
            return nil
        }
        return try handle(RemoteOpAnnouncement.decode(words))
    }

    /// Runs `op`, which the stream admitted.
    private func perform(_ op: RemoteModelOp) throws -> RemoteModelReply {
        switch op {
        case .describe:
            return .capabilities(modelCapabilities)
        case .begin, .end:
            return .done
        case .newCache(let id, let quantization):
            guard handles[id] == nil else { throw RemoteModelExecutorError.cacheExists(id) }
            handles[id] = try emptyHandle(quantization: try quantization?.configuration())
            return .cache(try status(of: id))
        case .forward(let tokens, let id, let scope):
            return try forward(tokens, cache: id, scope: scope)
        case .trim(let id, let count):
            return try trim(id, count: count)
        case .copy(let id, let target):
            return try copy(id, into: target)
        case .reset(let id):
            handles[id] = try emptyHandle(quantization: try handle(of: id).quantization)
            return .cache(try status(of: id))
        case .release(let id):
            _ = try handle(of: id)
            handles[id] = nil
            return .done
        case .speculate(let round):
            return try speculate(round)
        case .quantize(let id):
            return try quantize(id)
        }
    }

    /// The caches of an empty sequence under `quantization`, and the drafter state rank 0 keeps
    /// beside them.
    ///
    /// The model makes the caches from the quantization, as a local generation makes them from
    /// its parameters, so a model that quantizes a cache from its first position does so here.
    ///
    /// - Parameter quantization: The KV-cache configuration of the caches, or nil for full
    ///   precision.
    private func emptyHandle(quantization: KVCacheConfiguration?) throws -> RemoteCacheHandle {
        RemoteCacheHandle(
            layers: try model.newCache(
                parameters: quantization.map { GenerateParameters(kvCache: $0) }),
            quantization: quantization, drafter: drafting.map(RemoteDrafterTrack.init))
    }

    /// Converts each layer of the caches of `id` that passed the compression start of their
    /// quantization, as a local ``KVCachePlan`` converts its caches, and checks the
    /// compatibility policy of the quantization.
    ///
    /// A handle of no quantization, or one whose every layer converted, changes nothing.
    private func quantize(_ id: RemoteCacheID) throws -> RemoteModelReply {
        let handle = try handle(of: id)
        guard handle.awaitsQuantization, let configuration = handle.quantization else {
            return .cache(status(of: handle))
        }
        var layers = handle.layers
        let isTerminal: Bool
        do {
            isTerminal = try applyKVCacheConfigurationValidated(
                cache: &layers, configuration: configuration
            ).isTerminal
        } catch let error as KVCacheConfigurationError {
            throw RemoteModelExecutorError.cacheConfiguration(error.localizedDescription)
        }
        eval(layers.flatMap(\.evaluationState))
        handles[id]?.layers = layers
        handles[id]?.awaitsQuantization = !isTerminal
        return .cache(try status(of: id))
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
        // A verify keeps the snapshot slots of its tokens until the op after it.
        discardSpeculativePromptCacheCheckpoints(handle.layers)
        let position = Self.position(of: handle)
        let output = run(
            tokens, on: handle, allPositions: scope == .allPositions,
            state: handle.drafter?.emitting(handle.state) ?? handle.state)
        handles[id]?.drafter?.didForward(
            tokens, at: position, hidden: output.state?[mtpLastHiddenStatesKey])
        return .forwarded(
            try settle(id, after: output),
            logits: scope == .none ? nil : RemoteLogits(output.logits))
    }

    /// Runs `tokens` into the caches of `handle` with `state`: the logits of every position
    /// when `allPositions` is true, else of the last.
    private func run(
        _ tokens: [Int32], on handle: RemoteCacheHandle, allPositions: Bool,
        state: LMOutput.State?
    ) -> LMOutput {
        let input = LMInput.Text(tokens: MLXArray(tokens)[.newAxis])
        return allPositions
            ? model(input, cache: handle.layers, state: state)
            : model.nextTokenLogits(input, cache: handle.layers, state: state)
    }

    /// Keeps the model state `output` left for `id`, evaluates the logits, the caches and the
    /// drafter state, and returns the status of the caches.
    private func settle(_ id: RemoteCacheID, after output: LMOutput) throws -> RemoteCacheStatus {
        handles[id]?.state = output.state?.withoutMTPExchange
        let handle = try handle(of: id)
        eval(
            [output.logits] + handle.layers.flatMap(\.evaluationState)
                + (handle.drafter?.heldArrays ?? []))
        return status(of: handle)
    }

    /// Drops the newest `count` positions of the caches of `id`.
    private func trim(_ id: RemoteCacheID, count: Int) throws -> RemoteModelReply {
        guard count >= 0 else {
            throw RemoteModelProtocolError.invalidCount(Int32(clamping: count))
        }
        let layers = try handle(of: id).layers
        let trimmed = Self.rewind(layers, count: count)
        discardSpeculativePromptCacheCheckpoints(layers)
        handles[id]?.drafter?.didTrim(trimmed)
        return .trimmed(trimmed, status: try status(of: id))
    }

    /// Drops the newest `count` positions of `layers`: by a trim, or, for the drafts of a verify
    /// that a recurrent layer cannot trim, by its snapshot slots.
    ///
    /// - Returns: The positions dropped: `count`, or 0 when the layers cannot drop them.
    private static func rewind(_ layers: [KVCache], count: Int) -> Int {
        let trimmed = trimPromptCache(layers, numTokens: count)
        guard trimmed == 0, count > 0 else { return trimmed }
        return rewindSpeculativePromptCache(layers, numTokens: count)
    }

    /// Copies the caches, the model state and the drafter state of `id` under `target`.
    private func copy(_ id: RemoteCacheID, into target: RemoteCacheID) throws -> RemoteModelReply {
        let source = try handle(of: id)
        guard handles[target] == nil else { throw RemoteModelExecutorError.cacheExists(target) }
        discardSpeculativePromptCacheCheckpoints(source.layers)
        var copy = source
        copy.layers = source.layers.map { $0.copy() }
        copy.drafter = source.drafter?.copy()
        handles[target] = copy
        return .cache(try status(of: target))
    }

    // MARK: - Rounds

    /// Runs one speculative round on every rank: rewinds the drafts the round before rejected,
    /// takes the drafts of rank 0 through the draft exchange, and verifies the round's token and
    /// its drafts with one forward.
    ///
    /// Every rank checks the round the same way before the exchange, so a round that fails does
    /// so on every rank, and none of them waits in the exchange.
    private func speculate(_ round: RemoteDraftRound) throws -> RemoteModelReply {
        try round.validate()
        let id = round.cache
        let layers = try handle(of: id).layers
        let rewound = Self.rewind(layers, count: round.rejectedCount)
        guard rewound == round.rejectedCount else {
            throw RemoteModelExecutorError.rewind(requested: round.rejectedCount, rewound: rewound)
        }
        discardSpeculativePromptCacheCheckpoints(layers)
        let position = Self.position(of: try handle(of: id))
        let proposed = handles[id]?.drafter?.propose(
            after: round.token, rejectedCount: round.rejectedCount,
            draftLimit: drafting?.rewindableDraftLimit(round.draftLimit, cache: layers) ?? 0,
            minimumDraftProbability: round.minimumDraftProbability, position: position)
        let drafts =
            round.draftLimit > 0
            ? try draftExchange.drafts(proposed, draftLimit: round.draftLimit) : []
        let output = run(
            round.verifiedTokens(drafts: drafts), on: try handle(of: id),
            allPositions: !drafts.isEmpty, state: try verifyState(of: id, drafting: !drafts.isEmpty)
        )
        handles[id]?.drafter?.didVerify(
            round.token, drafts: drafts, at: position,
            hidden: output.state?[mtpLastHiddenStatesKey])
        return .drafted(
            try settle(id, after: output), drafts: drafts, logits: RemoteLogits(output.logits))
    }

    /// The model state of the verify of a round on `id`: on every rank, a verify of drafts keeps
    /// the snapshot slots of its recurrent layers, which the next op rewinds; on rank 0, it
    /// emits the hidden rows the drafter reads.
    private func verifyState(of id: RemoteCacheID, drafting drafts: Bool) throws
        -> LMOutput.State?
    {
        let handle = try handle(of: id)
        var state = handle.state
        if drafts {
            state = state ?? LMOutput.State()
            state?[mtpEmitFlagKey] = true
            state?[mtpVerifyKeepsSnapshotsKey] = true
        }
        return handle.drafter?.emitting(state) ?? state
    }

    /// The position the caches of `handle` stand at, as their status reports it.
    private static func position(of handle: RemoteCacheHandle) -> Int {
        handle.layers.first?.offset ?? 0
    }

    /// The caches and state of `id`.
    private func handle(of id: RemoteCacheID) throws -> RemoteCacheHandle {
        guard let handle = handles[id] else { throw RemoteModelExecutorError.unknownCache(id) }
        return handle
    }

    /// The status of the caches of `id`.
    private func status(of id: RemoteCacheID) throws -> RemoteCacheStatus {
        status(of: try handle(of: id))
    }

    /// The status of the caches of `handle`, whose bytes count the drafter state too.
    private func status(of handle: RemoteCacheHandle) -> RemoteCacheStatus {
        RemoteCacheStatus(describing: handle.layers, awaitsQuantization: handle.awaitsQuantization)
            .addingResidentBytes(handle.drafter?.residentByteCount ?? 0)
    }
}

extension RemoteModelExecutor {

    /// What the executor reports of its model beside the model itself.
    public struct Configuration: Sendable, Hashable {

        /// The rows of the vocabulary projection: the last dimension of the logits.
        public var vocabularySize: Int

        /// The number of ranks the model runs on.
        public var rankCount: Int

        /// The `model_type` of the loaded model, or nil when the worker names none.
        public var modelType: String?

        /// The default quantization of the loaded model, or nil for none.
        public var quantization: BaseConfiguration.Quantization?

        /// Makes a configuration.
        ///
        /// - Parameters:
        ///   - vocabularySize: The last dimension of the logits.
        ///   - rankCount: The number of ranks the model runs on.
        ///   - modelType: The `model_type` of the loaded model, or nil.
        ///   - quantization: The default quantization of the loaded model, or nil.
        public init(
            vocabularySize: Int, rankCount: Int = 1, modelType: String? = nil,
            quantization: BaseConfiguration.Quantization? = nil
        ) {
            self.vocabularySize = vocabularySize
            self.rankCount = rankCount
            self.modelType = modelType
            self.quantization = quantization
        }

        /// Reads the `config.json` of `modelDirectory`: the vocabulary from its `vocab_size`, or
        /// the `vocab_size` of its `text_config`, and its `model_type` and `quantization`.
        ///
        /// - Parameters:
        ///   - modelDirectory: The checkpoint directory the model loaded from.
        ///   - rankCount: The number of ranks the model runs on.
        /// - Throws: ``RemoteModelExecutorError/missingVocabularySize`` when neither field is
        ///   there, or the error of the read.
        public init(modelDirectory: URL, rankCount: Int = 1) throws {
            let fields = try CheckpointFields(modelDirectory: modelDirectory)
            guard let vocabularySize = fields.vocabularySize ?? fields.textConfig?.vocabularySize
            else {
                throw RemoteModelExecutorError.missingVocabularySize
            }
            self.init(
                vocabularySize: vocabularySize, rankCount: rankCount, modelType: fields.modelType,
                quantization: fields.quantization)
        }
    }
}

/// The fields of a `config.json` that a worker reports: the vocabulary, the `model_type` and the
/// quantization.
struct CheckpointFields: Decodable {

    /// The `vocab_size` of the configuration.
    let vocabularySize: Int?

    /// The `model_type` of the configuration.
    let modelType: String?

    /// The default quantization of the configuration. Its per-layer entries are not read.
    let quantization: BaseConfiguration.Quantization?

    /// Reads the `config.json` of `modelDirectory`.
    ///
    /// - Parameter modelDirectory: The checkpoint directory.
    /// - Throws: The error of the read or of the decode.
    init(modelDirectory: URL) throws {
        let data = try Data(contentsOf: modelDirectory.appending(component: "config.json"))
        self = try JSONDecoder.json5().decode(Self.self, from: data)
    }

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
        case modelType = "model_type"
        case quantization
        case textConfig = "text_config"
    }
}

/// The caches of every layer of one handle, the model state the last forward left, the
/// quantization of the caches, and the drafter state rank 0 keeps beside them.
private struct RemoteCacheHandle {

    /// The cache of each layer. A quantize replaces the layers it converts.
    var layers: [any KVCache]

    /// The model state the last forward left, or nil before one.
    var state: LMOutput.State?

    /// The KV-cache configuration the caches quantize under, or nil for full precision.
    let quantization: KVCacheConfiguration?

    /// Whether a layer still waits to convert under ``quantization``.
    var awaitsQuantization: Bool

    /// The drafter state of rank 0, or nil on a rank without a drafter.
    var drafter: RemoteDrafterTrack?

    /// Makes a handle of the empty `layers`, which wait to convert when they carry a
    /// quantization.
    init(
        layers: [any KVCache], quantization: KVCacheConfiguration?, drafter: RemoteDrafterTrack?
    ) {
        self.layers = layers
        self.quantization = quantization
        self.awaitsQuantization = quantization != nil
        self.drafter = drafter
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
            try Self.expect(open, found: request.requestID)
            state = .closed(open)
        case (_, .open(let current)), (_, .closed(let current)):
            try Self.expect(current, found: request.requestID)
        }
    }

    /// Takes a park of `requestID`, which must name the open request, or the last request when
    /// none is open. A park takes no sequence number and moves no state.
    ///
    /// - Parameter requestID: The request the park names.
    /// - Throws: ``RemoteModelExecutorError/requestMismatch(expected:found:)`` for another
    ///   request.
    func admitPark(_ requestID: RemoteRequestID) throws {
        switch state {
        case .open(let current), .closed(let current):
            try Self.expect(current, found: requestID)
        }
    }

    /// Throws unless `found` is `id`.
    private static func expect(_ id: RemoteRequestID, found: RemoteRequestID) throws {
        guard found == id else {
            throw RemoteModelExecutorError.requestMismatch(expected: id, found: found)
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

    /// The draft exchange gave words that do not hold the drafts of the round: their first word
    /// is `count`.
    case draftWords(count: Int)

    /// A round asked to rewind `requested` rejected drafts, and the caches could rewind
    /// `rewound`.
    case rewind(requested: Int, rewound: Int)

    /// The caches cannot quantize as their ``RemoteKVCacheQuantization`` asks: the reason
    /// names the ``KVCacheConfigurationError``.
    case cacheConfiguration(String)

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
        case .draftWords(let count):
            "The draft exchange gave words of \(count) drafts, which the round cannot hold."
        case .rewind(let requested, let rewound):
            "The worker rewound \(rewound) of the \(requested) rejected drafts of a round."
        case .cacheConfiguration(let reason):
            "The worker cannot quantize the KV cache: \(reason)"
        }
    }
}

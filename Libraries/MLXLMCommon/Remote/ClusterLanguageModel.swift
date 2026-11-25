// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

private let clusterLogger = MLXLogger(label: "cluster-language-model")

/// A language model whose forward pass and caches run in a worker process, reached through a
/// ``RemoteModelTransport``.
///
/// Everything around the forward pass stays in the app: sampling, logit processors, grammar,
/// stop strings, tool and reasoning parsing, and the prompt-cache policy. Put the model in a
/// ``ModelContext`` with the configuration, tokenizer and processor of the checkpoint (see
/// ``loadClusterModelContext(from:using:transport:)``), and ``ModelContainer``, ``ChatSession``
/// and the FoundationModels adapter run on it unchanged.
///
/// The model holds no weights. ``newCache(parameters:)`` makes one ``RemoteKVCache``, which
/// names the caches of every layer on the worker.
///
/// It refuses what a worker cannot run, each with a ``ClusterModelError``: image, video and
/// audio input, speculative decoding, a KV-cache configuration, and a batch of several
/// sequences. A prompt cache file cannot hold its cache.
///
/// A forward that fails cannot throw, because ``LanguageModel/callAsFunction(_:cache:state:)``
/// does not throw. The model then cancels the task that runs it, which ends the generation, and
/// keeps the failure: every later op fails, and the next throwing op reports it.
public final class ClusterLanguageModel: Module, LanguageModel, SuppressedTokensProviding {

    /// The stream of requests to the worker.
    let session: RemoteModelSession

    /// The tokens generation never samples: those of the model family and of
    /// `generation_config.json`, which ``loadClusterModelContext(from:using:transport:)`` sets.
    public var suppressedTokenIds: Set<Int> = []

    /// What the worker reported it runs.
    public let workerCapabilities: RemoteModelCapabilities

    /// Makes a model over `session`, whose worker reported `capabilities`.
    private init(session: RemoteModelSession, capabilities: RemoteModelCapabilities) {
        self.session = session
        self.workerCapabilities = capabilities
        super.init()
    }

    /// Asks the worker behind `transport` what it runs, and makes a model over it.
    ///
    /// - Parameter transport: The transport to the rank-0 worker.
    /// - Returns: The model.
    /// - Throws: ``ClusterModelError/protocolVersionMismatch(expected:found:)`` for a worker of
    ///   another protocol version, or the error of the op.
    public static func connect(over transport: any RemoteModelTransport) throws
        -> ClusterLanguageModel
    {
        let session = RemoteModelSession(transport: transport)
        let reply = try session.perform(.describe)
        guard case .capabilities(let capabilities) = reply else {
            throw ClusterModelError.unexpectedReply(
                op: "describe", reply: String(describing: reply))
        }
        guard capabilities.protocolVersion == RemoteModelProtocol.version else {
            throw ClusterModelError.protocolVersionMismatch(
                expected: RemoteModelProtocol.version, found: capabilities.protocolVersion)
        }
        return ClusterLanguageModel(session: session, capabilities: capabilities)
    }

    /// Tears down the workers through ``RemoteModelTransport/close()``, one time. Every later op
    /// fails with ``ClusterModelError/closed``.
    ///
    /// Call it when no generation runs on the model, for example from the eviction of the model
    /// container, which waits for the generation that holds the container.
    public func close() {
        session.close()
    }

    /// The forward pass runs in a worker: ``LMModelCapabilities/remoteForward``.
    public var capabilities: LMModelCapabilities { [.remoteForward] }

    /// The prefill step of the worker's model, at most one chunk.
    public var defaultPrefillStepSize: Int {
        min(workerCapabilities.defaultPrefillStepSize, RemoteModelProtocol.maximumChunkTokenCount)
    }

    /// The largest prefill step of the worker's model, at most one chunk.
    public var maximumPrefillStepSize: Int? {
        min(
            workerCapabilities.maximumPrefillStepSize ?? RemoteModelProtocol.maximumChunkTokenCount,
            RemoteModelProtocol.maximumChunkTokenCount)
    }

    /// Makes an empty cache on the worker.
    ///
    /// - Parameter parameters: The generation parameters. They must ask for no KV-cache
    ///   configuration and no `maxKVSize`: the worker makes its caches itself.
    /// - Returns: One ``RemoteKVCache``, which names the caches of every layer.
    /// - Throws: ``ClusterModelError/unsupported(_:)`` for a cache configuration, or the error
    ///   of the op.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        guard try parameters?.resolvedKVCacheConfiguration() == nil else {
            throw ClusterModelError.unsupported(.cacheConfiguration)
        }
        let id = session.makeCacheID()
        let status = try session.perform(.newCache(id)).cacheStatus(of: "newCache")
        return [RemoteKVCache(id: id, session: session, status: status)]
    }

    /// Begins a request on the worker, prefills all but the tail of the prompt in chunks, and
    /// leaves the tail to the caller, as a local model does.
    ///
    /// A prompt no longer than one step goes to the caller whole, so its one forward matches
    /// the forward of a local model.
    ///
    /// - Parameters:
    ///   - input: The prompt, text only.
    ///   - cache: The one ``RemoteKVCache`` of ``newCache(parameters:)``.
    ///   - state: Unused: the worker keeps the model state of each cache.
    ///   - prefill: The step size, the chunking and the progress observer.
    /// - Returns: The tokens the caller still has to run.
    /// - Throws: ``ClusterModelError/unsupported(_:)`` for media input or a batch,
    ///   ``ClusterModelError/foreignCache`` for another cache, the error of an op, or
    ///   `CancellationError`.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw ClusterModelError.unsupported(.mediaInput)
        }
        let text = input.text
        guard Self.holdsOneSequence(text.tokens) else {
            throw ClusterModelError.unsupported(.batchedInput)
        }
        let remote = try remoteCache(of: cache)
        try session.beginRequest()
        let stepSize = prefill.resolvedStepSize(
            defaultStepSize: defaultPrefillStepSize, maximumStepSize: maximumPrefillStepSize)
        guard text.tokens.size > stepSize else { return .tokens(text) }

        let tokens = Self.tokenIDs(of: text.tokens)
        var failure: (any Error)?
        let processed = try prefill.forEachChunk(
            total: tokens.count, reserving: prefill.chunking == .remainder ? stepSize : 1,
            defaultStepSize: defaultPrefillStepSize, maximumStepSize: maximumPrefillStepSize
        ) { range in
            guard failure == nil else { return }
            do {
                _ = try remote.forward(Array(tokens[range]), logits: .none)
            } catch {
                failure = error
            }
        }
        if let failure { throw failure }
        return .tokens(text[processed...])
    }

    /// Runs `input` on the worker, and returns the logits of every position.
    ///
    /// - Parameters:
    ///   - input: The tokens, shape `[1, tokens]`.
    ///   - cache: The one ``RemoteKVCache`` of ``newCache(parameters:)``.
    ///   - state: Unused: the worker keeps the model state of each cache.
    /// - Returns: The logits, shape `[1, tokens, vocabulary]`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(input.tokens, cache: cache, logits: .allPositions)
    }

    /// Runs `input` on the worker, and returns the logits of the last position alone.
    ///
    /// - Parameters:
    ///   - input: The tokens, shape `[1, tokens]`.
    ///   - cache: The one ``RemoteKVCache`` of ``newCache(parameters:)``.
    ///   - state: Unused: the worker keeps the model state of each cache.
    /// - Returns: The logits, shape `[1, 1, vocabulary]`.
    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(input.tokens, cache: cache, logits: .lastPosition)
    }

    /// Runs `inputs` on the worker, and returns the logits of every position.
    ///
    /// - Parameters:
    ///   - inputs: The tokens, shape `[1, tokens]`.
    ///   - cache: The one ``RemoteKVCache`` of ``newCache(parameters:)``.
    /// - Returns: The logits, shape `[1, tokens, vocabulary]`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, logits: .allPositions).logits
    }

    /// Runs `tokens` on the worker. A failure, or a batch of several sequences
    /// (``ClusterModelError/unsupported(_:)`` of ``ClusterModelFeature/batchedInput``), cancels
    /// the current task, which ends its generation, keeps the failure on the model, and returns
    /// zero logits of the asked shape.
    private func forward(
        _ tokens: MLXArray, cache: [KVCache]?, logits scope: RemoteLogitsScope
    ) -> LMOutput {
        do {
            guard Self.holdsOneSequence(tokens) else {
                throw ClusterModelError.unsupported(.batchedInput)
            }
            let remote = try remoteCache(of: cache ?? [])
            let logits = try remote.forward(Self.tokenIDs(of: tokens), logits: scope)
            return LMOutput(logits: logits ?? placeholderLogits(of: tokens, scope: .lastPosition))
        } catch {
            session.recordFailure(error)
            clusterLogger.error("A cluster forward failed, cancelling the generation: \(error)")
            withUnsafeCurrentTask { $0?.cancel() }
            return LMOutput(logits: placeholderLogits(of: tokens, scope: scope))
        }
    }

    /// Zero logits of the shape a forward of `tokens` returns for `scope`, which a failed
    /// forward returns: one row per sequence, and one position, or every position of
    /// ``RemoteLogitsScope/allPositions``.
    private func placeholderLogits(of tokens: MLXArray, scope: RemoteLogitsScope) -> MLXArray {
        let sequences = tokens.ndim < Self.batchRank ? 1 : tokens.dim(0)
        let positions = scope == .allPositions ? (tokens.ndim == 0 ? 1 : tokens.dim(-1)) : 1
        return MLXArray.zeros([sequences, positions, workerCapabilities.vocabularySize])
    }

    /// The one remote cache of `cache`, which this model made.
    private func remoteCache(of cache: [KVCache]) throws -> RemoteKVCache {
        guard cache.count == 1, let remote = cache.first as? RemoteKVCache,
            remote.session === session
        else {
            throw ClusterModelError.foreignCache
        }
        return remote
    }

    /// The rank of a batch of tokens, `[sequences, tokens]`. A lower rank holds one sequence.
    private static let batchRank = 2

    /// Whether `tokens` hold one sequence: a rank below ``batchRank``, or a batch of one.
    private static func holdsOneSequence(_ tokens: MLXArray) -> Bool {
        tokens.ndim < batchRank || tokens.dim(0) == 1
    }

    /// The token ids of `tokens`, one sequence of any rank.
    private static func tokenIDs(of tokens: MLXArray) -> [Int32] {
        tokens.flattened().asType(.int32).asArray(Int32.self)
    }
}

// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// The failure an ``InProcessRemoteModelWorker`` throws for the ops a test told it to fail.
public struct InProcessWorkerFault: Error, Equatable, CustomStringConvertible {

    /// Makes the fault.
    public init() {}

    /// The fault in words.
    public var description: String { "the in-process worker is down" }
}

/// A worker that runs a real local model in this process, behind the ``RemoteModelTransport``
/// a ``ClusterLanguageModel`` drives.
///
/// Each request travels as its ``RemoteOpAnnouncement`` words, as rank 0 announces it to the
/// other ranks, so a value the words cannot carry fails the test. The worker keeps the caches
/// and the model state of each handle, as a cluster worker does, and records every request it
/// received.
// swiftlint:disable:next no_unchecked_sendable every access to a stored var holds `lock`
public final class InProcessRemoteModelWorker: RemoteModelTransport, @unchecked Sendable {

    /// The model the worker runs.
    private let model: any LanguageModel

    /// What ``RemoteModelOp/describe`` answers.
    private let capabilities: RemoteModelCapabilities

    /// Guards the stored vars and runs one op at a time.
    private let lock = NSLock()

    /// The caches of each handle.
    private var caches: [RemoteCacheID: [any KVCache]] = [:]

    /// The model state the last forward left for each handle.
    private var states: [RemoteCacheID: LMOutput.State] = [:]

    /// Every request received, in order.
    private var received: [RemoteModelRequest] = []

    /// The requests the worker still runs before each op fails, or nil for no limit.
    private var remainingSuccesses: Int?

    /// Makes a worker over `model`.
    ///
    /// - Parameters:
    ///   - model: The local model the worker runs.
    ///   - vocabularySize: The last dimension of the logits of `model`.
    ///   - protocolVersion: The protocol version the worker reports.
    public init(
        model: any LanguageModel, vocabularySize: Int,
        protocolVersion: Int32 = RemoteModelProtocol.version
    ) {
        self.model = model
        self.capabilities = RemoteModelCapabilities(
            protocolVersion: protocolVersion, modelFamilies: ["qwen3_5_text"],
            supportsSharding: false, rankCount: 1, vocabularySize: vocabularySize,
            defaultPrefillStepSize: model.defaultPrefillStepSize,
            maximumPrefillStepSize: model.maximumPrefillStepSize)
    }

    /// Every request the worker received, in order.
    public var requests: [RemoteModelRequest] {
        lock.withLock { received }
    }

    /// The handles whose caches the worker holds.
    public var liveCacheCount: Int {
        lock.withLock { caches.count }
    }

    /// Makes every op after the next `count` throw ``InProcessWorkerFault``.
    ///
    /// - Parameter count: The ops that still succeed.
    public func failOps(after count: Int) {
        lock.withLock { remainingSuccesses = count }
    }

    /// Runs `request` on the model.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply.
    /// - Throws: ``InProcessWorkerFault`` after ``failOps(after:)``, or the error of the
    ///   announce words.
    public func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        try lock.withLock {
            let announced = try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.encode(request))
            received.append(announced)
            if let remaining = remainingSuccesses {
                guard remaining > 0 else { throw InProcessWorkerFault() }
                remainingSuccesses = remaining - 1
            }
            return try perform(announced.op)
        }
    }

    /// Runs `op`. The caller holds ``lock``.
    private func perform(_ op: RemoteModelOp) throws -> RemoteModelReply {
        switch op {
        case .describe:
            return .capabilities(capabilities)
        case .begin, .end:
            return .done
        case .newCache(let id), .reset(let id):
            caches[id] = try model.newCache(parameters: nil)
            states[id] = nil
            return cacheReply(id)
        case .forward(let tokens, let id, let scope):
            return forward(tokens, cache: id, scope: scope)
        case .trim(let id, let count):
            guard let layers = caches[id] else { return unknown(id) }
            let trimmed = trimPromptCache(layers, numTokens: count)
            return .trimmed(trimmed, status: RemoteCacheStatus(describing: layers))
        case .copy(let id, let target):
            guard let layers = caches[id] else { return unknown(id) }
            caches[target] = layers.map { $0.copy() }
            states[target] = states[id]
            return cacheReply(target)
        case .release(let id):
            caches[id] = nil
            states[id] = nil
            return .done
        }
    }

    /// Runs `tokens` into the caches of `id`, as a local generation would run them.
    private func forward(
        _ tokens: [Int32], cache id: RemoteCacheID, scope: RemoteLogitsScope
    ) -> RemoteModelReply {
        guard let layers = caches[id] else { return unknown(id) }
        let input = LMInput.Text(tokens: MLXArray(tokens)[.newAxis])
        let output =
            scope == .allPositions
            ? model(input, cache: layers, state: states[id])
            : model.nextTokenLogits(input, cache: layers, state: states[id])
        states[id] = output.state
        eval([output.logits] + layers.flatMap(\.evaluationState))
        return .forwarded(
            RemoteCacheStatus(describing: layers),
            logits: scope == .none ? nil : RemoteLogits(output.logits))
    }

    /// The status reply of the caches of `id`.
    private func cacheReply(_ id: RemoteCacheID) -> RemoteModelReply {
        caches[id].map { .cache(RemoteCacheStatus(describing: $0)) } ?? unknown(id)
    }

    /// The reply to an op on a handle the worker does not hold.
    private func unknown(_ id: RemoteCacheID) -> RemoteModelReply {
        .failed("the worker holds no \(id)")
    }
}

extension Qwen35MTPFixture {

    /// A worker that runs the target of the fixture.
    ///
    /// - Returns: The worker.
    public func makeWorker() -> InProcessRemoteModelWorker {
        InProcessRemoteModelWorker(model: target, vocabularySize: Self.vocabularySize)
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

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
public final class InProcessRemoteModelWorker: RemoteModelTransport, @unchecked Sendable {
    // swiftlint:enable no_unchecked_sendable

    /// The executor that runs each op.
    private let executor: RemoteModelExecutor

    /// The protocol version the worker reports in place of the executor's.
    private let protocolVersion: Int32

    /// Guards the stored vars and runs one op at a time.
    private let lock = NSLock()

    /// Every request received, in order.
    private var received: [RemoteModelRequest] = []

    /// The requests the worker still runs before each op fails, or nil for no limit.
    private var remainingSuccesses: Int?

    /// The error each op throws after ``remainingSuccesses`` runs out.
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
    public init(
        model: any LanguageModel, vocabularySize: Int,
        protocolVersion: Int32 = RemoteModelProtocol.version, modelType: String? = nil
    ) {
        self.executor = RemoteModelExecutor(
            model: model,
            configuration: .init(vocabularySize: vocabularySize, modelType: modelType))
        self.protocolVersion = protocolVersion
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

    /// Runs `request` on the executor.
    ///
    /// - Parameter request: The request.
    /// - Returns: The reply.
    /// - Throws: The fault of ``failOps(after:with:)`` after its ops, or the error of the
    ///   announce words.
    public func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        try lock.withLock {
            let announced = try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.encode(request))
            received.append(announced)
            if let remaining = remainingSuccesses {
                guard remaining > 0 else { throw fault }
                remainingSuccesses = remaining - 1
            }
            return reported(try executor.handle(announced))
        }
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
                modelType: capabilities.modelType, quantization: capabilities.quantization))
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

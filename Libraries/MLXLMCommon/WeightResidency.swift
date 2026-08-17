// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// What one request to raise the Metal wired-memory limit asked for, and what
/// the wired-memory manager applied.
public struct WiredMemoryOutcome: Sendable, Equatable {
    /// The number of bytes the request asked to keep wired.
    public let requestedBytes: Int

    /// The number of bytes the wired-memory manager applied.
    public let appliedBytes: Int

    /// Creates an outcome from what a request asked for and what the
    /// wired-memory manager applied.
    ///
    /// - Parameters:
    ///   - requestedBytes: The number of bytes the request asked for.
    ///   - appliedBytes: The number of bytes the manager applied.
    public init(requestedBytes: Int, appliedBytes: Int) {
        self.requestedBytes = requestedBytes
        self.appliedBytes = appliedBytes
    }

    /// A Boolean value that tells whether the manager applied the whole request.
    ///
    /// A short apply leaves weight buffers outside the Metal residency set,
    /// thus each decode step pays for those buffers again.
    public var isFullyApplied: Bool { appliedBytes >= requestedBytes }
}

/// Holds the Metal wired-memory limit high enough for the weights of every
/// model this process loads.
///
/// MLX gives its Metal residency set a capacity of zero until the process
/// raises the wired limit, and the one heap that set holds takes only buffers
/// of 1 MB or less. Every weight tensor is larger than that, thus each weight
/// buffer sits outside the residency set and Metal makes the whole weight set
/// resident again for each command buffer, which is once for each decode step.
///
/// Measured on an M3 Ultra with `mlx-community/DeepSeek-V4-Flash-4bit`, whose
/// weights hold 141 GiB: one decode step takes 2.10 s with the default limit
/// and 0.068 s with the limit raised, which is 31 times faster.
///
/// Order and direction both decide the result:
///
/// - A buffer joins the residency set when it is MADE, thus a limit raised
///   after the load changes nothing. ``MLXLMCommon/loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
///   therefore raises the limit before it reads the first weight file.
/// - A limit that falls again empties the residency set, thus this type never
///   lowers the limit. A request below the standing limit changes nothing.
///
/// The limit is one process-wide value, thus every instance drives the same
/// ``MLX/WiredMemoryManager/shared`` manager. Each instance keeps its own
/// high-water mark.
///
/// Where no wired-memory control exists — an older system, or a device that is
/// not a Metal GPU — every request is a no-op that returns `nil`.
public actor ModelWeightResidency {
    /// The residency the weight load of this library raises.
    public static let shared = ModelWeightResidency()

    /// The share of the weight bytes each request adds for the KV cache and
    /// for the working set of one decode step.
    ///
    /// The wired-memory manager takes the maximum across policy groups, never
    /// the sum, thus a ticket taken later at generation time adds nothing to
    /// this request unless it asks for more by itself. The request therefore
    /// carries its own headroom.
    private static let workingSetShareOfWeights = 0.25

    /// The largest number of bytes any request of this residency asked for.
    ///
    /// This is the standing limit, because a request below it changes nothing.
    /// It is zero until the first request that this device can apply.
    public private(set) var highWaterMarkBytes = 0

    /// The number of bytes the manager applied for the standing request.
    private var appliedBytes = 0

    /// The ticket that holds the standing limit. It never ends while this
    /// residency lives, because a limit that falls empties the residency set.
    private var heldTicket: WiredMemoryTicket?

    /// Creates a residency that has raised nothing yet.
    public init() {}

    /// What the standing request asked for and what the manager applied, or
    /// `nil` when this residency has raised nothing.
    public var outcome: WiredMemoryOutcome? {
        guard highWaterMarkBytes > 0 else { return nil }
        return WiredMemoryOutcome(
            requestedBytes: highWaterMarkBytes, appliedBytes: appliedBytes)
    }

    /// The number of bytes to ask the wired-memory manager for, for a
    /// checkpoint whose weights hold `weightBytes` bytes.
    ///
    /// The answer covers the weights and the working set above them. The caller
    /// clamps it to the recommended working-set size of the device.
    ///
    /// - Parameter weightBytes: The number of bytes of model weights.
    /// - Returns: The number of bytes to ask for.
    package static func limitBytes(forWeightBytes weightBytes: Int) -> Int {
        weightBytes + Int(Double(weightBytes) * workingSetShareOfWeights)
    }

    /// Raises the wired-memory limit of this process so that it covers
    /// `weightBytes`, and never lowers it.
    ///
    /// Call this before the first weight buffer is made. A buffer joins the
    /// Metal residency set when it is made, thus a call after the load gives no
    /// residency benefit at all.
    ///
    /// - Parameter weightBytes: The number of bytes of model weights the load
    ///   is about to allocate.
    /// - Returns: What the standing request asks for and what the manager
    ///   applied, or `nil` when this device has no wired-memory control and
    ///   this residency has raised nothing.
    @discardableResult
    public func raise(toCoverWeightBytes weightBytes: Int) async -> WiredMemoryOutcome? {
        guard weightBytes > 0, let recommended = recommendedWorkingSetBytes() else {
            return outcome
        }
        let requestedBytes = min(recommended, Self.limitBytes(forWeightBytes: weightBytes))
        guard requestedBytes > highWaterMarkBytes else { return outcome }

        let ticket = WiredMemoryTicket(
            size: requestedBytes,
            policy: WiredFixedPolicy(limit: requestedBytes),
            kind: .active)
        appliedBytes = await ticket.start()
        highWaterMarkBytes = requestedBytes

        // The new ticket is active before the ticket it replaces ends, thus the
        // applied limit never falls between the two.
        let replacedTicket = heldTicket
        heldTicket = ticket
        if let replacedTicket {
            _ = await replacedTicket.end()
        }
        return outcome
    }
}

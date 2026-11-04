// Copyright © 2026 Apple Inc.

import MLX

/// The fixed cadence at which a decode loop returns the freed buffers of the MLX pool to the
/// system.
///
/// The pool reuses a freed buffer only for a request of nearly its size, thus freed buffers of odd
/// or growing sizes stay there and nothing asks for them again. A KV cache that grows by whole
/// steps frees its whole old buffers at each step, so at a long context the pool would otherwise
/// grow by the size of the KV cache each time. The cadence counts emitted tokens and never reads
/// the speed of generation or the state of memory. It matches mlx-lm, which also clears on the
/// first token, after prefill.
package struct GenerationBufferPoolClearing: Sendable {

    /// The emitted tokens between two clears.
    private static let tokenInterval = 256

    /// Returns the freed buffers of the pool to the system.
    private let clearPool: @Sendable () -> Void

    /// Makes the cadence.
    ///
    /// - Parameter clearPool: Returns the freed buffers of the pool to the system.
    package init(clearPool: @escaping @Sendable () -> Void = { MLX.Memory.clearCache() }) {
        self.clearPool = clearPool
    }

    /// Clears the pool when the emitted tokens reach a whole number of intervals.
    ///
    /// - Parameter tokenCount: The tokens emitted before the token at hand.
    package func clearPoolIfDue(tokenCount: Int) {
        if tokenCount % Self.tokenInterval == 0 {
            clearPool()
        }
    }
}

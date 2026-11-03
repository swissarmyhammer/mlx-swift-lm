// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// The tokens of `count` successive draws of `sampler` from `logits`.
///
/// - Parameters:
///   - sampler: the sampler to draw with.
///   - logits: the logits of every draw.
///   - count: the number of draws.
///   - draw: runs one draw, given its index and the draw; the default runs it as it is. A test
///     wraps it, for example, in an ambient random state of its own.
/// - Returns: the drawn tokens, in order.
public func sampledTokens(
    _ sampler: any LogitSampler, logits: MLXArray, count: Int,
    draw: (_ index: Int, _ sample: () -> MLXArray) -> MLXArray = { $1() }
) -> [Int] {
    (0 ..< count).map { index in
        draw(index) { sampler.sample(logits: logits) }.item(Int.self)
    }
}

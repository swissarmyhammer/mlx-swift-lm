// Copyright © 2026 Apple Inc.

import MLX

/// The largest absolute elementwise difference between `a` and `b`, computed in float32 so that
/// a bfloat16 or float16 pair does not round the difference to its own precision.
///
/// - Parameters:
///   - a: the first array.
///   - b: the second array, of a shape that broadcasts with `a`.
/// - Returns: the largest absolute difference; it evaluates both arrays.
public func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

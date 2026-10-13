// Copyright © 2026 Apple Inc.

import MLX

/// Helpers that read MLX arrays back into Swift values for the DeepSeek-V4
/// test suites.
///
/// More than one suite compares array values against Swift literals or
/// against a second array. Each of these suites reads the values through
/// these shared helpers, so the reading stays the same in every suite.
///
/// A caller needs no `eval` and no `reshaped(-1)` before a read.
/// `MLXArray.asArray(_:)` evaluates the array itself, and it copies the
/// values out in row-major order for every memory layout: a transposed, a
/// sliced or a broadcast array gives the same list as its contiguous copy.
enum DeepSeekV4TestArrays {

    /// Reads an array back as a flat list of `Float`, in row-major order.
    ///
    /// - Parameter array: The array to read. Its dtype changes to float32
    ///   before the read.
    /// - Returns: Every value of the array, in float32.
    static func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// Reads an array back as a flat list of `Int32`, in row-major order.
    ///
    /// - Parameter array: The array to read. Its dtype changes to int32
    ///   before the read.
    /// - Returns: Every value of the array, in int32.
    static func integers(_ array: MLXArray) -> [Int32] {
        array.asType(.int32).asArray(Int32.self)
    }
}

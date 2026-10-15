// Copyright © 2026 Apple Inc.
//
// The tolerance of the attention kernel tests.
//
// An attention kernel adds the keys with an online softmax, in float32, in
// another order than the SDPA of MLX. Thus the two paths are not bit-exact,
// and each kernel test compares its kernel with the SDPA of MLX in float32
// inside these tolerances.

import MLX

/// The tolerance of an attention kernel against the SDPA of MLX.
enum AttentionKernelTolerance {

    /// The largest gap allowed between a kernel and the SDPA of MLX, as a
    /// part of the largest reference value, for float32 inputs. Both paths
    /// add in float32, each in its own order.
    static let float32: Float = 1e-4

    /// The largest gap for bfloat16 inputs. The reference reads the same
    /// bfloat16 inputs in float32. The kernel rounds its output, and some
    /// kernels also their softmax weights, to bfloat16, whose step is 2^-8
    /// of the value.
    static let bfloat16: Float = 8e-3

    /// The tolerance of one dtype.
    static func of(_ dtype: DType) -> Float {
        dtype == .float32 ? float32 : bfloat16
    }

    /// The largest gap between two outputs, as a part of the largest
    /// absolute value of the expected output.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let reference = expected.asType(.float32)
        let gap = abs(got.asType(.float32) - reference).max().item(Float.self)
        return gap / abs(reference).max().item(Float.self)
    }
}

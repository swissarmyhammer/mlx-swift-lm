// Copyright © 2026 Apple Inc.
//
// The tolerance and the shared operands of the attention kernel tests.
//
// An attention kernel adds the keys with an online softmax, in float32, in
// another order than the SDPA of MLX. Thus the two paths are not bit-exact,
// and each kernel test compares its kernel with the SDPA of MLX in float32
// inside these tolerances.

import MLX
import Testing

/// The tolerance of an attention kernel against the SDPA of MLX.
enum AttentionKernelTolerance {

    /// The dtypes that each attention kernel reads. Each tolerance test runs
    /// every one of them.
    static let dtypes: [DType] = [.float32, .float16, .bfloat16]

    /// The largest gap allowed between a kernel and the SDPA of MLX, as a
    /// part of the largest reference value, for float32 inputs. Both paths
    /// add in float32, each in its own order.
    static let float32: Float = 1e-4

    /// The largest gap for float16 inputs. The reference reads the same
    /// float16 inputs in float32. The kernel rounds its output, and some
    /// kernels also their softmax weights, to float16, whose step is 2^-11
    /// of the value.
    static let float16: Float = 1e-3

    /// The largest gap for bfloat16 inputs. The reference reads the same
    /// bfloat16 inputs in float32. The kernel rounds its output, and some
    /// kernels also their softmax weights, to bfloat16, whose step is 2^-8
    /// of the value.
    static let bfloat16: Float = 8e-3

    /// The tolerance of one dtype.
    static func of(_ dtype: DType) -> Float {
        switch dtype {
        case .float32: float32
        case .float16: float16
        default: bfloat16
        }
    }

    /// The largest gap between two outputs, as a part of the largest
    /// absolute value of the expected output.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let reference = expected.asType(.float32)
        let gap = abs(got.asType(.float32) - reference).max().item(Float.self)
        return gap / abs(reference).max().item(Float.self)
    }
}

/// One shape of an attention kernel test.
struct AttentionKernelCase: CustomTestStringConvertible, Sendable {

    /// The number of query heads of Qwen3.8.
    static let qwen38QueryHeads = 24

    /// The number of key and value heads of Qwen3.8.
    static let qwen38KVHeads = 4

    /// The width of one attention head of Qwen3.8.
    static let qwen38HeadDim = 256

    /// The number of sequences.
    let batch: Int
    /// The number of query heads.
    let queryHeads: Int
    /// The number of key and value heads.
    let kvHeads: Int
    /// The width of one head.
    let headDim: Int
    /// The number of query tokens.
    let queries: Int
    /// The number of cached keys, the new keys included.
    let keys: Int
    /// The number of rows the cache buffer holds. The keys are a slice of
    /// the first `keys` rows, as a `KVCacheSimple` gives them back.
    let capacity: Int

    /// The text the test report shows for this case.
    var testDescription: String {
        "\(batch) x \(queryHeads) heads over \(kvHeads) KV heads of \(headDim), "
            + "\(queries) queries over \(keys) keys of \(capacity) rows"
    }

    /// The Qwen3.8 attention shape of one sequence.
    ///
    /// - Parameters:
    ///   - queries: The number of query tokens.
    ///   - keys: The number of cached keys.
    ///   - capacity: The number of rows of the cache buffer; `keys` when it
    ///     is not given.
    /// - Returns: The case.
    static func qwen38(queries: Int, keys: Int, capacity: Int? = nil) -> AttentionKernelCase {
        AttentionKernelCase(
            batch: 1, queryHeads: qwen38QueryHeads, kvHeads: qwen38KVHeads,
            headDim: qwen38HeadDim, queries: queries, keys: keys, capacity: capacity ?? keys)
    }
}

/// The repeatable random operands of the attention kernel tests of one suite:
/// each suite gives its own seeds.
struct AttentionKernelRandomOperands: Sendable {
    /// The seed of the queries.
    let queries: UInt64
    /// The seed of the keys.
    let keys: UInt64
    /// The seed of the values.
    let values: UInt64

    /// Builds the random operands of one case in one dtype. The keys and the
    /// values are slices of a larger buffer, as a cache holds them.
    ///
    /// - Parameters:
    ///   - attentionCase: The shape of the operands.
    ///   - dtype: The dtype of the operands.
    /// - Returns: The operands.
    func operands(_ attentionCase: AttentionKernelCase, dtype: DType) -> AttentionKernelOperands {
        let bufferShape = [
            attentionCase.batch, attentionCase.kvHeads, attentionCase.capacity,
            attentionCase.headDim,
        ]
        let keyBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(keys)).asType(dtype)
        let valueBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(values)).asType(dtype)
        let queryBuffer = MLXRandom.normal(
            [
                attentionCase.batch, attentionCase.queries, attentionCase.queryHeads,
                attentionCase.headDim,
            ],
            key: MLXRandom.key(queries)
        ).asType(dtype)
        eval(keyBuffer, valueBuffer, queryBuffer)
        return AttentionKernelOperands(
            queries: queryBuffer.swappedAxes(
                AttentionKernelOperands.headAxis, AttentionKernelOperands.rowAxis),
            keys: keyBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            values: valueBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            scale: 1 / Float(attentionCase.headDim).squareRoot())
    }

    /// The random operands of the Qwen3.8 attention shape: `queries` new
    /// tokens over `keys` cached keys.
    ///
    /// - Parameters:
    ///   - queries: The number of query tokens.
    ///   - keys: The number of cached keys.
    ///   - dtype: The dtype of the operands; bfloat16, as the published
    ///     checkpoints hold it, when it is not given.
    /// - Returns: The operands.
    func qwen38(queries: Int, keys: Int, dtype: DType = .bfloat16) -> AttentionKernelOperands {
        operands(.qwen38(queries: queries, keys: keys), dtype: dtype)
    }
}

/// The operands of one attention kernel call.
struct AttentionKernelOperands {

    /// The axis a `(batch, heads, rows, width)` tensor holds its heads on.
    static let headAxis = 1

    /// The axis a `(batch, heads, rows, width)` tensor holds its rows on.
    static let rowAxis = 2

    /// The axis a `(batch, heads, rows, width)` tensor holds the numbers of
    /// one head on.
    static let widthAxis = 3

    /// The queries, shape `(batch, queryHeads, queries, headDim)`, a
    /// transposed view of a `(batch, queries, queryHeads, headDim)` buffer,
    /// as the model gives them.
    let queries: MLXArray
    /// The cached keys, a slice of shape `(batch, kvHeads, keys, headDim)`.
    let keys: MLXArray
    /// The cached values, of the shape of ``keys``.
    let values: MLXArray
    /// The factor the scores take before the softmax.
    let scale: Float

    /// The mask mode of a causal flag.
    static func maskMode(causal: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        causal ? .causal : .none
    }

    /// The operands with only the first `count` keys and values.
    func firstKeys(_ count: Int) -> AttentionKernelOperands {
        AttentionKernelOperands(
            queries: queries, keys: keys[.ellipsis, ..<count, 0...],
            values: values[.ellipsis, ..<count, 0...], scale: scale)
    }

    /// The operands with the same keys and values, read from a transposed
    /// buffer, thus the numbers of one head do not stand side by side.
    func withSpacedHeadNumbers() -> AttentionKernelOperands {
        let spaced = AttentionKernelOperands(
            queries: queries, keys: Self.spaced(keys), values: Self.spaced(values), scale: scale)
        eval(spaced.keys, spaced.values)
        return spaced
    }

    /// The same rows from a transposed buffer.
    private static func spaced(_ rows: MLXArray) -> MLXArray {
        rows.swappedAxes(rowAxis, widthAxis).contiguous().swappedAxes(rowAxis, widthAxis)
    }

    /// The SDPA of MLX over the operands, in float32.
    func reference(causal: Bool) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32), keys: keys.asType(.float32),
            values: values.asType(.float32), scale: scale, mask: Self.maskMode(causal: causal))
    }

    /// Checks that a kernel output over the operands has the shape and the
    /// dtype of the queries, and stays inside the tolerance of the SDPA of
    /// MLX in float32.
    ///
    /// - Parameters:
    ///   - kernel: The kernel output over the operands.
    ///   - causal: Tells if the kernel ran with a causal mask.
    ///   - sourceLocation: The location that a failure reports.
    func expectInsideTheTolerance(
        kernel: MLXArray, causal: Bool, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let reference = reference(causal: causal)
        let dtype = queries.dtype

        #expect(kernel.shape == queries.shape, sourceLocation: sourceLocation)
        #expect(kernel.dtype == dtype, sourceLocation: sourceLocation)
        #expect(
            abs(reference).max().item(Float.self) > 0, "every reference value was zero",
            sourceLocation: sourceLocation)
        let gap = AttentionKernelTolerance.worstGap(kernel, reference)
        #expect(
            gap <= AttentionKernelTolerance.of(dtype),
            "\(dtype), causal \(causal), worst gap \(gap)", sourceLocation: sourceLocation)
    }

    /// Checks that a kernel reads keys and values whose head numbers do not
    /// stand side by side, and stays inside the tolerance of the SDPA of MLX.
    ///
    /// - Parameters:
    ///   - causal: Tells if the kernel runs with a causal mask.
    ///   - sourceLocation: The location that a failure reports.
    ///   - kernel: The kernel call over some operands.
    func expectSpacedHeadNumbersStayInsideTheTolerance(
        causal: Bool, sourceLocation: SourceLocation = #_sourceLocation,
        kernel: (AttentionKernelOperands) -> MLXArray
    ) {
        let spaced = withSpacedHeadNumbers()

        #expect(
            spaced.keys.asData(access: .noCopy).strides[Self.widthAxis] != 1,
            sourceLocation: sourceLocation)
        spaced.expectInsideTheTolerance(
            kernel: kernel(spaced), causal: causal, sourceLocation: sourceLocation)
    }
}

// Copyright © 2026 Apple Inc.
//
// The tolerance, the shared operands and the shared checks of the attention
// kernel tests.
//
// An attention kernel adds the keys with an online softmax, in float32, in
// another order than the SDPA of MLX. Thus the two paths are not bit-exact,
// and each kernel test compares its kernel with the SDPA of MLX in float32
// inside these tolerances.

import MLX
import Testing

/// The tolerance of an attention kernel against the SDPA of MLX.
enum AttentionKernelTolerance {

    /// The tolerance of each dtype that the attention kernels read, one row
    /// for each dtype.
    private static let table: [(dtype: DType, tolerance: Float)] = [
        (.float32, float32), (.float16, float16), (.bfloat16, bfloat16),
    ]

    /// The dtypes that each attention kernel reads. Each tolerance test runs
    /// every one of them.
    static let dtypes = table.map(\.dtype)

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

    /// The largest tolerance of ``table``. A gap past it breaks the tolerance
    /// of each dtype.
    static let largest = table.map(\.tolerance).reduce(0, max)

    /// The tolerance of one dtype, from ``table``.
    ///
    /// - Parameter dtype: The dtype of the kernel inputs.
    /// - Returns: The tolerance, or `nil` when the kernels do not read the
    ///   dtype. No other dtype gives its tolerance in its place.
    static func of(_ dtype: DType) -> Float? {
        table.first { $0.dtype == dtype }?.tolerance
    }

    /// The largest gap between two outputs, as a part of the largest
    /// absolute value of the expected output.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let reference = expected.asType(.float32)
        let gap = abs(got.asType(.float32) - reference).max().item(Float.self)
        return gap / abs(reference).max().item(Float.self)
    }

    /// The check of a positive control: a wrong output is farther from the
    /// expected output than the tolerance of each dtype.
    ///
    /// - Parameters:
    ///   - wrong: The output of a kernel with a known error.
    ///   - expected: The correct output.
    ///   - sourceLocation: The location that a failure reports.
    static func expectPastEachTolerance(
        _ wrong: MLXArray, _ expected: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let gap = worstGap(wrong, expected)
        #expect(gap > largest, "worst gap \(gap)", sourceLocation: sourceLocation)
    }

    /// The check of a path that must give the output of a kernel bit for
    /// bit, with no tolerance: a route, or a model layer that calls the
    /// kernel.
    ///
    /// - Parameters:
    ///   - got: The output of the path.
    ///   - expected: The output that the path must give.
    ///   - sourceLocation: The location that a failure reports.
    static func expectBitForBit(
        _ got: MLXArray, _ expected: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(arrayEqual(got, expected).item(Bool.self), sourceLocation: sourceLocation)
    }
}

/// One attention kernel under test, with the mask of one causal flag. Each
/// kernel suite gives its kernel to the shared checks through this value.
struct AttentionKernelCall {

    /// Tells if a kernel runs over queries, keys and values.
    typealias Check = (_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray) -> Bool

    /// An attention kernel over queries, keys, values and a score factor.
    typealias Attention = (
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> MLXArray

    /// Tells if the kernel runs over some operands.
    let canRun: Check
    /// The kernel output over some operands.
    let attention: Attention
    /// Tells if the kernel applies a causal mask.
    let causal: Bool

    /// Checks that the kernel runs over each of `read`, and does not run over
    /// each of `skipped`.
    ///
    /// - Parameters:
    ///   - read: The operands that the kernel must run over.
    ///   - skipped: The operands that the kernel must not run over.
    ///   - sourceLocation: The location that a failure reports.
    func expectReads(
        _ read: [AttentionKernelOperands], skips skipped: [AttentionKernelOperands],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        for operands in read {
            #expect(
                operands.runs(by: canRun), "must run: \(operands.testDescription)",
                sourceLocation: sourceLocation)
        }
        for operands in skipped {
            #expect(
                !operands.runs(by: canRun), "must not run: \(operands.testDescription)",
                sourceLocation: sourceLocation)
        }
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

    /// The operands with other queries, and the same keys, values and scale.
    ///
    /// - Parameter newQueries: The queries of the new operands.
    /// - Returns: The operands.
    func withQueries(_ newQueries: MLXArray) -> AttentionKernelOperands {
        AttentionKernelOperands(queries: newQueries, keys: keys, values: values, scale: scale)
    }

    /// The operands with one change to the keys and to the values, and the
    /// same queries and scale.
    ///
    /// - Parameter change: The change that the keys and the values each get.
    /// - Returns: The operands.
    func withKeysAndValues(_ change: (MLXArray) -> MLXArray) -> AttentionKernelOperands {
        AttentionKernelOperands(
            queries: queries, keys: change(keys), values: change(values), scale: scale)
    }

    /// The operands with only the first `count` keys and values.
    func firstKeys(_ count: Int) -> AttentionKernelOperands {
        withKeysAndValues { $0[.ellipsis, ..<count, 0...] }
    }

    /// The operands with only the first `count` query heads.
    func firstQueryHeads(_ count: Int) -> AttentionKernelOperands {
        withQueries(queries[0..., ..<count])
    }

    /// The operands with only the first `width` numbers of each head of the
    /// queries, the keys and the values, and the same scale.
    func firstHeadNumbers(_ width: Int) -> AttentionKernelOperands {
        withQueries(queries[.ellipsis, ..<width]).withKeysAndValues { $0[.ellipsis, ..<width] }
    }

    /// The operands with the same keys and values, read from a transposed
    /// buffer, thus the numbers of one head do not stand side by side.
    func withSpacedHeadNumbers() -> AttentionKernelOperands {
        let spaced = withKeysAndValues(Self.spaced)
        eval(spaced.keys, spaced.values)
        return spaced
    }

    /// The same rows from a transposed buffer.
    private static func spaced(_ rows: MLXArray) -> MLXArray {
        rows.swappedAxes(rowAxis, widthAxis).contiguous().swappedAxes(rowAxis, widthAxis)
    }

    /// The text a failure shows for the operands.
    var testDescription: String {
        "queries \(queries.shape) over keys \(keys.shape) in \(queries.dtype)"
    }

    /// Tells if a kernel runs over the operands.
    ///
    /// - Parameter check: The check of the kernel.
    /// - Returns: The result of the check.
    func runs(by check: AttentionKernelCall.Check) -> Bool {
        check(queries, keys, values)
    }

    /// The output of an attention kernel over the operands.
    ///
    /// - Parameter attention: The kernel.
    /// - Returns: The kernel output.
    func output(of attention: AttentionKernelCall.Attention) -> MLXArray {
        attention(queries, keys, values, scale)
    }

    /// A route: an attention call over queries, keys, values, a score factor
    /// and a mask.
    typealias Route = (
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float,
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray

    /// The output of an attention call over the operands and a mask, on the
    /// route that a kernel picks.
    ///
    /// - Parameters:
    ///   - mask: The mask of the call.
    ///   - route: The `scaledDotProductAttention` of a kernel, which picks
    ///     the kernel or the SDPA of MLX.
    /// - Returns: The output of the route.
    func routed(
        mask: MLXFast.ScaledDotProductAttentionMaskMode, through route: Route
    ) -> MLXArray {
        route(queries, keys, values, scale, mask)
    }

    /// The SDPA of MLX over the operands, in their own dtype.
    func sdpa(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        routed(mask: mask) {
            MLXFast.scaledDotProductAttention(
                queries: $0, keys: $1, values: $2, scale: $3, mask: $4)
        }
    }

    /// The SDPA of MLX over the operands, in float32.
    func reference(causal: Bool) -> MLXArray {
        withQueries(queries.asType(.float32))
            .withKeysAndValues { $0.asType(.float32) }
            .sdpa(mask: Self.maskMode(causal: causal))
    }

    /// Checks that a route over the operands picks a kernel: it gives the
    /// kernel output bit for bit, with the mask of the causal flag of the
    /// kernel.
    ///
    /// - Parameters:
    ///   - route: The route under test.
    ///   - call: The kernel that the route must pick.
    ///   - sourceLocation: The location that a failure reports.
    func expectTheRoute(
        _ route: Route, picks call: AttentionKernelCall,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        expectTheRoute(
            route, mask: Self.maskMode(causal: call.causal), gives: output(of: call.attention),
            sourceLocation: sourceLocation)
    }

    /// Checks that a route over the operands keeps the SDPA of MLX: it gives
    /// the SDPA output bit for bit, with the same mask.
    ///
    /// - Parameters:
    ///   - route: The route under test.
    ///   - mask: The mask of the call.
    ///   - sourceLocation: The location that a failure reports.
    func expectTheRoute(
        _ route: Route, keepsTheSDPAWith mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        expectTheRoute(route, mask: mask, gives: sdpa(mask: mask), sourceLocation: sourceLocation)
    }

    /// Checks that a route over the operands and a mask gives an expected
    /// output bit for bit.
    private func expectTheRoute(
        _ route: Route, mask: MLXFast.ScaledDotProductAttentionMaskMode, gives expected: MLXArray,
        sourceLocation: SourceLocation
    ) {
        AttentionKernelTolerance.expectBitForBit(
            routed(mask: mask, through: route), expected, sourceLocation: sourceLocation)
    }

    /// Checks that a kernel runs over the operands, that its output has the
    /// shape and the dtype of the queries, and that its output stays inside
    /// the tolerance of the SDPA of MLX in float32.
    ///
    /// - Parameters:
    ///   - call: The kernel under test.
    ///   - sourceLocation: The location that a failure reports.
    /// - Throws: A failed requirement when the kernel does not run over the
    ///   operands; the kernel call would then stop on its precondition.
    func expectInsideTheTolerance(
        of call: AttentionKernelCall, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try #require(
            runs(by: call.canRun), "must run: \(testDescription)", sourceLocation: sourceLocation)
        let kernel = output(of: call.attention)
        let causal = call.causal
        let reference = reference(causal: causal)
        let dtype = queries.dtype

        #expect(kernel.shape == queries.shape, sourceLocation: sourceLocation)
        #expect(kernel.dtype == dtype, sourceLocation: sourceLocation)
        #expect(
            abs(reference).max().item(Float.self) > 0, "every reference value was zero",
            sourceLocation: sourceLocation)
        let tolerance = try #require(
            AttentionKernelTolerance.of(dtype), "no tolerance for \(dtype)",
            sourceLocation: sourceLocation)
        let gap = AttentionKernelTolerance.worstGap(kernel, reference)
        #expect(
            gap <= tolerance,
            "\(dtype), causal \(causal), worst gap \(gap)", sourceLocation: sourceLocation)
    }

    /// Checks that a kernel reads keys and values whose head numbers do not
    /// stand side by side, and stays inside the tolerance of the SDPA of MLX.
    ///
    /// - Parameters:
    ///   - call: The kernel under test.
    ///   - sourceLocation: The location that a failure reports.
    /// - Throws: A failed requirement of ``expectInsideTheTolerance(of:sourceLocation:)``.
    func expectSpacedHeadNumbersStayInsideTheTolerance(
        of call: AttentionKernelCall, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let spaced = withSpacedHeadNumbers()

        #expect(
            spaced.keys.asData(access: .noCopy).strides[Self.widthAxis] != 1,
            sourceLocation: sourceLocation)
        try spaced.expectInsideTheTolerance(of: call, sourceLocation: sourceLocation)
    }
}

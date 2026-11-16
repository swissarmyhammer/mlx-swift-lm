// Copyright © 2026 Apple Inc.
//
// Parity tests for ``ResidualRMSNormKernel``.
//
// The kernel replaces a residual add and the RMS norm that reads its sum. It
// must give the SAME BITS as the two MLX operations it replaces: one unit in
// the last place of one hidden state reaches every later layer and, through
// the caches, every later token, thus a tolerance would hide a real
// difference. Every parity test below compares the bits.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ResidualRMSNormKernelTests {

    // MARK: - The shapes

    /// The row widths of the parity cases: a row narrower than one SIMD
    /// group of norm reads, a row that leaves lanes of its last SIMD group
    /// empty, one SIMD group, the widest row MLX reads in one pass, the hidden
    /// width of the Qwen3.8 checkpoints, which MLX reads in a loop, and a row
    /// whose loop ends inside a pass.
    private static let widths = [4, 72, 128, 4096, 5120, 6000]

    /// The dtypes of the parity cases: the float32 of the tests, the
    /// bfloat16 of the checkpoints, and float16.
    private static let dtypes: [DType] = [.float32, .bfloat16, .float16]

    /// The rows of the parity cases: one decode token, and many rows, whose
    /// sums of squares a rounding that differs from the MLX kernel misses in
    /// few rows only.
    private static let rowCounts = [1, 64]

    /// The epsilon of the norm, the `rms_norm_eps` of Qwen3.8.
    private static let normEps: Float = 1e-6

    /// The spread of the random norm weights around one.
    private static let normWeightSpread: Float = 0.25

    /// The spread of the random residual and addend values. A wide spread
    /// gives the sum and the norm large values to round.
    private static let valueSpread: Float = 4

    /// The width of the rows of the cases that need only one width.
    private static let plainWidth = 128

    /// The seeds of the random arrays, one for each role.
    private enum Seed {
        static let residual: UInt64 = 51
        static let addend: UInt64 = 52
        static let weight: UInt64 = 53
    }

    /// One parity case of the kernel.
    struct NormCase: CustomTestStringConvertible, Sendable {
        /// The numbers of one row.
        let width: Int
        /// The dtype of the residual, of the addend and of the norm weight.
        let dtype: DType
        /// The rows: the tokens of one sequence.
        let rows: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "width \(width), \(dtype), \(rows) rows"
        }
    }

    /// Each width with each dtype and row count.
    static let normCases = widths.flatMap { width in
        dtypes.flatMap { dtype in
            rowCounts.map { NormCase(width: width, dtype: dtype, rows: $0) }
        }
    }

    // MARK: - Helpers

    /// A random array of `shape`, spread by `spread`, in `dtype`.
    private static func randomArray(
        _ shape: [Int], seed: UInt64, spread: Float, dtype: DType
    ) -> MLXArray {
        (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * spread).asType(dtype)
    }

    /// The residual and the addend of `normCase`, shape `[1, rows, width]`.
    private static func operands(_ normCase: NormCase) -> (residual: MLXArray, addend: MLXArray) {
        let shape = [1, normCase.rows, normCase.width]
        return (
            randomArray(shape, seed: Seed.residual, spread: valueSpread, dtype: normCase.dtype),
            randomArray(shape, seed: Seed.addend, spread: valueSpread, dtype: normCase.dtype)
        )
    }

    /// A norm of `width` numbers whose weight is random around one, in
    /// `dtype`.
    private static func norm(width: Int, dtype: DType) throws -> RMSNorm {
        let norm = RMSNorm(dimensions: width, eps: normEps)
        let weight =
            1 + randomArray([width], seed: Seed.weight, spread: normWeightSpread, dtype: .float32)
        try norm.update(
            parameters: ModuleParameters.unflattened(["weight": weight.asType(dtype)]),
            verify: .all)
        return norm
    }

    /// Checks that two arrays hold the same dtype, the same shape and the
    /// same bits.
    private func expectSameBits(
        _ kernel: MLXArray, _ reference: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(kernel.dtype == reference.dtype, "\(what): dtype", sourceLocation: sourceLocation)
        #expect(kernel.shape == reference.shape, "\(what): shape", sourceLocation: sourceLocation)
        let kernelValues = DeepSeekV4TestArrays.floats(kernel)
        let referenceValues = DeepSeekV4TestArrays.floats(reference)
        let firstGap = zip(kernelValues, referenceValues).enumerated().first {
            $0.element.0.bitPattern != $0.element.1.bitPattern
        }
        #expect(
            firstGap == nil,
            """
            \(what)[\(firstGap?.offset ?? 0)]: kernel \(firstGap?.element.0 ?? 0), \
            op path \(firstGap?.element.1 ?? 0)
            """,
            sourceLocation: sourceLocation)
    }

    // MARK: - The kernel

    @Test(arguments: normCases)
    func theKernelGivesTheBitsOfTheOpPath(normCase: NormCase) throws {
        let (residual, addend) = Self.operands(normCase)
        let norm = try Self.norm(width: normCase.width, dtype: normCase.dtype)
        let kernel = try #require(ResidualRMSNormKernel(width: normCase.width, eps: Self.normEps))

        let fused = try #require(kernel(residual: residual, addend: addend, weight: norm.weight))

        let sum = residual + addend
        expectSameBits(fused.sum, sum, "sum")
        expectSameBits(fused.normed, norm(sum), "normed")
    }

    @Test func aNormWeightOfAnotherDtypeTakesNoKernel() throws {
        let normCase = NormCase(width: Self.plainWidth, dtype: .bfloat16, rows: 1)
        let (residual, addend) = Self.operands(normCase)
        let norm = try Self.norm(width: normCase.width, dtype: .float32)
        let kernel = try #require(ResidualRMSNormKernel(width: normCase.width, eps: Self.normEps))

        #expect(kernel(residual: residual, addend: addend, weight: norm.weight) == nil)
    }

    @Test func aRowOfAnotherWidthTakesNoKernel() throws {
        let normCase = NormCase(width: Self.plainWidth, dtype: .bfloat16, rows: 1)
        let (residual, addend) = Self.operands(normCase)
        let norm = try Self.norm(width: normCase.width, dtype: .bfloat16)
        let kernel = try #require(
            ResidualRMSNormKernel(width: normCase.width + 1, eps: Self.normEps))

        #expect(kernel(residual: residual, addend: addend, weight: norm.weight) == nil)
    }

    @Test func anEmptyRowHasNoKernel() {
        #expect(ResidualRMSNormKernel(width: 0, eps: Self.normEps) == nil)
    }

    // MARK: - The norm

    /// An RMS norm that doubles its output: a subclass whose norm the kernel
    /// does not reproduce.
    private final class DoublingRMSNorm: RMSNorm {
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            super.callAsFunction(x) * 2
        }
    }

    @Test func theNormOfASumRunsTheKernel() throws {
        let normCase = NormCase(width: Self.plainWidth, dtype: .bfloat16, rows: 1)
        let (residual, addend) = Self.operands(normCase)
        let norm = try Self.norm(width: normCase.width, dtype: normCase.dtype)
        eval(residual, addend, norm)

        let fused = norm.normalizingSum(of: residual, adding: addend)

        #expect(DeepSeekV4TestArrays.nodeCount([fused.sum, fused.normed], primitive: "Add") == 0)
    }

    @Test func theNormOfASumKeepsTheNormOfASubclass() {
        let normCase = NormCase(width: Self.plainWidth, dtype: .bfloat16, rows: 1)
        let (residual, addend) = Self.operands(normCase)
        let norm = DoublingRMSNorm(dimensions: normCase.width, eps: Self.normEps)

        let fused = norm.normalizingSum(of: residual, adding: addend)

        expectSameBits(fused.normed, norm(residual + addend), "normed")
    }
}

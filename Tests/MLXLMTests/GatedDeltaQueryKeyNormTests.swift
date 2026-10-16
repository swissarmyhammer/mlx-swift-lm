// Copyright © 2026 Apple Inc.
//
// The contract of the shared q and k RMSNorm of the gated delta net layers
// (``GatedDeltaQueryKeyNorm``). Each row of q comes out with a root mean
// square of `1 / keyHeadDim`, and each row of k with a root mean square of
// `1 / sqrt(keyHeadDim)`, in the dtype of the factors.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

struct GatedDeltaQueryKeyNormTests {

    /// The shared operands of the GDN decode kernel tests.
    typealias Fixture = GDNDecodeKernelFixture

    /// The shape of the q and k inputs, without the head width:
    /// `(batch, steps, heads)`.
    private static let leadingShape = [2, 3, 4]

    /// The seed of the q input.
    private static let querySeed: UInt64 = 20_261_021

    /// The seed of the k input.
    private static let keySeed: UInt64 = 20_261_022

    /// The key head widths of the GDN decode kernel tests.
    static let keyHeadDims = Fixture.headCases.map(\.keyHeadDim)

    /// The float32 reference: each row divided by its root mean square, then
    /// multiplied by `factor`.
    ///
    /// - Parameters:
    ///   - rows: The input rows.
    ///   - factor: The factor of the normalized rows.
    /// - Returns: The normalized rows in float32.
    static func reference(_ rows: MLXArray, factor: Float) -> MLXArray {
        let rows = rows.asType(.float32)
        let meanSquare = mean(rows * rows, axis: -1, keepDims: true)
        let eps = GatedDeltaQueryKeyNorm.eps(keyHeadDim: rows.dim(-1))
        return factor * rows * rsqrt(meanSquare + eps)
    }

    @Test(arguments: keyHeadDims, Fixture.dtypes)
    func qAndKComeOutWithTheirRootMeanSquares(keyHeadDim: Int, dtype: DType) {
        let shape = Self.leadingShape + [keyHeadDim]
        let query = MLXRandom.normal(shape, key: MLXRandom.key(Self.querySeed)).asType(dtype)
        let key = MLXRandom.normal(shape, key: MLXRandom.key(Self.keySeed)).asType(dtype)
        let width = Float(keyHeadDim)
        let expectedQuery = Self.reference(query, factor: 1 / width)
        let expectedKey = Self.reference(key, factor: 1 / width.squareRoot())

        let normalized = GatedDeltaQueryKeyNorm.normalized(query: query, key: key, dtype: dtype)

        #expect(normalized.query.dtype == dtype)
        #expect(normalized.key.dtype == dtype)
        #expect(normalized.query.shape == shape)
        #expect(normalized.key.shape == shape)
        let queryGap = Fixture.worstGap(normalized.query, expectedQuery)
        let keyGap = Fixture.worstGap(normalized.key, expectedKey)
        #expect(queryGap <= Fixture.tolerance(dtype), "worst q gap \(queryGap)")
        #expect(keyGap <= Fixture.tolerance(dtype), "worst k gap \(keyGap)")
    }
}

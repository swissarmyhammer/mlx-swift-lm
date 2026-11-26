// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

/// A sorted quantized gather matmul reads only inside its weights, scales and biases.
///
/// With a reduction depth `K` that is not a multiple of the NAX tile depth (64), the NAX
/// loaders of mlx dequantized the whole tile depth in the tail of `K`, so for the last row of
/// the last expert they read past the end of the buffers. A NaN scale there turned the output
/// NaN, though the zero-padded input multiplies those columns. The tiny Inkling 2 target, whose
/// expert down projection has `K` 32, gave NaN logits from it in parallel test runs. These
/// tests put bytes of `0xFF`, a NaN scale in every format, after each buffer.
@Suite("A sorted quantized gather matmul reads only inside its buffers")
struct QuantizedGatherGuardBytesTests {

    /// The experts of the weights.
    private static let experts = 4

    /// The output columns of each expert.
    private static let outputs = 64

    /// The rows of the input, at least 16 and four per expert, so that the gather takes the
    /// sorted right-hand-side kernel.
    private static let rows = 80

    /// The scale of the random weights.
    private static let weightScale: Float = 0.3

    /// The bytes after each buffer.
    private static let guardByteCount = 4096

    /// Guard bytes that every scale format reads as NaN.
    private static let nanGuard: UInt8 = 0xFF

    /// Guard bytes that every scale format reads as zero.
    private static let zeroGuard: UInt8 = 0

    /// The seed of the random weights.
    private static let weightSeed: UInt64 = 3

    /// The seed of the random input.
    private static let inputSeed: UInt64 = 4

    /// A quantization of the weights.
    struct Format: Sendable, CustomStringConvertible {
        let mode: QuantizationMode
        let groupSize: Int
        let bits: Int

        var description: String { "\(mode) \(bits)-bit, groups of \(groupSize)" }
    }

    /// The formats under test: an affine one and NVFP4 with a global scale.
    static let formats = [
        Format(mode: .affine, groupSize: 32, bits: 4),
        Format(mode: .nvfp4, groupSize: 16, bits: 4),
    ]

    /// Reduction depths that are not a multiple of the NAX tile depth.
    static let unalignedDepths = [32, 96]

    /// `array` as a view of the front of a buffer whose last `guardByteCount` bytes are `fill`.
    private static func followed(by fill: UInt8, _ array: MLXArray) -> MLXArray {
        let bytes = array.reshaped(-1).view(dtype: .uint8)
        let padded = concatenated([
            bytes, MLXArray.full([guardByteCount], values: MLXArray(fill)),
        ])
        eval(padded)
        return padded[0 ..< bytes.size].view(dtype: array.dtype).reshaped(array.shape)
    }

    /// The gather matmul of `x` with the quantized `weights`, each buffer followed by `fill`.
    private static func product(
        _ x: MLXArray, weights: (MLXArray, MLXArray, MLXArray?), globalScale: MLXArray?,
        indices: MLXArray, format: Format, fill: UInt8
    ) -> MLXArray {
        let product = gatherQuantizedMM(
            x, followed(by: fill, weights.0), scales: followed(by: fill, weights.1),
            biases: weights.2.map { followed(by: fill, $0) }, rhsIndices: indices,
            transpose: true, groupSize: format.groupSize, bits: format.bits, mode: format.mode,
            globalScale: globalScale, sortedIndices: true)
        eval(product)
        return product
    }

    /// The products of one format and depth over zero guard bytes and over NaN guard bytes.
    private static func products(format: Format, depth: Int) -> (zero: MLXArray, nan: MLXArray) {
        let dense = withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            MLXRandom.normal([experts, outputs, depth]) * weightScale
        }.asType(.bfloat16)
        let x = withRandomState(MLXRandom.RandomState(seed: inputSeed)) {
            MLXRandom.normal([rows, 1, depth])
        }.asType(.bfloat16)
        let indices = MLXArray((0 ..< rows).map { Int32($0 * experts / rows) })
        // NVFP4 quantizes with one global scale, and the gather takes one per expert.
        let tensorScale = format.mode == .nvfp4 ? abs(dense).max().asType(.float32) : nil
        let globalScale = tensorScale.map { MLX.broadcast($0, to: [experts]) }
        let weights = MLX.quantized(
            dense, groupSize: format.groupSize, bits: format.bits, mode: format.mode,
            globalScale: tensorScale)
        eval(x, indices, weights.wq, weights.scales)
        func run(_ fill: UInt8) -> MLXArray {
            product(
                x, weights: (weights.wq, weights.scales, weights.biases),
                globalScale: globalScale, indices: indices, format: format, fill: fill)
        }
        return (run(zeroGuard), run(nanGuard))
    }

    @Test(
        "NaN guard bytes give the product of zero guard bytes",
        arguments: formats, unalignedDepths)
    func guardBytesDoNotReachTheProduct(format: Format, depth: Int) {
        let (zero, nan) = Self.products(format: format, depth: depth)
        #expect(
            zero.shape == [Self.rows, 1, Self.outputs] && arrayEqual(zero, nan).item(Bool.self))
    }
}

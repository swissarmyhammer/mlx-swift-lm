// Copyright © 2026 Apple Inc.
//
// The contract of ``MultiRowQuantizedMatvec``: each output row of a
// speculative verify of 3 to 8 rows has the same bits as the matvec of MLX
// over that row alone, the plain decode of one row.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class MultiRowQuantizedMatvecTests: XCTestCase {

    /// One quantization format a checkpoint can carry.
    private struct Format: CustomStringConvertible {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        var description: String { "\(mode) \(bits)-bit g\(groupSize)" }
    }

    /// Every format the kernel reads: affine 4 and 8 bits with groups of 32,
    /// 64 and 128, and mxfp4.
    private static let formats =
        [4, 8].flatMap { bits in
            [32, 64, 128].map { Format(groupSize: $0, bits: bits, mode: .affine) }
        } + [Format(groupSize: 32, bits: 4, mode: .mxfp4)]

    private static let dtypes: [DType] = [.bfloat16, .float16]

    /// The input width of the small layers, two blocks of the 4-bit kernel.
    private static let inputSize = 1024

    /// Two small output widths, each a whole number of output tiles.
    private static let smallOutputSizes = [48, 96]

    /// The vocabulary width of Qwen3.8, the widest projection of a step.
    private static let vocabularySize = 248_320

    /// The input width of the vocabulary test, one block of the 4-bit kernel.
    private static let vocabularyInputSize = 512

    /// The row counts a verify can run, and one beyond them.
    private static let verifyRows = 1 ... 8
    private static let tooManyRows = 9

    /// The row counts whose rows have the bits of a single-row decode: one
    /// row, which keeps the matvec of MLX, and the rows the kernel runs.
    private static let bitIdenticalRows =
        [1] + Array(MultiRowQuantizedMatvec.fewestRows ... MultiRowQuantizedMatvec.mostRows)

    /// The bound of the random weights, about the scale of a trained projection.
    private static let weightScale: Float = 0.02

    private func makeLayer(
        _ format: Format, inputs: Int, outputs: Int, dtype: DType, seed: UInt64
    ) -> QuantizedLinear {
        MLXRandom.seed(seed)
        let weight = (MLXRandom.normal([outputs, inputs]) * Self.weightScale).asType(dtype)
        let layer = QuantizedLinear(
            weight: weight, bias: nil, groupSize: format.groupSize, bits: format.bits,
            mode: format.mode)
        eval(layer)
        return layer
    }

    private func makeRows(_ rows: Int, width: Int, dtype: DType) -> MLXArray {
        let x = MLXRandom.normal([rows, width]).asType(dtype)
        eval(x)
        return x
    }

    /// The matvec of MLX over each row alone, the rows of a plain decode.
    private func rowByRowReference(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray {
        concatenated((0 ..< x.dim(0)).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0)
    }

    private func assertEveryRowMatches(
        _ layer: QuantizedLinear, width: Int, dtype: DType, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        for rows in Self.bitIdenticalRows {
            let x = makeRows(rows, width: width, dtype: dtype)
            let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer)
            let want = rowByRowReference(x, layer)
            eval(got, want)
            XCTAssertEqual(got.shape, want.shape, "\(label) M=\(rows)", file: file, line: line)
            XCTAssertEqual(got.dtype, want.dtype, "\(label) M=\(rows)", file: file, line: line)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0,
                "\(label) M=\(rows): rows differ from their own matvec", file: file, line: line)
        }
    }

    func testEveryRowIsBitIdenticalToItsOwnMatvecForEveryFormat() {
        var seed: UInt64 = 1
        for format in Self.formats {
            for dtype in Self.dtypes {
                for outputs in Self.smallOutputSizes {
                    seed += 1
                    let layer = makeLayer(
                        format, inputs: Self.inputSize, outputs: outputs, dtype: dtype,
                        seed: seed)
                    assertEveryRowMatches(
                        layer, width: Self.inputSize, dtype: dtype,
                        label: "\(format) \(dtype) N=\(outputs)")
                }
            }
        }
    }

    func testVocabularyWidthRowsAreBitIdenticalToTheirOwnMatvec() {
        let formats = [
            Format(groupSize: 64, bits: 4, mode: .affine),
            Format(groupSize: 32, bits: 4, mode: .mxfp4),
        ]
        for (offset, format) in formats.enumerated() {
            let layer = makeLayer(
                format, inputs: Self.vocabularyInputSize, outputs: Self.vocabularySize,
                dtype: .bfloat16, seed: UInt64(100 + offset))
            assertEveryRowMatches(
                layer, width: Self.vocabularyInputSize, dtype: .bfloat16,
                label: "\(format) vocabulary")
        }
    }

    func testTheKernelRunsThreeToEightRowsAndLeavesOtherCountsToMLX() {
        for format in Self.formats {
            let layer = makeLayer(
                format, inputs: Self.inputSize, outputs: Self.smallOutputSizes[0],
                dtype: .bfloat16, seed: 7)
            for rows in Self.verifyRows {
                let x = makeRows(rows, width: Self.inputSize, dtype: .bfloat16)
                XCTAssertEqual(
                    MultiRowQuantizedMatvec.canRun(x, layer),
                    rows >= MultiRowQuantizedMatvec.fewestRows,
                    "\(format) M=\(rows)")
            }
            let tooMany = makeRows(Self.tooManyRows, width: Self.inputSize, dtype: .bfloat16)
            XCTAssertFalse(MultiRowQuantizedMatvec.canRun(tooMany, layer), "\(format)")
        }
    }

    func testVerifyShapedInputKeepsItsLeadingAxes() {
        let format = Format(groupSize: 64, bits: 4, mode: .affine)
        let layer = makeLayer(
            format, inputs: Self.inputSize, outputs: Self.smallOutputSizes[1],
            dtype: .bfloat16, seed: 11)
        let tokens = 3
        let x = makeRows(tokens, width: Self.inputSize, dtype: .bfloat16)
        let verify = x.reshaped(1, tokens, Self.inputSize)

        let got = MultiRowQuantizedMatvec.quantizedMatmulRows(verify, layer)
        let want = rowByRowReference(x, layer).reshaped(1, tokens, Self.smallOutputSizes[1])
        eval(got, want)

        XCTAssertTrue(MultiRowQuantizedMatvec.canRun(verify, layer))
        XCTAssertEqual(got.shape, [1, tokens, Self.smallOutputSizes[1]])
        XCTAssertEqual(BitwiseComparison.mismatchCount(got, want), 0)
    }

    func testABiasIsAddedToEveryRow() {
        let format = Format(groupSize: 64, bits: 4, mode: .affine)
        let outputs = Self.smallOutputSizes[0]
        MLXRandom.seed(13)
        let weight = (MLXRandom.normal([outputs, Self.inputSize]) * Self.weightScale)
            .asType(.bfloat16)
        let bias = MLXRandom.normal([outputs]).asType(.bfloat16)
        let layer = QuantizedLinear(
            weight: weight, bias: bias, groupSize: format.groupSize, bits: format.bits,
            mode: format.mode)
        eval(layer)

        XCTAssertTrue(
            MultiRowQuantizedMatvec.canRun(
                makeRows(
                    MultiRowQuantizedMatvec.fewestRows, width: Self.inputSize, dtype: .bfloat16),
                layer))
        assertEveryRowMatches(
            layer, width: Self.inputSize, dtype: .bfloat16, label: "with bias")
    }

    func testAnInputWidthOutsideTheFastMatvecKeepsTheMLXPath() {
        let format = Format(groupSize: 64, bits: 4, mode: .affine)
        let narrowWidth = 320
        let layer = makeLayer(
            format, inputs: narrowWidth, outputs: Self.smallOutputSizes[0], dtype: .bfloat16,
            seed: 17)
        let x = makeRows(4, width: narrowWidth, dtype: .bfloat16)

        let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer)
        let want = layer(x)
        eval(got, want)

        XCTAssertFalse(MultiRowQuantizedMatvec.canRun(x, layer))
        XCTAssertEqual(BitwiseComparison.mismatchCount(got, want), 0)
    }

    func testASubclassKeepsItsOwnCall() {
        final class ScaledQuantizedLinear: QuantizedLinear {
            override func callAsFunction(_ x: MLXArray) -> MLXArray {
                super.callAsFunction(x) * 2
            }
        }
        MLXRandom.seed(19)
        let weight =
            (MLXRandom.normal([Self.smallOutputSizes[0], Self.inputSize])
            * Self.weightScale).asType(.bfloat16)
        let layer = ScaledQuantizedLinear(weight: weight, bias: nil)
        eval(layer)
        let x = makeRows(3, width: Self.inputSize, dtype: .bfloat16)

        let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer)
        let want = layer(x)
        eval(got, want)

        XCTAssertFalse(MultiRowQuantizedMatvec.canRun(x, layer))
        XCTAssertEqual(BitwiseComparison.mismatchCount(got, want), 0)
    }
}

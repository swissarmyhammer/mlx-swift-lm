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
    struct Format: CustomStringConvertible {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        var description: String { "\(mode) \(bits)-bit g\(groupSize)" }

        /// Seeded random weights of about `weightScale`.
        static func makeWeight(
            inputs: Int, outputs: Int, dtype: DType, seed: UInt64, weightScale: Float
        ) -> MLXArray {
            MLXRandom.seed(seed)
            return (MLXRandom.normal([outputs, inputs]) * weightScale).asType(dtype)
        }

        /// A layer of this format with seeded random weights of about `weightScale`.
        func makeLayer(
            inputs: Int, outputs: Int, dtype: DType, seed: UInt64, weightScale: Float,
            bias: MLXArray? = nil
        ) -> QuantizedLinear {
            let weight = Self.makeWeight(
                inputs: inputs, outputs: outputs, dtype: dtype, seed: seed,
                weightScale: weightScale)
            let layer = QuantizedLinear(
                weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
            eval(layer)
            return layer
        }

        /// The matvec of MLX over each row alone, the rows of a plain decode.
        static func rowByRowReference(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray {
            concatenated((0 ..< x.dim(0)).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0)
        }
    }

    /// Every format the kernel reads: affine 4 and 8 bits with groups of 32,
    /// 64 and 128, and mxfp4.
    static let formats =
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

    /// An output width of whole tiles of 32 rows, the tile of 8 SIMD groups.
    private static let wideTileOutputs = 4096

    /// An output width of whole tiles of 16 rows and not of 32.
    private static let mediumTileOutputs = 48

    /// An output width of whole tiles of 8 rows and not of 16.
    private static let narrowTileOutputs = 40

    /// An input width of three blocks of the 4-bit kernel and six of the
    /// 8-bit kernel: the prefetch of the next block runs, and the last block
    /// has no next block.
    private static let oddBlockInputSize = 1536

    /// Output widths that take each threadgroup width of the kernel: 8, 4 and
    /// 2 SIMD groups.
    private static let everyTileOutputs = [64, mediumTileOutputs, narrowTileOutputs]

    /// The seeds before the first layer of the every-format test and of the
    /// every-threadgroup-width test, far enough apart that no layer shares a seed.
    private static let everyFormatSeed: UInt64 = 1
    private static let everyThreadgroupWidthSeed: UInt64 = 200

    /// The seed before the first layer of the generic-layout test.
    private static let genericLayoutSeed: UInt64 = 300

    private func makeLayer(
        _ format: Format, inputs: Int, outputs: Int, dtype: DType, seed: UInt64,
        bias: MLXArray? = nil
    ) -> QuantizedLinear {
        format.makeLayer(
            inputs: inputs, outputs: outputs, dtype: dtype, seed: seed,
            weightScale: Self.weightScale, bias: bias)
    }

    private func makeRows(_ rows: Int, width: Int, dtype: DType) -> MLXArray {
        let x = MLXRandom.normal([rows, width]).asType(dtype)
        eval(x)
        return x
    }

    private func rowByRowReference(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray {
        Format.rowByRowReference(x, layer)
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

    /// Checks every row of a layer of each format and dtype, for each output
    /// width of `outputSizes` and the input width `inputs` gives for the bits
    /// of the format. The layer seeds count up from one past `seed`.
    private func assertEveryFormatAndDtypeMatches(
        outputSizes: [Int], inputs inputWidth: (Int) -> Int, seed: UInt64,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        var seed = seed
        for format in Self.formats {
            let inputs = inputWidth(format.bits)
            for dtype in Self.dtypes {
                for outputs in outputSizes {
                    seed += 1
                    let layer = makeLayer(
                        format, inputs: inputs, outputs: outputs, dtype: dtype, seed: seed)
                    assertEveryRowMatches(
                        layer, width: inputs, dtype: dtype,
                        label: "\(format) \(dtype) N=\(outputs) K=\(inputs)", file: file,
                        line: line)
                }
            }
        }
    }

    func testEveryRowIsBitIdenticalToItsOwnMatvecForEveryFormat() {
        assertEveryFormatAndDtypeMatches(
            outputSizes: Self.smallOutputSizes, inputs: { _ in Self.inputSize },
            seed: Self.everyFormatSeed)
    }

    func testEveryRowIsBitIdenticalToItsOwnMatvecForEveryThreadgroupWidth() {
        assertEveryFormatAndDtypeMatches(
            outputSizes: Self.everyTileOutputs, inputs: { _ in Self.oddBlockInputSize },
            seed: Self.everyThreadgroupWidthSeed)
    }

    /// An input width that MLX runs through its generic `qmv` for one row at
    /// `bits` bits: whole blocks of `qmv` and not of `qmv_fast`. For 4 bits the
    /// 2304 of Inkling 2 (nine blocks of 256), for 8 bits half of it (nine
    /// blocks of 128).
    private static func genericLayoutWidth(bits: Int) -> Int {
        bits == 4 ? 2304 : 1152
    }

    func testEveryRowIsBitIdenticalToItsOwnMatvecInTheGenericLayout() {
        for format in Self.formats {
            let inputs = Self.genericLayoutWidth(bits: format.bits)
            XCTAssertEqual(
                MultiRowQuantizedMatvec.MatvecLayout.of(width: inputs, bits: format.bits),
                .generic, "\(format) K=\(inputs)")
        }
        assertEveryFormatAndDtypeMatches(
            outputSizes: Self.everyTileOutputs, inputs: Self.genericLayoutWidth(bits:),
            seed: Self.genericLayoutSeed)
    }

    func testTheInputWidthPicksTheMatvecLayoutOfMLX() {
        typealias Layout = MultiRowQuantizedMatvec.MatvecLayout
        XCTAssertEqual(Layout.of(width: 4096, bits: 4), .fast)
        XCTAssertEqual(Layout.of(width: 2304, bits: 4), .generic)
        XCTAssertEqual(Layout.of(width: 2304, bits: 8), .fast)
        XCTAssertEqual(Layout.of(width: 1152, bits: 8), .generic)
        XCTAssertNil(Layout.of(width: 1152, bits: 4))
        XCTAssertNil(Layout.of(width: 320, bits: 4))
        XCTAssertNil(Layout.of(width: 0, bits: 4))
    }

    func testAnOutputWidthOfWholeWideTilesTakesEightSimdGroups() {
        XCTAssertEqual(MultiRowQuantizedMatvec.simdGroups(outputs: Self.wideTileOutputs), 8)
    }

    func testAnOutputWidthOfWholeMediumTilesTakesFourSimdGroups() {
        XCTAssertEqual(MultiRowQuantizedMatvec.simdGroups(outputs: Self.mediumTileOutputs), 4)
    }

    func testAnOutputWidthOfWholeNarrowTilesTakesTwoSimdGroups() {
        XCTAssertEqual(MultiRowQuantizedMatvec.simdGroups(outputs: Self.narrowTileOutputs), 2)
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
        let seed: UInt64 = 13
        MLXRandom.seed(seed)
        let bias = MLXRandom.normal([outputs]).asType(.bfloat16)
        let layer = makeLayer(
            format, inputs: Self.inputSize, outputs: outputs, dtype: .bfloat16, seed: seed,
            bias: bias)

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
        let weight = Format.makeWeight(
            inputs: Self.inputSize, outputs: Self.smallOutputSizes[0], dtype: .bfloat16, seed: 19,
            weightScale: Self.weightScale)
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

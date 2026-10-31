// Copyright © 2026 Apple Inc.
//
// The range of inputs in which ``MultiRowQuantizedMatvec`` keeps the bits of the
// matvec of MLX, for every format, with each magnitude in rows of its own.
//
// bfloat16: each input is 0 or has a magnitude of at least 2^-114, up to the
// largest finite number. float16: every finite input. In both, each sum of four
// inputs of a pack stays finite and no output is NaN.
//
// The lower edge comes from the 4-bit affine `qmv_fast`, which divides an input
// by 4096 where the kernel does not. A GPU that flushes a subnormal float, as
// the M5 Ultra does, then loses a smaller input in MLX and keeps it in the kernel.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class MultiRowQuantizedMatvecExactProductTests: XCTestCase {

    /// The columns of a row that hold the magnitude. Every other column is 0.
    private enum Columns {
        /// Every column, each with its own sign.
        case every
        /// One column of each lane, at another place of the pack from lane to lane, so a
        /// magnitude near the limit of float does not overflow the sum of a lane.
        case onePerLane
        /// One column of the first pack, at another place of the pack from row to row.
        case onePerRow
        /// Two neighbors of opposite sign in the first pack, at another place from row to row.
        case cancellingPair
        /// The four columns of one pack of a single lane, at another pack from row to row.
        case fullPack

        /// The columns one lane of the 4-bit kernel reads.
        private static let lane = 16

        /// The numbers of one pack of the 4-bit kernel.
        private static let pack = 4

        /// The first places of a pair of neighbors in a pack.
        private static let pairPlaces = pack - 1

        /// The packs of one lane of the 4-bit kernel.
        private static let packsPerLane = lane / pack

        /// The factor of the magnitude at `column` of `row`.
        func coefficient(row: Int, column: Int) -> Float {
            let lanePlace = column % Self.lane
            switch self {
            case .every:
                return 1
            case .onePerLane:
                return lanePlace == (column / Self.lane) % Self.pack ? 1 : 0
            case .onePerRow:
                return column == row % Self.pack ? 1 : 0
            case .cancellingPair:
                let first = row % Self.pairPlaces
                return column == first ? -1 : column == first + 1 ? 1 : 0
            case .fullPack:
                return column / Self.pack == row % Self.packsPerLane ? 1 : 0
            }
        }

        /// The shape of the random signs of `shape`: a pair shares one sign, which keeps
        /// its two numbers opposite, and a full pack shares one sign, so that products
        /// past the limit of float do not meet as opposite infinities.
        func signShape(_ shape: [Int]) -> [Int] {
            switch self {
            case .every, .onePerLane, .onePerRow:
                return shape
            case .cancellingPair, .fullPack:
                return [shape[0], 1]
            }
        }
    }

    /// One row of the range: a magnitude, its columns, and the outputs it gives.
    private struct Magnitude {
        let name: String
        let value: Float
        let columns: Columns
        /// The bound of the random weights, large enough that a small input reaches
        /// the output.
        var weightScale = MultiRowQuantizedMatvecExactProductTests.trainedWeightScale
        /// False for an input whose products overflow float, where an output can be inf.
        var outputsAreFinite = true
    }

    /// The bound of the random weights of a trained projection.
    private static let trainedWeightScale: Float = 0.02

    /// The weights that lift the smallest float16 input above the rounding of the output.
    private static let weightScaleOfTheSmallestFloat16: Float = 20

    /// The smallest bfloat16 magnitude in the range: over 4096 it is the smallest normal float.
    private static let smallestBfloat16 = Float(sign: .plus, exponent: -114, significand: 1)

    /// The largest finite bfloat16, 8 significant bits below the exponent limit.
    private static let largestBfloat16 = Float(bitPattern: 0x7f7f_0000)

    /// The largest bfloat16 whose four inputs of one sign fit a bfloat16 sum.
    private static let largestBfloat16OfAFullPack = largestBfloat16 / 4

    /// Near the largest input whose four products with 4-bit weights sum to a finite float:
    /// the largest float over four times the highest nibble.
    private static let largestBfloat16OfAFinite4BitPack: Float = 5.6e36

    /// Near the largest input whose product with every 4-bit weight stays finite.
    private static let largestBfloat16With4BitFiniteProducts: Float = 2.2e37

    /// Near the largest input whose product with every 8-bit weight stays finite.
    private static let largestBfloat16With8BitFiniteProducts: Float = 1.3e36

    /// Overflows float by a nibble of 10, not by a nibble of 9.
    private static let bfloat16OverflowingAtTheTenthNibble: Float = 3.59e37

    /// The smallest positive float16, a subnormal.
    private static let smallestFloat16 = Float(sign: .plus, exponent: -24, significand: 1)

    /// The largest float16 whose four inputs of one sign fit a float16 sum.
    private static let largestFloat16OfAFullPack: Float = 16376

    /// The largest finite float16.
    private static let largestFloat16: Float = 65504

    private static let bfloat16Magnitudes = [
        Magnitude(name: "lower edge", value: smallestBfloat16, columns: .every),
        Magnitude(name: "small", value: 1e-30, columns: .every),
        Magnitude(name: "unit", value: 1, columns: .every),
        Magnitude(name: "large", value: 1e30, columns: .every),
        Magnitude(
            name: "finite products", value: largestBfloat16With8BitFiniteProducts,
            columns: .onePerLane),
        Magnitude(
            name: "highest nibble", value: largestBfloat16With4BitFiniteProducts,
            columns: .onePerRow, outputsAreFinite: false),
        Magnitude(
            name: "overflowing pair", value: bfloat16OverflowingAtTheTenthNibble,
            columns: .cancellingPair, outputsAreFinite: false),
        Magnitude(
            name: "finite full pack", value: largestBfloat16OfAFinite4BitPack,
            columns: .fullPack, outputsAreFinite: false),
        Magnitude(
            name: "full pack", value: largestBfloat16OfAFullPack, columns: .fullPack,
            outputsAreFinite: false),
        Magnitude(
            name: "upper edge", value: largestBfloat16, columns: .onePerRow,
            outputsAreFinite: false),
    ]

    private static let float16Magnitudes = [
        Magnitude(
            name: "lower edge", value: smallestFloat16, columns: .every,
            weightScale: weightScaleOfTheSmallestFloat16),
        Magnitude(name: "unit", value: 1, columns: .every),
        Magnitude(name: "full pack", value: largestFloat16OfAFullPack, columns: .every),
        Magnitude(name: "upper edge", value: largestFloat16, columns: .onePerLane),
    ]

    /// The input width of the layer, two blocks of the 4-bit kernel.
    private static let inputSize = 1024

    /// The output width of the layer, a whole number of output tiles.
    private static let outputSize = 48

    /// The seed of the first layer.
    private static let firstSeed: UInt64 = 20_261_031

    /// The seeds one magnitude takes, one for each format and more.
    private static let seedsPerMagnitude = 100

    /// The share of the magnitude the largest output must reach: a single column by a
    /// trained weight.
    private static let smallestReach: Float = 0.005

    /// The sign of a column falls on either side of this draw from 0 to 1.
    private static let signThreshold: Float = 0.5

    /// Rows of `magnitude`, built on the CPU, which keeps the subnormals the GPU can flush.
    private func makeRows(_ rows: Int, _ magnitude: Magnitude, dtype: DType) -> MLXArray {
        let shape = [rows, Self.inputSize]
        let draws = MLXRandom.uniform(0 ..< 1, magnitude.columns.signShape(shape))
        let signs = MLX.which(
            draws .< MLXArray(Self.signThreshold), MLXArray(Float(-1)), MLXArray(Float(1)))
        let coefficients = (0 ..< rows).flatMap { row in
            (0 ..< Self.inputSize).map {
                magnitude.columns.coefficient(row: row, column: $0)
            }
        }
        let x = (signs * MLXArray(coefficients, shape) * magnitude.value)
            .asType(dtype, stream: .cpu)
        eval(x)
        return x
    }

    /// Tells that the outputs prove something: none is NaN, the magnitude reaches them,
    /// and they are finite when the products of the magnitude are.
    private func assertOutputsCarry(
        _ magnitude: Magnitude, _ want: MLXArray, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let outputs = want.asType(.float32).asArray(Float.self)
        XCTAssertFalse(outputs.contains { $0.isNaN }, "\(label): NaN", file: file, line: line)
        XCTAssertTrue(
            !magnitude.outputsAreFinite || outputs.allSatisfy { $0.isFinite },
            "\(label): an output is not finite", file: file, line: line)
        XCTAssertGreaterThanOrEqual(
            outputs.map { abs($0) }.max() ?? 0, magnitude.value * Self.smallestReach,
            "\(label): the magnitude does not reach the output", file: file, line: line)
    }

    private func assertRowsKeepTheirBits(
        _ magnitudes: [Magnitude], dtype: DType,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let rowCounts = MultiRowQuantizedMatvec.fewestRows ... MultiRowQuantizedMatvec.mostRows
        for (index, magnitude) in magnitudes.enumerated() {
            for (offset, format) in MultiRowQuantizedMatvecTests.formats.enumerated() {
                let layer = format.makeLayer(
                    inputs: Self.inputSize, outputs: Self.outputSize, dtype: dtype,
                    seed: Self.firstSeed + UInt64(index * Self.seedsPerMagnitude + offset),
                    weightScale: magnitude.weightScale)
                for rows in rowCounts {
                    let label = "[\(magnitude.name)] \(dtype) \(format) M=\(rows)"
                    let x = makeRows(rows, magnitude, dtype: dtype)
                    XCTAssertTrue(
                        MultiRowQuantizedMatvec.canRun(x, layer),
                        "\(label): the kernel did not run", file: file, line: line)
                    let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer)
                    let want = MultiRowQuantizedMatvecTests.Format.rowByRowReference(x, layer)
                    eval(got, want)
                    XCTAssertEqual(
                        BitwiseComparison.mismatchCount(got, want), 0,
                        "\(label): the bits differ", file: file, line: line)
                    assertOutputsCarry(magnitude, want, label: label, file: file, line: line)
                }
            }
        }
    }

    func testBfloat16RowsKeepTheBitsOfTheirOwnMatvecOverTheWholeRange() {
        assertRowsKeepTheirBits(Self.bfloat16Magnitudes, dtype: .bfloat16)
    }

    func testFloat16RowsKeepTheBitsOfTheirOwnMatvecOverTheWholeRange() {
        assertRowsKeepTheirBits(Self.float16Magnitudes, dtype: .float16)
    }
}

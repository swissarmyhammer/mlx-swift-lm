import MLX

/// Bit-exact comparison of two arrays.
///
/// The float32 upcast is exact for float16, bfloat16 and float32 numbers, thus
/// two upcast numbers with the same bit pattern come from two numbers with the
/// same bits.
enum BitwiseComparison {
    /// The count of the positions where `got` and `want` have different bits.
    ///
    /// - Parameters:
    ///   - got: The array under test.
    ///   - want: The reference array, of the shape of `got`.
    /// - Returns: The count of the different numbers. `0` means the same bits.
    static func mismatchCount(_ got: MLXArray, _ want: MLXArray) -> Int {
        let gotNumbers = got.asType(.float32).asArray(Float.self)
        let wantNumbers = want.asType(.float32).asArray(Float.self)
        return zip(gotNumbers, wantNumbers).lazy
            .filter { $0.bitPattern != $1.bitPattern }
            .count
    }
}

import MLXTestPrecision
import Testing

/// The float32 tests of this bundle compare two computation paths with float32
/// tolerances. On a GPU with neural accelerators (for example the Apple M5),
/// MLX computes float32 matrix operations in TF32 for some shapes only, and
/// the two paths then differ by several TF32 units of roundoff (2^-11).
///
/// The `MLXTestPrecision` target sets `MLX_ENABLE_TF32=0` when the bundle
/// loads. This suite makes sure that the setting is in effect. A failure here
/// explains the tolerance failures in the model tests.
@Suite("Float32 precision of the test process")
struct Float32PrecisionTests {

    @Test("MLX computes float32 matrix operations in float32, not TF32")
    func tf32IsDisabledForTheTestProcess() {
        #expect(mlx_test_precision_tf32_is_disabled())
    }
}

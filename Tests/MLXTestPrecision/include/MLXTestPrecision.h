#ifndef MLX_TEST_PRECISION_H
#define MLX_TEST_PRECISION_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Tells if the float32 matrix operations of MLX in this process use float32
/// arithmetic, and not TF32.
///
/// A GPU with neural accelerators (NAX, for example the Apple M5) computes a
/// float32 matrix multiplication, a quantized matrix multiplication and the
/// fused attention in TF32 when `MLX_ENABLE_TF32` is not `0`. TF32 has a
/// 10-bit mantissa. MLX uses the NAX kernel only for some shapes, thus two
/// paths that must give the same float32 result give different results.
///
/// When the test bundle loads, this target sets `MLX_ENABLE_TF32` to `0` if
/// the variable is not set. MLX reads the variable one time, at the first
/// matrix operation, which occurs after the load.
///
/// - Returns: `true` when `MLX_ENABLE_TF32` is `0` in this process.
bool mlx_test_precision_tf32_is_disabled(void);

#ifdef __cplusplus
}
#endif

#endif

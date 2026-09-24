#include "MLXTestPrecision.h"

#include <stdlib.h>
#include <string.h>

/// The environment variable that MLX reads to enable TF32 on NAX kernels.
static const char *const tf32Variable = "MLX_ENABLE_TF32";

/// The value of `tf32Variable` that keeps float32 arithmetic in float32.
static const char *const tf32Disabled = "0";

/// The `setenv` overwrite flag that keeps a value that the caller set.
static const int keepCallerValue = 0;

/// Sets `MLX_ENABLE_TF32` to `0` when the test bundle loads.
///
/// The float32 tests compare two computation paths with float32 tolerances.
/// A value that the caller set stays, thus a person can run the tests with
/// TF32 on purpose.
__attribute__((constructor)) static void mlx_test_precision_disable_tf32(void) {
    setenv(tf32Variable, tf32Disabled, keepCallerValue);
}

bool mlx_test_precision_tf32_is_disabled(void) {
    const char *value = getenv(tf32Variable);
    return value != NULL && strcmp(value, tf32Disabled) == 0;
}

# mlx-swift-lm

## How to run the tests

Do not use `swift test`. It stops the whole test process at the first GPU test
with `MLX error: Failed to load the default metallib`. Both build systems fail
the same way (`--build-system swiftbuild`, the default, and `--build-system
native`). SwiftPM puts the metallib in
`<Target>.xctest/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/`,
but the test binary is in `<Target>.xctest/Contents/MacOS/`. All five probe
paths in the mlx loader (`Cmlx/mlx/backend/metal/device.cpp`) miss it.

Use the two steps that CI uses (see `.github/workflows/pull_request.yml`).

1. Build the tests:

   ```sh
   xcodebuild build-for-testing -skipPackagePluginValidation \
       -scheme mlx-swift-lm-Package -destination 'platform=macOS'
   ```

2. Run a test bundle:

   ```sh
   D=$(echo ~/Library/Developer/Xcode/DerivedData/mlx-swift-lm*/Build/Products/Debug)
   xcrun xctest "$D/MLXLMTests.xctest"
   ```

`xcodebuild` also puts `mlx-swift_Cmlx.bundle` beside the `.xctest` bundles in
`Build/Products/Debug/`. The mlx loader finds the metallib there, so no symlink
and no test bootstrap are necessary.

There are five test bundles:

- `MLXLMTests`
- `MLXGuidedGenerationTests`
- `MLXFoundationModelsTests`
- `CXGrammarTests`
- `MLXHuggingFaceMacrosTests`

CI runs only `MLXLMTests`. Run all five before you say that the tests are green.
A full run is approximately 1520 tests: 1040 with Swift Testing and 480 with
XCTest.

Three tests stay skipped on purpose. `ConstraintCachingTests` is `.disabled`
because `GrammarMatcher::Fork()` needs xgrammar v0.1.34 or later, and
`Libraries/MLXCXGrammar/xgrammar/VERSION` pins v0.1.30.

## How to build

`swift build` is correct for the libraries alone. Use `swift build
--build-tests` when you only want to find compile errors and warnings in the
test targets quickly. Do not run the tests that this command builds.

## Before you commit

`.pre-commit-config.yaml` runs `swift-format` on every Swift file:

```sh
swift-format format --in-place --configuration .swift-format --recursive .
```

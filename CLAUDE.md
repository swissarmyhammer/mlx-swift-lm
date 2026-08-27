# mlx-swift-lm

## How to run the tests

Do not use `swift test`. It stops the whole test process at the first GPU test
with `MLX error: Failed to load the default metallib`. Both build systems fail
the same way (`--build-system swiftbuild`, the default, and `--build-system
native`).

Build the tests with SwiftPM, but run them with `xctest`.

1. Build the tests:

   ```sh
   swift build --build-tests
   ```

2. Run a test bundle:

   ```sh
   xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest
   ```

One `xctest` command runs the XCTest tests and the Swift Testing tests of that
bundle together. The SwiftPM build products are correct as they are, so neither
a symlink nor a test bootstrap is necessary. (`.build/out/Products/Debug/` is
the product directory of the `swiftbuild` build system, which is the default.)

The two steps that CI uses (see `.github/workflows/pull_request.yml`) also
work, and they put the bundles in DerivedData instead:

```sh
xcodebuild build-for-testing -skipPackagePluginValidation \
    -scheme mlx-swift-lm-Package -destination 'platform=macOS'
D=$(echo ~/Library/Developer/Xcode/DerivedData/mlx-swift-lm*/Build/Products/Debug)
xcrun xctest "$D/MLXLMTests.xctest"
```

### Why `swift test` fails

The cause is the test runner, not the layout of the build products. `swift
test` runs the two test libraries in two different processes:

- The XCTest tests run in the `xctest` tool. That tool opens the bundle with
  `NSBundle`, so `NSBundle.allBundles` contains the bundle. The mlx loader
  finds `mlx-swift_Cmlx.bundle` in the resources of that bundle. These tests
  pass.
- The Swift Testing tests run in `swiftpm-testing-helper` (in the toolchain, at
  `usr/libexec/swift/pm/`). That helper uses `dlopen`, which does not record
  the bundle with `NSBundle`. The bundle probe of the mlx loader
  (`load_swiftpm_library` in `Cmlx/mlx/backend/metal/device.cpp`) thus finds
  nothing. The four other probes look only in
  `<Target>.xctest/Contents/MacOS/` and in the current directory, and the
  metallib is in `<Target>.xctest/Contents/Resources/mlx-swift_Cmlx.bundle/
  Contents/Resources/`. All five probes fail.

The mlx-swift repository does not have this problem, because all of its own
tests use XCTest. A fix must come from upstream: the mlx loader must also look
for the SwiftPM bundle relative to the binary that contains it, and not only
through `NSBundle`.

There are five test bundles:

- `MLXLMTests`
- `MLXGuidedGenerationTests`
- `MLXFoundationModelsTests`
- `CXGrammarTests`
- `MLXHuggingFaceMacrosTests`

CI runs only `MLXLMTests`. Run all five before you say that the tests are green.
A full run is approximately 1520 tests: 1040 with Swift Testing and 480 with
XCTest.

No test is skipped. `Libraries/MLXCXGrammar/xgrammar/VERSION` pins v0.1.34,
which supplies `GrammarMatcher::Fork()`, thus `ConstraintCachingTests` runs.

## How to build

`swift build` is correct for the libraries alone. `swift build --build-tests`
builds the test targets also. Use it to find compile errors and warnings in the
test targets quickly, and to make the bundles that step 2 above runs.

## Before you commit

`.pre-commit-config.yaml` runs `swift-format` on every Swift file:

```sh
swift-format format --in-place --configuration .swift-format --recursive .
```

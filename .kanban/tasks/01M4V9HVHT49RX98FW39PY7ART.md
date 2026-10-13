---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: 'DeepSeek-V4 tests: use one shared reader for the static floats helper with eval'
---
## What

Three DeepSeek-V4 test files hold the same private helper `private static func floats(_ array: MLXArray) -> [Float]` with the body `eval(array); return array.asType(.float32).reshaped(-1).asArray(Float.self)`:
- `Tests/MLXLMTests/DeepSeekV4ModelTests.swift`
- `Tests/MLXLMTests/DeepSeekV4CacheTests.swift`
- `Tests/MLXLMTests/DeepSeekV4CompressorTests.swift`

Task ^rnnxh8d moved the other variant (no `eval`, no `reshaped`) into `Tests/MLXLMTests/DeepSeekV4TestArrays.swift` as `DeepSeekV4TestArrays.floats(_:)` and `DeepSeekV4TestArrays.integers(_:)`. That task kept each call's behaviour identical, thus it did not touch these three files.

Also: `Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift` reads a bool mask two times with the inline expression `visible.asType(.int32).asArray(Int32.self)`. That is the body of `DeepSeekV4TestArrays.integers(_:)`.

## Decide and do

- Make the three files call one shared helper. Either add the `eval` form to `DeepSeekV4TestArrays` with a name that states the difference, or show that `asArray` already evaluates and flattens (it calls `eval()` and copies to a contiguous row-major buffer in mlx-swift 0.32.3, `MLXArray+Bytes.swift`) and call `DeepSeekV4TestArrays.floats(_:)`.
- Replace the two inline int32 reads in `DeepSeekV4MathHelpersTests.swift` with `DeepSeekV4TestArrays.integers(_:)`.

## Acceptance Criteria

- [ ] No DeepSeek-V4 test file defines its own `floats` helper.
- [ ] The DeepSeek-V4 unit tests pass (`swift build --build-tests`, then `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`).
#deepseek-v4 #tests
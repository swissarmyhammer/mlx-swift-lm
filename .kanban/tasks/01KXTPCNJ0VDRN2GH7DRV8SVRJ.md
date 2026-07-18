---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'Pre-existing SwiftPM build warnings: unhandled Documentation.docc + mutated node'
---
Full `swift test` run (2026-07-18, branch foundationmodels-fixes) is 100% green (0 test failures across MLXLMTests, MLXHuggingFaceMacrosTests, MLXFoundationModelsTests (241/42), MLXGuidedGenerationTests, and swift-testing suites — ~795 total tests), but the build emits 4 warnings unrelated to the tests under review:

```
warning: 'mlx-swift-lm': found 1 file(s) which are unhandled; explicitly declare them as resources or exclude from the target
    Libraries/MLXLLM/Documentation.docc
warning: 'mlx-swift-lm': found 1 file(s) which are unhandled; explicitly declare them as resources or exclude from the target
    Libraries/MLXLMCommon/Documentation.docc
warning: 'mlx-swift-lm': found 1 file(s) which are unhandled; explicitly declare them as resources or exclude from the target
    Libraries/MLXHuggingFace/Documentation.docc
warning: missing creator for mutated node: ('/Users/wballard/github/swissarmyhammer/mlx-swift-lm/.build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/MacOS')
```

These predate this branch's PromptCache/MLXLanguageModel work — Package.swift and the three Documentation.docc directories were not touched by that change, and the docc folders' files are dated Jul 16-17 (before today). Toolchain in use is Xcode-beta.app, so this may be a beta SwiftPM DocC-catalog auto-discovery regression rather than a manifest bug.

What I considered: adding `exclude: ["Documentation.docc"]` to the MLXLLM/MLXLMCommon/MLXHuggingFace target definitions (mirroring the existing `exclude: ["README.md"]` pattern used on sibling targets) would silence the warning, but risks breaking `swift package generate-documentation` (via the swift-docc-plugin dependency already in Package.swift) if the plugin relies on SwiftPM's declared target file list to locate the catalog. Didn't want to guess at that without verifying the docc-plugin workflow still works after the change.

Next step: verify whether `swift package --allow-writing-to-directory ... generate-documentation --target MLXLLM` (etc.) still finds the catalog after adding the exclude, and whether the same warnings reproduce on a stable (non-beta) toolchain — if they don't, this is toolchain noise and can be closed as won't-fix / tracked against the Xcode beta. #test-failure
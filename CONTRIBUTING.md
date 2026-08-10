# Contributing to MLX Swift Examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications IntegrationTesting
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests run without any special hardware and do not download models.
Note: `swift test` [does not work yet](https://github.com/ml-explore/mlx-swift?tab=readme-ov-file#xcodebuild) — use `xcodebuild` instead:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

Integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. These
tests do not run in CI.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS'

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

## Attribution for Ported Code

Sometimes we port code into this repository from a different project. This
section is the decision about how to attribute that code. Obey it. Do not make
the decision again.

### There is no git ancestry

The reference repositories are not GitHub forks of this repository. Git does not
know about a relation between them and this repository. You cannot merge from
them, and you cannot rebase on them.

Thus each port is a manual transcription. A person reads the source file and
writes the code again in this repository. That same person must add the
attribution by hand. No tool does this for you.

### Header block for each new ported file

Put this block at the top of each new file you port:

```swift
// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   <source path> @ <sha>
// Manual transcription; no git ancestry.
```

Do these three steps:

1. Replace `<source path>` with the path of the source file.
2. Replace `<sha>` with the commit SHA that you read.
3. Keep the other lines exactly as they are.

Give the copyright to Osaurus AI. Do not give the copyright to `scouzi1966`.
The DeepSeek-V4 files in `scouzi1966/mlx-swift-lm` carry the Osaurus AI header,
which shows that the code comes from `osaurus-ai/vmlx-swift-lm`.

This block is only for a file you port. A file that you write yourself keeps the
usual `// Copyright © <year> Apple Inc.` header.

### Notice file

`THIRD-PARTY-NOTICES.md` is at the root of the repository. It lists each
reference repository and gives the full license text for each one. Add a new
reference repository to that file when you port from a new source.

Do not change `LICENSE`. The header block and the notice file are sufficient.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift Examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

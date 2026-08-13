# Contributing to MLX Swift LM

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
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation
```

Integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. They
are not part of the pull request checks, so they never block a merge. In
`ml-explore/mlx-swift-lm` they run nightly on a self-hosted macOS runner
(`.github/workflows/integration_tests.yml`), and failures are reported on a
tracking issue labeled `ci-failure`. Because nothing runs them on your branch,
run them locally when you change model loading, generation, or tokenizer
behavior.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

CI also verifies that DocC documentation builds without warnings for every
library target. Run the same check locally with:

```bash
scripts/verify-docs.sh
```

## Attribution for Ported Code

This section is the decision about how to attribute code that we port into this
repository from a different project. Obey it. Do not make the decision again.

### There is no git ancestry

The reference repositories are not GitHub forks of this repository. Git does not
know about a relation between them and this repository. You cannot merge from
them, and you cannot rebase on them.

Thus each port is a manual transcription. A person reads the source file and
writes the code again in this repository. That same person must add the
attribution by hand. No tool does this for you.

### Header block for each new ported DeepSeek-V4 file

Put this block at the top of each new DeepSeek-V4 file you port:

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

Give the copyright of each ported DeepSeek-V4 file to Osaurus AI. Do not give
the copyright to `scouzi1966`. The DeepSeek-V4 code comes from
`osaurus-ai/vmlx-swift-lm`. `THIRD-PARTY-NOTICES.md` names the file that shows
this.

Keep the name `Osaurus AI` in the block above. `Osaurus AI` and `Osaurus
contributors` are two different names, and each name comes from a different
file. `Osaurus AI` comes from the header of a source file.
`Osaurus contributors` comes from the LICENSE of `osaurus-ai/vmlx-swift-lm`, and
`THIRD-PARTY-NOTICES.md` keeps that name because a notice must agree with the
LICENSE. Both names are correct where they are. Do not change one name into the
other.

Before you port a file, read the header in that file, and find its commit SHA.
Write those two items in the header block of the new file. This document does
not tell you about the headers of the other files in that project, thus do not
use a statement about them.

This block is only for a DeepSeek-V4 file that you port from
`osaurus-ai/vmlx-swift-lm`. A file that you port from a different source keeps
the name in the header of that source file. A file that you write yourself keeps
the usual `// Copyright © <year> Apple Inc.` header.

### Notice file

`THIRD-PARTY-NOTICES.md` is at the root of the repository. It lists each
reference repository and gives the full license text for each one. Add a new
reference repository to that file when you port from a new source.

Do not change `LICENSE`. The header block and the notice file are sufficient.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift LM, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

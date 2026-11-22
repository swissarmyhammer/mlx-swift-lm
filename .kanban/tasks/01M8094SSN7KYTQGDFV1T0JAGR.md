---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'LFM2 checkpoint suite: use the shared drain helper'
---
`IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift` keeps a private `drain<Iterator: TokenIteratorProtocol>` that is the same loop as the public `drain` now in `Libraries/IntegrationTestHelpers/TokenIteratorDrain.swift` (added for ^04yn8q4 review). Delete the LFM2 copy, import `IntegrationTestHelpers` in that file, and call the shared helper. Run `LFM2CheckpointIntegrationTests` (Release, one suite per xcodebuild call, under the GPU lock) to confirm.

#inkling
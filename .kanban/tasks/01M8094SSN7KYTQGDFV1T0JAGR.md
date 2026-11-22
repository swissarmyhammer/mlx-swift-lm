---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m80c7yxn6hx9r433eaj6my9q
  text: 'Done under ^04yn8q4 (finding 2026-11-21 18:16, TokenIteratorDrain.swift:10): the private `drain` in LFM2CheckpointIntegrationTests.swift is deleted, the file imports IntegrationTestHelpers and calls the shared `drain`. Release LFM2CheckpointIntegrationTests: the drain users (the DSpark parity and throughput tests) pass. 2 factory-load tests fail on `reasoningConfig` assertions that do not touch drain. Follow-up card filed.'
  timestamp: 2026-11-22T00:59:33.685755+00:00
position_column: done
position_ordinal: ffffcf80
title: 'LFM2 checkpoint suite: use the shared drain helper'
---
`IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift` keeps a private `drain<Iterator: TokenIteratorProtocol>` that is the same loop as the public `drain` now in `Libraries/IntegrationTestHelpers/TokenIteratorDrain.swift` (added for ^04yn8q4 review). Delete the LFM2 copy, import `IntegrationTestHelpers` in that file, and call the shared helper. Run `LFM2CheckpointIntegrationTests` (Release, one suite per xcodebuild call, under the GPU lock) to confirm.

#inkling
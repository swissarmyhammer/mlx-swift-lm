---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyvnyff3sjkhwc82mr16jpa
  text: |-
    ### Closed — the official upstream replaced the code under the test

    The catch-up to `ml-explore/mlx-swift-lm` on 2026-08-13 brought:

    `4c7874b` **Prefill: balanced chunking behind a new PrefillParameters (~9% at 32K) (#470)**

    That commit writes a new `Libraries/MLXLMCommon/PrefillParameters.swift` and changes the chunking of `Gemma4.swift`, `Gemma3.swift`, `Gemma3Text.swift`, `Gemma3nText.swift` and `Paligemma.swift`. It also changes `Tests/MLXLMTests/Gemma4ChunkedPrefillTests.swift` itself by 28 lines.

    Thus the chunking that made the invariance test unstable is not the chunking of this repository any more. Over the whole merge the test file took 53 added lines and 18 removed lines, and it is upstream's form of the test now.

    `swift test` is green: 1068 tests in 114 suites, 0 failures, and the invariance test did not fail.

    **One honest limit:** a test that is unstable does not always fail. One green run does not prove that the instability is gone; it proves that the code beneath the test is not the code the card was written against. Open this card again if the test flickers, and write the new evidence against upstream's chunking rather than ours.
  timestamp: 2026-08-14T00:45:17.167379+00:00
position_column: done
position_ordinal: f780
title: Make Gemma4ChunkedPrefillTests chunk-size invariance test stable
---
## What

`Gemma4ChunkedPrefillTests.chunkSizeInvariance(chunkSize:)` fails from time to
time in a full `swift test` run. It passed on a second full run and on three
runs of `swift test --filter Gemma4ChunkedPrefill`.

The failure seen:

```
Test "Chunked and single-pass prefill agree on logits and cache offsets"
recorded an issue with 1 argument chunkSize -> 16 at
Gemma4ChunkedPrefillTests.swift:104:9: Expectation failed: close
Final-position logits differ between windowSize=16 and single-pass prefill
```

## Cause

`makeTinyModel()` calls `Gemma4(config)`, which gives the model random weights.
No test sets a random seed. Each run thus gets different weights, and the
comparison `allClose(chunked.logits, singlePass.logits, rtol: 1e-4, atol: 1e-5)`
fails when a draw makes the logits large.

The cache offset comparisons in the same test are exact and did not fail.

## Acceptance Criteria

- [ ] The test gives the same result on each run.
- [ ] The test still shows that chunked prefill and single-pass prefill agree.
- [ ] The fix does not loosen the tolerance to hide a real difference. Set the
      random seed, or make the weights from a fixed table.

## Notes

Found during task ^t1g41y9 (a rename in `DeepseekV4Configuration`). The rename
cannot reach Gemma4, and the full suite passed on the next run, thus the fault
was in this test and not in that change.

#test-stability

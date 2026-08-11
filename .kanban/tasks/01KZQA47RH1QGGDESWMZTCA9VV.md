---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
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

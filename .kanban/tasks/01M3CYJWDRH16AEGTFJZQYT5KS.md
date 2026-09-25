---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3d4xb9jcqr5ktghaqr0ebjs
  text: |-
    Research and red test.

    - Upstream `mlx_lm/models/baichuan_m1.py` (main) makes two masks: `global_mask = create_attention_mask(x, c_global)` from the KV child of the first global layer, and `swa_mask = create_attention_mask(x, c_swa, window_size=self.sliding_window)` from the KV child of the first sliding-window layer. Each layer uses the mask of its kind.
    - Swift gives `cache?.first` (a `CacheList`) to `createAttentionMask`. `CacheList` uses the base `makeMask`, which gives `.causal` and no window. A prefill longer than the window thus attends to all tokens on the sliding-window layers, and a decode attends only to the window.
    - Discovery: the tiny BaichuanM1 fixture had zero `conv_k` / `conv_v` weights (the model starts them with zeros). Zero weights make each key and each value zero, thus each attention output was zero and no test could see a mask or attention-cache defect. With window 4 and zero weights, the new case passed (diff 2.4e-7). The fixture now gives these weights random values.
    - Red evidence (new case `baichuanM1ShortWindow`, window 4, prompt 13 tokens): cold prefill against a decode of one token at a time differs by 1.20 (logits up to 1.88). `liveCachesContinueTheNextPrompt` and `restoredPrefillDecodesAsTheLiveCaches` step 0 also fail for this case. The window-64 case agrees (4.5e-7).
  timestamp: 2026-09-25T20:43:04.370447+00:00
- actor: claude-code
  id: 01m3d506r4gqxgxg3gdghbgx2n
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/BaichuanM1.swift (a global mask from the KV child of the first global layer and a windowed mask from the KV child of the first sliding-window layer, as in mlx-lm), Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift (new fixture `baichuanM1ShortWindow` with window 4, nonzero conv weights for both BaichuanM1 fixtures, new test `coldPrefillAgreesWithSingleTokenDecode`). Suite HybridRecurrentCacheOffsetTests: 6 tests pass; before the fix 3 cases failed for `baichuanM1ShortWindow` (diff 1.20).
    - next: /test on all five bundles.
  timestamp: 2026-09-25T20:44:38.020070+00:00
- actor: claude-code
  id: 01m3d5g62e0y6f7v9ptxt9mnnr
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. `xcrun xctest` on each bundle: MLXLMTests 695 XCTest + 1217 Swift Testing (one new test), 0 failures, 3 runs; MLXGuidedGenerationTests 70; MLXFoundationModelsTests 312; CXGrammarTests 7; MLXHuggingFaceMacrosTests 5. 0 failures, 0 skipped.
    - next: /commit.
  timestamp: 2026-09-25T20:53:21.614819+00:00
position_column: doing
position_ordinal: '80'
title: Check the attention mask of the BaichuanM1 sliding-window layers
---
## What

Found during ^znyes82. `BaichuanM1ModelInner.callAsFunction` (`Libraries/MLXLLM/Models/BaichuanM1.swift`) makes one mask for all layers with `createAttentionMask(h: x, cache: cache?.first)`. It gives the `CacheList` and no `windowSize`. The layers in `sliding_window_layers` use a `RotatingKVCache` with the window, thus a prefill longer than the window possibly attends to more tokens than a decode on the same layer. Compare with `mlx_lm/models/baichuan_m1.py`, which can make a separate mask for the sliding-window layers.

## Acceptance Criteria

- [x] A test with a tiny BaichuanM1 whose window is shorter than the prompt compares a cold prefill with a split prefill and decode.
- [x] If they differ, the sliding-window layers get a windowed mask as in mlx-lm.
- [x] All five unit bundles pass.

#prompt-cache
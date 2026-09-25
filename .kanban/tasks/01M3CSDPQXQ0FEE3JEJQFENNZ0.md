---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cwdthdxdzkvwxge44h6tbc
  text: |-
    Research done.
    - All seven models call `ArraysCache.advance(_:)` on a `MambaCache` in one place only: the general forward of the recurrent (conv or SSM) layer. None of the seven has a compiled decode path (no `CompiledDecodeSegments` use; the only `compile` match is `compiledSiluProduct` in the LFM2MoE MLP). Thus one line for each model is the fix.
    - No model reads the offset of a `MambaCache` for a mask or for RoPE. The attention masks read the first attention cache, and `createSSMMask` reads only `leftPadding`/`lengths`. Thus a moved offset does not change the logits.
    - `reconcilePromptCache` (PromptCacheReusePolicy.swift:448) sets `mainCacheIsAligned` only when every cache offset equals the ledger length. With a recurrent offset of 0, the live caches of these models never extend the ledger.
    - `CachedForwardSmokeTests.assertCacheAdvancesOnce` skips the offset of a `MambaCache` ("SSM/Mamba caches carry recurrent state rather than a token offset"). That comment is wrong after this fix.
    - Discovery outside this task: `CacheList` does not override `offset`, thus the top-level offset of a `CacheList` stays 0. BaichuanM1 and FalconH1 give a `CacheList` for each layer. I record this as a separate task.
  timestamp: 2026-09-25T18:14:47.085246+00:00
- actor: claude-code
  id: 01m3cwynakqx9nbhzsk4rm5vct
  text: |-
    Implementation landed.

    Red proof (before the fix, `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift`, 4 parameterized tests x 7 models): 107 issues. For each of NemotronH, Jamba, Mamba2, GraniteMoeHybrid, LFM2, LFM2MoE and LFM2VL, the offset of each `MambaCache` was 0 after the prefill, after each of 4 decode steps, after a save/restore and after the restored decode. `reconcilePromptCache` gave no `.extend` reuse for any of the 7 models (7 issues).

    Fix: each of the 7 recurrent layers calls `MambaCache.advancePosition(by:)` in place of `advance(_:)`. None of the 7 models has a compiled decode path, thus the general path is the only site. After the fix all 28 test cases pass, and the offsets are the prompt length after the prefill and prompt length + n after n decode steps.

    Warm continuation: the live caches extend the ledger (`PromptCacheReuse(suffixStart: 10, ..., kind: .extend)`), and the caches restored from `savePromptCache`/`loadPromptCacheSnapshot` have each offset and decode as the live caches (<= 1e-6) and as a cold prefill.

    Cold tolerance: the restored-vs-cold logits differ by 3.1e-3 (Mamba2) and 1.8e-3 (NemotronH), identical before and after the fix. The SSM layers send many tokens through a chunked scan and one token through a step kernel. The test uses 5e-3 and records the measurement.

    Existing test corrected: `CachedForwardSmokeTests.assertCacheAdvancesOnce` skipped the offset of a `MambaCache` ("SSM/Mamba caches carry recurrent state rather than a token offset"). That statement was the defect. The `MambaCache` case now also checks the offset (fallthrough to the offset check).

    Judgement on `ArraysCache.advance(_:)`: it is a trap. The name says "advance", but it moves only `lengths`/`leftPadding` and not `offset`. Eight models (these 7 and Qwen3-Next) fell into it. I do not rename it: it is public API, it mirrors `ArraysCache.advance` of mlx-lm (Python), and after this change no model calls it directly (`rg "\.advance\(" Libraries` gives only FalconH1, which moves `offset` by hand on the next line, and `advancePosition(by:)` itself). The change adds a doc comment to `advance(_:)` that says it does NOT move `offset` and names `MambaCache.advancePosition(by:)`. A rename or a deprecation is a separate decision for a person.

    Discovered work: ^ (01M3CWE2TJSX4H90T9YZNYES82) — `CacheList` top-level offset (BaichuanM1, FalconH1).
  timestamp: 2026-09-25T18:23:58.803387+00:00
- actor: claude-code
  id: 01m3cwyxaty64zb4peegg4wpgx
  text: |-
    ### implement — changed
    - evidence: 11 files — Libraries/MLXLLM/Models/{NemotronH,Jamba,Mamba2,GraniteMoeHybrid,LFM2,LFM2MoE}.swift, Libraries/MLXVLM/Models/LFM2VL.swift, Libraries/MLXLMCommon/KVCache.swift (doc comment of `advance(_:)`), Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift (new), Tests/MLXLMTests/CachedForwardSmokeTests.swift. MLXLMTests: 695 XCTest 0 failures, 1215 Swift Testing passed. The discovered task in the comment above is ^znyes82.
    - next: /test (all five bundles)
  timestamp: 2026-09-25T18:24:07.002898+00:00
- actor: claude-code
  id: 01m3cxbvqg8w971evcsqweth5d
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. `xcrun xctest` on all five bundles: MLXLMTests 695 XCTest 0 failures + 1215 Swift Testing passed (3 runs, all green; the flaky ChatSessionTests test of ^k8hm41j did not fail); MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 312 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed. 0 skipped.
    - next: /commit
  timestamp: 2026-09-25T18:31:11.344762+00:00
- actor: claude-code
  id: 01m3cxcq4nmktw3vt0j13m5z9w
  text: |-
    ### commit — changed
    - evidence: 314b5a3 fix(hybrid): move the MambaCache offset in each recurrent layer
    - next: /review qfennz0 HEAD~1..HEAD
  timestamp: 2026-09-25T18:31:39.413543+00:00
- actor: claude-code
  id: 01m3cxhfm5te9bd7semn13kjq2
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 5 findings: Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:79, :80, :81, :82, :83 `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
    - next: /implement (name the literals of the NemotronH configuration initializer call)
  timestamp: 2026-09-25T18:34:15.557989+00:00
- actor: claude-code
  id: 01m3cxhhneba8k86r3p003wtt1
  text: |-
    ### finish iteration 1 — findings
    - implement: changed (7 models call advancePosition(by:); new HybridRecurrentCacheOffsetTests; CachedForwardSmokeTests checks the MambaCache offset; doc comment on ArraysCache.advance(_:))
    - test: green (all five bundles; MLXLMTests 3 runs)
    - commit: 314b5a3
    - review: findings — 5 x code-hygiene/magic-numbers-swift at HybridRecurrentCacheOffsetTests.swift:79-83
  timestamp: 2026-09-25T18:34:17.646475+00:00
- actor: claude-code
  id: 01m3cxpb07jw1yqv7tv6keyajc
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift. The NemotronH fixture was a `NemotronHConfiguration(...)` initializer call with numeric literals at lines 79-83 (`code-hygiene/magic-numbers-swift`). It is now the JSON text `nemotronHJSON`, decoded with the same `decode` helper as the six other fixtures, with the same values. No other numeric literal in the file stands outside a named `static let` or the carve-out values 0, 1 and -1. MLXLMTests: 695 XCTest 0 failures, 1215 Swift Testing passed.
    - next: /test
  timestamp: 2026-09-25T18:36:54.663256+00:00
- actor: claude-code
  id: 01m3cxzfdvsan8rfm4ewzctav2
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. All five bundles: MLXLMTests 695 XCTest 0 failures + 1215 Swift Testing passed (3 runs in this iteration, all green); MLXGuidedGenerationTests 70; MLXFoundationModelsTests 312; CXGrammarTests 7; MLXHuggingFaceMacrosTests 5. 0 skipped.
    - next: /commit
  timestamp: 2026-09-25T18:41:54.107454+00:00
- actor: claude-code
  id: 01m3cy05rrbdbd6vr7d2bdhdk3
  text: |-
    ### commit — changed
    - evidence: 20a347b test(hybrid): give the NemotronH fixture as JSON text
    - next: /review qfennz0 HEAD~1..HEAD
  timestamp: 2026-09-25T18:42:16.984217+00:00
- actor: claude-code
  id: 01m3cy27fpmfr2cc18cv6v7ar0
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (20a347b) — 0 findings, 0 confirmed, 0 refuted; all 5 prior items of 2026-09-25 13:31 are checked.
    - next: none (task moved to done)
  timestamp: 2026-09-25T18:43:24.278619+00:00
- actor: claude-code
  id: 01m3cy29kx33jb7hqt6hd863yd
  text: |-
    ### finish iteration 2 — clean
    - implement: changed (NemotronH fixture as JSON text; removes the magic-numbers-swift findings at HybridRecurrentCacheOffsetTests.swift:79-83)
    - test: green (all five bundles; MLXLMTests 3 runs)
    - commit: 20a347b
    - review: clean (0 findings) — task moved to done
  timestamp: 2026-09-25T18:43:26.461562+00:00
position_column: done
position_ordinal: ffab80
title: Check the recurrent-cache offset of the other hybrid models (NemotronH, Jamba, Mamba2, GraniteMoeHybrid, LFM2, LFM2MoE, LFM2VL)
---
## What

^qr0p806 found that Qwen3-Next called `MambaCache.advance(_:)` and not `MambaCache.advancePosition(by:)`. `advance(_:)` moves only the batch bookkeeping, thus the `MambaCache` offset stayed at 0. The prompt cache file then records offset 0, and `ExecutorPromptCacheFile.read` refuses the file because its offsets are not the ledger length.

These models also call `advance(_:)` on a recurrent cache and do not move `offset`:

- `Libraries/MLXLLM/Models/NemotronH.swift:235`
- `Libraries/MLXLLM/Models/Jamba.swift:330`
- `Libraries/MLXLLM/Models/Mamba2.swift:208`
- `Libraries/MLXLLM/Models/GraniteMoeHybrid.swift:184`
- `Libraries/MLXLLM/Models/LFM2.swift:226`, `Libraries/MLXLLM/Models/LFM2MoE.swift:235`, `Libraries/MLXVLM/Models/LFM2VL.swift:412`

`FalconH1.swift:526-527` moves `offset` by hand.

## Acceptance Criteria

- [x] For each model, a test with a tiny model shows the offset of each recurrent cache after a prefill.
- [x] Each model whose recurrent offset is not the prompt length moves its offset (for a `MambaCache`, with `advancePosition(by:)`), or the task records why the offset must stay.
- [x] All five unit bundles pass.

#prompt-cache

## Review Findings (2026-09-25 13:31)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3CSDPQXQ0FEE3JEJQFENNZ0.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3CSDPQXQ0FEE3JEJQFENNZ0.md` — no validator matches this file
> - `.kanban/tasks/01M3CWE2TJSX4H90T9YZNYES82.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3CWE2TJSX4H90T9YZNYES82.md` — no validator matches this file

- [x] `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:79` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:80` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:81` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:82` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift:83` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
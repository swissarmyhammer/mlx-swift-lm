---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6gggh2dcpmcqyh4fkdc3zm7
  text: |-
    ### review — findings
    - evidence: review sha 75a36b26..b7177dec — 4 findings (4 confirmed, 4 refuted): MTPVerifyGraphBuildTests.swift:177, Libraries/MLXVLM/Models/Qwen35.swift:1170, :1370, :1416. Hand check passed: verify traces keyed per CompiledVerifyShape and dropped on fused GDN projection change (KeyedCompiledDecodeSegmentCaches.invalidate drops every key); M-RoPE, KV write and attention run outside the traces with the call's positionIds; the shared rotary is local to one verifyStep call and computed only when positionIds is non-nil, so it is the same ids for every layer and does not outlive a rollback; decode and non-verify steps keep their previous paths; no runtime MTP on/off heuristic.
    - next: implementer resolves the 4 findings; subtask 3 re-measure pending; card stays in review.
  timestamp: 2026-11-03T10:50:35.981313+00:00
- actor: claude-code
  id: 01m6gvk3tg0zm8c4e1wfmnyr6b
  text: |-
    ### implement + test + review — clean
    - commits (cherry-picked onto mtp): 42dad4b9, ec4d7e8e, b74297a1 (worktree eb56804d, b7177dec, e09e3872)
    - worktree: swift build --build-tests clean; 5 bundles 0 failures (CXGrammar 7, Macros 7, Guided 72, FM 471, MLXLMTests 930 XCTest + 2116 Swift Testing, 1 known issue)
    - Release guards on e09e3872: qwen38FourBitMTPGeneration pass (bs4 pMin0.75 89.8 tok/s vs best bs5 90.7, allowance 3.3%); qwen38FourBitMTPSystemPromptCache pass (edit turn cached 2169, prefilled 123, exact plain tokens, 97.8 tok/s = 1.77x plain)
    - ABBA router, 4 runs/arm, base 75a36b26 vs e09e3872: verify graph build 32k 2.9 -> 1.6 ms, 100k 3.3 -> 1.5 ms; verify forward 100k 3.4–8% faster (4/4); MTP tok/s 32k 75.65 -> 79.08 (+2.5–8.9%, 4/4 pairs), 100k 52.16 -> 52.73 (+3.2–5.1% in 3 pairs, one pair −7.9% with normal forward); plain decode within ±2.5% (noise floor)
    - review sha b7177dec..e09e3872: 7 items, none apply — Qwen35.swift:254–262 are the named constants the change introduces; :1307–1308 (modelType/kvHeads assign-only) are lines this change did not touch. Earlier commits were reviewed in the worktree pass.
    - full sweep and MTPCPULoadTests left to ^j2tnzpn
  timestamp: 2026-11-03T14:04:15.056848+00:00
position_column: done
position_ordinal: ffffa280
title: 'MTP router path: compile the MLXVLM Qwen35 verify forward (2.0–2.4 ms of host graph build per round while the GPU idles; the text path''s compiled verify builds in 1.0 ms)'
---
From ^1abqd92's round profile (Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, bs4 pMin0.75, after the buffer-pool and device-side causal-mask fixes): on the router load path (MLXVLM `Qwen35`, VLM drafter) the verify forward still takes the general per-layer path, and building its graph costs 2.03 ms per round at 32k and 2.44 ms at 100k, against 1.02-1.03 ms for the text path's compiled verify schedule (`Qwen35TextModelInner.verifyStep` + `KeyedCompiledDecodeSegmentCaches`). The GPU is idle during that build: the drafter chain's last confidence read has just drained it. That is the largest host-only piece left in a router round (36.6 ms at 32k: pre 0.19, chain build 0.39, verify build 2.03, commit 0.40, book 0.18).

- [x] Give MLXVLM `Qwen35Language.Model` the compiled verify schedule the MLXLLM text model has (M-RoPE position ids stay outside the trace with the attention step), or show why it cannot be shared; coordinate with ^8p0rfxx (router verify mask as `.causal`).
- [x] Verify output within `AttentionKernelTolerance` of the general path for S = 2..6; greedy MTP equals plain on the VLM fixture.
- [ ] Re-measure the router sweep and MTPCPULoadTests (verify build per round, tok/s at 32k/100k).

#mtp #perf

## Results so far (eb56804d, b7177dec on wt-yh9ens5; base 75a36b26)

- Router verify build: 3.1-3.9 ms to 1.4-1.5 ms per round, 4/4 ABBA pairs.
- Router MTP tok/s: +2.3% at 32k and at 100k for eb56804d against base; plain within ±2%.
- Edit turn 123: intact.
- `qwen38FourBitMTPGeneration`: one throughput check missed by 0.7 tok/s under load 12-14; rerun pending.
- Subtask 3 (router sweep and MTPCPULoadTests re-measure) is open: measurement is pending.

## Review Findings (2026-11-03 04:44)

> Scope: `review sha 75a36b26..b7177dec` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 0 not reviewed.

- [ ] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPVerifyGraphBuildTests.swift:177` `reuse/reuse` — `median(_ samples: [Double])` reimplements a median helper that already exists in the same test target. Four copies now sit side by side, and a fix to one will not reach the others. Move one shared `median([Double])` into the test support folder (for example beside `GPUWallTime.swift`, which already holds a `median` helper for a different contract) and call it from this file, so this file does not carry its own copy.
- [ ] `Libraries/MLXVLM/Models/Qwen35.swift:1170` `duplication/duplication` — The state closure of the new `compiledVerifySegments` cache repeats the state closure of `compiledSegments` (line 1161-1163) token for token, differing only in which cache property it passes to `segmentTraceState(at:of:)`. Both closures carry the same logic, so a later change to one can drift from the other. Extract one private helper that returns the state closure, parameterized by the trace object (for example `segmentTraceState(of:)` returning `(Model, Int) -> [Module]`), and use it in both initializers. Leave the two cache types and their body closures as they are, since the verify variant is keyed by shape and the two cache classes differ.
- [ ] `Libraries/MLXVLM/Models/Qwen35.swift:1370` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [ ] `Libraries/MLXVLM/Models/Qwen35.swift:1416` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.

> Engine note: the code-hygiene Swift rules (disallowed-constructs, function-length, idioms, magic-numbers, missing-docs) found no file at `MTPVerifyGraphBuildTests.swift`, `Tests/MLXLMTests/CompiledVerifyOutcome.swift` and `Tests/MLXLMTests/Qwen35VLMCompiledVerifyTests.swift` (they exist only on wt-yh9ens5, not in the main checkout), so those rules did not read those three files.
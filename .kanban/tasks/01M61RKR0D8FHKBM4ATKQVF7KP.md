---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m61xeqjpc9bj2e442da4yf2w
  text: |-
    Research (implement):
    - LLM Qwen35 (`Qwen35TextModelInner`) holds `ScheduleCaches` / `scheduleCaches` / `linearCaches` / `runSchedule` / `commit`; decode and verify both go through it. Model-specific parts there: SSM-mask check on `ssmIdx`, the attention mask from `faIdx` (decode needs `.none`, verify passes it to `attentionCacheStep(mask:)`), and segment 0 embedding inside the trace (carry = token ids).
    - Qwen3Next `decodeStep`: embeds outside the trace (carry = embedded), fp16-only gate, SSM + attention mask `.none` checks, then a hand loop with `mambaCaches[i]!`, `mambaCache[0]!/[1]!`, `cache[pre]!`, literal output indices `1 + 2 * i`, `outputIndex + 2/+3/+1`. Attention step = rope at `cache.ropeOffset` + `attentionWithCacheUpdate(mask: .none)`.
    - VLM `Qwen35Language.Model.decodeStep`: same loop shape, plus `supportsCompiledDecode` per layer and `supportsExactCompiledDecode(dtype)`; attention = `layers[pre].attentionCacheStep(... positionIds:)` (M-RoPE). `DecoderLayer.decodeLinearLayer` force-unwraps `cache[0]!, cache[1]!` after the caller tested `!= nil`.
    - Plan: new `CompiledDecodeSchedule` (MLXLMCommon, beside `CompiledDecodeSegment`) owning segments + layer count, `caches(_:)` (count, GDN states present, attention caches present and on the plain route), and `run(_:caches:verify:segment:attention:)` with the commit (states, snapshot slots for a verify, advance). Models keep their own eligibility gates (dtype, masks, supportsCompiledDecode) and pass their attention step as a non-escaping closure, so no new per-step allocation beyond what LLM Qwen35 already did.
    - No ARCHITECTURE.md in the repo.
  timestamp: 2026-10-28T18:48:12.118945+00:00
- actor: claude-code
  id: 01m61yjnpm6q4ybk8dwzd7tm7h
  text: |-
    Implementation landed (not committed):
    - New `CompiledDecodeSchedule` (Libraries/MLXLMCommon/CompiledDecodeSchedule.swift): owns the segments and layer count; `caches(_:)` gathers the GDN states per segment and the attention cache of each head (nil on wrong length, a GDN layer with no state, or an attention cache missing / not on the plain route); `run(_:caches:verify:segment:attention:)` is the old LLM `runSchedule` + `commit` (outputs indexed with `statesPerLinearLayer`, `Snapshots.traceArrayCount`, `AttentionOutput`). Both closures are non-escaping, so no new per-step allocation.
    - LLM Qwen35: `ScheduleCaches` / `linearCaches` / `runSchedule` / `commit` / `AttentionOutput` typealias deleted; `scheduleCaches` keeps only the SSM-mask gate, `attentionMask(_:_:)` and `attend(_:mask:)` are the model-specific parts. Qwen3Next: hand loop deleted, fp16 + mask gates kept, `attentionCacheStep(_ step:)` takes the schedule step. VLM: hand loop deleted, `needsNoMask(_:_:)` holds the SSM/attention mask gate, attention closure passes `positionIds`; `DecoderLayer.decodeLinearLayer` now takes the states its caller bound (`let convState = mambaCache[0], let recState = mambaCache[1]`). No force unwraps left in any of the three decode paths.
    - Removed `CompiledDecodeSegment.attentionOutputOffset` (no-arg var): its only production callers were the deleted loops; the one test site uses `attentionOutputOffset(outputsPerLinearLayer:)`.
    - TDD: 6 new tests in HybridDecodeLayerTests (caches refuse wrong length / no GDN state / quantized attention cache / missing attention cache; run of a decode step commits states + attends between segments; run of a verify keeps snapshot slots and advances by its tokens) — RED was "cannot find 'CompiledDecodeSchedule'", then GREEN.
    - Bit-identity proof (temporary dump test, deleted): logits + every cache inner state + offsets for Qwen3Next (fp16 compiled, bf16 general, 8 tokens), VLM Qwen35 (fp16 dense, fp16 MoE, bf16 dense; 4 layers, M-RoPE positions, 7 tokens) and LLM Qwen35 (fp16 kernel fixture: 15 single-token steps, then compiled verifies of 2...6 with a rollback each). 20 runs before, 20 after: Qwen3Next one hash (88982c1a…) before and after, VLM one hash (6a1eeae1…) before and after. LLM: before {ae874149 x12, d31fcd3b x7, 95bc2be8 x1}, after {ae874149 x19, d31fcd3b x1} — every after outcome is a before outcome.
    - Discovery (did not work as a single-dump proof): the LLM Qwen35 kernel fixture is not bitwise deterministic across processes even with no code change (bf16 batched prefill, and plain general decode from step 8). Filed ^g8fn2bf. Do not use a single before/after dump for that model; compare outcome sets.
    - Pre-existing skip: `MTPGreedyVerificationTests.benchmarkVerification()` (MTP_GREEDY_BENCHMARK env gate) shows skipped in every MLXLMTests run; untouched here, filed ^smn05mj.
  timestamp: 2026-10-28T19:07:49.844417+00:00
- actor: claude-code
  id: 01m61ykbtede795r741432vwvs
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXLMCommon/CompiledDecodeSchedule.swift (new), Libraries/MLXLMCommon/CompiledDecodeSegments.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/HybridDecodeLayerTests.swift. `swift build --build-tests`: 0 compiler warnings (only the pre-existing swiftbuild "missing creator for mutated node" notice). `xcrun xctest`: MLXLMTests 897 XCTest + 1923 Swift Testing pass (1 pre-existing env-gated skip, benchmarkVerification, ^smn05mj), MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures. Bit-identity dumps: Qwen3Next and VLM Qwen35 one identical hash over 20 runs before and 20 after; LLM Qwen35 after-outcomes a subset of its (pre-existing nondeterministic, ^g8fn2bf) before-outcomes. Release qwen38FourBitMTPGeneration(): plain 59.6/59.6/59.5, MTP 92.6/83.1/81.0 tok/s math/code/prose (target ~92/83/80, within 3%), matches plain on all three. Release qwen38FourBitMTPSystemPromptCache(): passed, 0 known issues, edit turn prefilled 123 tok with MTP on (cached 2169).
    - next: /review
  timestamp: 2026-10-28T19:08:12.494967+00:00
- actor: claude-code
  id: 01m61yvf1a38b0nvzwyv69p291
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (08b56ad7) — 6 findings (6 confirmed, 2 refuted): Libraries/MLXLLM/Models/Qwen35.swift:930, Libraries/MLXLLM/Models/Qwen3Next.swift:661 (x3), Libraries/MLXLLM/Models/Qwen3Next.swift:664, Libraries/MLXLLM/Models/Qwen3Next.swift:668
    - next: implement the 6 unchecked items in ## Review Findings (2026-10-28 14:08); task stays in review
  timestamp: 2026-10-28T19:12:37.930584+00:00
- actor: claude-code
  id: 01m61yvxm7v81rbp27kqn308tk
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 6 files
    - test: green (implementer run) — 5 bundles (MLXLMTests 897 + 1923); Release MTP 92.6/83.1/81.0 = plain; cache guard pass, edit turn 123 tok
    - commit: 08b56ad7
    - review: findings — Libraries/MLXLLM/Models/Qwen35.swift:930, Qwen3Next.swift:661 (x3), Qwen3Next.swift:664, Qwen3Next.swift:668
  timestamp: 2026-10-28T19:12:52.871948+00:00
- actor: claude-code
  id: 01m61zpa34dvwn3bpvjyckshr5
  text: |-
    Review findings (2026-10-28 14:08) implemented (not committed):
    - `CompiledDecodeSchedule` (MLXLMCommon) now owns every rule the three models repeated: `init(layers:)` (a `HybridDecodeLayer` is linear when it holds a gated delta net), `firstLinearLayer` / `firstAttentionLayer`, `caches(_:over:)` (the old checks plus the SSM-mask refusal, last in its guard so the check order is the old one), `attentionMask(_:_:)` (hoisted from LLM Qwen35), and `decode(_:cache:segments:of:attention:)` (caches + `.none` mask guard + `run` with the compiled segment cache). Non-escaping closures as before, so no new per-step allocation.
    - LLM Qwen35: `scheduleCaches` and `attentionMask` deleted; `decodeStep` is one `decode` call; `verifyStep` uses `caches(_:over:)` + `attentionMask`; `ssmIdx` / `faIdx` come from the schedule. Qwen3Next: `decodeStep` keeps only the fp16 guard, then `decode` (the mid-chain `if … { return nil }` is gone). VLM Qwen35: `needsNoMask` deleted, `decodeStep` keeps the dtype + `supportsCompiledDecode` guard, then `decode`; `ssmIdx` / `faIdx` from the schedule.
    - No behaviour change: the masks only read axis 1 of their input (1 for a decode step either way), and the VLM's dropped `returnArray: true` is moot for one token (`BaseKVCache.makeMask` and `RotatingKVCache.makeMask` with no window return `.none` before reading it; `caches(_:over:)` already restricts attention caches to the plain route).
    - TDD: 6 new tests in HybridDecodeLayerTests (schedule of layers, SSM-mask refusal, attention mask one token / verify, decode refuses a masked step, decode through compiled segments equals the eager run and commits the same states). RED = compile errors (`init(layers:)`, `caches(_:over:)`, `attentionMask`, `decode` missing), then GREEN.
    - Note: `swift test --filter` cannot run these here (default metallib not found); use `xcrun xctest` on the bundle.
  timestamp: 2026-10-28T19:27:17.604279+00:00
- actor: claude-code
  id: 01m61zpefrjt17krbb19qsevhp
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/CompiledDecodeSchedule.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/HybridDecodeLayerTests.swift. All 6 findings of ## Review Findings (2026-10-28 14:08) flipped to [x]. `swift build --build-tests`: 0 compiler warnings (only the pre-existing swiftbuild "missing creator for mutated node" notice). `xcrun xctest`: MLXLMTests 897 XCTest + 1929 Swift Testing pass (1 pre-existing env-gated skip, benchmarkVerification, ^smn05mj), MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures. Release qwen38FourBitMTPGeneration(): plain 59.6/59.6/59.5, MTP 93.0/83.2/80.8 tok/s math/code/prose (target ~92/83/81, within 3%), matches plain: true on all three. Release qwen38FourBitMTPSystemPromptCache(): passed, 0 known issues, edit turn prefilled 123 tok with MTP on (cached 2169).
    - next: /review
  timestamp: 2026-10-28T19:27:22.104089+00:00
- actor: claude-code
  id: 01m61zv07q10ysjy2ka7wqhmc3
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (40daff66) — 0 findings, 0 confirmed, 1 refuted, 14 pairs attempted, 0 failed; all 6 prior items from 2026-10-28 14:08 are checked
    - next: task moved to done
  timestamp: 2026-10-28T19:29:51.351073+00:00
- actor: claude-code
  id: 01m61zvdeq7v06nej8azjdmwh6
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 5 files
    - test: green (implementer run) — 5 bundles (MLXLMTests 897 + 1929); Release MTP 93.0/83.2/80.8 = plain; cache guard pass, edit turn 123 tok
    - commit: 40daff66
    - review: clean — task in done
  timestamp: 2026-10-28T19:30:04.887361+00:00
position_column: done
position_ordinal: fff080
title: Route Qwen3Next and VLM Qwen35 decodeStep through a shared schedule runner, drop their cache force unwraps
---
## What
Found while doing ^5fq4w2k. `Qwen3NextModelInner.decodeStep` (Libraries/MLXLLM/Models/Qwen3Next.swift) and `Qwen35Language.Model.decodeStep` (Libraries/MLXVLM/Models/Qwen35.swift) each re-implement the schedule loop that `Qwen35TextModelInner.runSchedule` / `scheduleCaches` / `linearCaches` / `commit` already hold in Libraries/MLXLLM/Models/Qwen35.swift: gather `[carry] + pendingAttention + states`, run the segment, write the states back, run the KV write between segments.

The copies still carry `mambaCaches[layerIndex]!`, `mambaCache[0]!`, `mambaCache[1]!`, `cache[pre]!`, and index the outputs with literals (`1 + 2 * stateIndex`, `outputIndex + 2`) instead of `CompiledDecodeSegment.statesPerLinearLayer` / `CompiledDecodeSegment.AttentionOutput`. The VLM `DecoderLayer.decodeLinearLayer` also force-unwraps `cache[0]!, cache[1]!`.

- [x] Move the schedule runner (cache gathering, segment loop, state commit) into MLXLMCommon next to `CompiledDecodeSegment`, parameterized by the attention cache step of each model (Qwen3Next: rope + `attentionWithCacheUpdate`; VLM: M-RoPE `attend` with `positionIds`; LLM Qwen35: `attentionCacheStep(mask:)`).
- [x] Route all three models through it; delete the per-model loops.
- [x] No force unwraps left in the three decode paths.

## Acceptance Criteria
- [x] Plain decode logits bit-identical before/after (Qwen3NextCompiledDecodeTests, Qwen35VLMCompiledDecodeTests, Qwen35 MTP tests).
- [x] Release `qwen38FourBitMTPGeneration` within 3% of the same-session baseline; `qwen38FourBitMTPSystemPromptCache` passes with no known issues.

## Tests
- [x] `swift build --build-tests` zero warnings; all five bundles green via `xcrun xctest`.

## Workflow
- Use `/tdd`. #mtp #perf

## Review Findings (2026-10-28 14:08)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 8 not reviewed.

> 8 file(s) not reviewed — no validator matched:
> - `.kanban/tags/01M61YHAYPES4WBCGV72FMTTZ5.jsonl` — no validator matches this file
> - `.kanban/tags/01M61YHAYPES4WBCGV72FMTTZ5.yaml` — no validator matches this file
> - `.kanban/tasks/01M61RKR0D8FHKBM4ATKQVF7KP.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61RKR0D8FHKBM4ATKQVF7KP.md` — no validator matches this file
> - `.kanban/tasks/01M61YHAYN5CDD36653G8FN2BF.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61YHAYN5CDD36653G8FN2BF.md` — no validator matches this file
> - `.kanban/tasks/01M61YHNPNARJ8AQABVSMN05MJ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61YHNPNARJ8AQABVSMN05MJ.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/Qwen35.swift:930` `duplication/duplication` — The three-line schedule setup in the Qwen35 initializer is identical to the one in the Qwen3Next initializer. Each builds `CompiledDecodeSchedule(linearLayers: layers.map(\.isLinear))`, reads `.segments`, and stores the schedule. Both copies must change together if the construction changes. Add a convenience initializer to CompiledDecodeSchedule that takes the decoder layers' `isLinear` flags, or takes the layers themselves. Each model's initializer then makes a single call. Keep the `segments` read where the compiled cache needs it.
- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:661` `duplication/duplication` — Qwen3Next.decodeStep repeats the SSM-mask bail-out that Qwen35.scheduleCaches already applies: it returns nil whenever `createSSMMask` produces a mask for the GDN cache. The two copies must change together if the rule for when a compiled step is refused changes. Put the SSM-mask refusal in CompiledDecodeSchedule, for example in `caches(_:)` with the SSM index and the inputs passed in. Or put it in a shared helper that both models call. Then Qwen3Next.decodeStep no longer repeats the check inline.
- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:661` `reuse/reuse` — Qwen3Next.decodeStep re-implements the SSM-mask gate that Qwen35's scheduleCaches already performs. Both models decide whether a step can take the compiled schedule by checking `createSSMMask(h:cache:) != nil` against the MambaCache at `ssmIdx`. Each model keeps its own copy of this rule, so a change to the gate must be made in two places. The gate belongs with the shared CompiledDecodeSchedule.caches logic, or in one helper both models call. Move the SSM-mask gate into a shared helper, for example an overload of `CompiledDecodeSchedule.caches(_:)` that takes the `createSSMMask` result or the SSM-cache lookup. Both `Qwen3NextModelInner.decodeStep` and `Qwen35TextModelInner.scheduleCaches` then call it and keep only their model-specific checks (the float16 gate in Qwen3Next).
- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:661` `swift/preconditions` — The SSM-mask precondition in `decodeStep` is a flat `if ... { return nil }` in the middle of a guard chain. The guards on the lines above (`guard embedded.dtype == .float16, let caches = ...`) and below (`guard case .none = createAttentionMask(...)`) state their requirements at the head of the scope. This one does not, so a reader has to notice a different check shape to find the requirement. Write the check as a guard in the same chain: `guard createSSMMask(h: inputs, cache: cache[ssmIdx] as? MambaCache) == nil else { return nil }`, keeping the three requirements in one guard sequence at the head of the function.
- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:664` `reuse/reuse` — Qwen3Next.decodeStep inlines the full-attention mask gate, `guard case .none = createAttentionMask(h:cache: cache[faIdx])`. Qwen35 has a dedicated `attentionMask(_:_:)` helper for the same lookup (`faIdx.flatMap { cache[$0] }.map { createAttentionMask(...) } ?? .none`). The two models compute the same mask, and the shared schedule has no single place that owns it. Have Qwen3Next use the same mask helper, or move `attentionMask` into `CompiledDecodeSchedule` as a function taking the layer index and the cache list. Both models then call it, and the gate reads the same way in each.
- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:668` `duplication/duplication` — Qwen3Next.decodeStep repeats the decode-step wiring of Qwen35.decodeStep: the same `decodeSchedule.run(_:caches:verify:segment:attention:)` call with a `segment:` closure forwarding to the compiled segment cache and an `attention:` closure forwarding to the layer attention. The two copies can drift, for example if the runner's argument order or the segment-cache call changes in one model but not the other. Move the call into one helper on CompiledDecodeSchedule (or a shared extension) that takes the inputs, the caches, the verify shape, the compiled segment cache closure and the attention closure. Both models then call that helper and only supply what differs: the embedding guard and the attention step.

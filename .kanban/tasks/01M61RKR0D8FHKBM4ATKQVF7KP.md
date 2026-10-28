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
position_column: doing
position_ordinal: '8280'
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
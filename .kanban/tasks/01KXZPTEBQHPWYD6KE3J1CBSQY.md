---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01ky5zq9ynbk52f6b020b22yfs
  text: 'CLOSED STALE (user decision 2026-07-22) — the re-scope checkpoint this card mandated has been reached and answered. er33v06 (stable-boundary checkpoints) + 05zt40g (preserve_thinking replay) landed and were measured: real Qwen3.6-27B-mxfp4 round-2 reuse hits the strong bound (>= prompt+output-8) in both suppressed and thinking modes, and the user confirms caching is working in the downstream FoundationModelsRouter. The residual value of block-boundary snapshots (forked/edited-history conversations) does not currently justify the MLXLMCommon prefill restructure. If fork-heavy workloads become important, this card''s design notes (attention-in-chunk-store + constant-size Mamba snapshots at chunk boundaries, vLLM/SGLang/llama.cpp prior art) remain the starting point.'
  timestamp: 2026-07-22T22:39:13.365894+00:00
depends_on:
- 01KXY1XEP9NAAMMRN1KER33V06
- 01KXZP6CTSETM3YJS3E05ZT40G
position_column: done
position_ordinal: ca80
title: 'PromptCache: block-boundary hybrid state snapshots married to the chunk store (vLLM-style prefix caching)'
---
## What

Long-term generalization of hybrid prompt caching, matching what vLLM/SGLang/llama.cpp do: instead of (or in addition to) single whole-round / transcript-stable checkpoints, capture **Mamba/recurrent state snapshots at fixed block boundaries during prefill** (e.g. every `chunkSize` tokens, aligned with the existing chunk store), so a future prompt that diverges ANYWHERE can restore from the nearest boundary at or before the divergence and recompute only the tail.

This subsumes the template-specific fixes (^er33v06 stable boundary, ^05zt40g preserve_thinking) and additionally handles: edited/rewritten history, forked conversations, tool-result substitution, and any template family's quirks — without knowing anything about templates.

**Design constraints (learned from this codebase, do not hand-wave past them):**
- A naive per-block whole-stack `HybridCheckpoint` duplicates attention K/V enormously (each snapshot owns a full copy up to its boundary). The right shape marries the two existing stores: attention layers stored ONCE via the existing chunk store (`PromptCacheChunks` — already 64/512-token sliced, hash-chained, LRU'd), plus small fixed-size **Mamba-state-only snapshots** keyed to chunk-chain positions. Restore = assemble attention chunks (existing `assemble`) + seed Mamba layers from the boundary snapshot. Mamba state is constant-size per snapshot (conv + SSM state), so N boundaries cost N×constant, not N×prefix-length.
- Boundary capture requires prefill to actually HALT at chunk boundaries for hybrid models. `e892fc8`'s commit message flagged this as the blast radius: "checkpointed Mamba state reuse … require[s] restructuring MLXLMCommon's prefill loop". Two candidate paths: (a) executor-side chunked prefill for hybrid models only (the executor already owns a split-prefill mechanism once ^er33v06 lands — generalize it to snapshot at every boundary it crosses); (b) a prefill callback/hook in `MLXLMCommon`'s `TokenIterator` prefill step. Prefer (a): zero MLXLMCommon surface change, hybrid-only cost.
- Eviction: Mamba boundary snapshots must be evicted WITH their attention chunk lineage (extend `evictChunkAndDescendants`) and accounted in `totalStoredBytes`.
- `resolveHybridCheckpoint` grows a nearest-boundary-≤-divergence lookup instead of exact-whole-prefix only.

**Do not start until ^er33v06 and ^05zt40g have landed and been measured** — if those already deliver "recompute only the new user message" for the primary Qwen3.6 workload, this task's residual value is edited-history/fork scenarios; re-scope its priority with the user then. Cite prior art in the implementation: vLLM automatic prefix caching + its hybrid/mamba block-boundary state snapshots, SGLang RadixAttention, llama.cpp recurrent state checkpoint ring.

## Acceptance Criteria

- [ ] A hybrid session whose round-2 prompt diverges mid-history (not just at the tail) reuses up to the nearest block boundary before the divergence (unit test with tiny hybrid models, shrunk chunk size)
- [ ] Memory: storing K boundaries for one round costs one attention chunk-chain + K constant-size Mamba snapshots (assert byte accounting, no duplicated attention K/V)
- [ ] Existing behavior preserved: whole-round and stable-boundary checkpoints still work; pure-attention chunk path untouched; full `swift test` green
- [ ] Real-weights: Qwen3.6 fork/edited-history scenario shows nonzero reuse where current HEAD shows zero (gated integration test)

## Tests

- [ ] Extend `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift`: mid-history divergence restore, boundary accounting, eviction lineage
- [ ] Gated integration: fork/edit scenario against `mlx-community/Qwen3.6-27B-mxfp4`
- [ ] Run: `swift test --filter MLXFoundationModelsTests` → green; integration cases pass

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass.
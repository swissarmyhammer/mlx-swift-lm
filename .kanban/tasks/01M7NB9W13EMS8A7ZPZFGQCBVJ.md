---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'Qwen3.8: register and verify the 8-bit MTP drafter (mlx-community/Qwen3.8-27B-MTP-8bit) with the 8-bit target'
---
## Why
ACPAgent (for the user) wants to run `mlx-community/Qwen3.8-27B-8bit` with the 8-bit MTP drafter `mlx-community/Qwen3.8-27B-MTP-8bit`. `MTPDrafterRegistry` in Libraries/MLXLMCommon/MTPDrafterModelFactory.swift only registers `qwen3_8_27b_mtp_4bit`. Router may switch its `large` preset to this pair. Write it TDD.

## What
- Register `qwen3_8_27b_mtp_8bit` (`mlx-community/Qwen3.8-27B-MTP-8bit`) next to the 4-bit entry and in `all()`.
- Check the 8-bit drafter checkpoint's config (quantization bits/group, layer naming) loads through the existing Qwen MTP drafter type (both `shared` and `visionLanguage` registries as the 4-bit uses) — fix any quantization assumption that hard-codes 4-bit (e.g. fused projections, MultiRowQuantizedMatvec, decode-heads kernels that may assume bits 4).
- The 8-bit target `mlx-community/Qwen3.8-27B-8bit` must also go through the fast paths correctly or fall back exactly (fused q/k/v and gate/up at load, multi-row verify kernel) — any kernel that only supports 4-bit must route 8-bit weights to the generic path.

## Acceptance
- [ ] Unit tests first (failing first): registry lookup of the 8-bit id; any 8-bit kernel/fallback routing.
- [ ] Real model (if downloadable within the ~65 GiB weight ceiling, 8-bit 27B ≈ 29 GB plus drafter): target + drafter load and draft through the VLM Qwen35 path; MTP greedy ids equal plain greedy ids on a short prompt; acceptance rate reported; with kvBits 8 too. Real-model runs take scratchpad bench.lock.
- [ ] All 5 test bundles green; swift-format; pushed to stable; SHA sent to ACPAgent (foundationmodelsacpagent-22) and Router (foundationmodelsrouter-da). #mtp-8bit
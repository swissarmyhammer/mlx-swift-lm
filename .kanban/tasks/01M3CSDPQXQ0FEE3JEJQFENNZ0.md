---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
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

- [ ] For each model, a test with a tiny model shows the offset of each recurrent cache after a prefill.
- [ ] Each model whose recurrent offset is not the prompt length moves its offset (for a `MambaCache`, with `advancePosition(by:)`), or the task records why the offset must stay.
- [ ] All five unit bundles pass.

#prompt-cache
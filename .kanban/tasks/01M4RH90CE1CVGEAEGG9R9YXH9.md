---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: 'DeepSeek-V4 speed: fused Metal kernel for the hyper-connection Sinkhorn steps'
---
## What

The hyper-connection Sinkhorn normalization runs as many small MLX ops. With `hc_sinkhorn_iters: 20`, each hyper-connection runs about 120 kernels. DeepSeek-V4-Flash has 86 hyper-connections, thus about 10,000 kernel dispatches for each decoded token.

Our code: `Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift:89-132` (the Sinkhorn loop at line 126), called from `DeepSeekV4HyperConnection.swift:255`.

## Evidence from other ports

- ml-explore/mlx-lm PR #1189: the fused Sinkhorn kernel took decode on an M3 Ultra (4-bit) from 11 to 20 tok/s.
- mlx-lm PR #1192: the changes that include Sinkhorn fusion took decode from 22.4 to 26.7 tok/s.
- Reference kernels: `deepseek_v41_sinkhorn` in mlx-lm main `mlx_lm/models/deepseek_v41.py:150-216` (merged PR #1895), and `hc_sinkhorn_split` / `hc_sinkhorn_collapse` in Blaizzy/mlx-vlm `mlx_vlm/models/deepseek_v4/hyper_connection.py`.
- This repo already uses `MLXFast.metalKernel` in `Libraries/MLXLMCommon/TurboQuantKernels.swift`.

## Acceptance Criteria

- [ ] One Metal kernel (or a small fixed number) does all Sinkhorn iterations of one hyper-connection.
- [ ] A parity test shows that the kernel gives the same result as the current op path, within float32 tolerance.
- [ ] The existing DeepSeek-V4 unit tests pass.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after.
#deepseek-v4 #performance
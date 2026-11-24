---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: Shard-check the Router's actual checkpoints (mxfp4, oQ4e-mtp/nvfp4)
---
Sharding was validated only on the plain affine 4-bit checkpoints (Qwen3.8-27B-4bit, Inkling-2-Mini-4bit). The Router's model lists also use mlx-community/Qwen3.8-27B-mxfp4 and mlx-community/Inkling-2-Mini-oQ4e-mtp. The latter has mixed quantization: 8-bit affine gs64 attention, embed and lm_head, plus nvfp4 gs16 experts. Its per-layer quantization entries in config.json also differ between in_proj and q/k/v layers.

- [ ] Run `scripts/shard-check.sh` (one Mac, ring) on Qwen3.8-27B-mxfp4 and Inkling-2-Mini-oQ4e-mtp. Gate: tokens + argmax at every step + KL ≤ 0.05.
- [ ] Check that each quantized split respects that format's group size: after splitting, (in/N) % groupSize == 0 for sharded-to-all and the row split for all-to-sharded, for mxfp4, nvfp4 and affine. Make `shard()` throw `.indivisible` naming the layer when it doesn't hold.
- [ ] Unit-test the plan for a mixed-quantization config.
- [ ] Ask the FoundationModelsCluster session (te3zhxt runner) to repeat each on 2 Macs over JACCL with FAST_SYNCH.
- [ ] Record results on this task. #cluster
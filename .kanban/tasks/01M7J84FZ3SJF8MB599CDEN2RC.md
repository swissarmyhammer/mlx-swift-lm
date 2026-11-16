---
assignees:
- claude-code
depends_on:
- 01M7J84C1V3YZ5VGSXGMN44V3N
position_column: todo
position_ordinal: 8b80
title: 'Qwen3.8: measure an 8-bit KV cache for the full-attention layers (opt-in go/no-go)'
---
## Why
At 64k context the KV cache adds 4.3 GB of reads per token. At 8-bit that would save about 1.6 ms per decode step, and more per verify. It changes the output, so it can never become the default without the user's sign-off.

## What
- Measure decode and MTP tok/s with an 8-bit quantized KV cache (group size 64) on the 16 full-attention layers only, at 32k and 64k.
- Measure top-1 agreement against the bf16 KV cache on the golden prompts.
- Check that a quantized cache trims, rewinds and restores correctly under MTP verify and multi-turn prompt-cache reuse.

## Acceptance
- [ ] A table of speed and agreement on this card, with a GO/NO-GO recommendation.
- [ ] Nothing changes by default. If it ships at all, it ships opt-in, and `qwencheck.sh` stays identical with the default settings. #qwen-perf
---
assignees:
- claude-code
depends_on:
- 01M7FHFQT76AN9817CJJ4PQTWR
- 01M7FHFWPMBDPGDSD9BG68BHJ1
position_column: todo
position_ordinal: '8280'
title: 'Flint: compile the one-token decode step (router math, combine, residuals)'
---
## Why
- **Router:** on oQ4e the router alone costs ~1.1 ms/token, about 23 µs per MoE layer. That covers sigmoid, bias add, the fused top-k, normalization (sum, add, divide) and the casts.
- **CPU:** building the graph costs 1.5 ms/token, and the GPU sits idle 26% of each step (Metal System Trace).
- **Precedent:** compiling the elementwise chains fuses them and cuts the CPU cost of building the graph. DeepSeek-V4 did this in ebd865dc (`perf(deepseek-v4): compile the one-token decode step`).

## What
- Add compiled functions, used only in the single-token decode shape, for:
  - the router pre and post steps around `fusedRouterTopK`;
  - the weighted expert combine plus the shared-expert add and its casts;
  - the residual adds that can join them.
- Prefill keeps the op path.
- The compiled path must be bit-exact with the op path. Add a test like the DeepSeek-V4 one.

## Acceptance
- Greedy ids are identical on oQ4e.
- The op path and the compiled path are bit-exact in a unit test on a tiny config.
- ms/token and graph-build time improve (flintcli `graph` mode prints build, asyncEval and wait per token).
- All five bundles are green. #flint-perf
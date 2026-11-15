---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7gmej30v9p45w0jxsxt0gvs
  text: |-
    Research + go/no-go measurements (M5, oQ4e, prompt_706, same build per A/B, env-gated scratch edits in KVCacheSimple.update / RotatingKVCache.updateInPlace / Flint attention, all reverted; Libraries/ == HEAD 383cf981 afterwards). Scripts: scratchpad/wqtab.sh (alternating base/test arms in one binary, 3 rounds, medians), scratchpad/wqtbench (49-layer synthetic decode step with asyncEval pipelining).

    MLX scheduling facts that decide this (mlx checkout: transforms.cpp eval_impl, backend/metal/device.cpp CommandEncoder):
    - The tape is a width-limited (MLX_BFS_MAX_WIDTH=20) reverse BFS from the eval outputs ([token] + every cache array, TokenIterator.evaluateTokenAndLogProbabilities). A write that attention does not read is ordered only by its own inputs, so its place in the tape is not controllable from Swift except by a data dependency.
    - Barriers are per-buffer: set_input_array barriers on a RAW hit in prev_outputs, register_output_array on a WAR hit in prev_inputs. With the write in front of SDPA a layer runs heads | B | writeK+writeV | B | SDPA | B | o_proj. Moving the write behind attention either takes a WAR barrier of its own or (if fed from the attention output) joins o_proj's epoch — one barrier less per layer at best.
    - Donation of the cache buffer to the SliceUpdate needs attention (which then also reads the old array) to be evaluated and detached first; otherwise the update copies the whole buffer.

    Real-model probes (decode ms/tok at 2k / 16k, base vs test in the same binary):
    1. C — attention (MLX SDPA, sinks) reads the pre-write buffer view (same row count), the write stays, nothing reads it but the cache: base 6.26 / 7.00, test 6.31 / 7.01 — no gain. This is the schedule of the card's design (prefix + new row as separate inputs; the write no longer a dependency of attention).
    2. B — same, but the K/V write takes its rows from the attention output (a view, no extra dispatch), so the write is ordered after attention and shares o_proj's epoch (the schedule a kernel that also emits the new rows would force): base 6.24 / 7.02, test 6.34 / 7.30 — worse.
    3. F — the write's subscript sets are still built on the CPU but never evaluated (no GPU write at all; attention reads the unwritten buffer): base 6.26 / 7.01, test 6.23 / 6.99 — -0.03 / -0.02 only.
    Prefill unchanged in all arms (8950-9020 / 8180-8220 tok/s); peak memory identical (25.15 / 25.39 GB), so no donation-failure copies.

    Synthetic 49-layer step (wqtbench, 2k / 16k ms/step): A write-then-SDPA 3.58 / 4.66; C SDPA on old view + independent write 3.45 / 4.68; B write fed by attention 3.50 / 4.72; D no write 3.05 / 4.10; F write built but not evaluated 3.45-3.47 (2k). In the synthetic (GPU underloaded) the write's cost is almost all CPU graph build, and its position is worth <= 0.13 ms at 2k and 0 at 16k.

    Conclusion: the 0.31 ms of ^y5d37gd ("skip both writes") needs BOTH the CPU build and the GPU dispatch of the writes gone. Removing only the GPU side (F) gives ~0.02 ms; removing only CPU (^y5d37gd putAlong / no-ellipsis) gives 0; moving the write off attention's dependency chain (C, B) gives 0 or a loss. Decode is a balanced CPU/GPU handoff (^nanqj4p), so a change that keeps the write (as this card requires: "the write still happens, just not in front of attention") cannot recover the 0.31 ms. A decode attention kernel taking the prefix + the new row would reproduce schedule C or B, plus a custom-kernel call in place of SDPA (which ^nanqj4p showed is not faster in-model). Not built: the go/no-go measurement says no, per the card's escape clause. The remaining lever is removing the write work itself on both sides — e.g. an upstream in-place / donated custom-kernel output in mlx + mlx-swift so the heads kernel writes the cache row (no separate SliceUpdate node, no separate dispatch).
  timestamp: 2026-11-15T22:15:07.616125+00:00
- actor: claude-code
  id: 01m7gmerpbd2vvmhpvtgcnfk7d
  text: |-
    ### implement — no-change
    - evidence: go/no-go probes of the card's schedule in the full model (oQ4e, same-binary A/B, decode ms/tok 2k/16k): C attention on the pre-write buffer + independent write 6.26/7.00 -> 6.31/7.01; B write ordered after attention (fed from its output, shares o_proj's epoch) 6.24/7.02 -> 6.34/7.30; F writes built on CPU but not run on GPU 6.26/7.01 -> 6.23/6.99. Prefill and peak memory unchanged in every arm. Synthetic 49-layer step agrees (write position <= 0.13 ms at 2k, 0 at 16k). The 0.31 ms needs the write's CPU build and GPU dispatch both gone, which a path that keeps the write cannot do, so no kernel was built and no tests were written (no production code). All probe edits reverted: git status shows only .kanban; no build, test suite, flintcheck or swift-format run was needed for an empty code diff.
    - next: human decides whether to close as measured / no-change, or to pursue an upstream in-place (donated) custom-kernel output in mlx + mlx-swift so the heads kernel writes the cache row itself.
  timestamp: 2026-11-15T22:15:14.379244+00:00
- actor: claude-code
  id: 01m7gmfvw3ccz2mqh8wrp8nmev
  text: |-
    ### commit — changed
    - evidence: local commit (not pushed) of the .kanban/ measurement record, subject: `chore(kanban): record that moving the Flint decode cache write off the attention path gives no speedup (^wqt7w37)`; files: .kanban/tasks/01M7GCNQTVHPRSGCP27Y5D37GD.{jsonl,md}, .kanban/tasks/01M7GKHRNX075K8G9T2WQT7W37.{jsonl,md}. No code changed. The sha is in `git log` under that subject.
    - next: the 0.31 ms needs the cache write gone entirely, which needs an in-place custom-kernel output upstream in mlx/mlx-swift.
  timestamp: 2026-11-15T22:15:50.403144+00:00
position_column: doing
position_ordinal: '80'
title: 'Flint: take the L=1 cache write off the attention critical path (~0.31 ms/token)'
---
## Why
- Found in ^y5d37gd (oQ4e, M5, same-session flintbench). If both L=1 cache writes are skipped (a timing-only probe), decode drops by 0.31 ms/token (6.25 -> 5.92 at 2k, 7.00 -> 6.70 at 16k). If only the V write is skipped, it drops by just 0.05 ms. Cutting the Swift CPU cost of the write (index forms without an ellipsis, putAlong with a shared index) changed nothing. So the cost is the serial step that the write adds to each layer: heads kernel -> slice update -> SDPA.
- MLX Swift has no public API that removes that step. `MLXFast.metalKernel` has no output that aliases or donates an input, so the heads kernel cannot write the cache row itself unless it relies on undefined aliasing.

## What
Evaluate an attention path that reads the cache prefix (rows < offset) plus the new K/V row as separate inputs. The cache write then runs at the same time instead of in front of the attention. Candidates are a decode attention kernel (decbench StreamAttention2 was bit-exact against MLX SDPA with sinks in earlier work) and an upstream request for in-place custom-kernel outputs in mlx / mlx-swift.

## Acceptance
- Bit-exact against MLX SDPA, so greedy ids stay unchanged on oQ4e and oQ8e.
- Decode ms/token improves against a same-session baseline at 2k and 16k. Prefill does not regress.
- RotatingKVCache and KVCacheSimple semantics stay unchanged. All five bundles are green. #flint-perf
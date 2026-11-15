---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7gfmm4pwc77dv659hfd2sgc
  text: |-
    Research (mlx-swift Cmlx, .build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx):
    - `metal::Device` constructor (backend/metal/device.cpp) picks defaults from the last char of the GPU arch string: 'p' 20/40, 'g' (base/pro) 40/40, 's' (max) 50/50, 'd' (ultra) 50/50, other 40/40. This machine is an Apple M5 Ultra -> 50 ops / 50 MB.
    - It then calls `env::max_ops_per_buffer(default)` / `env::max_mb_per_buffer(default)` (utils.h). Each reads `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` once into a function-local static. The Device is a lazily-made singleton (`static Device* metal_device = new Device`), so the values are fixed at the first GPU use in the process. setenv works only before that; afterward it is ignored silently.
    - `CommandEncoder::needs_commit()` commits when buffer_ops_ > max_ops OR (bytes of inputs bound to the buffer >> 20) > max_mb. Checked in metal eval after each primitive.
    - There is no C/Swift API to set either value, and nothing per stream: `get_max_ops_mb_per_buffer()` is a const device getter, mlx-c exposes nothing. The only supported knob is the env var, process-wide, before the device is created.
    - No library code in this repo sets MLX env vars; the only precedent is Tests/MLXTestPrecision (C constructor, setenv with overwrite=0) for TF32, test-only.
    Measurement: scratchpad flintcli bench now also prints peak memory; grid script scratchpad/bufgrid.sh (configs interleaved per rep, 5 reps, medians).
  timestamp: 2026-11-15T20:51:03.446581+00:00
- actor: claude-code
  id: 01m7ggkszk7hymky3t8y9q4ef9
  text: |-
    Flint oQ4e grid (scratchpad/bufgrid.sh, flintcli release bench, prompt_706, 200 decode tok, 5 reps interleaved, medians; d = env var unset = 50 ops / 50 MB on M5 Ultra). Raw: scratchpad/grid_flint.txt.

    cfg(ops/mb) | dec2k ms | dec8k ms | dec16k ms | pf2k tok/s | pf8k tok/s | pf16k tok/s | peak16k GB
    d/d     | 6.23 | 6.62 | 7.00 | 8998 | 9020 | 8212 | 25.39
    d/50    | 6.22 (-0.2%) | 6.60 (-0.3%) | 6.99 (-0.1%) | 9034 (+0.4%) | 8981 (-0.4%) | 8259 (+0.6%) | 25.39
    d/100   | 6.22 (-0.2%) | 6.44 (-2.7%) | 6.83 (-2.4%) | 8809 (-2.1%) | 8958 (-0.7%) | 8346 (+1.6%) | 26.22
    d/200   | 5.92 (-5.0%) | 6.25 (-5.6%) | 6.84 (-2.3%) | 8549 (-5.0%) | 8906 (-1.3%) | 8369 (+1.9%) | 27.49
    50/d    | 6.23 (+0.0%) | 6.63 (+0.2%) | 7.02 (+0.3%) | 8973 (-0.3%) | 9019 (-0.0%) | 8240 (+0.3%) | 25.39
    50/50   | 6.25 (+0.3%) | 6.60 (-0.3%) | 7.01 (+0.1%) | 8993 (-0.0%) | 8962 (-0.6%) | 8185 (-0.3%) | 25.39
    50/100  | 6.20 (-0.5%) | 6.42 (-3.0%) | 6.82 (-2.6%) | 8832 (-1.8%) | 9007 (-0.1%) | 8338 (+1.5%) | 26.22
    50/200  | 5.91 (-5.1%) | 6.24 (-5.7%) | 6.83 (-2.4%) | 8706 (-3.2%) | 8934 (-1.0%) | 8354 (+1.7%) | 27.49
    100/d   | 6.24 (+0.2%) | 6.58 (-0.6%) | 6.99 (-0.1%) | 8951 (-0.5%) | 8944 (-0.8%) | 8234 (+0.3%) | 25.39
    100/50  | 6.26 (+0.5%) | 6.63 (+0.2%) | 7.01 (+0.1%) | 9008 (+0.1%) | 8945 (-0.8%) | 8212 (+0.0%) | 25.39
    100/100 | 6.19 (-0.6%) | 6.43 (-2.9%) | 6.84 (-2.3%) | 8752 (-2.7%) | 8953 (-0.7%) | 8364 (+1.9%) | 26.22
    100/200 | 5.92 (-5.0%) | 6.31 (-4.7%) | 6.90 (-1.4%) | 8624 (-4.1%) | 8960 (-0.7%) | 8410 (+2.4%) | 28.08
    200/d   | 6.23 (+0.0%) | 6.57 (-0.8%) | 6.99 (-0.1%) | 8952 (-0.5%) | 8961 (-0.7%) | 8184 (-0.3%) | 25.39
    200/50  | 6.25 (+0.3%) | 6.58 (-0.6%) | 6.99 (-0.1%) | 8945 (-0.6%) | 8940 (-0.9%) | 8230 (+0.2%) | 25.39
    200/100 | 6.23 (+0.0%) | 6.44 (-2.7%) | 6.86 (-2.0%) | 8788 (-2.3%) | 8930 (-1.0%) | 8361 (+1.8%) | 26.22
    200/200 | 5.92 (-5.0%) | 6.30 (-4.8%) | 6.90 (-1.4%) | 8620 (-4.2%) | 8920 (-1.1%) | 8413 (+2.5%) | 28.08
    400/d   | 6.24 (+0.2%) | 6.58 (-0.6%) | 6.98 (-0.3%) | 8987 (-0.1%) | 9131 (+1.2%) | 8231 (+0.2%) | 25.39
    400/50  | 6.26 (+0.5%) | 6.60 (-0.3%) | 7.00 (+0.0%) | 8961 (-0.4%) | 9041 (+0.2%) | 8208 (-0.0%) | 25.39
    400/100 | 6.18 (-0.8%) | 6.43 (-2.9%) | 6.83 (-2.4%) | 8847 (-1.7%) | 8951 (-0.8%) | 8342 (+1.6%) | 26.22
    400/200 | 5.93 (-4.8%) | 6.31 (-4.7%) | 6.89 (-1.6%) | 8609 (-4.3%) | 8905 (-1.3%) | 8419 (+2.5%) | 28.08

    Reading: the ops cap has no effect at any value (each row equals its d/MB row within noise), so for Flint the MB cap always commits first. All of the gain comes from MB: 200 MB gives decode -5% at 2k/8k and -1.4..-2.4% at 16k, but 2k prefill -3..-5%, and peak memory +2.1..2.7 GB (+8..11%). 100 MB gives about half the decode gain (only at 8k/16k), with 2k prefill -2% and peak +0.8 GB. The ^nanqj4p 2k number (5.99 vs 6.35) reproduces as 5.92 vs 6.23.
  timestamp: 2026-11-15T21:08:05.235335+00:00
- actor: claude-code
  id: 01m7ghfbk7dfmmpdrjpt22ab1j
  text: |-
    Qwen3.8-27B-4bit (plain decode, in HF cache) grid, same bench, 4 reps interleaved, medians. Raw: scratchpad/grid_qwen.txt.

    cfg(ops/mb) | dec2k ms | dec8k ms | dec16k ms | pf2k tok/s | pf8k tok/s | pf16k tok/s | peak16k GB
    d/d     | 17.95 | 18.16 | 19.26 | 1376 | 1285 | 1158 | 17.47
    d/100   | 17.54 (-2.3%) | 18.05 (-0.6%) | 18.92 (-1.8%) | 1389 (+0.9%) | 1301 (+1.2%) | 1168 (+0.8%) | 18.13
    d/200   | 17.48 (-2.6%) | 18.30 (+0.7%) | 19.12 (-0.8%) | 1401 (+1.8%) | 1316 (+2.4%) | 1183 (+2.1%) | 19.17
    200/d   | 17.98 (+0.2%) | 18.17 (+0.0%) | 19.30 (+0.2%) | 1372 (-0.3%) | 1284 (-0.1%) | 1155 (-0.3%) | 17.47
    200/200 | 17.48 (-2.6%) | 18.27 (+0.6%) | 19.12 (-0.7%) | 1375 (-0.1%) | 1296 (+0.9%) | 1168 (+0.8%) | 19.47
    400/400 | 17.61 (-1.9%) | 18.76 (+3.3%) | 19.97 (+3.7%) | 1385 (+0.6%) | 1299 (+1.1%) | 1169 (+0.9%) | 22.13

    Again ops alone does nothing. MB 100/200 gives -2.5% decode at 2k, but at 8k/16k it is within noise or worse, and peak memory goes up 0.7..2.0 GB. 400/400 regresses decode at 8k/16k (+3.3/+3.7%) and adds +4.7 GB peak.

    Decision: no library change.
    1. The card's acceptance says that prefill and memory must not regress. Every MB value above the 50 MB default raises peak memory for both models (Flint +0.8 GB at 100 / +2.1 GB at 200; Qwen3.8 +0.7 / +1.7 GB). For Flint, 2k prefill also drops by 2% at 100 and 3-5% at 200. No cell meets the acceptance. The ops cap has no measurable effect on either model.
    2. It cannot be applied in a narrow way. MLX reads the value once into a C++ static when the Metal device singleton is first made. There is no API for it, and nothing per stream or per model. A setenv at model load is silently ignored if the app (or another model, or a test) has already touched the GPU. When it does take effect, it changes every model in the process, including the Qwen3.8 8k/16k decode regression at larger values. A GenerateParameters/ModelConfiguration opt-in would therefore be a lie in most processes.
    Recommendation for the user/apps: an app that is Flint-only, decode-bound and has memory headroom can launch with `MLX_MAX_MB_PER_BUFFER=200` (the ops var is not needed): Flint decode -5% at 2k/8k and -2% at 16k, for +2 GB peak memory and -3..5% short-prompt prefill. Do not go above 200 (Qwen3.8 regresses at 400). This must be set in the process environment before launch (or before any MLX GPU use).
    Side note: scratchpad flintcli `bench` now also prints `peak <GB>` per ctx (backup in scratchpad/main.swift.bak_twc); flintbench.sh still parses it.
  timestamp: 2026-11-15T21:23:08.007941+00:00
- actor: claude-code
  id: 01m7ghfhxmwhhh3ec3z4db98bp
  text: |-
    ### implement — no-change
    - evidence: Flint oQ4e 20-cell grid (5 reps) and Qwen3.8-27B-4bit 6-cell grid (4 reps), tables above. Only MLX_MAX_MB_PER_BUFFER matters. It raises peak memory on both models, regresses Flint 2k prefill, and regresses Qwen3.8 8k/16k decode at 400. MLX gives no per-process API and no per-stream API (the env var is read once at Metal device creation), so a library opt-in cannot be applied safely. No repo files changed, so tests and flintcheck were not needed.
    - next: the user decides whether apps should document `MLX_MAX_MB_PER_BUFFER=200` for Flint-only, decode-bound deployments. The acceptance (A/B table + decision with evidence) is met. Task left in doing.
  timestamp: 2026-11-15T21:23:14.484814+00:00
- actor: claude-code
  id: 01m7ghgf6dhgs16kj31j34cwxp
  text: |-
    ### commit — changed
    - evidence: commit subject "chore(kanban): record the MLX command-buffer sizing measurements on M5 (^twc3zh6)"; files: .kanban/tasks/01M7GCNJFYATF646QNT5SMNPG3.{jsonl,md}, .kanban/tasks/01M7GCNVYNJNS73KPQGTWC3ZH6.{jsonl,md}; local commit only, not pushed; no code changed
    - next: none
  timestamp: 2026-11-15T21:23:44.461677+00:00
- actor: claude-code
  id: 01m7ghha1ynkyfdz572v43dy89
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (2c68203d) — 0 findings; 4 .kanban/ files in the delta, no validator matches them (0 files reviewed). No prior Review Findings sections.
    - next: none — moved to done
  timestamp: 2026-11-15T21:24:11.966298+00:00
- actor: claude-code
  id: 01m7ghhq422z3ycjt54venaad9
  text: |-
    ### finish iteration 1 — clean
    - implement: no-change — measured grid on Flint + Qwen3.8-27B; no safe library change (process-wide, read once, costs memory/prefill)
    - test: skipped — empty code diff
    - commit: 2c68203d (kanban evidence only)
    - review: clean — 0 findings; task done
  timestamp: 2026-11-15T21:24:25.346972+00:00
position_column: done
position_ordinal: ffffad80
title: 'Decode: measure MLX command-buffer sizing (MLX_MAX_OPS/MB_PER_BUFFER) for decode on M5'
---
## Why
- Found in ^nanqj4p. On oQ4e at 2k (flintcli graph), `MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200` gives 5.99 ms/tok against 6.35 at the default 50/50 (-6%); 400/400 gives 5.98, and 25/25 gives 7.04.
- A decode token makes ~41 command buffers of ~111 us. MLX allows 10 in flight (MAX_ACTIVE_TASKS), and the GPU idles ~1.1-1.5 ms at each token boundary.
- This is a process-wide MLX setting, not a model change, so it needs its own decision.

## What
1. Measure decode and prefill across buffer sizes for Flint and Qwen3.8 at 2k/16k, plus peak memory.
2. Decide whether the library should set it (and where), or document it for apps.

## Acceptance
- A recorded A/B table, and a decision with its evidence. Prefill and memory must not regress. #flint-perf
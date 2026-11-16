---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jyy27ky05m13mkd5eyrc3r
  text: |-
    Research:
    - VLM `Qwen35.prepare` (Libraries/MLXVLM/Models/Qwen35.swift) picks `window = prefill.resolvedStepSize()` (512) for single-shot vs windowed, and `prepareContinuation` calls `prefill.forEachChunk(total:)` with the generic 512 default. Both need the model default. Single-shot path already projects the last token only (`lastTokenOnly: true`); continuation drops chunk logits un-evaluated, so lm_head does not run on prefill.
    - LLM `Qwen35TextModel` / `Qwen35Model` (Libraries/MLXLLM/Models/Qwen35.swift) have no `prepare` of their own; they use the generic `LLMModel.prepare` extension, which reads `resolvedStepSize()` and `forEachChunk` with the generic default. A class cannot call the protocol-extension default, so the generic body needs a helper that takes the default step.
    - Checkpoint alignment: ChatSession's hybrid checkpoints are taken by `PromptCacheCheckpoint.prefillToCheckpoint`, which feeds the prompt up to the message boundary through `PromptCacheCheckpoint.prefill` (model `callAsFunction` + `forEachChunk` with the generic 512 default, reserving 0) and takes the checkpoint after that call. Only the tail after the boundary reaches `model.prepare`. So the checkpoint position is the message boundary regardless of the prepare step; the GDN state is never snapshotted at a prepare-chunk boundary. The prepare step change thus cannot misalign checkpoints. Consequence: on ChatSession turns most of the prompt still prefills at 512 via the checkpoint feed (out of scope here, candidate follow-up).
    - `WiredMemoryUtils.tune` reports `prefill.resolvedStepSize()` (generic 512) as the measured step; Flint left it as is too.
    - qwencli (scratchpad harness) loads through VLMModelFactory, so qwencheck/qwenbench exercise the VLM path; `bench` passes `stepSize: nil` by default and prints peak GB per context.
  timestamp: 2026-11-16T19:56:50.291468+00:00
- actor: claude-code
  id: 01m7jzcnksqn37g92404dpx817
  text: |-
    Implementation landed (uncommitted): `Qwen35PrefillPolicy.defaultStepSize = 2048` (new, Libraries/MLXLMCommon/Qwen35PrefillPolicy.swift); VLM `Qwen35.prepare` window and `prepareContinuation` forEachChunk use it; generic `LLMModel.prepare` body moved into internal `prepareInChunks(...defaultStepSize:)`, which `Qwen35TextModel.prepare` calls with 2048 and `Qwen35Model.prepare` forwards to the text model. Caller step still wins (`resolvedStepSize(defaultStepSize:)`). Unit tests: Tests/MLXLMTests/Qwen35PrefillStepTests.swift (RED `[512,1024,1536,2048(,2049)]`, GREEN `[2048(,2049)]` on VLM, text and wrapped text). Flint's private ProgressLog moved to MLXLMTestSupport as `PrefillProgressLog` so both suites share it.

    qwencheck.sh (out qwen_golden_mn44v3n.json): short/code/toolchat plain+MTP MATCH (all 257 ids); SESSION PASS (plain and MTP, turns 2/3 warm: cached 139/203 tok, prefilled 32/29 tok, i.e. checkpoint restore on follow-ups works); TOOLCHAT both PASS. CHECK FAIL only because the 29127-token "32k" prompt diverges, as expected from the chunk change (512 -> 2048 balanced chunks):
    - 32k plain: first difference at step 82: id 8 vs golden 310.
    - 32k mtp: first difference at step 212: id 553 vs golden 8.

    Near-tie evidence (new scratchpad `qwencli neartie` mode: TokenIterator with a sampler that teacher-forces the golden ids and prints the top-2 raw logits at the given step):
    - step 82 on golden plain ids, step 2048: top1 8 = 26.875, top2 310 = 26.875, gap 0.0. Same point at step 512: top1 310 = 26.875, top2 8 = 26.75, gap 0.125 (reproduces golden 310).
    - step 212 on golden MTP ids: top1 553 = 27.0, top2 8 = 26.875, gap 0.125 at both step 2048 and step 512 (the golden MTP id 8 was already the near-tie runner-up of the verify path).
    - For reference, the golden itself has plain vs MTP diverging at step 113: gap 0.125 (2048) / 0.0 (512) between 4936 and 6397.
    All gaps are <= 0.125 (one bf16 ulp at |logit|~27), well under 0.5. qwen_golden_base.json NOT overwritten; user to decide whether to re-baseline.
  timestamp: 2026-11-16T20:04:48.889348+00:00
- actor: claude-code
  id: 01m7k06n0nyttz71txtpd8cy3z
  text: |-
    qwenbench.sh (out qwen_bench_mn44v3n.txt, medians of 3):
    - prefill tok/s 2k / 16k / 64k: 1470 / 1269 / 797. Delta vs stable (qwen_bench_base.txt 1334 / 1152 / 722): +10.2% / +10.2% / +10.4%. Delta vs HEAD before this change (qwen_bench_nsh_c1.txt 1339 / 1158 / 722): +9.8% / +9.6% / +10.4%.
    - decode unchanged vs HEAD: plain 17.14 / 19.92 / 22.82 ms/tok (HEAD 17.10 / 19.84 / 22.74), MTP 9.46 / 12.50 / 13.59 (HEAD 9.35 / 12.50 / 13.58), tokens/verify 3.29 / 2.40 / 2.65 identical. (Vs stable: plain -6.1/-3.3/-2.6%, MTP -7.0/-4.0/-4.3%, from earlier commits.)
    - Peak memory (bench `peak`, Memory.peakMemory across prefill + 256 decode): 64k prefill 22.44 GB vs 20.88 GB at HEAD (+1.56 GB); 2k 18.13 GB vs 16.69 GB (+1.44 GB). Matches the predicted ~1.6 GB.
  timestamp: 2026-11-16T20:19:00.245887+00:00
- actor: claude-code
  id: 01m7k0kw6heg09at3bdyx97akd
  text: |-
    ### implement — changed
    - evidence: 9 files — Libraries/MLXLMCommon/Qwen35PrefillPolicy.swift (new), Libraries/MLXLLM/LLMModel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Libraries/MLXLMCommon/PrefillParameters.swift (doc), Tests/MLXLMTestSupport/PrefillProgressLog.swift (new), Tests/MLXLMTests/Qwen35PrefillStepTests.swift (new), Tests/MLXLMTests/FlintTests.swift (uses the shared log). `swift build --build-tests` clean. xctest: MLXLMTests 1120 XCTest (0 failures; 4 skipped are the pre-existing env-gated LFM2 real-checkpoint tests from the upstream merge 6dc1e4e7) + 2369 Swift Testing passed (3 pre-existing known issues); MLXGuidedGenerationTests 78, MLXFoundationModelsTests 496, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 — all pass. qwencheck: 32k golden diverges (near-tie, gap <= 0.125); qwenbench prefill +10.2/+10.2/+10.4% vs stable; 64k peak +1.56 GB.
    - next: /review. User decision: re-baseline qwen_golden_base.json for the 32k prompt (not done). Follow-up ^237qeb2 filed: checkpoint feeds (most of a ChatSession prompt) still prefill at step 512.
  timestamp: 2026-11-16T20:26:13.585120+00:00
depends_on:
- 01M7J847TB7CBFPZ2BQNSH02J2
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 prefill: default step 2048'
---
## Why
Measured prefill at 2k: step 512 runs at 1326 tok/s and step 2048 at 1427 tok/s (+8–10%), for about 1.6 GB more peak memory. lm_head already does not run on prefill. Flint commit 6adfaa83 set FlintModel.prepare to step 2048, and commit 792cd657 made guided passes use the model's own prefill step.

## What
- Give Qwen35 (VLM and LLM) a default prefill step of 2048 in `prepare`, keeping any caller-specified step.
- Check that the hybrid checkpoint and prompt-cache boundaries still line up. GDN state is snapshotted at chunk boundaries.

## Acceptance
- [x] `qwencheck.sh`: greedy ids identical in plain and MTP modes, and cache reuse passes, including checkpoint restore on follow-ups. bf16 output can depend on chunk boundaries; if ids change, record the near-tie evidence here. (short/code/toolchat identical; session PASS; 32k diverges at plain step 82 / MTP step 212, both near-ties with top-2 gap 0.0 / 0.125 — see comments. Base golden not overwritten; user decides on re-baseline.)
- [x] Peak memory at a 64k prefill is recorded. All five bundles are green. `qwenbench.sh` delta on this card. #qwen-perf
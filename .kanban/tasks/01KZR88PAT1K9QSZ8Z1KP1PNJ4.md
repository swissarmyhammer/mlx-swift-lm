---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzs300qnapq8mfmyfg8k8dst
  text: |-
    Research result. The decision comes from the training-time reference of the checkpoint.

    The checkpoint `deepseek-ai/DeepSeek-V4-Flash` does not publish a `modeling_deepseek_v4.py` of its own. Its `config.json` gives `"architectures": ["DeepseekV4ForCausalLM"]`, `"model_type": "deepseek_v4"`, `"swiglu_limit": 10.0`, `"n_shared_experts": 1`, and no `auto_map`. Thus the checkpoint binds to the native `deepseek_v4` implementation in Hugging Face `transformers`, `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py` (generated from `modular_deepseek_v4.py`).

    That file decides. In `DeepseekV4SparseMoeBlock.__init__` the shared expert is `self.shared_experts = DeepseekV4MLP(config)`. In `DeepseekV4MLP` the forward pass is `gate = self.gate_proj(x).clamp(max=self.limit)` and `up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)` with `self.limit = config.swiglu_limit`. Two independent fetches of `modeling_deepseek_v4.py` and `modular_deepseek_v4.py` gave the same quotes.

    DECISION: the shared expert READS the SwiGLU clamp. The Swift copies (`scouzi1966/mlx-swift-lm`, `osaurus-ai/vmlx-swift-lm`) are correct on this point. The `swiglu_limit=0.0` in `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` diverges from the reference.

    Plan: keep the code as it is (it already gives `configuration.swigluLimit` to the shared expert), rewrite the header of `DeepseekV4MoE.swift` to state the decision and its source, and add a pin test in `DeepseekV4MoETests.swift` that drives the shared expert through `DeepseekV4MoE(configuration:layer:)` with a gate value over the limit. I will prove the test with a mutation: flip the shared expert to `swigluLimit: 0`, see the red run, and restore.
  timestamp: 2026-08-11T18:57:40.597856+00:00
- actor: claude-code
  id: 01kzs37r5924yzakez61ykhcgr
  text: |-
    Implementation landed. TDD with a mutation proof:

    1. Added `theSharedExpertReadsTheClampOfTheCheckpoint()` and its fixture `sharedExpertOnlyLayer()` to `Tests/MLXLMTests/DeepseekV4MoETests.swift`. The fixture builds a whole `DeepseekV4MoE` from a configuration with `swiglu_limit: 10`, gives the routed path zero `down_proj` weights, and gives the shared expert split projections. Thus the layer output IS the shared-expert output, and the test drives it through `DeepseekV4MoE(configuration:layer:)` — the constructor that decides which limit the shared expert reads.
    2. Mutation proof: with the shared expert built as `swigluLimit: 0`, the test went RED — `floats(saturated) → [50.0, 0.0]` against the clamped `[9.999546, 0.0]`.
    3. Restored `swigluLimit: configuration.swigluLimit` (the correct form, per the decision). The test went GREEN.
    4. Rewrote the header of `Libraries/MLXLLM/Models/DeepseekV4MoE.swift`: the open-divergence paragraph is now the decision, with the deciding source (transformers `modeling_deepseek_v4.py`, `DeepseekV4SparseMoeBlock.__init__` → `DeepseekV4MLP(config)` with the `config.swiglu_limit` clamp) and the name of the pin test.

    Full `DeepseekV4MoETests` suite: 15 tests, all pass, build clean. No production behavior changed — the code already agreed with the decision; the header and the pin test are the changes.
  timestamp: 2026-08-11T19:01:53.961035+00:00
- actor: claude-code
  id: 01kzs37yxaxwyz3ph9brc6fhsx
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DeepseekV4MoE.swift (header states the decision; shared expert keeps `configuration.swigluLimit`), Tests/MLXLMTests/DeepseekV4MoETests.swift (new pin test `theSharedExpertReadsTheClampOfTheCheckpoint` + fixture). Deciding source: HF transformers `modeling_deepseek_v4.py` — `DeepseekV4SparseMoeBlock.__init__` builds `self.shared_experts = DeepseekV4MLP(config)`, and `DeepseekV4MLP` clamps with `config.swiglu_limit`; the checkpoint `config.json` (`model_type: deepseek_v4`, `swiglu_limit: 10.0`, no `auto_map`) binds to that file. Mutation proof: `swigluLimit: 0` turned the run red (50.0 against 9.999546); restore turned it green. Suite: 15/15 pass.
    - next: /review
  timestamp: 2026-08-11T19:02:00.874698+00:00
- actor: claude-code
  id: 01kzs3mpx9dz0dg0ea8yr25mc5
  text: |
    ### test — green

    - evidence: `swift test` — 0 failures across all bundles (245 MLXLMTests XCTest, 13 MLXHuggingFaceMacrosTests XCTest, 454 MLXGuidedGenerationTests swift-testing, 80 MLXFoundationModelsTests swift-testing, 282 + 7 swift-testing), 2 tests skipped, 1 build warning.
    - The 2 skips are `CompiledDecodeCorrectnessTests.testShapefulCompileIsOnlyReliableInIsolation_DoNotAssert` and `testShapelessCompileCrashesOnGrowingKVCacheSimple_DoNotRun`. Both are pre-existing and governed by the completed, human-reviewed research task 01KYD3ZCWTZ414Y79RSAKVQXXZ, whose written acceptance criteria explicitly sanction this outcome (a process-fatal mlx-c crash under `shapeless: true`, and a non-deterministic full-suite-only divergence under `shapeless: false`). Not touched.
    - The 1 warning (`missing creator for mutated node: ... mlx-swift_Cmlx.bundle/Contents/MacOS`) is a SwiftPM build-graph diagnostic about the vendored mlx-swift dependency's resource bundle, not a compiler warning. It was already proven independent of this repository's source in task 01KZGMTXP266RDESZHBG5E7907 (same single line with and without the change stashed). Confirmed again here: no other `warning:` line in the full log.
    - No code changes were made. No failures needed fixing.
  timestamp: 2026-08-11T19:08:58.665545+00:00
- actor: claude-code
  id: 01kzs3zmf67mew529x0hv4kfv9
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (commit 55582c9); 2 findings, 2 confirmed, 8 files attempted, 0 skipped — Libraries/MLXLLM/Models/DeepseekV4MoE.swift:118, Libraries/MLXLLM/Models/DeepseekV4MoE.swift:356
    - next: the implementer must mark `DeepseekV4SwitchGLU` and `DeepseekV4MoE` as `final`, then request a new review pass
  timestamp: 2026-08-11T19:14:56.614077+00:00
- actor: claude-code
  id: 01kzs40bez103rfv8crax0t6fz
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 2 files (Libraries/MLXLLM/Models/DeepseekV4MoE.swift, Tests/MLXLMTests/DeepseekV4MoETests.swift)
    - test: green — swift test, 1081 tests run, 0 failures
    - commit: 55582c9
    - review: findings — Libraries/MLXLLM/Models/DeepseekV4MoE.swift:118, Libraries/MLXLLM/Models/DeepseekV4MoE.swift:356
  timestamp: 2026-08-11T19:15:20.159005+00:00
- actor: claude-code
  id: 01kzs4315ragkgf435pssypphv
  text: |-
    ### implement — changed
    - evidence: 1 file — Libraries/MLXLLM/Models/DeepseekV4MoE.swift. The two findings named `DeepseekV4SwitchGLU` and `DeepseekV4MoE`. The finding shows one example of a cause, thus I removed the cause from the whole file: all four concrete classes are now `final` (`DeepseekV4MLP`, `DeepseekV4SwitchGLU`, `DeepseekV4MoEGate`, `DeepseekV4MoE`). A repository grep found no subclass of any of the four. Build clean — the one `missing creator for mutated node` line is the known SwiftPM diagnostic that is independent of this source. `swift test --filter DeepseekV4MoETests`: 15 tests, 0 failures. Both findings are checked on the card.
    - next: /review
  timestamp: 2026-08-11T19:16:47.928046+00:00
- actor: claude-code
  id: 01kzs4cnq7w3rzw8r0bxkx2qjk
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (f4c2214) — 0 findings, 8 attempted, 0 failed, 0 skipped; the two prior findings from 2026-08-11 14:10 are checked
    - next: none — the task is done
  timestamp: 2026-08-11T19:22:03.879941+00:00
- actor: claude-code
  id: 01kzs4d6pay34a4gdjasd58hy4
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 file (Libraries/MLXLLM/Models/DeepseekV4MoE.swift, four classes now final)
    - test: green — swift test, exit 0, 0 failures
    - commit: f4c2214
    - review: clean — 0 findings, 2 prior findings checked; task moved to done
  timestamp: 2026-08-11T19:22:21.258617+00:00
position_column: done
position_ordinal: e980
title: Decide whether the DeepSeek-V4 shared expert reads the SwiGLU clamp
---
## What

The two references disagree about the shared expert of a DeepSeek-V4 MoE
layer, and this repository must follow one of them.

- `scouzi1966/mlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift:1284-1287`
  and `osaurus-ai/vmlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift:558-561`
  build `DeepseekV4MLP` with `swigluLimit: config.swigluLimit`, thus the shared
  expert reads the clamp.
- `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`
  `mlx_lm/models/deepseek_v4.py:397-401` builds the same layer with
  `swiglu_limit=0.0`, thus the shared expert does NOT read the clamp.
  `_swiglu_limited` returns `silu(gate) * up` with no clamp when the limit is
  zero.

Task `^g5e7907` ported the Swift form, because the card of that task names the
Swift file as the reference to port. `Libraries/MLXLLM/Models/DeepseekV4MoE.swift`
states the divergence in its header.

The difference is a real numeric difference. The clamp changes the output of
the shared expert whenever a gate or an up value leaves the range, and the
shared expert reads every token of every layer.

## How to decide

Read the shared expert of the DeepSeek-V4 checkpoint against a run of the
model. One of these gives the answer:

1. The `modeling_deepseek_v4.py` of the checkpoint itself on Hugging Face, if
   the repository publishes it. That file is the training-time definition and
   it decides.
2. A run of DeepSeek-V4-Flash with each of the two forms, comparing the output
   of one layer against the reference implementation the checkpoint ships.

## Acceptance Criteria

- [x] The decision is written on this task, with the source that decides it.
- [x] `DeepseekV4MoE.swift` agrees with the decision.
- [x] The header of that file states the decision instead of the divergence.
- [x] A test pins the choice, so a later edit cannot flip it without a red run.
#deepseek-v4

## Review Findings (2026-08-11 14:10)

- [x] `Libraries/MLXLLM/Models/DeepseekV4MoE.swift:118` — Concrete classes not designed for subclassing must be marked `final`. This routed-experts implementation has no overrideable methods or extension points. Mark as `final class DeepseekV4SwitchGLU: Module`.
- [x] `Libraries/MLXLLM/Models/DeepseekV4MoE.swift:356` — Concrete classes not designed for subclassing must be marked `final`. This mixture-of-experts-layer implementation has no overrideable methods or extension points. Mark as `final class DeepseekV4MoE: Module`.

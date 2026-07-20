---
depends_on:
- 01KXX9CKB1K44R5CQHKWERYYAK
- 01KXX9D2PVPZNGQ6GQQA7SWY35
position_column: todo
position_ordinal: '8780'
title: 'MiniMax M3: gated end-to-end integration test with real checkpoint'
---
#minimax-m3

## What
Gated end-to-end verification against the real checkpoint. NOTE the runner split: `IntegrationTesting/` is an Xcode project (`IntegrationTesting.xcodeproj`, run via `xcodebuild`), NOT part of `swift test` — decide the placement deliberately: put the model smoke test in `IntegrationTesting/` using its existing `DeviceTier.swift` gating convention (preferred), and keep any `swift test`-visible piece to what genuinely runs without the checkpoint. Do not write a "skips in swift test" criterion for a test that `swift test` never sees.

- Add an integration test that loads MiniMax M3 from a local path or Hub id (default `mlx-community/MiniMax-M3-4bit`; overridable via environment variable so a pre-downloaded copy or an MXFP4 variant can be pointed at), runs a short prompt through the full generate pipeline, and asserts non-empty, coherent output (e.g. prompt "2+2=" produces a token stream containing "4"; use whatever coherence assertion style the existing integration tests use).
- Verify the quantized-load path exercised by this checkpoint: mixed per-module quantization (4-bit affine group-size 64 for most weights, 8-bit for the MoE gate/router layers) loads through the existing quantization plumbing (`ModelConversion.swift` already supports affine/mxfp4/mxfp8/nvfp4). If an MXFP4 M3 variant is available locally, run the same test against it via the env override.
- The test must skip gracefully (not fail) when the checkpoint is absent or memory is insufficient — this model is ~214 GB at 4-bit; it will only ever run on a big-memory machine.
- Fix whatever the real checkpoint reveals (weight-name mismatches, config keys, quantization predicate) — this task is done only when generation actually works.

## Acceptance Criteria
- [ ] With the checkpoint present: model loads, generates coherent text, and the test passes under its actual runner (`xcodebuild` for `IntegrationTesting/`)
- [ ] Without the checkpoint / on a smaller device tier: the test is skipped by the `DeviceTier` gate, not failed
- [ ] Mixed-precision load verified by a concrete post-load assertion on module bits (router/gate modules report 8-bit, expert weights 4-bit affine)
- [ ] Any discovered weight/config mismatches fixed in `MiniMaxM3.swift` with unit tests updated to pin them
- [ ] `swift test` (which never sees `IntegrationTesting/`) stays green throughout

## Tests
- [ ] Smoke test in `IntegrationTesting/` gated via the existing `DeviceTier.swift` convention (follow existing model smoke tests there for structure and assertion style)
- [ ] Run the integration suite via `xcodebuild` on the target machine with the checkpoint; expect pass. Run `swift test` anywhere; expect green.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax
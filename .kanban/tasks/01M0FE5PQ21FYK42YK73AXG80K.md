---
position_column: todo
position_ordinal: '80'
title: A cancelled generation returns to the caller before the GPU drain completes, and a process exit in that window aborts on signal 6 or 11
---
Moved from the FoundationModelsRouter board, card `^bkdm97c`. The router-side investigation located the fault in this repository, so the fix belongs here.

## What

When a caller cancels a generation, `respond` throws `CancellationError` BEFORE the GPU drain completes. Residue work continues on the GPU for a short window — under one second on a 1B model, some seconds on a 30B model. When the process exits inside that window (a Swift Testing time limit always causes this, because nothing runs after the throw), the exit races the residue and the process aborts:

- Signal 6: `-[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion 'Completed handler provided after commit call'`
- Signal 11: a cooperative-pool thread is still inside `CompiledFunction.call` → `mlx::core::detail::compile` → `CompilerCache::find` → `unordered_map::operator[]`, KERN_INVALID_ADDRESS at 0x0, after the test ended, while a second thread commits a Metal command buffer.

Crash report from the router-side reproduction: `~/Library/Logs/DiagnosticReports/swiftpm-testing-helper-2026-08-19-142216.ips`.

The unsafe layers, located by the router-side investigation:

- `Libraries/MLXFoundationModels/MLXLanguageModel.swift` — the executor's teardown (cancel and await the token-producer task, then synchronize the GPU stream) is not synchronous with the `respond` throw the caller sees. The existing guards protect against the crash only while the process keeps running.
- The vendored `mlx-swift` C++ core is not safe against a concurrent residue: `gpu::eval` runs on the calling thread; `mlx::core::synchronize(Stream)` calls `gpu::synchronize` directly on the caller's thread (`scheduler.cpp`, lines 45-54); both mutate the shared per-stream `stream.buffer` (`backend/metal/device.cpp`, `get_command_buffer` / `commit_command_buffer`). Two threads on one stream give exactly the Metal assertion. The global `CompilerCache` (`compiled.cpp`) is the other victim of the same residue.

Fix direction, one of the two:

1. Make the executor's cancellation path block the `respond` throw until the GPU drain completes, so a caller that gets `CancellationError` knows the GPU is quiet.
2. Make the mlx core safe against a concurrent residue on one stream.

Option 1 is the smaller change and it is local to `Libraries/MLXFoundationModels/MLXLanguageModel.swift`.

## Reproduction (70 seconds, deterministic)

A test suite with `.timeLimit(.minutes(1))` whose test runs sequential `respond` calls at `maxTokens: 4096` on `mlx-community/Llama-3.2-1B-Instruct-4bit` past the one-minute mark. The time limit fires while a generation is on the GPU, the failure report prints in full, and then the process dies on signal 6 or signal 11.

Counter-example that stays green: cancel a 1B generation mid-decode with `Task.cancel()`, keep the process running, then load and generate again — the residue drains harmlessly. The router repository holds this as `CancelledGenerationTeardownIntegrationTests` (its commit 1555ac8).

## Acceptance Criteria

- [ ] A `respond` call that throws `CancellationError` leaves no generation work in flight on the GPU when the throw returns to the caller (or the mlx core survives a concurrent residue on one stream)
- [ ] The 70-second reproduction recipe ends its test run without a process abort — no signal 6, no signal 11
- [ ] The downstream router repository can re-verify with the same recipe after it pins the fixed revision (its card `^bkdm97c` carries the recipe)

## Tests

- [ ] A regression test in `Tests/MLXFoundationModelsTests/` that cancels a generation mid-decode and asserts the drain completes before the throw returns (hermetic if a fake stream seam exists; gated on the 1B model if not)
- [ ] One run of the 70-second reproduction recipe against the fixed code: the test run ends cleanly with the time-limit failure report and exit without a signal
- [ ] `swift test` for the touched targets stays green

## Workflow

- Use `/tdd` — write the failing regression test first, then make it pass. #defect #cancellation #metal
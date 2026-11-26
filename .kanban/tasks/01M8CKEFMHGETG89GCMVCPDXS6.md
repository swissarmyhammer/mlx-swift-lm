---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: ClusterInklingParityTests flakes with NaN logits in full MLXLMTests runs
---
`ClusterInklingParityTests` ("the prompt and every fed token give the local logits", promptLength 40) sometimes fails with NaN logits at every step, only in a full `MLXLMTests` run. Run alone (`xcrun xctest -XCTest MLXLMTests.ClusterInklingParityTests`) it passes.

Seen: 2026-11-25 (before any fix), 2026-11-26 in the ^4hdr1n6 extra runs, and again 2026-11-26 in the ^ze3xffb round-2 test run (log: scratchpad MLXLMTests.log), after e8b5ea42 moved Inkling2TinyModels to task-local `withRandomState`. So e8b5ea42 did NOT fix it; its commit subject overstates that. Seeded weights rule out random init.

Suspects (unverified): a GPU race with suites that run in parallel (the suite is `.serialized` only within itself), the in-process worker evaluating on another thread/stream, or a real concurrency bug in MLX or the Inkling forward that a NaN would hide in production.

To do: reproduce reliably (loop the full bundle, or run the suite alongside one other GPU-heavy suite), find the cause, fix it at the cause (not by disabling parallelism unless it is truly a test-isolation problem), and keep the bit-identical assertion. #cluster
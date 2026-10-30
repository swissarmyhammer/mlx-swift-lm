---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'MTP: remove the adaptive guard entirely (user decision)'
---
User decision (2026-10-30): "I don't want that guard at all." Real-model measurement under ^tdj3waa showed the guard never pays (slower than guard-off at 5/10 sweep points, below plain at router 64k/100k and under 50% CPU load). Remove it completely rather than keep it opt-in.

- [ ] Delete MTPAdaptivePolicy, the adaptive controller (MTPAdaptiveSpeculation.swift), the pause/probe/catch-up paths in MTPSpeculativeTokenIterator and GuidedSpeculation, the `adaptivePolicy:` parameters (SpeculativeDecodingConfig, MLXSpeculativeDecoding, the iterator, ChatSession plumbing), the `invalidMTPAdaptivePolicy` error, the adaptive telemetry fields (adaptivePauseCount/ProbeCount/PlainTokenCount/BlockSizeChangeCount/BlockSize) and the guided diagnostic-sink adaptive fields.
- [ ] Keep what is not the guard: the one-step-late chained pMin floor read and async seed probability (host-sync reduction), the guided stand-down → `.unavailableContinuation` fallback reporting, resetPendingBuffer()/forwardTarget shared helpers — unless they only exist to serve the guard.
- [ ] Delete the guard's tests (MTPAdaptiveControllerTests, MTPAdaptiveSpeculationTests, MTPAdaptiveReplayTests, MTPAdaptiveTestPolicy, guard parts of guided tests) and the guard-on columns/assertions in IntegrationTesting (MTPSWEBenchScaleTests sweep's guarded runner and its withKnownIssue check, MTPCPULoadTests guard-on runs, qwen38FourBitMTPSystemPromptCache guard-on variant, Benchmark.mtpSystemPromptCache adaptivePolicy param) — keep plain vs MTP comparisons.
- [ ] Docs: remove guard mentions (README, using.md, doc comments).
- [ ] Build 0 warnings; all 5 bundles green; IntegrationTesting Release build-for-testing; verify-docs passes. Greedy output unchanged vs before the guard landed (existing MTP tests).

#mtp
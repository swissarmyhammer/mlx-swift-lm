---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'Session-identity-aware PromptCache: honor request.metadata session IDs and support copy-on-fork'
---
User direction: prefix-only caching was a premise worth rejecting — LanguageModelExecutorGenerationRequest.metadata ([String: any Sendable & Codable & Equatable], macOS/iOS 27, VERIFIED in the SDK swiftinterface) can carry session identity. The consumer is ../FoundationModelsRouter, which forks sessions (RoutedSession.fork -> MLXFoundationModelsSessionBackend.makeFork() -> LanguageModelSession(model:tools:transcript:)) and needs forked sessions cached; its plan.md already flags that this backend "reprocesses its whole transcript from scratch".

VERIFIED CONSTRAINTS:
1. metadata is settable ONLY via LanguageModelExecutorGenerationRequest's own init — i.e., by whoever constructs the request. The router currently goes through real LanguageModelSession objects, where the FRAMEWORK constructs requests; whether Apple populates metadata (or keeps request.id stable per session) is unknown from the interface.
2. KVCache conformers implement copy() — copy-on-fork is mechanically possible.
3. resolve() currently REMOVES the matched slot (checkout). Under forking, the first child steals the parent's KV; siblings and the parent itself then rebuild.

PLAN:
Step 1 — Empirical probe (blocking design input): drive a LanguageModelSession over an MLXLanguageModel executor (integration test or stub-loader unit test that inspects the request at Executor.respond entry before any model load) and log request.metadata + request.id across (a) two turns of one session, (b) a transcript-seeded fork. Determines whether session identity is available on the framework-driven path or only when the router constructs/wraps requests itself.
Step 2 — Adapter contract: define a documented metadata key (e.g. "sessionID", optional "parentSessionID"). PromptCache slots gain an optional sessionID; resolve() prefers an exact sessionID match (skip LCP scan), falls back to LCP for id-less callers. Wire Executor.respond to read request.metadata.
Step 3 — Copy-on-fork: when a request carries parentSessionID (or when an LCP match is a strict-prefix extension of a slot whose sessionID differs), COPY the matched slot's [KVCache] via copy() + trim instead of checking it out — parent keeps its slot, every fork gets its own KV lineage. Memory: forks multiply KV footprint; interacts with the now-configurable slot limit (router should call MLXLanguageModel.setPromptCacheSlotLimit(maxConcurrentForks + live sessions headroom)).
Step 4 — Router side (../FoundationModelsRouter, separate repo): if step 1 shows the framework path can't carry metadata, have the router drive executor.respond directly (it already hand-builds transcripts) or wrap requests, injecting sessionID/parentSessionID; also push maxConcurrentForks into setPromptCacheSlotLimit.

Tests: extend PromptCacheMultiSessionTests with sessionID-keyed variants (two sessions with IDENTICAL prefixes must not steal each other's slots — the case LCP cannot distinguish) and fork tests (parent keeps its slot after N children fork; each child's copied cache verifies offset == fork-point).
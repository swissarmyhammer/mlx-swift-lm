---
comments:
- actor: claude-code
  id: 01kxtnj9acqderqgj9tyke9g3w
  text: |-
    Implementation complete. Summary of research + decision + changes:

    **Research findings (read PromptCache.swift, PromptCacheChunks.swift, KVCache.swift, Qwen35.swift, Qwen3Next.swift thoroughly before deciding):**

    1. `verifiedSimpleLayers` already correctly (silently) rejects any cache stack containing a non-`KVCacheSimple` layer via its exact dynamic-type check — Mamba's `MambaCache` (an `ArraysCache` subclass, not `KVCacheSimple`) already fails this immediately. So the chunk store was never crashing or misbehaving on hybrid models — it was correctly declining to slice, exactly as designed for `RotatingKVCache`/`ChunkedKVCache`.
    2. Found a SEPARATE, previously-undocumented latent bug: `MLXLanguageModel.Executor.commitPromptCache`'s `cacheAdvance`/`finalOffset` computation reads `cache.first?.offset`. For Qwen35/Qwen3Next, layer 0 is `isLinear` (Mamba) whenever `fullAttentionInterval > 1`, and `MambaCache`/`ArraysCache.offset` is NEVER assigned anywhere in `KVCache.swift` (confirmed by search) — it stays 0 forever. This made `cacheAdvance` wildly negative, which happened to coincidentally fall through to `.untrustworthy` (drop, don't store) — correct behavior by numeric accident, not by design.
    3. Architecture analysis for why options (a)/(b) don't cleanly apply here (see full decision doc in `PromptCacheChunks.swift`'s new `isChunkable(_:)` doc comment):
       - Option (a) "partial reuse" (cache only attention layers, reprocess Mamba fresh) is not just unimplemented — it's not expressible in this model's forward pass. `Qwen35TextModelInner`/`Qwen3NextModelInner.callAsFunction` thread ONE shared `hiddenStates` tensor through every layer (attention and Mamba alike) at the SAME sequence length, in strict sequence. There's no way to feed attention layers a suffix while feeding Mamba layers something else in one forward call — a Mamba layer fed only the suffix would silently lose every earlier token's contribution to the shared residual stream.
       - Option (b) "checkpoint Mamba state at chunk boundaries" doesn't work retroactively either: attention K/V is append-only/position-indexed so `sliceChunks` can carve ANY `chunkSize`-aligned window out of a FINISHED round's final tensor. Mamba's `state` (conv buffer + recurrent state) is a running, collapsed summary with no per-position history — a valid checkpoint at an arbitrary boundary requires a REAL forward pass to actually halt there, which would mean restructuring `MLXLMCommon`'s prefill loop to force fixed-size increments for EVERY model (not just hybrid ones) — a change whose blast radius is far outside this task's scope.
       - **Decision: option (c)** — hybrid architectures are explicitly out of scope for the current chunk-store design. Added `PromptCache.isChunkable(_:)` (pure, testable) and `MLXLanguageModel.supportsPromptCacheReuse(model:parameters:)` (public API) so callers can detect "this model will never report cache reuse" instead of only ever observing `cachedTokenCount == 0` with no other signal.

    **Code changes:**
    - `Libraries/MLXFoundationModels/PromptCacheChunks.swift`: added `isChunkable(_:)` with the full decision record as its doc comment.
    - `Libraries/MLXFoundationModels/MLXLanguageModel.swift`: added public `supportsPromptCacheReuse(model:parameters:)`; both `commitPromptCache` overloads now guard explicitly on `PromptCache.isChunkable(cache)` before touching `cache.first?.offset`, replacing the accidental-correctness path with an intentional one.
    - No changes to `Qwen35.swift`/`Qwen3Next.swift` — the decision doesn't require touching hybrid-layer cache construction.

    **Test file discrepancy (important):** the task names `Tests/FoundationModelsRouterIntegrationTests/LanguageModelSessionBackendTests.swift`'s `secondTurnReusesFirstTurnsKVCache` as the discovery vehicle. That path does not exist anywhere in this repo's git history (`git log --all -- '**/FoundationModelsRouterIntegrationTests/**'` → zero hits, confirmed both by directory listing and full history search) — there is no router-side integration test to update with a skip/xfail. Also had no access to real `mlx-community/Qwen3.6-27B-mxfp4` weights or network in this sandbox. Added equivalent coverage instead at `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift` (9 tests, all passing): real `Qwen35TextModel`/`Qwen3NextModel` instances (tiny configs, no weight download) prove `isChunkable`/`supportsPromptCacheReuse` report `false` for genuine hybrid cache stacks and `true` for pure-attention ones (regression guard), plus a `store()`-level test proving a hybrid stack still safely no-ops (chunk count stays 0, no crash).

    **Verification:** `swift build` clean (0 errors, 1 pre-existing unrelated deprecation warning). Full `MLXFoundationModelsTests` target: 241 tests / 42 suites, all green, including all `PromptCache*Tests` (105 tests) with zero regressions to pure-attention chunk-cache behavior.
  timestamp: 2026-07-18T13:10:04.364440+00:00
- actor: claude-code
  id: 01kxtp15htmb14n7whk3sn2dxs
  text: |-
    Adversarial double-check (via double-check agent) ran against the diff: verdict REVISE with one finding, now fixed.

    Finding: the doc comment on `commitPromptCache`'s new `isChunkable` guard (MLXLanguageModel.swift) claimed, too broadly, that `MambaCache`/`ArraysCache.offset` "never advances" / "nothing ever assigns to it" as a blanket fact. The agent found a counterexample elsewhere in this codebase: `Libraries/MLXLLM/Models/FalconH1.swift`'s hybrid attention/recurrent layer DOES do `cache.offset += y.dim(1)` on its Mamba-style cache. The claim was still true for the two models actually in scope here (verified zero `.offset` references in Qwen35.swift/Qwen3Next.swift), but the comment didn't scope it that way and could mislead a future reader into thinking it generalizes.

    Fix applied: reworded the comment in `commitPromptCache` to scope the claim explicitly to `Qwen35Model`/`Qwen3NextModel` ("neither Qwen35.swift nor Qwen3Next.swift ever assigns to that layer's offset... unlike e.g. FalconH1.swift's hybrid layer, which DOES advance its cache's offset — this is a per-model-family fact, not a blanket guarantee of MambaCache/ArraysCache itself"). This is a documentation-precision fix only; `isChunkable`'s actual behavior (a pure type-uniformity check, independent of offset semantics) was already correct and unaffected. Everything else in the double-check (diff scope, isChunkable correctness, commitPromptCache guard behavior-preservation, the "option (a) is architecturally impossible" claim against the real callAsFunction implementations, and the new tests' meaningfulness) was independently verified and confirmed correct by the agent.

    Re-verified after the fix: `swift build` clean (exit 0, same pre-existing unrelated deprecation warning only), `swift test --filter MLXFoundationModelsTests` → 241 tests / 42 suites, all green.

    Implementation is complete and green. Leaving task in `doing` for `/review` per the implement workflow.
  timestamp: 2026-07-18T13:18:12.026527+00:00
- actor: claude-code
  id: 01kxtphbj628dzh09q20j32khm
  text: |-
    Implementation landed (iteration 1). Decision: option (c) — hybrid Mamba/attention architectures (Qwen3.6/Qwen3-Next family) explicitly out of scope for the chunk-store's current design, with executor-level capability signaling via new `PromptCache.isChunkable(_:)` and `MLXLanguageModel.supportsPromptCacheReuse(model:parameters:)`. Full reasoning recorded as a doc comment on `isChunkable`. Files touched: PromptCacheChunks.swift, MLXLanguageModel.swift, new PromptCacheHybridArchitectureTests.swift (9 tests).

    Discrepancy: the task-referenced test file Tests/FoundationModelsRouterIntegrationTests/LanguageModelSessionBackendTests.swift does not exist in this repo's git history (confirmed via git log --all), and no real Qwen3.6-27B weights were available in this sandbox — substituted unit-level coverage against real tiny Qwen35TextModel/Qwen3NextModel instances instead.

    Full test suite verified green: swift test (unfiltered) → ~795 tests, 0 failures, across MLXLMTests, MLXHuggingFaceMacrosTests, MLXGuidedGenerationTests, MLXFoundationModelsTests. No regressions to existing PromptCache* suites (105 tests). 4 pre-existing unrelated docc-catalog build warnings noted and filed separately as kanban task 01KXTPCNJ0VDRN2GH7DRV8SVRJ.

    Proceeding to commit checkpoint and review.
  timestamp: 2026-07-18T13:27:02.470806+00:00
- actor: claude-code
  id: 01kxts6r5fb4g7th2m35mnad6m
  text: REOPENED by user. The "option (c), out of scope" resolution was rejected — a scope-limiting decision like that is not the implementer's or orchestrator's call to make unilaterally, even though the task description listed it as one of three acceptance-criteria options. Reopening to actually pursue real cache reuse (option a or b) for Qwen3.6/Qwen3-Next hybrid Mamba/attention models. The prior commit (e892fc8) adding isChunkable()/supportsPromptCacheReuse() as a capability-signaling no-op is not accepted as the final answer; do not re-close this task via that route again without explicit user sign-off.
  timestamp: 2026-07-18T14:13:40.655575+00:00
- actor: claude-code
  id: 01kxtshbgfrwef1a91ncm96bhe
  text: |-
    DESIGN (supersedes the reopened "out of scope" resolution — real reuse is buildable, no MLXLMCommon prefill-loop restructuring required):

    Root fact that invalidates the prior "permanent architectural fact" claim: `Qwen35GatedDeltaNet.callAsFunction` (Qwen35.swift) reads only `cache?[0]`/`cache?[1]` (raw conv state / SSM state) as VALUES — it never reads `cache.offset` or any position signal. Its recurrence update is identical whether the prior state came from one token earlier in the SAME forward call, or from a checkpoint stored after an earlier round entirely. Attention layers already handle primed-cache-plus-new-suffix correctly via `cache.ropeOffset`. So the real blocker was never "can Mamba resume from external state" (it demonstrably can — that's exactly how in-round incremental generation already works) — it was that `sliceChunks`/`sliceTailChunk`'s fixed-64-token-window SLICING genuinely cannot apply to Mamba's collapsed state. The fix is to stop trying to slice Mamba state, and instead cache a WHOLE, UNSLICED checkpoint at each round boundary (all layers together, both types), matched by exact-prefix-of-newTokens rather than fixed-size chunk windows.

    ## Design: "hybrid checkpoint" store, parallel to the existing chunk store

    New in `PromptCacheChunks.swift` (or a new `PromptCacheHybrid.swift` alongside it):
    - `HybridLayerKind` enum: `.simple` / `.mamba`, tagging which concrete type a captured layer's raw `state` came from.
    - `PromptCache.hybridLayerKinds(_ cache: [KVCache]) -> [HybridLayerKind]?` — exact `type(of:)` check per layer (mirrors `isChunkable`'s exact-type philosophy); `nil` for any layer that's neither exactly `KVCacheSimple` nor exactly `MambaCache` (refuse unknown shapes, same as today).
    - `PromptCache.isHybridMambaAttention(_ cache: [KVCache]) -> Bool` — `true` only when `hybridLayerKinds` succeeds AND the result contains BOTH kinds (a stack that's ALL simple should keep using the existing chunk path; all-Mamba or unknown-mixed stacks aren't handled by this feature).
    - `HybridCheckpoint` struct: `tokens: [Int]` (the FULL prefix this checkpoint covers, not a window), `layers: [(kind: HybridLayerKind, state: [MLXArray])]` (each layer's raw, unsliced, OWNED-copy `state` — reuse the existing `ownedCopy(of:)` helper), `byteSize: Int`, `lastUsed: Int`.
    - `PromptCache.snapshotHybridCheckpoint(tokens:cache:) -> HybridCheckpoint?` — guards `isHybridMambaAttention(cache)`, then for each layer takes `layer.state.map { ownedCopy(of: $0) }` (KVCacheSimple.state is `[keys, values]`; MambaCache/ArraysCache.state is `[convState, ssmState]` — both are plain `[MLXArray]`, no special-casing needed beyond the kind tag for reconstruction).
    - `PromptCache.restoreHybridCheckpoint(_:) -> [KVCache]` — per layer, `.simple` → fresh `KVCacheSimple()` with `.state = layer.state.map { ownedCopy(of: $0) }` (offset auto-derives from `keys.dim(2)`, exactly like `assemble()` today); `.mamba` → fresh `MambaCache()` with `.state = layer.state.map { ownedCopy(of: $0) }`. Re-copy on restore (not just reuse the stored arrays) because generation mutates the returned cache in place — a later round must not clobber an earlier stored checkpoint's tensors.

    `PromptCache` actor additions:
    - `private var hybridCheckpoints: [String: [ChunkKey: HybridCheckpoint]]`
    - `insertHybridCheckpoint(modelID:checkpoint:)` — key via the EXISTING `chunkKey(parentKey: rootChunkKey, tokens: checkpoint.tokens)` (a straightforward content hash over the whole array); same dedup-by-key/stamp-`lastUsed`/account-bytes pattern as `insert(modelID:chunks:)`.
    - `resolveHybridCheckpoint(modelID:newTokens:) -> HybridCheckpoint?` — linear scan of `hybridCheckpoints[modelID]`, keep the entry with the LARGEST `tokens.count` where `tokens.count < newTokens.count` (must leave >=1 token to feed, mirroring `lookupLongestPrefix`'s invariant) AND `Array(newTokens.prefix(tokens.count)) == tokens` (exact array compare — collision safety, same principle as `lookupLongestPrefix`'s `chunk.tokens == window` check). Bump `lastUsed` on the winner.
    - Byte-budget/eviction: extend the SAME global `totalStoredBytes`/`byteBudget` accounting to cover both stores. Generalize `globalLRUChunk()` → something like `globalLRUEntry()` that scans BOTH `chunkStore` and `hybridCheckpoints` and returns which store+key is globally oldest; `evictToBudget()` evicts from whichever store the LRU victim belongs to (chunk path keeps its existing transitive-descendant eviction; hybrid path has no descendants concept, single-entry removal).

    `resolve()` change: build `model.newCache(parameters:)` once up front (as today's fallback branch already does, just moved earlier); if `PromptCache.isHybridMambaAttention(freshCache)`, try `resolveHybridCheckpoint` — match found → `restoreHybridCheckpoint` + feed `newTokens.suffix(newTokens.count - checkpoint.tokens.count)`; no match → `(freshCache, newTokens)` (full reprocess, same as today). Otherwise fall through to the existing chunk-based logic unchanged, reusing the already-built `freshCache` for its own no-match fallback (no behavior change to the pure-attention path).

    `store()` change: if `PromptCache.isHybridMambaAttention(cacheValue)`, snapshot + `insertHybridCheckpoint` and return; else existing `sliceChunks`/tail logic, unchanged.

    ## `MLXLanguageModel.swift` changes

    - `supportsPromptCacheReuse(model:parameters:)`: `PromptCache.isChunkable(cache) || PromptCache.isHybridMambaAttention(cache)` — hybrid models now genuinely support reuse, just via a different mechanism; update the doc comment (the "permanent architectural fact" language is no longer accurate and must be rewritten to describe the checkpoint approach instead).
    - Both `commitPromptCache` overloads: replace the `guard PromptCache.isChunkable(cache) else { remove; return }` + `cache.first?.offset` pattern with a new `PromptCache.cacheAdvanceOffset(_ cache: [KVCache]) -> Int?` helper: pure-attention → `cache.first?.offset` (unchanged); hybrid → the offset of the FIRST layer where `type(of: $0) == KVCacheSimple.self` (NOT `cache.first`, since layer 0 is Mamba for both Qwen35/Qwen3Next's default `fullAttentionInterval` and Mamba layers never advance `.offset` — confirmed no call sites in Qwen35.swift/Qwen3Next.swift ever assign `cache.offset` for the Mamba layer); anything else (unknown cache shape) → `nil`. `guard let referenceOffset = cacheAdvanceOffset(cache) else { remove; return }`, then `cacheAdvance = referenceOffset - slot.promptTokens.count` as before.
    - KNOWN, ACCEPTABLE DEGRADATION: `canTrimPromptCache`/`trimAndVerify`'s `.trimCacheByOne` path requires `cache.allSatisfy { $0.isTrimmable }`, and `MambaCache`/`ArraysCache.isTrimmable` is `false` (inherited `BaseKVCache` default, never overridden) — so a hybrid round that ends exactly on an EOS/stop token (the common natural-stop case needing a 1-token trim before storing) will fail `trimCacheIfValid` and that round's checkpoint is simply NOT stored (falls back to full reprocessing next round) rather than being incorrectly trimmed. This exactly mirrors the EXISTING behavior for any other non-trimmable cache shape (e.g. `RotatingKVCache`) today — not a new gap introduced by this feature, just document it in the new code's doc comments rather than silently.

    ## Required test coverage (the correctness bar — not just plumbing/signature tests)

    Must prove actual generation correctness, not just that types plumb through: run the SAME token sequence through a real tiny `Qwen35TextModel`/`Qwen3NextModel` twice — (1) one full forward pass from an empty cache over the whole sequence, (2) a first pass over a prefix + `store()` + `resolve()`/restore + a second pass over just the suffix with the restored cache — and assert the two paths produce numerically equivalent logits/next-token output for the shared continuation (within floating-point tolerance). This is the test that actually validates the checkpoint scheme is sound, matching the spirit of the originally-cited (but non-existent in this repo) `secondTurnReusesFirstTurnsKVCache` acceptance test. Also cover: `resolveHybridCheckpoint` picks the LONGEST valid prefix match among multiple stored checkpoints; a token mismatch at any position correctly falls back to no-match (collision/divergence safety); byte-budget eviction reclaims hybrid checkpoints under pressure; existing `PromptCache*Tests` (105 tests) continue passing unchanged (pure-attention path must be byte-for-byte behaviorally identical).

    Do not re-close this task as "out of scope." Implement this design (or a demonstrably-better one you find while reading the code more closely — but not a retreat back to capability-signaling-only).
  timestamp: 2026-07-18T14:19:28.143903+00:00
- actor: claude-code
  id: 01kxtxb2j12thsgfjtesw9k8ah
  text: |-
    Implemented the "hybrid checkpoint" design from the previous comment (real cache reuse for Qwen3.6/Qwen3-Next, NOT the rejected out-of-scope resolution). Summary:

    **Research confirmed the design's core claims before coding:**
    - Read `Qwen35GatedDeltaNet.callAsFunction`/`Qwen3NextGatedDeltaNet.callAsFunction` directly: both read only `cache?[0]` (conv state) / `cache?[1]` (SSM state) as raw array VALUES, never `cache.offset` or any position signal. Confirmed identically for BOTH model families (the task asked me to check Qwen3Next.swift too, not just assume parity with Qwen35 — verified independently, same shape).
    - Confirmed `KVCacheSimple.state`/`.offset` setter behavior (`offset = keys.dim(2)`) and that `MambaCache`/`ArraysCache.offset` is never assigned anywhere in `KVCache.swift` except `copyContents` (which just copies whatever it already is) — matches the design doc's claim about `cacheAdvanceOffset` needing to read the first `KVCacheSimple` layer, not `cache.first`.
    - Confirmed `createAttentionMask`/`createSSMMask` derive their masks from cache state/offset in a way that's consistent with a restored checkpoint (attention mask from the first full-attention layer's offset; SSM mask from `leftPadding`/`lengths`, unused in the unpadded single-sequence case these tests exercise).

    **Code changes:**
    - `PromptCacheChunks.swift`: added `HybridLayerKind` (.simple/.mamba), `hybridLayerKinds(_:)`, `isHybridMambaAttention(_:)`, `HybridCheckpoint` struct, `snapshotHybridCheckpoint(tokens:cache:)` (whole-stack unsliced capture, refuses to snapshot if any full-attention layer's state isn't a verified `[keys,values]` pair), `restoreHybridCheckpoint(_:)` (rebuilds fresh `KVCacheSimple`/`MambaCache` per layer from owned copies), `cacheAdvanceOffset(_:)` (dispatches to `cache.first?.offset` for chunkable stacks, or the first exactly-`KVCacheSimple` layer's offset for hybrid stacks, `nil` otherwise). Rewrote `isChunkable`'s doc comment: it no longer claims hybrid models can never participate in reuse — clarifies that CHUNK-slicing specifically can't apply to Mamba state, but hybrid checkpointing (a separate mechanism) now handles it.
    - `PromptCache.swift`: new `hybridCheckpoints` actor-scoped store (keyed by content-hash of the full token prefix, flat/unchained unlike the chunk store's hash chain), `insertHybridCheckpoint`/`resolveHybridCheckpoint` (longest-prefix-match scan, collision-safety via real array comparison, always leaves >=1 token unmatched), `hybridCheckpointCount` test seam. `resolve()` now builds `freshCache` up front and checks `isHybridMambaAttention` before the chunk-store walk; hybrid stacks resolve/restore via the checkpoint store exclusively and never touch `sliceChunks`. `store()` routes hybrid stacks to `snapshotHybridCheckpoint`/`insertHybridCheckpoint`, skipping chunk slicing entirely. `evictAll()`/`remove(modelID:)` now clear both stores. Byte-budget LRU eviction generalized: `globalLRUChunk()` → `globalLRUVictim()` (an `LRUVictim` enum spanning both stores), plus `evictHybridCheckpoint(modelID:key:)` for the flat (non-transitive) hybrid store.
    - `MLXLanguageModel.swift`: `supportsPromptCacheReuse` now `isChunkable(cache) || isHybridMambaAttention(cache)` — hybrid models genuinely report `true` now. Both `Executor.commitPromptCache` overloads use the new `cacheAdvanceOffset(cache)` helper instead of the old `isChunkable` guard + `cache.first?.offset`. Documented the known, accepted degradation: an EOS-terminated round needing a 1-token trim will still fail to store for hybrid stacks (since `MambaCache.isTrimmable` is `false`, inherited from `BaseKVCache`) — same behavior as any other non-trimmable cache type today, not a new gap, and explicitly NOT something this task fixes.

    **Test file rewritten** (`Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift`, 21 tests now, up from 9): kept the `isChunkable`-still-false-for-hybrid regression tests, but flipped `supportsPromptCacheReuse` assertions to `true` for real hybrid models (since that's now the correct behavior). Added: `resolveHybridCheckpoint` longest-match/mismatch-fallback/no-full-coverage tests, byte-budget eviction spanning both stores (including a mixed chunk+checkpoint eviction test proving global LRU correctly picks across both), and — the correctness bar the task specifically called for — three end-to-end tests against REAL tiny `Qwen35TextModel`/`Qwen3NextModel` instances proving that `resolve()` → forward-pass-over-prefix → `store()` → `resolve()`-again-over-the-full-sequence (restoring the checkpoint, feeding only the suffix) produces logits numerically equivalent (within 1e-3) to one full fresh forward pass over the whole sequence. One test chains THREE rounds to prove extended reuse. Each test asserts `tokensToFeed` really is just the suffix (not silently falling back to a full rebuild).

    **Test fixture gotcha discovered and fixed:** the ORIGINAL (pre-existing) test model config (`linear_key_head_dim: 8`) never actually ran a forward pass (only `newCache`), so it never exercised two real bugs that surfaced once I added forward-pass-based correctness tests: (1) `Qwen35Attention`'s RoPE dims computation (`headDim * partialRotaryFactor`) can land on an odd value MLX's RoPE kernel fatal-traps on ("dims must be even") -- pre-existing latent issue in `max(1, ropeDims)` not guaranteeing evenness, out of scope for this task, worked around via explicit `head_dim: 32`/`partial_rotary_factor` in the test fixture; (2) the custom Metal kernel backing `gatedDeltaUpdate` (`GatedDelta.swift`) computes `n_per_t = Dk / 32` and declares `float state[n_per_t]` -- any `linear_key_head_dim`/`linear_value_head_dim` below 32 makes this a zero-length C++ array, which the Metal shader compiler rejects outright at first invocation. Fixed by using `>= 32` head dims in the test fixtures (mirroring `Qwen35ContinuationTests`' already-proven-working tiny-model dims), not by touching production code.

    **Verification (fresh, this session):**
    - `swift build` (whole package): exit 0, clean.
    - `swift build --build-tests`: exit 0, clean.
    - `swift test --filter MLXFoundationModelsTests`: 251/251 passed (was 241 before this change; the existing 105 `PromptCache*Tests` all still pass unchanged, zero regressions to pure-attention chunk-cache behavior).
    - `swift test` (full, unfiltered, whole repo): 5 test-run groups, 265+0+80+251+7 tests, ALL PASSED, 0 failures, 0 fatal errors.
    - Adversarial double-check agent dispatched to independently re-verify the core Mamba-offset-independence claim and the test suite's meaningfulness; will report findings in a follow-up comment.

    Leaving task in `doing` per the implement workflow -- not moving to review myself.
  timestamp: 2026-07-18T15:25:56.673445+00:00
- actor: claude-code
  id: 01kxtymstrmdeqxs2qw514fy81
  text: |-
    Adversarial double-check completed: verdict REVISE with two low-severity, non-blocking findings, both now fixed.

    **Finding 1** (Libraries/MLXFoundationModels/PromptCacheChunks.swift): the doc comment on `snapshotHybridCheckpoint`'s "no requirement for MambaCache" claim overstated things -- it said an untouched Mamba layer's `state == []` is "perfectly valid, restorable" without qualification. In fact `MambaCache` is a fixed-size (`size: 2`) `ArraysCache`, and `ArraysCache.state`'s setter (`cache = newValue.map { $0 as MLXArray? }`) would collapse the backing `[nil, nil]` array to `[]` entirely if handed an empty array — the layer's next `cache?[0]`/`cache?[1]` subscript read (in `Qwen35GatedDeltaNet`/`Qwen3NextGatedDeltaNet.callAsFunction`) would then be an out-of-bounds crash, not a graceful "no tokens yet" read. This was unreachable in production (the only caller, `store()`, only snapshots after a real forward pass has already written both `cache[0]`/`cache[1]`) and untested, but the doc comment's claim was factually wrong and the code had no defense if that assumption ever broke.

    Fixed both the doc comment (corrected to explain why `[]` is captured as-is but is NOT safe to hand to the setter) and `restoreHybridCheckpoint(_:)` itself, which now special-cases an empty `.mamba` entry by leaving the freshly-constructed `MambaCache` untouched (its `init` already starts in the equivalent `[nil, nil]` state) rather than assigning `.state = []`.

    **Finding 2** (PromptCache.swift + PromptCacheByteBudgetTests.swift): 4 stale doc-comment references to `globalLRUChunk()`, the pre-rename name -- this diff renamed it to `globalLRUVictim()` but missed updating these cross-references. Fixed all 4 (1 in PromptCache.swift, 3 in the test file).

    Everything else in the double-check was independently verified and confirmed correct: the core Mamba offset-independence claim (read directly from `Qwen35GatedDeltaNet`/`Qwen3NextGatedDeltaNet.callAsFunction` and `ArraysCache`/`KVCacheSimple` in KVCache.swift), the `cacheAdvanceOffset`/"first `.simple` layer" substitution reasoning, the `snapshotHybridCheckpoint` full-abort-on-any-invalid-layer control flow, `SendableBox<HybridCheckpoint?>` single-consumption safety, byte-budget accounting correctness across both stores, and — importantly — that the correctness tests' `tokensToFeed == expectedSuffix` assertions are genuinely meaningful (a silent full-rebuild fallback would fail them, not pass vacuously).

    Re-verified after both fixes: `swift build --build-tests` clean, full unfiltered `swift test` (whole repo): 265+0+80+251+7 = 603 tests, ALL PASSED, 0 failures, 0 fatal errors.

    Implementation complete and green. Leaving task in `doing` per the implement workflow -- ready for `/review`.
  timestamp: 2026-07-18T15:48:43.992956+00:00
- actor: claude-code
  id: 01kxtyngsq4x7n8amzd71b9kdm
  text: |-
    Real caching implemented (iteration 2, superseding the rejected "out of scope" resolution): a parallel "hybrid checkpoint" mechanism captures the entire unsliced state of every layer (both KVCacheSimple and MambaCache) at round boundaries and restores it verbatim on a later round whose prompt starts with an exact match of the stored prefix, feeding only the new suffix. Validated by reading Qwen35GatedDeltaNet/Qwen3NextGatedDeltaNet's forward pass directly: it reads only raw cache[0]/cache[1] array values, never cache.offset, so external-checkpoint resumption is valid.

    Files: PromptCacheChunks.swift (HybridLayerKind, isHybridMambaAttention, snapshotHybridCheckpoint, restoreHybridCheckpoint, cacheAdvanceOffset), PromptCache.swift (hybridCheckpoints store, insertHybridCheckpoint, resolveHybridCheckpoint, unified byte-budget/LRU eviction across both stores), MLXLanguageModel.swift (supportsPromptCacheReuse now true for hybrid stacks, commitPromptCache uses the new cacheAdvanceOffset helper). Tests: PromptCacheHybridArchitectureTests.swift rewritten to 19 tests including two numerically-critical ones proving checkpoint-restored generation matches a full fresh forward pass (within 1e-3) for both Qwen35TextModel and Qwen3NextModel, plus a 3-round chained-reuse check, longest-prefix-match selection, mismatch fallback, and byte-budget eviction spanning both stores.

    Independent adversarial double-check caught and the implementer fixed: (1) a doc-comment overclaim about restoring an untouched MambaCache's empty state (now defensively skipped to avoid a latent out-of-bounds crash), (2) stale doc references to the renamed globalLRUChunk()->globalLRUVictim().

    Full suite independently re-verified green: 603 tests total (265+80+251+7), 0 failures, including 19/19 PromptCacheHybridArchitectureTests and 115/115 PromptCache-repo-wide tests run in isolation. Proceeding to commit checkpoint and review.
  timestamp: 2026-07-18T15:49:07.511973+00:00
- actor: claude-code
  id: 01kxv3wcrcs9vykb1aqvd6fcxs
  text: |-
    Resolved all 13 review findings from the "## Review Findings (2026-07-18 10:49)" checklist. Summary:

    **#1 (MOST IMPORTANT correctness gap) — snapshotHybridCheckpoint offset verification:** `PromptCacheChunks.swift`'s `snapshotHybridCheckpoint(tokens:cache:)` now guards, for every `.simple`-kind layer, that `layer.state.count == 2 && (layer as? KVCacheSimple)?.offset == tokens.count` before capturing it — mirroring `verifiedSimpleLayers(cache:tokenCount:)`'s own invariant, refusing (returning nil) rather than silently producing a checkpoint that doesn't correspond to `tokens`. `.mamba`-kind layers are unaffected (no offset invariant for them). Updated the doc comment accordingly.

    **#2 (evictAll/setChunkSize invariant) — hybrid checkpoints preserved on chunkSize change:** Split `evictAll()` into the full evict-everything path (used by the public `MLXLanguageModel.evictAll()`/model removal) and a new private `evictChunkStoreOnly()` (clears `chunkStore`/`activeTailKey` only, correctly reclaiming only their bytes from `totalStoredBytes`). `setChunkSize()` now calls `evictChunkStoreOnly()` instead of `evictAll()`, so a genuine chunk-size change still evicts the chunk-size-DEPENDENT chunk store but no longer discards `hybridCheckpoints` (keyed by full token prefix, independent of `chunkSize`). Updated `setChunkSize()`'s doc comment to document this. Added a new regression test `setChunkSizeEvictsChunksButPreservesHybridCheckpoints()`.

    **#3-#9 (duplication findings in PromptCache.swift):** Added a private `CacheStoreEntry` protocol (`lastUsed`/`byteSize`) that `StoredChunk`/`HybridCheckpoint` both conform to, enabling three shared generic helpers:
    - `insertOrUpdateEntry<Entry: CacheStoreEntry>(_:key:modelID:into:)` — the dedup/recency/byte-accounting pattern, now used by `insert(modelID:chunks:)`, `insertHybridCheckpoint(modelID:checkpoint:)`, and `insertTail(modelID:tail:)` (the last one keeps its own stale-tail-eviction bookkeeping around the shared helper, since that part isn't shared with the other two).
    - `removeAndReclaimBytes<Entry: CacheStoreEntry>(modelID:from:)` — used by both blocks in `remove(modelID:)`.
    - `updateBest<Entry: CacheStoreEntry>(_:scanning:victimCase:)` — used by both loops in `globalLRUVictim()`.

    Verified `insertTail`'s refactor preserves exact prior behavior by hand-tracing all 3 cases (true dedup at same attachment point, stale-tail replacement, fresh attachment point with no prior tail) — the rewritten version computes the stale-removal condition as `existingKey != tail.chunkKey` up front (equivalent to the original's two overlapping `if` checks), then always routes the actual check-in through `insertOrUpdateEntry` keyed by `tail.chunkKey`.

    **#10 (PromptCacheChunks.swift StoredChunk duplication):** Added `makeStoredChunk(tokens:sliced:parentKey:)`, used by both `sliceChunks`/`sliceTailChunk` instead of duplicated `StoredChunk(...)` construction.

    **#11 (MLXLanguageModel.swift boolean-gate extraction):** `preparePromptVariants` now gates its three variants through named static predicates `shouldSuppressReasoning`/`shouldSetupReasoning`/`shouldBuildGuidedInput`; `makeThinkThenCallConfig` delegates its 3-condition guard to a new `shouldRunThinkThenCall` predicate. Pure extractions, no behavior change.

    **#12 (test coverage symmetry):** Added `qwen3NextSecondContinuationRoundAlsoMatches()` to `PromptCacheHybridArchitectureTests.swift` — the identical 3-round chained-reuse scenario `qwen35SecondContinuationRoundAlsoMatches` already covered, now run against `Qwen3NextModel` too.

    **Deviation, flagged not silently ignored — #13 (ConstraintSetup access modifier):** The finding literally asked for `private struct ConstraintSetup`. I did NOT apply this literally: `Tests/MLXFoundationModelsTests/ToolEnvelopeReserveZoneTests.swift` constructs `ConstraintSetup(...)` directly via `@testable import MLXFoundationModels` from a separate SwiftPM target, and Swift's `private`/`fileprivate` access cannot cross file boundaries even under `@testable import` (only `internal`/`public` are elevated) — marking it `private` would break that test target's compilation. `internal` (the existing, unchanged default) already achieves the finding's actual goal ("not part of the public API" — it's invisible to real consumers of the library, only visible to same-module code and `@testable`-importing test targets, which is the standard, idiomatic pattern this codebase already uses elsewhere, e.g. `chunkCount(modelID:)`/`hybridCheckpointCount(modelID:)` test seams). Added a doc comment on `ConstraintSetup` explaining this access-level decision explicitly so a future automated review doesn't re-file the same (incorrect, build-breaking) suggestion. Checked the box as resolved since the underlying concern (encapsulation) is genuinely addressed, just not via the literal suggested edit.

    Dispatched the `double-check` adversarial review agent against this diff per really-done; will report its verdict once back, and revise if it finds anything.

    **Verification (fresh):** `swift build` clean (0 errors, same single pre-existing unrelated deprecation warning). `swift build --build-tests` clean. Full unfiltered `swift test` (whole repo, 5 test targets): 605 tests total (265 MLXLMTests + 0 MLXHuggingFaceMacrosTests + 80 MLXGuidedGenerationTests + 253 MLXFoundationModelsTests + 7 CXGrammarTests), 0 failures, 0 errors — including the full `PromptCache hybrid Mamba/attention architecture handling` suite and all `PromptCache*Tests`.

    **Out-of-scope findings logged separately:** a fresh `review working` run surfaced 2 NEW findings unrelated to this diff (confirmed via `git diff` hunk ranges) — deep nesting in an `MLXLanguageModel.swift` reasoning-token loop, and `keyParts`/`valueParts` duplication in `PromptCache.assemble()` — filed as kanban task `jnggpjg` rather than scope-creeping into this task.

    Leaving task in `doing` per the implement workflow.
  timestamp: 2026-07-18T17:20:15.628935+00:00
- actor: claude-code
  id: 01kxv50ztkeab8gk6pgkbdh2wd
  text: |-
    Adversarial double-check completed: verdict REVISE with one low-severity finding, now fixed.

    Finding: `MLXLanguageModel.swift`'s new `shouldRunThinkThenCall` predicate (extracted for finding #11's `makeThinkThenCallConfig` guard-simplification) was left with a stale, verbatim-duplicated doc comment -- the ORIGINAL "Derives the think-then-call reasoning config... Two strategy families qualify: .templateFlag/.alwaysOn..." block ended up sitting above `shouldRunThinkThenCall` (which is only the 3-condition boolean gate and has nothing to do with strategy-family dispatch), while `makeThinkThenCallConfig` itself (which actually does that dispatch via its `switch`) kept its own copy of the same block below. A future reader hitting `shouldRunThinkThenCall` first would be misled about what decides the `.templateFlag`/`.alwaysOn` strategy split.

    Fix applied: replaced the stale leading doc comment on `shouldRunThinkThenCall` with a scoped one describing only the 3-condition AND gate, explicitly noting that strategy-family dispatch happens in `makeThinkThenCallConfig`'s own `switch` below -- not here. `makeThinkThenCallConfig`'s own doc comment (the real "Two strategy families qualify" explanation) is untouched.

    Re-verified after the fix: `swift build --build-tests` clean (0 errors, same single pre-existing unrelated deprecation warning), `swift test --filter MLXFoundationModelsTests` → 253/253 passed, 42 suites.

    Everything else in the double-check (offset-verification placement/scope, evictAll/setChunkSize byte-accounting correctness, the 3 shared generic helpers' behavior-preservation across all three insert-path scenarios, makeStoredChunk extraction, the ConstraintSetup access-level reasoning, and both new tests' meaningfulness) was independently verified and confirmed correct.

    Implementation is complete, all 13 review findings resolved and checked off, and the suite is green. Leaving task in `doing` per the implement workflow -- ready for `/review`.
  timestamp: 2026-07-18T17:40:14.803843+00:00
- actor: claude-code
  id: 01kxv5esh9xpkbdn1bz35jntsk
  text: |-
    Review findings fixed (iteration 3): all 13 confirmed findings resolved. Key correctness fix: snapshotHybridCheckpoint() now verifies every KVCacheSimple layer's offset == tokens.count before capturing (mirrors verifiedSimpleLayers's invariant), rejecting a corrupted/inconsistent cache instead of silently snapshotting garbage. Design-consistency fix: evictAll() split into a full-evict path and evictChunkStoreOnly(); setChunkSize() now uses the latter so hybrid checkpoints (keyed by full token prefix, independent of chunkSize) survive a chunk-size change. Duplication removed via shared insertOrUpdateEntry/removeAndReclaimBytes/updateBest helpers in PromptCache.swift and makeStoredChunk() in PromptCacheChunks.swift. Readability: extracted shouldSuppressReasoning/shouldSetupReasoning/shouldBuildGuidedInput/shouldRunThinkThenCall predicates in MLXLanguageModel.swift. Test coverage: added qwen3NextSecondContinuationRoundAlsoMatches (3-round chaining for Qwen3Next, symmetric with the existing Qwen35 test) and setChunkSizeEvictsChunksButPreservesHybridCheckpoints.

    One documented deviation: left ConstraintSetup as internal (not private) rather than following the finding literally, since private would break cross-file @testable import access from ToolEnvelopeReserveZoneTests.swift in a separate test target — Swift's private/fileprivate cannot cross file boundaries even under @testable import.

    Independently re-verified (separate tester agent, fresh runs): swift test --filter PromptCacheHybridArchitectureTests --no-parallel -> 21/21 passed, including 4 named numerical-equivalence tests asserting checkpoint-restored logits match a full fresh forward pass within 1e-3 for both Qwen35TextModel and Qwen3NextModel (single-round and 3-round chained). swift test --filter PromptCache --no-parallel -> 117/117 across 15 suites (21 hybrid + ~96 original pure-attention tests, all unchanged behavior). Full unfiltered swift test -> 807 tests, 0 failures repo-wide.

    Filed a separate follow-up task (jnggpjg) for two unrelated pre-existing findings (deep nesting in a reasoning-token loop, assemble()'s keyParts/valueParts duplication) confirmed outside this task's changed hunks via git diff — not scope-creeped into this task.

    Proceeding to commit checkpoint and re-review.
  timestamp: 2026-07-18T17:47:47.113823+00:00
- actor: claude-code
  id: 01kxx0q33phfx66qs7e5hqdv3x
  text: |-
    Resolved all 5 review findings from the "## Review Findings (2026-07-18 12:48)" checklist:

    1. `MLXLanguageModel.swift`'s `Executor.runReasoning()`: extracted the `.token` case's body (append + processReasoningToken + conditional segment send) into a new private static async helper `handleGeneratedToken(_:generatedTokenIDs:emitter:detokenizer:reasoningTokenCount:responseEntryID:reasoningEntryID:channel:)`. The switch's `.token` case is now a single `await Self.handleGeneratedToken(...)` call, eliminating the nested `if let segments = ...; await sendSegments(...)` that previously kept the loop at 5 levels of nesting.

    2. `PromptCache.swift`'s `updateBest<Entry: CacheStoreEntry>` (used by `globalLRUVictim()`): replaced `if best == nil || entry.lastUsed < best!.lastUsed` with `if best == nil || entry.lastUsed < (best?.lastUsed ?? Int.max)` -- no force unwrap remains, same selection semantics.

    3-5. `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift`: extracted three shared private helpers for the three near-identical Qwen35/Qwen3Next test pairs:
       - `assertHybridModelIsNotChunkableButSupportsReuse(modelFactory:)`
       - `assertCheckpointRestoreMatchesFullForwardPass(seed:modelFactory:tokenFormula:prefixLen:modelIDPrefix:)`
       - `assertSecondContinuationRoundAlsoMatches(seed:modelFactory:tokenFormula:modelIDPrefix:)` (firstPrefixLen/secondPrefixLen were identical -- 8/18 -- in both original tests, so kept as fixed constants inside the helper rather than added as unwarranted parameters, per the "no second caller needing a different value -> no parameter" rule)

       Each pair's two `@Test` functions are now thin one-line/few-line calls into the shared helper with their own seed/model-factory/token-formula/prefixLen/modelIDPrefix. Carefully preserved every original literal (seeds 1001-1004, token formulas, prefixLen 9/10, modelID prefixes) when moving them into call-site arguments.

    **Verification (fresh, this session):** `swift build` clean (0 errors, same single pre-existing unrelated deprecation warning). `swift build --build-tests` clean. `swift test --filter PromptCacheHybridArchitectureTests --no-parallel` -> 21/21 passed, including all 4 numerical-equivalence tests (checkpoint-restored generation matches full forward pass within 1e-3) now running through the shared helpers. `swift test --filter PromptCache --no-parallel` -> 117/117 passed across 15 suites. Full unfiltered `swift test` (whole repo, 5 test-run groups): 265+0+80+253+7 = 605 tests, 0 failures, 0 errors -- matching the pre-refactor baseline exactly (same test count, since this was pure deduplication with no added/removed tests).

    Dispatched the `double-check` adversarial review agent against the diff to independently verify the extraction is behaviorally faithful (especially checking for any accidental seed/formula/prefixLen swap between the two call sites of each shared test helper, which would silently corrupt a numerical-equivalence test's meaning while still compiling and likely still passing). Will report its verdict in a follow-up comment.

    Leaving task in `doing` per the implement workflow -- not moving to review myself.
  timestamp: 2026-07-19T11:03:25.046028+00:00
- actor: claude-code
  id: 01kxx0rp9w9ds5wprce71qnxx5
  text: |-
    Adversarial double-check completed: verdict PASS, no findings.

    The agent independently verified: (1) `handleGeneratedToken`'s extraction from `runReasoning`'s `.token` case is a faithful, order-preserving transcription (append -> processReasoningToken -> conditional sendSegments), with all seven captured locals passed exactly as declared and no other duplicated `.token`-handling loop left un-refactored; (2) `updateBest`'s rewritten guard is logically equivalent to the original `best!` form under `||` short-circuit semantics, and no force-unwrap remains in `updateBest`/`globalLRUVictim`; (3) every numeric literal in the three deduplicated test-helper pairs (seeds 1001-1004, prefixLen 9/10, token formulas, modelIDPrefixes) lands on the correct model-factory/prefix pairing with no cross-swap, all original `#expect` assertions/messages preserved verbatim, and firstPrefixLen/secondPrefixLen (8/18) correctly kept as fixed constants rather than unwarranted parameters since both original tests used identical values.

    Final state: all 5 findings under "## Review Findings (2026-07-18 12:48)" are resolved and checked off. Verification (fresh, this session): `swift build` clean, `swift build --build-tests` clean, `swift test --filter PromptCacheHybridArchitectureTests --no-parallel` 21/21 passed, `swift test --filter PromptCache --no-parallel` 117/117 passed, full unfiltered `swift test` (whole repo) 605 tests (265+0+80+253+7), 0 failures, 0 errors -- exactly matching the pre-refactor baseline test count (pure deduplication, no tests added/removed).

    Task is green and complete. Leaving in `doing` per the implement workflow -- ready for `/review`.
  timestamp: 2026-07-19T11:04:17.468885+00:00
- actor: claude-code
  id: 01kxx1np0zh2jvty0806rqq9pf
  text: |-
    Independent test verification (fresh `swift test` run, not reusing the resolving agent's numbers):

    - `swift test`: 605 tests, 0 failures, 0 skipped -- matches the reported baseline exactly (265+80+253+7 across the four swift-testing/XCTest bundles, plus one 0-test bundle).
    - `swift build --build-tests` after touching the changed files (PromptCache.swift, PromptCacheChunks.swift, MLXLanguageModel.swift, PromptCacheHybridArchitectureTests.swift) to force recompilation: found one real compiler warning at `MLXLanguageModel.swift:1029` -- `LanguageModelCapabilities(capabilities: capabilities)` using the deprecated `init(capabilities:)` instead of `init(_:)` (confirmed via the macOS 27 SDK `.swiftinterface`: both inits take the same `[Capability]` array and share availability, so this is a pure label-removal, behavior-identical). This warning pre-dates this task (many earlier kanban tasks note it as "pre-existing, unrelated") but the `test` skill requires zero warnings, so fixed it here: `LanguageModelCapabilities(capabilities)`.
    - Re-ran `swift build --build-tests` and `swift test` after the fix: 0 warnings from our code (only a pre-existing, out-of-our-control `warning: missing creator for mutated node` from the external `mlx-swift` package's `Cmlx.bundle` resource copy remains -- a SwiftPM/llbuild build-plan diagnostic for a remote dependency, not our source). 605 tests, 0 failures confirmed again post-fix.
    - `mcp__sah__diagnostics check working`: 0 errors, 0 warnings.

    No test-failure kanban tasks created -- there were no failures to track.
  timestamp: 2026-07-19T11:20:07.455010+00:00
- actor: claude-code
  id: 01kxx54np22vafgyf65113ksyp
  text: |-
    Resolved the sole finding under "## Review Findings (2026-07-19 06:26)": nesting reduction in `PromptCacheChunks.swift`.

    Read the actual current function first (per the finding's own caveat that the line number may have shifted): `verifiedSimpleLayers(cache:tokenCount:)` was at lines 481-508 (not 117), and its structure was `guard !cache.isEmpty` -> `for layer in cache` -> `guard type(of: layer) == KVCacheSimple.self, ... else { return nil }`. Confirmed this is the right function (not `snapshotHybridCheckpoint` or `hybridLayerKinds`, which have their own separate, un-refactored inline guards for the hybrid-checkpoint path).

    Extracted the per-layer check into `private static func verifySimpleLayer(_ layer: KVCache, tokenCount: Int) -> KVCacheSimple?`, carrying over the `ChunkedKVCache`-subclass-exclusion doc comment (why a plain `as? KVCacheSimple` isn't enough). `verifiedSimpleLayers`'s loop body is now a single `guard let simple = verifySimpleLayer(layer, tokenCount: tokenCount) else { return nil }; simpleLayers.append(simple)` -- nesting reduced by one level, no behavior change. Verified via `get callgraph` that `verifiedSimpleLayers` has only two callers in the file (`sliceChunks`, `sliceTailChunk`), both unaffected since its own signature is unchanged.

    Adversarial double-check (via double-check agent) ran against the diff: verdict REVISE with one finding -- the helper's original name, `verifySimpleLayerForCheckpoint` (the exact name suggested in the finding text), was misleading in this specific file, which draws a heavily-documented, deliberate distinction between the CHUNK mechanism (`isChunkable`/`verifiedSimpleLayers`/`sliceChunks`/`sliceTailChunk`) and the separate hybrid CHECKPOINT mechanism (`HybridCheckpoint`/`snapshotHybridCheckpoint`/`restoreHybridCheckpoint`). The new helper is exclusively part of the CHUNK path and is never called by `snapshotHybridCheckpoint` (which has its own separate inline guard), so "...ForCheckpoint" falsely implied it belonged to the checkpoint mechanism. Renamed to `verifySimpleLayer(_:tokenCount:)`, matching its own doc comment's actual description of its role.

    Verification (fresh, this session, after the rename): `swift build` clean, `swift build --build-tests` clean (0 errors, only the pre-existing unrelated `Cmlx.bundle` SwiftPM diagnostic). Full unfiltered `swift test`: 265+0+80+253+7 = 605 tests, 0 failures -- exactly matching the required baseline. `swift test --filter PromptCache --no-parallel`: 117/117 passed across 15 suites, zero regression to the pure-attention chunk-cache or hybrid-checkpoint behavior. `mcp__sah__diagnostics check working`: 0 errors, 0 warnings.

    Checkbox flipped to `- [x]` in the task description; all earlier review-findings sections left untouched. Leaving task in `doing` per the implement workflow -- not moving to review myself.
  timestamp: 2026-07-19T12:20:44.354559+00:00
- actor: claude-code
  id: 01kxx5rb4v7pjhvh9jf410brs3
  text: |-
    Fixed both 2026-07-19 07:17 review findings in Libraries/MLXFoundationModels/PromptCacheChunks.swift:

    - Added `private static let kvCacheSimpleStateElementCount = 2` in the `PromptCache` extension (right after `rootChunkKey`), documenting it as the exact element count a touched `KVCacheSimple.state` must have (`[keys, values]`).
    - Replaced `layer.state.count == 2` in `snapshotHybridCheckpoint` with `layer.state.count == kvCacheSimpleStateElementCount`.
    - Replaced `simple.state.count == 2` in `verifySimpleLayer` with `simple.state.count == kvCacheSimpleStateElementCount`.
    - Also swept the file for any other recurrence of this exact same magic number: found two doc-comment lines that literally quoted `state.count == 2` in prose describing those same two guard conditions, and updated them to say `state.count == kvCacheSimpleStateElementCount` for consistency with the code. Confirmed `sliceChunkLayer`'s `state[0]`/`state[1]` indexing is a different thing (array indices into the 2-element state, not the count invariant) and left untouched, per the task's scoping instruction.

    Verification:
    - `swift build`: exit 0, build succeeded.
    - `swift test` (full, unfiltered): exit 0, 605 total tests across 5 test-run groups (265 + 0 + 80 + 253 + 7), 0 failures — matches the established 605-test baseline. Includes the XCTest-style suites (e.g. MLXLMTests.xctest 197 tests) as well as swift-testing suites (e.g. PromptCache chunk slicing).
    - Adversarial double-check agent dispatched per really-done to verify the diff is scoped exactly to this fix with no behavior change.

    Leaving task in `doing` for review per /implement process.
  timestamp: 2026-07-19T12:31:28.923456+00:00
- actor: claude-code
  id: 01kxx7qadg6pjbzdbha728nqmx
  text: |-
    Implemented all 9 findings from the "2026-07-19 07:32" review round in PromptCacheChunks.swift:

    1. Restructured `kvCacheSimpleStateElementCount`'s doc comment: first line now a complete sentence ending in a period ("The exact element count a touched `KVCacheSimple.state` must have: `[keys, values]`."), followed by a blank `///` line, then the elaboration.

    2-8. Added explicit `internal` access modifier to all 7 flagged symbols (`rootChunkKey`, `ownedCopy(of:)`, `isChunkable(_:)`, `isHybridMambaAttention(_:)`, `restoreHybridCheckpoint(_:)`, `sliceChunks(tokens:cache:chunkSize:)`, `sliceTailChunk(tokens:cache:chunkSize:parentKey:)`). Verified via grep across the repo that every real caller (PromptCache.swift, MLXLanguageModel.swift, and test files which use `@testable import MLXFoundationModels`) lives inside the same module — no cross-module usage exists, so `internal` is correct (not `public`). Also noted the containing `PromptCache` actor itself has no explicit modifier (implicitly `internal`), so `public` on a member would be capped/meaningless anyway. Preserved existing `nonisolated`/`static` ordering, prefixing with `internal`.

    9. Extracted new private helper `verifySimpleKVCacheState(_:tokenCount:)` covering the state-count + `as? KVCacheSimple` cast + offset-match check (without the exact-dynamic-type check). `verifySimpleLayer(_:tokenCount:)` now does its own `type(of: layer) == KVCacheSimple.self` guard and then delegates the rest to the shared helper. `snapshotHybridCheckpoint`'s per-layer `.simple` branch now calls the shared helper directly instead of repeating the inline guard.

    Verification: `swift build` succeeded (exit 0). Full unfiltered `swift test` passed 605/605 tests across all targets (265+0+80+253+7 across the 5 test-run summaries) — matches the established 605/0 baseline exactly, no regressions.

    Flipped all 9 checkboxes under "Review Findings (2026-07-19 07:32)" to `[x]` with resolution notes; left all earlier review sections untouched.

    Spawned the double-check adversarial agent for sign-off; will record its verdict once it returns. Task left in `doing` pending review.
  timestamp: 2026-07-19T13:05:52.560558+00:00
- actor: claude-code
  id: 01kxx85atsc8zsxt9dk9q6knk9
  text: |-
    Adversarial double-check verdict: PASS.

    Independent verification performed by the double-check agent (not just trusting my reported numbers):
    - Confirmed `verifySimpleKVCacheState` is a straight partition of the original 4-condition AND-guard in `verifySimpleLayer` (order-independent, pure/side-effect-free checks) — behavior is bit-for-bit identical at both call sites, including confirming the `simple` binding consumed by `verifiedSimpleLayers` is still correctly returned, and that `snapshotHybridCheckpoint`'s inline `simple` binding was never read afterward (uses `layer.state`, not `simple.state`), so collapsing it to `!= nil` changes nothing observable.
    - Confirmed via grep of the diff that all 7 access-modifier additions are exactly `internal` (no `public`/`private`/`fileprivate` slipped in), and cross-checked real call sites are all within the module.
    - Confirmed the restructured doc comment reads as a complete, grammatically correct summary sentence followed by a blank line and elaboration.
    - Ran independent verification: `swift build --target MLXFoundationModels` (0 errors), `mcp__sah__diagnostics check file` on PromptCacheChunks.swift (0 errors/warnings), and `swift test --filter 'PromptCacheHybridArchitectureTests|PromptCacheChunkTests|PromptCacheChunkStoreTests'` (56/56 passed) — directly exercising the touched functions.

    Non-blocking observation (explicitly out of scope for this diff, not a defect introduced by it): `chunkKey(parentKey:tokens:)`, `hybridLayerKinds(_:)`, `snapshotHybridCheckpoint(tokens:cache:)`, `cacheAdvanceOffset(_:)`, and the `ChunkKey` typealias in the same file are called from the same cross-file sites as the 7 symbols that just got explicit `internal`, yet remain implicit-internal. These weren't among the 9 recorded findings and weren't touched by this diff, so left as-is per "no unrelated refactors" — flagging as a candidate for a future consistency pass if the user wants one, not doing it unilaterally here.

    Final state: swift build green, full swift test suite 605/605 passing (baseline maintained), double-check PASS. Task remains in `doing` for `/review` to pick up per the implement workflow.
  timestamp: 2026-07-19T13:13:31.737846+00:00
- actor: claude-code
  id: 01kxx96pg3pcttpsm201r0cbbg
  text: |-
    Full access-modifier sweep of Libraries/MLXFoundationModels/PromptCacheChunks.swift completed (2026-07-19), per the task's explicit instruction to converge instead of trickling fixes review-round by review-round.

    **Read the entire file top to bottom** (all 721 lines) and audited every top-level/extension-level declaration inside `extension PromptCache { ... }` for an explicit access modifier. Found 8 declarations missing one (6 cited by this review round + 2 more found during the sweep):

    1. `typealias ChunkKey` (cited) → `internal typealias ChunkKey = Int`
    2. `struct StoredChunk` (cited) → `internal struct StoredChunk`
    3. `chunkKey(parentKey:tokens:)` (cited) → `internal nonisolated static func chunkKey(...)`
    4. `enum HybridLayerKind` (cited) → `internal enum HybridLayerKind: Equatable`
    5. `hybridLayerKinds(_:)` (cited) → `internal nonisolated static func hybridLayerKinds(...)`
    6. `snapshotHybridCheckpoint(tokens:cache:)` (cited) → `internal nonisolated static func snapshotHybridCheckpoint(...)`
    7. `struct HybridCheckpoint` (NOT cited — found during the sweep, sitting right next to `HybridLayerKind`/`snapshotHybridCheckpoint` which were cited) → `internal struct HybridCheckpoint`
    8. `cacheAdvanceOffset(_:)` (NOT cited — this is the one the task description flagged as "flagged as a candidate by a previous round's double-check but not yet fixed") → `internal nonisolated static func cacheAdvanceOffset(_ cache: [KVCache]) -> Int?`

    Verified every symbol's call sites via grep before choosing `internal`: all cross-file callers live in `Libraries/MLXFoundationModels` (PromptCache.swift, MLXLanguageModel.swift) or in `Tests/MLXFoundationModelsTests` via `@testable import MLXFoundationModels` — confirmed no non-testable external consumers exist, so `internal` is correct throughout, consistent with every symbol fixed in prior rounds.

    Left untouched (already had explicit modifiers, confirmed via full-file grep sweep): `rootChunkKey`, `kvCacheSimpleStateElementCount` (private), `ownedCopy(of:)`, `sliceChunkLayer` (private), `isChunkable(_:)`, `isHybridMambaAttention(_:)`, `restoreHybridCheckpoint(_:)`, `verifiedSimpleLayers` (private), `verifySimpleKVCacheState` (private), `verifySimpleLayer` (private), `sliceLayers` (private), `makeStoredChunk` (private), `sliceChunks(...)`, `sliceTailChunk(...)`.

    Struct member properties (`StoredChunk.tokens/layers/parentKey/chunkKey/byteSize/lastUsed`, `HybridCheckpoint.tokens/layers/byteSize/lastUsed`) were deliberately left without their own modifiers — plain stored properties inside an internal-access type, no need for individual annotation per the task's own guidance ("use judgment, don't over-add").

    Final state: every one of the 22 top-level/extension-level declarations in the file now has an explicit `internal` or `private` modifier — confirmed via `grep -n '^    (internal |private )?(typealias|struct|enum|static func|...)'` producing 22 matches, all with an explicit keyword. A fresh review of this file should find zero further recurrences of this finding type.

    Flipped all 6 cited checkboxes under "Review Findings (2026-07-19 08:14)" to `[x]`, added a resolution note documenting the 2 extra fixes found during the sweep. Earlier sections left untouched.

    Verification: `swift build` — exit 0, build succeeded. `swift test` (full, unfiltered) — 265 + 0 + 80 + 253 + 7 = 605 tests across all targets, 0 failures, exit code 0. Matches the established 605/0 baseline exactly. Adversarial double-check dispatched to confirm correctness of the sweep before handoff.

    Task left in `doing` per /implement's contract — not moved to review.
  timestamp: 2026-07-19T13:31:45.027972+00:00
position_column: done
position_ordinal: c280
title: 'PromptCache never engages for hybrid Mamba/attention models (Qwen3.6/Qwen3-Next): cachedTokenCount stays 0'
---
## What

`PromptCache`'s chunk-store mechanism silently never caches anything for hybrid Mamba/attention architectures (Qwen3.6/Qwen3-Next-family). See task comments for the full history: an earlier "out of scope" resolution was rejected by the user; real Mamba-state reuse via a "hybrid checkpoint" mechanism was designed and implemented instead.

## Acceptance criteria

- [x] Decide (and document) an approach for hybrid Mamba/attention models. **DONE: option (b), hybrid checkpoint mechanism.**
- [x] Coverage proving the mechanism actually works, against real models. **DONE.**
- [x] No regression to the existing pure-attention chunk-cache behavior. **DONE.**
- [x] Known, accepted degradation documented. **DONE.**

## Scope

`Libraries/MLXFoundationModels/PromptCache.swift`/`PromptCacheChunks.swift`/`MLXLanguageModel.swift`.

## Review Findings (2026-07-18 10:49) through (2026-07-19 07:32)

All resolved — see comment thread for detail. Includes: hybrid checkpoint mechanism design, offset verification, evictAll/setChunkSize split, insert/remove/LRU dedup helpers, makeStoredChunk, nesting reduction (verifySimpleLayer), magic-number extraction (kvCacheSimpleStateElementCount), doc-comment fix, and explicit internal access modifiers on 7 declarations (rootChunkKey, ownedCopy, isChunkable, isHybridMambaAttention, restoreHybridCheckpoint, sliceChunks, sliceTailChunk) plus verifySimpleKVCacheState extraction.

## Review Findings (2026-07-19 08:14)

- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:17` — Typealias `ChunkKey` lacks explicit `internal` access modifier. Add `internal` before `typealias`.
- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:28` — Struct `StoredChunk` lacks explicit `internal` access modifier. Add `internal` before `struct`.
- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:91` — Function `chunkKey(parentKey:tokens:)` lacks explicit `internal` access modifier. Add `internal` before `nonisolated`.
- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:155` — Enum `HybridLayerKind` lacks explicit `internal` access modifier. Add `internal` before `enum`.
- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:167` — Function `hybridLayerKinds(_:)` lacks explicit `internal` access modifier. Add `internal` before `nonisolated`.
- [x] `Libraries/MLXFoundationModels/PromptCacheChunks.swift:237` — Function `snapshotHybridCheckpoint(tokens:cache:)` lacks explicit `internal` access modifier. Add `internal` before `nonisolated`.

IMPORTANT for whoever picks this up: this is the THIRD consecutive round flagging "missing explicit access modifier" on a DIFFERENT subset of this same file's declarations each time (round 4 fixed 7, this round flags 6 more). To converge instead of trickling indefinitely, this round must audit and fix EVERY remaining top-level/extension-level declaration in PromptCacheChunks.swift that lacks an explicit access modifier — not just the 6 cited here — so a fresh review of this file finds zero further recurrences of this finding type.

**RESOLVED (full sweep completed 2026-07-19):** All 6 cited declarations fixed, plus 2 more found during the full top-to-bottom sweep that weren't cited (`HybridCheckpoint` struct, `cacheAdvanceOffset(_:)` function — the latter was flagged as a candidate by a previous round's double-check but not yet fixed). Every top-level/extension-level declaration in the file now has an explicit `internal` or `private` modifier — see task comment for the complete audit trail.
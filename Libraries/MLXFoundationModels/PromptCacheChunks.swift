// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

extension PromptCache {

    /// Key type for the chunk hash chain: a `Hasher`-derived, per-process value
    /// (see ``chunkKey(parentKey:tokens:)``), never persisted or compared across
    /// process launches. Named distinctly from a bare `Int` for readability at
    /// the chunk-store call sites (``insert(modelID:chunks:)``,
    /// ``lookupLongestPrefix(modelID:newTokens:chunkSize:)``).
    internal typealias ChunkKey = Int

    /// One fixed-size token-range slice of a verified `KVCacheSimple` stack, with
    /// its own tensors materialized as OWNED, evaluated, contiguous copies (see
    /// `ownedCopy(of:)` -- the copies never alias the source cache's buffers).
    ///
    /// Chunks form a hash chain via ``parentKey``/``chunkKey`` (see
    /// ``PromptCache/chunkKey(parentKey:tokens:)``): identical token prefixes
    /// produce identical keys, and keys diverge from the first point two token
    /// sequences differ. `Hasher` is per-process seeded, so a `chunkKey` must
    /// never be persisted or compared across process launches.
    internal struct StoredChunk {
        /// The token ids this chunk covers -- exactly `chunkSize` long for a
        /// FULL chunk (see `sliceChunks`), or a variable, shorter-than-
        /// `chunkSize` span for a TAIL chunk (see `sliceTailChunk`, which
        /// slices exactly the trailing partial span `sliceChunks` itself
        /// leaves uncovered).
        var tokens: [Int]

        /// Per-layer key/value tensor slices, in the same layer order as the
        /// source `[KVCache]`. Each tensor is an owned copy (see
        /// ``PromptCache/ownedCopy(of:)``), independent of the source cache's
        /// buffers.
        var layers: [(keys: MLXArray, values: MLXArray)]

        /// The chunk chain's previous link -- `PromptCache.rootChunkKey` for the
        /// first chunk of a sequence, otherwise the preceding chunk's `chunkKey`.
        var parentKey: Int

        /// This chunk's own hash-chain key: `hash(parentKey, tokens)` (see
        /// ``PromptCache/chunkKey(parentKey:tokens:)``).
        var chunkKey: Int

        /// Real retained memory footprint of ``layers``' owned tensors, in
        /// bytes -- computed from the owned copies themselves (`MLXArray.nbytes`),
        /// so it reflects what's actually retained, not the source's footprint.
        var byteSize: Int

        /// Recency stamp for eviction bookkeeping. `sliceChunks` is a pure
        /// function with no clock/counter of its own, so every freshly sliced
        /// chunk starts at `0`; the storage layer that checks chunks into its
        /// cache is responsible for stamping a real recency value.
        var lastUsed: Int
    }

    /// Root parent key for the first chunk in a sequence -- a fixed seed distinct
    /// from any real chunk key (which are `Hasher`-derived and, in practice,
    /// unpredictable, making an accidental collision with this sentinel
    /// vanishingly unlikely).
    internal static let rootChunkKey = 0

    /// The exact element count a touched `KVCacheSimple.state` must have:
    /// `[keys, values]`.
    ///
    /// (see `KVCacheSimple.state` in `KVCache.swift`). Named so every
    /// verification site that checks this shape invariant
    /// (``snapshotHybridCheckpoint(tokens:cache:)``, ``verifySimpleLayer(_:tokenCount:)``)
    /// shares one point of change rather than repeating the literal `2`.
    private static let kvCacheSimpleStateElementCount = 2

    /// Computes the next link in a chunk's hash chain from its parent's key and
    /// its own token ids.
    ///
    /// `Hasher`'s per-process randomized seed means two chunks with the same
    /// `parentKey`/`tokens` produce the same `chunkKey` only within the same
    /// process (see `StoredChunk`'s doc comment) -- this is a chain identity
    /// check for prefix reuse within one running process, not a stable content
    /// hash.
    ///
    /// - Parameters:
    ///   - parentKey: The preceding chunk's `chunkKey`, or `rootChunkKey` for
    ///     the first chunk in a sequence.
    ///   - tokens: This chunk's token ids.
    /// - Returns: This chunk's `chunkKey`.
    internal nonisolated static func chunkKey(parentKey: Int, tokens: [Int]) -> Int {
        var hasher = Hasher()
        hasher.combine(parentKey)
        for token in tokens {
            hasher.combine(token)
        }
        return hasher.finalize()
    }

    /// Materializes `array` as a genuinely OWNED, contiguous, already-evaluated
    /// copy, backed by fresh Swift-native memory that shares nothing with
    /// `array`'s underlying buffer.
    ///
    /// This matters because an MLX index slice (e.g.
    /// `array[.ellipsis, a..<b, 0...]`) always evaluates via
    /// `shared_buffer_slice`/`copy_shared_buffer` (see
    /// `mlx/backend/common/slicing.cpp`) -- a zero-copy view that shares the
    /// SOURCE array's entire underlying allocator buffer, unconditionally, not
    /// merely when the slice happens to be contiguous. Calling `.eval()` on
    /// such a slice does not copy any data either: it only detaches the
    /// *compute graph* (`array::detach()` in `array.cpp` clears the array's
    /// `inputs`), which is orthogonal to the *buffer* its data still points
    /// into. So a naive "slice, then eval" chunk keeps the ENTIRE source
    /// cache's buffer allocated for as long as the chunk lives, even after the
    /// source `KVCacheSimple` itself is unreachable -- defeating both eviction
    /// and `byteSize` (confirmed empirically: `PromptCacheChunkTests
    /// .chunkCopiesReleaseSourceBufferMemory` fails against exactly this naive
    /// implementation, showing active memory retained at ~the full source
    /// cache's footprint).
    ///
    /// Round-tripping through `asData(access: .copy)` (a `Data` the docs
    /// guarantee is an independent, contiguous copy, unconditionally) and back
    /// through `MLXArray(data:)` sidesteps the shared buffer entirely: the
    /// result is built from `mlx_array_new_data`, which copies its input bytes
    /// into a fresh `mlx::core::array` allocation with no graph and no shared
    /// buffer.
    ///
    /// Not `private`: also used by `PromptCache.assemble(chunks:layerCount:)`
    /// (`PromptCache.swift`) to force a genuinely fresh buffer after
    /// `concatenated(...)` -- necessary because `mlx::core::concatenate`
    /// special-cases a SINGLE input array by returning that exact array
    /// unchanged (`mlx/ops.cpp`: `if (arrays.size() == 1) { return arrays[0]; }`),
    /// never allocating a fresh buffer or invoking the `Concatenate` primitive
    /// in that case -- so a lone matched chunk's concatenation result would
    /// otherwise alias the chunk store's own owned tensor by reference.
    internal static func ownedCopy(of array: MLXArray) -> MLXArray {
        let owned = MLXArray(data: array.asData(access: .copy))
        owned.eval()
        return owned
    }

    /// Slices and copies a single layer's key/value tensors for one chunk's
    /// token range, producing OWNED, evaluated, non-aliasing tensors (see
    /// ``ownedCopy(of:)`` -- the same buffer-independence guarantee applies
    /// here, since this is exactly the per-layer slice `sliceChunks` used to
    /// perform inline).
    ///
    /// - Parameters:
    ///   - layer: The source layer's verified `KVCacheSimple` state.
    ///   - start: The chunk's starting token offset (inclusive).
    ///   - end: The chunk's ending token offset (exclusive).
    /// - Returns: The layer's owned `keys`/`values` slices for this chunk,
    ///   plus their combined retained byte footprint (`MLXArray.nbytes`).
    private static func sliceChunkLayer(
        layer: KVCacheSimple, start: Int, end: Int
    ) -> (keys: MLXArray, values: MLXArray, byteSize: Int) {
        let state = layer.state
        let keys = ownedCopy(of: state[0][.ellipsis, start ..< end, 0...])
        let values = ownedCopy(of: state[1][.ellipsis, start ..< end, 0...])
        return (keys: keys, values: values, byteSize: keys.nbytes + values.nbytes)
    }

    /// Whether `cache`'s layer composition can participate in this store's
    /// CHUNK-based reuse (fixed-size token-range slicing of a finished
    /// round's K/V tensors) -- a structural, round-independent capability
    /// check, unlike ``verifiedSimpleLayers(cache:tokenCount:)`` (which
    /// additionally requires every layer's `offset` to match a SPECIFIC
    /// round's `tokenCount`). `true` only when every layer is exactly
    /// `KVCacheSimple` (mirroring `verifiedSimpleLayers`'s own exact-type
    /// rule -- see that function's doc comment for why a plain
    /// `as? KVCacheSimple` isn't enough).
    ///
    /// DECISION (kanban `r9rf5g7`): hybrid Mamba/attention architectures
    /// (Qwen3.6/Qwen3-Next family: `Qwen35Model`/`Qwen35MoEModel`/
    /// `Qwen3NextModel` in `Libraries/MLXLLM/Models/Qwen35.swift`/
    /// `Qwen3Next.swift`, via `Qwen35DecoderLayer.isLinear`/
    /// `Qwen3NextDecoderLayer.isLinear`) build a `[KVCache]` mixing
    /// `KVCacheSimple` (full-attention layers) with `MambaCache`
    /// (Gated-DeltaNet linear-attention layers), so `isChunkable` always
    /// returns `false` for them: CHUNK slicing genuinely cannot apply to
    /// Mamba's collapsed recurrent state. `sliceChunks` carves ANY
    /// `chunkSize`-aligned window out of a FINISHED round's tensor by plain
    /// array indexing -- that only works because attention's key/value
    /// tensors are append-only and addressable by absolute token position.
    /// A Mamba layer's `state` (conv buffer + Gated-DeltaNet recurrent
    /// state -- see `ArraysCache`/`MambaCache` in `MLXLMCommon/KVCache.swift`)
    /// is a running, COLLAPSED summary with no per-position history at
    /// all -- there is no boundary inside it to slice.
    ///
    /// This does NOT mean hybrid architectures can never participate in
    /// prompt-cache reuse at all -- an earlier revision of this comment
    /// claimed exactly that, and the claim was wrong (a scope-limiting
    /// decision, not an architectural impossibility). ``isHybridMambaAttention(_:)``/
    /// ``snapshotHybridCheckpoint(tokens:cache:)``/``restoreHybridCheckpoint(_:)``
    /// (below) implement a SEPARATE, parallel "hybrid checkpoint" mechanism:
    /// rather than slicing a fixed window out of a layer's state, it
    /// captures the ENTIRE, unsliced state of every layer (both
    /// `KVCacheSimple`'s `[keys, values]` and `MambaCache`'s `[convState,
    /// ssmState]`) at a round boundary, and restores all of it verbatim on
    /// a later round whose prompt tokens start with an exact match of the
    /// stored prefix. This works because `Qwen35GatedDeltaNet.callAsFunction`/
    /// `Qwen3NextGatedDeltaNet.callAsFunction` (the Mamba layer's forward
    /// pass) read ONLY `cache?[0]`/`cache?[1]` (the raw conv/SSM state
    /// arrays) as VALUES -- never `cache.offset` or any other position
    /// signal -- so the recurrence update is identical whether the prior
    /// state came from one token earlier in the SAME forward call (ordinary
    /// incremental generation) or from a checkpoint captured after an
    /// entirely earlier round. See `PromptCache.resolveHybridCheckpoint(modelID:newTokens:)`/
    /// `PromptCache.insertHybridCheckpoint(modelID:checkpoint:)` for how the
    /// actor stores and matches checkpoints, and
    /// `MLXLanguageModel.supportsPromptCacheReuse(model:parameters:)` for
    /// the combined public capability signal (`isChunkable(_:) ||
    /// isHybridMambaAttention(_:)`).
    ///
    /// - Parameter cache: The cache stack to check, one entry per model layer.
    /// - Returns: `true` when every layer is exactly `KVCacheSimple`;
    ///   `false` for an empty stack or any non-`KVCacheSimple`/subclass/
    ///   mixed-type layer -- including a genuine hybrid stack, which
    ///   instead participates in reuse via ``isHybridMambaAttention(_:)``'s
    ///   checkpoint mechanism, never this one.
    internal nonisolated static func isChunkable(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { type(of: $0) == KVCacheSimple.self }
    }

    // MARK: - Hybrid Mamba/Attention Checkpoints

    /// Tags which concrete `KVCache` subclass a hybrid checkpoint's captured
    /// layer state came from, so ``restoreHybridCheckpoint(_:)`` knows which
    /// concrete type to reconstruct -- ``HybridCheckpoint/layers`` stores
    /// plain `[MLXArray]` state (the same shape both `KVCacheSimple.state`
    /// and `MambaCache.state` expose), which alone doesn't say which setter
    /// to feed it back into.
    internal enum HybridLayerKind: Equatable {
        /// Captured from an exactly-`KVCacheSimple` layer: `state` is
        /// `[keys, values]`.
        case simple
        /// Captured from an exactly-`MambaCache` layer: `state` is
        /// `[convState, ssmState]` (or `[]` for an untouched cache -- see
        /// `MambaCache`/`ArraysCache.state` in `KVCache.swift`).
        case mamba
    }

    /// One layer per entry, tagging `cache`'s concrete shape -- `nil` if any
    /// layer is neither exactly `KVCacheSimple` nor exactly `MambaCache` (an
    /// unknown shape this mechanism refuses to reason about, mirroring
    /// ``isChunkable(_:)``'s own exact-type philosophy).
    ///
    /// - Parameter cache: The cache stack to classify, one entry per model layer.
    /// - Returns: One ``HybridLayerKind`` per layer, in the same order as
    ///   `cache`; `nil` for an empty stack or any layer of an unrecognized type.
    internal nonisolated static func hybridLayerKinds(_ cache: [KVCache]) -> [HybridLayerKind]? {
        guard !cache.isEmpty else { return nil }
        var kinds: [HybridLayerKind] = []
        kinds.reserveCapacity(cache.count)
        for layer in cache {
            if type(of: layer) == KVCacheSimple.self {
                kinds.append(.simple)
            } else if type(of: layer) == MambaCache.self {
                kinds.append(.mamba)
            } else {
                return nil
            }
        }
        return kinds
    }

    /// Whether `cache` is a genuine hybrid Mamba/attention stack -- every
    /// layer exactly `KVCacheSimple` or `MambaCache` (see
    /// ``hybridLayerKinds(_:)``), with AT LEAST ONE of each. A stack that's
    /// entirely `KVCacheSimple` should keep using the ordinary chunk store
    /// (``isChunkable(_:)``); an all-`MambaCache` or unrecognized-mixed
    /// stack isn't handled by either mechanism.
    ///
    /// - Parameter cache: The cache stack to check, one entry per model layer.
    /// - Returns: `true` when `cache` mixes at least one `KVCacheSimple` and
    ///   at least one `MambaCache` layer.
    internal nonisolated static func isHybridMambaAttention(_ cache: [KVCache]) -> Bool {
        guard let kinds = hybridLayerKinds(cache) else { return false }
        return kinds.contains(.simple) && kinds.contains(.mamba)
    }

    /// A whole-stack, UNSLICED snapshot of a hybrid cache's raw layer state
    /// at a round boundary, keyed by the full token prefix it covers.
    ///
    /// Unlike ``StoredChunk``, which slices a fixed token-range WINDOW out
    /// of a verified `KVCacheSimple` layer's finished tensor,
    /// `HybridCheckpoint` captures each layer's ENTIRE `state` verbatim --
    /// this is what makes Mamba's collapsed recurrent state (which has no
    /// per-position history to slice) reusable at all: rather than carving
    /// a window out of it after the fact, a checkpoint is captured whole
    /// and restored whole.
    internal struct HybridCheckpoint {
        /// The FULL token prefix this checkpoint covers, from the start of
        /// the sequence -- not a window; every layer's `state` here
        /// reflects having processed exactly this many tokens.
        var tokens: [Int]

        /// Every layer's raw, unsliced, OWNED-copy `state` (see
        /// ``PromptCache/ownedCopy(of:)``), tagged with the concrete type it
        /// came from (see ``HybridLayerKind``) so
        /// ``restoreHybridCheckpoint(_:)`` can reconstruct the right
        /// concrete `KVCache` subclass per layer, in the same layer order
        /// as the source `[KVCache]`.
        var layers: [(kind: HybridLayerKind, state: [MLXArray])]

        /// Real retained memory footprint of every layer's owned state
        /// arrays, in bytes (`MLXArray.nbytes`, summed).
        var byteSize: Int

        /// Recency stamp for eviction bookkeeping -- see
        /// ``StoredChunk/lastUsed``'s doc comment for the same convention
        /// (fresh snapshots start at `0`; the storing actor stamps a real
        /// value).
        var lastUsed: Int
    }

    /// Captures a completed round's hybrid `[KVCache]` as a whole,
    /// UNSLICED ``HybridCheckpoint``, for later exact-prefix-match restore
    /// (see ``restoreHybridCheckpoint(_:)``/`PromptCache.resolveHybridCheckpoint(modelID:newTokens:)`).
    ///
    /// Refuses to snapshot a `KVCacheSimple` layer whose `state` isn't
    /// exactly `[keys, values]` (an untouched, never-updated layer, whose
    /// `state` getter returns `[]` -- see `KVCacheSimple.state` in
    /// `KVCache.swift`) OR whose `offset` doesn't exactly match
    /// `tokens.count` -- mirroring ``verifiedSimpleLayers(cache:tokenCount:)``'s
    /// own `state.count == kvCacheSimpleStateElementCount && offset == tokenCount` requirement, so a
    /// corrupted or inconsistent cache (e.g. one whose layers have silently
    /// drifted out of sync with `tokens`) can never silently produce a
    /// checkpoint that doesn't actually correspond to `tokens` -- restoring
    /// such a checkpoint later and feeding it a suffix computed against the
    /// WRONG prefix length would corrupt every subsequent token's
    /// generation. A `MambaCache` layer's `state` has no such requirement
    /// here: an untouched layer's `[]` is captured as-is (`store()` only
    /// calls this after a real round of generation, so in practice every
    /// layer has already been touched at least once). `[]` is NOT, however,
    /// a state `restoreHybridCheckpoint(_:)` can hand to `MambaCache.state`'s
    /// setter directly -- see that function's doc comment for why it
    /// special-cases this.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence `cache` currently reflects.
    ///   - cache: The hybrid cache stack to snapshot, one entry per model layer.
    /// - Returns: The captured ``HybridCheckpoint``, or `nil` when `cache`
    ///   isn't a genuine hybrid stack (``isHybridMambaAttention(_:)``) or
    ///   any `KVCacheSimple` layer's `state` isn't a verified `[keys,
    ///   values]` pair whose `offset` matches `tokens.count`.
    internal nonisolated static func snapshotHybridCheckpoint(
        tokens: [Int], cache: [KVCache]
    ) -> HybridCheckpoint? {
        guard isHybridMambaAttention(cache), let kinds = hybridLayerKinds(cache) else {
            return nil
        }

        var layers: [(kind: HybridLayerKind, state: [MLXArray])] = []
        layers.reserveCapacity(cache.count)
        var byteSize = 0
        for (layer, kind) in zip(cache, kinds) {
            if kind == .simple {
                // Mirrors `verifiedSimpleLayers(cache:tokenCount:)`'s own
                // `state.count == kvCacheSimpleStateElementCount && offset == tokenCount` requirements: a
                // `KVCacheSimple` layer whose `offset` doesn't match
                // `tokens.count` reflects a DIFFERENT token position than
                // what this snapshot claims to cover -- e.g. a corrupted or
                // inconsistent cache passed in by a caller bug -- and
                // capturing it anyway would produce a checkpoint that
                // silently does not correspond to `tokens`, restorable only
                // into a wrong future round. Refuse rather than guess.
                guard verifySimpleKVCacheState(layer, tokenCount: tokens.count) != nil
                else { return nil }
            }
            let owned = layer.state.map { ownedCopy(of: $0) }
            byteSize += owned.reduce(0) { $0 + $1.nbytes }
            layers.append((kind: kind, state: owned))
        }
        return HybridCheckpoint(tokens: tokens, layers: layers, byteSize: byteSize, lastUsed: 0)
    }

    /// Rebuilds a fresh, PRIVATE `[KVCache]` from a whole-stack
    /// ``HybridCheckpoint``, one fresh `KVCacheSimple`/`MambaCache` instance
    /// per layer (per ``HybridCheckpoint/layers``' `kind` tag), each seeded
    /// with an owned COPY of the checkpoint's stored state.
    ///
    /// Re-copies (rather than handing the checkpoint's own arrays straight
    /// to the restored cache) because generation mutates its cache in
    /// place: a later round must not clobber an earlier stored checkpoint's
    /// tensors -- exactly the same independence guarantee
    /// `PromptCache.assemble(chunks:layerCount:)` provides for the ordinary
    /// chunk-store path.
    ///
    /// An EMPTY `.mamba` entry (an untouched layer's snapshot -- see
    /// `snapshotHybridCheckpoint(tokens:cache:)`'s doc comment) is special-
    /// cased: a fresh `MambaCache` already starts in that same untouched
    /// state (`ArraysCache.init(size:leftPadding:)` fills its backing array
    /// with `nil`s), so this leaves it alone rather than assigning `.state
    /// = []`. Assigning would actually be WORSE than a no-op: `MambaCache`
    /// is fixed-size (`ArraysCache(size: 2, ...)`), and `ArraysCache.state`'s
    /// setter does `cache = newValue.map { $0 as MLXArray? }` -- replacing
    /// the fixed-size `[nil, nil]` backing array with an empty one entirely,
    /// which would make the layer's next `cache?[0]`/`cache?[1]` subscript
    /// access (in `Qwen35GatedDeltaNet`/`Qwen3NextGatedDeltaNet.callAsFunction`)
    /// an out-of-bounds crash instead of a graceful "no tokens yet" read.
    ///
    /// - Parameter checkpoint: The checkpoint to restore.
    /// - Returns: One freshly constructed `KVCacheSimple`/`MambaCache` per
    ///   layer, in the same order as `checkpoint.layers`, each seeded with
    ///   an owned copy of that layer's checkpointed state.
    internal nonisolated static func restoreHybridCheckpoint(_ checkpoint: HybridCheckpoint) -> [KVCache] {
        checkpoint.layers.map { entry in
            let restoredState = entry.state.map { ownedCopy(of: $0) }
            switch entry.kind {
            case .simple:
                let cache = KVCacheSimple()
                cache.state = restoredState
                return cache
            case .mamba:
                let cache = MambaCache()
                if !restoredState.isEmpty {
                    cache.state = restoredState
                }
                return cache
            }
        }
    }

    /// The ground-truth offset to compute a round's generated-token advance
    /// from, for WHICHEVER cache shape `cache` actually is -- shared by both
    /// `Executor.commitPromptCache` overloads (`MLXLanguageModel.swift`).
    ///
    /// For a pure-attention (``isChunkable(_:)``) stack, this is simply
    /// `cache.first?.offset`, exactly as before. For a genuine hybrid stack
    /// (``isHybridMambaAttention(_:)``), it is NOT `cache.first?.offset`:
    /// layer 0 is a `MambaCache` whenever `fullAttentionInterval > 1` (the
    /// default for both `Qwen35Model`/`Qwen3NextModel`), and neither
    /// `Libraries/MLXLLM/Models/Qwen35.swift` nor `Qwen3Next.swift` ever
    /// assigns to a Mamba layer's `offset` (unlike, e.g., `FalconH1.swift`'s
    /// hybrid attention/recurrent layer, which DOES advance its cache's
    /// `offset` -- this is a per-model-family fact about `Qwen35Model`/
    /// `Qwen3NextModel` specifically, not a blanket guarantee of
    /// `MambaCache`/`ArraysCache` itself). So this reads the offset of the
    /// FIRST exactly-`KVCacheSimple` layer instead, which always exists (by
    /// ``isHybridMambaAttention(_:)``'s own definition) and always advances
    /// normally.
    ///
    /// KNOWN, ACCEPTABLE DEGRADATION: an EOS-terminated round that needs a
    /// 1-token trim before storing (`Executor.trimCacheIfValid`) will fail
    /// to store for a hybrid stack too, exactly like any other
    /// non-trimmable cache shape today (e.g. `RotatingKVCache`) -- not a
    /// new gap this mechanism introduces. `canTrimPromptCache(_:)` requires
    /// EVERY layer's `isTrimmable`, and `MambaCache`/`ArraysCache.isTrimmable`
    /// is `false` (the inherited `BaseKVCache` default, never overridden),
    /// so a hybrid round needing that trim is simply not stored -- the next
    /// round falls back to a full reprocess from wherever the last valid
    /// checkpoint left off, same as any dropped round.
    ///
    /// - Parameter cache: The cache stack this round generated with.
    /// - Returns: The offset to treat as ground truth, or `nil` when `cache`
    ///   is neither a pure-attention nor a genuine hybrid stack (an unknown
    ///   shape this function refuses to reason about -- the caller should
    ///   drop the round rather than guess).
    internal nonisolated static func cacheAdvanceOffset(_ cache: [KVCache]) -> Int? {
        if isChunkable(cache) {
            return cache.first?.offset
        }
        if isHybridMambaAttention(cache), let kinds = hybridLayerKinds(cache) {
            guard let simpleIndex = kinds.firstIndex(of: .simple) else { return nil }
            return cache[simpleIndex].offset
        }
        return nil
    }

    /// Verifies that every layer in `cache` is a chunkable, fully-offset
    /// `KVCacheSimple` layer -- shared by `sliceChunks` and `sliceTailChunk`,
    /// since both need the identical verification before slicing any span of
    /// `cache`'s tensors.
    ///
    /// Requires every layer to be a `KVCacheSimple` whose `offset` exactly
    /// matches `tokenCount` and whose `state` is the expected `[keys,
    /// values]` pair -- mirroring `isTrimmable`'s degradation pattern for
    /// `RotatingKVCache`/`ChunkedKVCache`, this function returns `nil` rather
    /// than guess at a layer type it can't safely slice.
    ///
    /// - Parameters:
    ///   - cache: The cache stack to verify, one entry per model layer.
    ///   - tokenCount: The token count every layer's `offset` must exactly
    ///     match.
    /// - Returns: One verified `KVCacheSimple` per layer, in the same order
    ///   as `cache`; `nil` if `cache` is empty or any layer isn't a verified,
    ///   fully-offset `KVCacheSimple`.
    private static func verifiedSimpleLayers(
        cache: [KVCache], tokenCount: Int
    ) -> [KVCacheSimple]? {
        guard !cache.isEmpty else { return nil }

        var simpleLayers: [KVCacheSimple] = []
        simpleLayers.reserveCapacity(cache.count)
        for layer in cache {
            guard
                let simple = verifySimpleLayer(layer, tokenCount: tokenCount)
            else { return nil }
            simpleLayers.append(simple)
        }
        return simpleLayers
    }

    /// Verifies a cache layer's `state` is exactly `[keys, values]` and its
    /// `offset` exactly matches `tokenCount`, WITHOUT requiring `layer` to
    /// be exactly `KVCacheSimple` (as opposed to a subclass) -- callers that
    /// need that additional exact-type guarantee (see
    /// ``verifySimpleLayer(_:tokenCount:)``) check it themselves before
    /// calling this.
    ///
    /// Shared by ``snapshotHybridCheckpoint(tokens:cache:)`` (whose
    /// `KVCacheSimple` layers, in a genuine hybrid stack, are never
    /// `ChunkedKVCache`, so the exact-type check ``verifySimpleLayer(_:tokenCount:)``
    /// needs doesn't apply there) and ``verifySimpleLayer(_:tokenCount:)``
    /// itself (which layers on the exact-type check before delegating the
    /// rest of the verification here).
    ///
    /// - Parameters:
    ///   - layer: The cache layer to verify.
    ///   - tokenCount: The token count `layer`'s `offset` must exactly
    ///     match.
    /// - Returns: `layer` downcast to `KVCacheSimple` if its `offset`
    ///   matches `tokenCount` and its `state` is the expected `[keys,
    ///   values]` pair; `nil` otherwise.
    private static func verifySimpleKVCacheState(
        _ layer: KVCache, tokenCount: Int
    ) -> KVCacheSimple? {
        guard layer.state.count == kvCacheSimpleStateElementCount,
            let simple = layer as? KVCacheSimple,
            simple.offset == tokenCount
        else { return nil }
        return simple
    }

    /// Verifies a single cache layer is a chunkable, fully-offset
    /// `KVCacheSimple` -- the per-layer check `verifiedSimpleLayers` applies
    /// to every entry in `cache`.
    ///
    /// `as? KVCacheSimple` alone is not enough: `ChunkedKVCache` is a
    /// SUBCLASS of `KVCacheSimple` (see `Libraries/MLXLMCommon/KVCache.swift`)
    /// that overrides `update`/`trim`/`copy`/`metaState` but not
    /// `state`/`isTrimmable`. Once its `maybeTrimFront()` has physically
    /// trimmed `keys`/`values` down to its own chunk size while `offset`
    /// keeps tracking the full logical token position, the inherited
    /// `state` getter's `offset == keys.dim(2)` branch no longer holds,
    /// and slicing against it would use the wrong physical extent. An
    /// exact dynamic-type check excludes every such subclass, matching
    /// this function's own "not chunkable" degradation for any cache
    /// shape it can't safely reason about. That exact-type check is this
    /// function's own addition on top of ``verifySimpleKVCacheState(_:tokenCount:)``,
    /// which the rest of the verification is delegated to.
    ///
    /// - Parameters:
    ///   - layer: The cache layer to verify.
    ///   - tokenCount: The token count `layer`'s `offset` must exactly
    ///     match.
    /// - Returns: `layer` downcast to `KVCacheSimple` if it's exactly a
    ///   `KVCacheSimple` whose `offset` matches `tokenCount` and whose
    ///   `state` is the expected `[keys, values]` pair; `nil` otherwise.
    private static func verifySimpleLayer(
        _ layer: KVCache, tokenCount: Int
    ) -> KVCacheSimple? {
        guard type(of: layer) == KVCacheSimple.self else { return nil }
        return verifySimpleKVCacheState(layer, tokenCount: tokenCount)
    }

    /// Slices every layer in `simpleLayers` over the shared `[start, end)`
    /// token span and accumulates their owned tensors and combined byte
    /// footprint -- the per-chunk inner loop shared by ``sliceChunks(tokens:cache:chunkSize:)``
    /// (called once per fixed `chunkSize` span) and
    /// ``sliceTailChunk(tokens:cache:chunkSize:parentKey:)`` (called once
    /// over the trailing partial span); the two differ only in which
    /// `[start, end)` span they pass.
    ///
    /// - Parameters:
    ///   - simpleLayers: The verified layer stack to slice (see
    ///     ``verifiedSimpleLayers(cache:tokenCount:)``).
    ///   - start: The span's starting token offset (inclusive).
    ///   - end: The span's ending token offset (exclusive).
    /// - Returns: One owned `(keys, values)` pair per layer, in the same
    ///   order as `simpleLayers`, plus their combined retained byte
    ///   footprint (`MLXArray.nbytes`).
    private static func sliceLayers(
        simpleLayers: [KVCacheSimple], start: Int, end: Int
    ) -> (layers: [(keys: MLXArray, values: MLXArray)], byteSize: Int) {
        var layers: [(keys: MLXArray, values: MLXArray)] = []
        layers.reserveCapacity(simpleLayers.count)
        var byteSize = 0
        for simple in simpleLayers {
            let sliced = sliceChunkLayer(layer: simple, start: start, end: end)
            byteSize += sliced.byteSize
            layers.append((keys: sliced.keys, values: sliced.values))
        }
        return (layers, byteSize)
    }

    /// Builds a freshly sliced ``StoredChunk``, stamped with `lastUsed: 0`
    /// (see ``StoredChunk/lastUsed``'s doc comment for why a pure slicing
    /// function never stamps a real recency value) -- the identical field
    /// wiring ``sliceChunks(tokens:cache:chunkSize:)`` and
    /// ``sliceTailChunk(tokens:cache:chunkSize:parentKey:)`` both need,
    /// differing only in which tokens/parent-key/sliced-layers they pass.
    ///
    /// - Parameters:
    ///   - tokens: This chunk's token ids (a full `chunkSize`-token span for
    ///     `sliceChunks`, or the trailing partial span for `sliceTailChunk`).
    ///   - sliced: The layers/byte-footprint ``sliceLayers(simpleLayers:start:end:)``
    ///     already computed for this span.
    ///   - parentKey: This chunk's hash-chain attachment point.
    /// - Returns: The freshly built `StoredChunk`, keyed via
    ///   ``chunkKey(parentKey:tokens:)``.
    private static func makeStoredChunk(
        tokens: [Int],
        sliced: (layers: [(keys: MLXArray, values: MLXArray)], byteSize: Int),
        parentKey: Int
    ) -> StoredChunk {
        let key = chunkKey(parentKey: parentKey, tokens: tokens)
        return StoredChunk(
            tokens: tokens, layers: sliced.layers, parentKey: parentKey, chunkKey: key,
            byteSize: sliced.byteSize, lastUsed: 0)
    }

    /// Cuts `tokens`/`cache` into fixed-size, non-overlapping token-range
    /// chunks, one covering each full `chunkSize`-token span; any trailing
    /// partial span (fewer than `chunkSize` tokens) is left uncovered here --
    /// see ``sliceTailChunk(tokens:cache:chunkSize:parentKey:)``, which slices
    /// exactly that remainder into its own variable-length chunk.
    ///
    /// Requires every layer in `cache` to be a verified, fully-offset
    /// `KVCacheSimple` (see ``verifiedSimpleLayers(cache:tokenCount:)``);
    /// returns `nil` rather than guess at a layer type it can't safely slice.
    ///
    /// Every returned chunk's tensors are owned, evaluated copies (see
    /// ``ownedCopy(of:)``), and `chunkKey`s form a hash chain (see
    /// ``chunkKey(parentKey:tokens:)``) so identical token prefixes -- even
    /// sliced from different `[KVCache]` instances -- produce identical keys.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence `cache` currently reflects.
    ///   - cache: The cache stack to slice, one entry per model layer.
    ///   - chunkSize: How many tokens each stored chunk covers.
    /// - Returns: One `StoredChunk` per full `chunkSize`-token span (possibly
    ///   empty, if `tokens.count < chunkSize`), or `nil` if any layer isn't a
    ///   verified, fully-offset `KVCacheSimple`.
    internal nonisolated static func sliceChunks(
        tokens: [Int], cache: [KVCache], chunkSize: Int
    ) -> [StoredChunk]? {
        guard chunkSize > 0,
            let simpleLayers = verifiedSimpleLayers(cache: cache, tokenCount: tokens.count)
        else { return nil }

        let chunkCount = tokens.count / chunkSize
        var parentKey = rootChunkKey
        var chunks: [StoredChunk] = []
        chunks.reserveCapacity(chunkCount)

        for index in 0 ..< chunkCount {
            let start = index * chunkSize
            let end = start + chunkSize
            let chunkTokens = Array(tokens[start ..< end])

            let sliced = sliceLayers(simpleLayers: simpleLayers, start: start, end: end)
            let chunk = makeStoredChunk(tokens: chunkTokens, sliced: sliced, parentKey: parentKey)
            chunks.append(chunk)
            parentKey = chunk.chunkKey
        }

        return chunks
    }

    /// Slices `tokens`/`cache`'s trailing PARTIAL span -- the `tokens.count %
    /// chunkSize` tokens left over after the last full `chunkSize`-token
    /// chunk (exactly the remainder ``sliceChunks(tokens:cache:chunkSize:)``
    /// leaves uncovered) -- into its own variable-length "tail" `StoredChunk`,
    /// an SGLang-style radix-node shorter than `chunkSize`.
    ///
    /// Keyed exactly like any other chunk in the hash chain (see
    /// ``chunkKey(parentKey:tokens:)``), chained from the caller-supplied
    /// `parentKey` -- typically the last full chunk's `chunkKey`, or
    /// `PromptCache.rootChunkKey` when `tokens.count < chunkSize` (no full
    /// chunks at all). This is what makes a conversation prefix SHORTER than
    /// one full chunk -- previously stored as NOTHING, since `sliceChunks`
    /// alone slices zero chunks for it -- reusable: even a single stored tail
    /// chunk gives a future lookup something to match a longest-common-prefix
    /// against.
    ///
    /// Reuses ``sliceChunkLayer(layer:start:end:)`` -- the tail's tensors are
    /// owned, evaluated copies exactly like any full chunk's (see that
    /// function's own doc comment for the ownership guarantee).
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence `cache` currently reflects.
    ///   - cache: The cache stack to slice, one entry per model layer.
    ///   - chunkSize: How many tokens each FULL chunk covers -- the tail is
    ///     whatever remains after `tokens.count / chunkSize` full chunks.
    ///   - parentKey: This tail's hash-chain attachment point (the preceding
    ///     full chunk's `chunkKey`, or `PromptCache.rootChunkKey`).
    /// - Returns: The trailing partial-span `StoredChunk`, or `nil` when the
    ///   tail span is empty (`tokens.count` is an exact multiple of
    ///   `chunkSize` -- nothing to store, not an error) or `cache`'s layers
    ///   aren't a verified, chunkable `KVCacheSimple` stack (mirrors
    ///   `sliceChunks`'s own degradation).
    internal nonisolated static func sliceTailChunk(
        tokens: [Int], cache: [KVCache], chunkSize: Int, parentKey: Int
    ) -> StoredChunk? {
        guard chunkSize > 0 else { return nil }
        let start = (tokens.count / chunkSize) * chunkSize
        guard start < tokens.count,
            let simpleLayers = verifiedSimpleLayers(cache: cache, tokenCount: tokens.count)
        else { return nil }

        let tailTokens = Array(tokens[start...])
        let sliced = sliceLayers(simpleLayers: simpleLayers, start: start, end: tokens.count)
        return makeStoredChunk(tokens: tailTokens, sliced: sliced, parentKey: parentKey)
    }
}

#endif
#endif

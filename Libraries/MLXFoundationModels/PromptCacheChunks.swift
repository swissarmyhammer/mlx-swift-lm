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
    typealias ChunkKey = Int

    /// One fixed-size token-range slice of a verified `KVCacheSimple` stack, with
    /// its own tensors materialized as OWNED, evaluated, contiguous copies (see
    /// `ownedCopy(of:)` -- the copies never alias the source cache's buffers).
    ///
    /// Chunks form a hash chain via ``parentKey``/``chunkKey`` (see
    /// ``PromptCache/chunkKey(parentKey:tokens:)``): identical token prefixes
    /// produce identical keys, and keys diverge from the first point two token
    /// sequences differ. `Hasher` is per-process seeded, so a `chunkKey` must
    /// never be persisted or compared across process launches.
    struct StoredChunk {
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
    static let rootChunkKey = 0

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
    nonisolated static func chunkKey(parentKey: Int, tokens: [Int]) -> Int {
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
    static func ownedCopy(of array: MLXArray) -> MLXArray {
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

    /// Whether `cache`'s layer composition can EVER participate in this
    /// store's chunk-based reuse -- a structural, round-independent
    /// capability check, unlike ``verifiedSimpleLayers(cache:tokenCount:)``
    /// (which additionally requires every layer's `offset` to match a
    /// SPECIFIC round's `tokenCount`). `true` only when every layer is
    /// exactly `KVCacheSimple` (mirroring `verifiedSimpleLayers`'s own
    /// exact-type rule -- see that function's doc comment for why a plain
    /// `as? KVCacheSimple` isn't enough).
    ///
    /// DECISION (kanban `r9rf5g7`): hybrid Mamba/attention architectures
    /// (Qwen3.6/Qwen3-Next family: `Qwen35Model`/`Qwen35MoEModel` in
    /// `Libraries/MLXLLM/Models/Qwen35.swift`, via
    /// `Qwen35DecoderLayer.isLinear`) build a `[KVCache]` mixing
    /// `KVCacheSimple` (full-attention layers) with `MambaCache`
    /// (Gated-DeltaNet linear-attention layers), so this always returns
    /// `false` for them. That is a PERMANENT architectural fact about the
    /// current chunk-store design, not a temporary gap:
    ///
    /// - Genuine PARTIAL reuse (chunk/cache only the attention layers,
    ///   leave Mamba layers to reprocess fresh every turn) is not simply
    ///   an optimization this store hasn't implemented yet -- it isn't
    ///   expressible in this model family's forward pass at all.
    ///   `Qwen35TextModelInner.callAsFunction`/`Qwen3NextModelInner.callAsFunction`
    ///   thread ONE shared `hiddenStates` tensor through every layer in
    ///   strict sequence, attention and Mamba layers alike, all at the
    ///   SAME sequence length. There is no way to feed a short suffix to
    ///   the attention layers while feeding the Mamba layers something
    ///   else in the same forward call -- a Mamba layer fed only the
    ///   suffix would compute its recurrent update as though those
    ///   tokens were the START of the conversation, silently losing
    ///   every earlier token's contribution to the shared residual
    ///   stream (Mamba layers are interleaved IN that stream, not routed
    ///   around it).
    ///
    /// - Genuine checkpointing of Mamba's recurrent state AT a chunk
    ///   boundary (the way `sliceChunks` retroactively carves a
    ///   FINISHED round's attention K/V tensor into `chunkSize`-aligned
    ///   windows, by plain array indexing) doesn't work either.
    ///   Attention's key/value tensors are append-only and addressable
    ///   by absolute token position, so any earlier boundary can be
    ///   recovered after the fact from the round's final tensor. A
    ///   Mamba layer's `state` (conv buffer + Gated-DeltaNet recurrent
    ///   state -- see `ArraysCache`/`MambaCache` in
    ///   `MLXLMCommon/KVCache.swift`) is a running, COLLAPSED summary
    ///   with no per-position history: the only way to obtain a valid
    ///   checkpoint at a specific `chunkSize`-aligned boundary is for a
    ///   REAL forward pass to actually halt there. That requires
    ///   restructuring `MLXLMCommon`'s prefill loop to force every
    ///   generation call -- not just hybrid models' -- to advance in
    ///   fixed `chunkSize` increments, a change whose blast radius
    ///   reaches far past `PromptCache`/`PromptCacheChunks`/`Qwen35.swift`
    ///   and is disproportionate to this store's scope.
    ///
    /// So the chosen fix is not a change to how chunking works, but
    /// making "this model will never report cache reuse" an explicit,
    /// queryable fact (`MLXLanguageModel.supportsPromptCacheReuse(model:parameters:)`)
    /// instead of leaving a caller to infer it from
    /// `LanguageModelSession.Usage.input.cachedTokenCount` staying `0`
    /// forever with no other signal.
    ///
    /// - Parameter cache: The cache stack to check, one entry per model layer.
    /// - Returns: `true` when every layer is exactly `KVCacheSimple`;
    ///   `false` for an empty stack or any non-`KVCacheSimple`/subclass/
    ///   mixed-type layer.
    nonisolated static func isChunkable(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { type(of: $0) == KVCacheSimple.self }
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
            // `as? KVCacheSimple` alone is not enough: `ChunkedKVCache` is a
            // SUBCLASS of `KVCacheSimple` (see `Libraries/MLXLMCommon/KVCache.swift`)
            // that overrides `update`/`trim`/`copy`/`metaState` but not
            // `state`/`isTrimmable`. Once its `maybeTrimFront()` has physically
            // trimmed `keys`/`values` down to its own chunk size while `offset`
            // keeps tracking the full logical token position, the inherited
            // `state` getter's `offset == keys.dim(2)` branch no longer holds,
            // and slicing against it would use the wrong physical extent. An
            // exact dynamic-type check excludes every such subclass, matching
            // this function's own "not chunkable" degradation for any cache
            // shape it can't safely reason about.
            guard type(of: layer) == KVCacheSimple.self,
                let simple = layer as? KVCacheSimple,
                simple.offset == tokenCount,
                simple.state.count == 2
            else { return nil }
            simpleLayers.append(simple)
        }
        return simpleLayers
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
    nonisolated static func sliceChunks(
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
            let key = chunkKey(parentKey: parentKey, tokens: chunkTokens)
            chunks.append(
                StoredChunk(
                    tokens: chunkTokens, layers: sliced.layers, parentKey: parentKey,
                    chunkKey: key, byteSize: sliced.byteSize, lastUsed: 0))
            parentKey = key
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
    nonisolated static func sliceTailChunk(
        tokens: [Int], cache: [KVCache], chunkSize: Int, parentKey: Int
    ) -> StoredChunk? {
        guard chunkSize > 0 else { return nil }
        let start = (tokens.count / chunkSize) * chunkSize
        guard start < tokens.count,
            let simpleLayers = verifiedSimpleLayers(cache: cache, tokenCount: tokens.count)
        else { return nil }

        let tailTokens = Array(tokens[start...])
        let sliced = sliceLayers(simpleLayers: simpleLayers, start: start, end: tokens.count)
        let key = chunkKey(parentKey: parentKey, tokens: tailTokens)
        return StoredChunk(
            tokens: tailTokens, layers: sliced.layers, parentKey: parentKey, chunkKey: key,
            byteSize: sliced.byteSize, lastUsed: 0)
    }
}

#endif
#endif

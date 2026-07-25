// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// Real-weights integration tests proving `MiniMaxM3KVCache` (the custom
/// sparse-attention KV cache, kanban `^8dbc476`) is correct across multi-turn
/// generation on `mlx-community/MiniMax-M3-4bit`.
///
/// Scope note: this suite is entirely about `MLXLMCommon.ChatSession`/`KVCache`
/// -- the ordinary in-process, in-session cache every model uses across
/// `respond()` calls. It has nothing to do with `MLXFoundationModels`'
/// cross-session `PromptCache` reuse system, which `^8dbc476` documented
/// `MiniMaxM3KVCache` deliberately does NOT participate in (`PromptCache
/// .isChunkable`/`.isHybridMambaAttention` don't recognize it, so
/// `MLXLanguageModel.supportsPromptCacheReuse` correctly reports `false` for
/// M3). Nothing here exercises or asserts on that system.
///
/// Shares `MiniMaxM3CoherenceIntegrationTests`' gating helpers
/// (`minimaxM3RequiredMemoryBytes`, `resolveMiniMaxM3Configuration()`,
/// `checkpointIsAvailable(_:)`) and its skip-gracefully-rather-than-fail
/// pattern for insufficient memory, a missing checkpoint override, or a
/// checkpoint load failure. Run explicitly via:
/// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/MiniMaxM3CacheIntegrationTests`
@Suite(.serialized, .timeLimit(.minutes(240)))
struct MiniMaxM3CacheIntegrationTests {

    /// Loads the MiniMax-M3 container, or returns `nil` after printing a skip
    /// message -- mirrors `MiniMaxM3CoherenceIntegrationTests.minimax_m3()`'s
    /// gating exactly so both suites skip under identical conditions.
    private func loadContainerOrSkip(testName: String) async -> LLModelContainer? {
        guard ProcessInfo.processInfo.physicalMemory >= minimaxM3RequiredMemoryBytes else {
            print(
                "Skipping \(testName): physical memory "
                    + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                    + "\(minimaxM3RequiredMemoryBytes) bytes")
            return nil
        }

        let configuration = resolveMiniMaxM3Configuration()
        guard checkpointIsAvailable(configuration) else {
            print("Skipping \(testName): checkpoint not found at \(configuration.name)")
            return nil
        }

        do {
            return try await models.vlmContainer(for: configuration)
        } catch {
            print("Skipping \(testName): failed to load checkpoint \(configuration.name): \(error)")
            return nil
        }
    }

    @Test
    func saveAndRestoreCacheContinuesWithFactRecall() async throws {
        guard let container = await loadContainerOrSkip(testName: "saveAndRestoreCacheContinuesWithFactRecall")
        else {
            return
        }

        let sessionA = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 30, temperature: 0))
        _ = try await sessionA.respond(to: "Remember this number: 8214. Just acknowledge briefly.")

        let cacheURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }
        try await sessionA.saveCache(to: cacheURL)

        let (restoredCache, _) = try loadPromptCache(url: cacheURL)
        let sessionB = ChatSession(
            container, cache: restoredCache,
            generateParameters: GenerateParameters(maxTokens: 30, temperature: 0))

        var response = ""
        for try await token in sessionB.streamResponse(to: "What number did I just ask you to remember?")
        {
            response += token
        }

        #expect(
            response.contains("8214"),
            """
            expected the cache-restored session to recall '8214' from the serialized/restored \
            MiniMaxM3KVCache, got: \(response)
            """
        )
    }

    @Test
    func incrementalCacheMatchesFreshFullContextRebuild() async throws {
        guard let container = await loadContainerOrSkip(testName: "incrementalCacheMatchesFreshFullContextRebuild")
        else {
            return
        }

        let greedyParameters = GenerateParameters(maxTokens: 30, temperature: 0)
        let turn1Prompt = "Say 'ready' and nothing else."
        let turn2Prompt = "Now say 'done' and nothing else."

        // Path A: one session, turn 2 reuses turn 1's live, in-place
        // MiniMaxM3KVCache via the session's normal incremental-decode path.
        let incrementalSession = ChatSession(container, generateParameters: greedyParameters)
        let turn1Output = try await incrementalSession.respond(to: turn1Prompt)
        let incrementalOutput = try await incrementalSession.respond(to: turn2Prompt)

        // Path B: a brand-new session seeded with the exact same growing
        // transcript via the `history:` initializer, which forces a full
        // fresh prefill of the whole accumulated transcript with no
        // incrementally-carried cache state (see `ChatSession.streamMap`'s
        // `.history` case).
        let freshSession = ChatSession(
            container,
            history: [
                .user(content: turn1Prompt),
                .assistant(content: turn1Output),
            ],
            generateParameters: greedyParameters)
        let freshOutput = try await freshSession.respond(to: turn2Prompt)

        #expect(
            incrementalOutput == freshOutput,
            """
            incremental MiniMaxM3KVCache decode (\(incrementalOutput)) diverged from a fresh, \
            full-context rebuild of the identical transcript (\(freshOutput)) -- the sparse-\
            attention cache's incremental decode path produced different generated tokens than \
            a from-scratch recomputation
            """
        )
    }
}

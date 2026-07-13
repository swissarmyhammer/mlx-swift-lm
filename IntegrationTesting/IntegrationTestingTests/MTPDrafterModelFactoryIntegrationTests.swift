// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLXLMCommon
import MLXVLM
import Testing

// MARK: - Factory load (gated on checkpoint presence)

/// The 31B `gemma-4-31B-it-assistant-bf16` drafter checkpoint this test
/// requires -- a ~33GB download not fetched by default. `.enabled(if:)`
/// reports SKIPPED (not FAILED) when it's absent, matching the
/// `MLX_RUN_VLM_INTEGRATION` idiom used elsewhere (see
/// `VisionIntegrationTests`, `PromptCacheMultimodalBoundaryTests`).
///
/// To exclude this test from an `xcodebuild test-without-building`
/// invocation by name, use
/// `-skip-testing:"IntegrationTestingTests/testMTPDrafterFactoryLoadFromDirectoryWhenCheckpointPresent()"`
/// -- experimentally verified: for this free-function Swift Testing test
/// (declared at file scope, not inside a `@Suite` struct), the identifier's
/// last path segment MUST include the trailing `()`; the bare name without
/// parens matches zero tests. See the header comment in
/// `MTPRung4TokenParityTests.swift` for the full verified `-skip-testing`
/// invocation covering every checkpoint-guarded suite together.
@Test(
    .enabled(
        if: hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16") != nil)
)
func testMTPDrafterFactoryLoadFromDirectoryWhenCheckpointPresent() async throws {
    // The `.enabled(if:)` trait above already guarantees the checkpoint is
    // present before this test body runs.
    let snapshot = try #require(
        hfSnapshotDir(modelId: "mlx-community/gemma-4-31B-it-assistant-bf16"),
        "31B-assistant-bf16 checkpoint unexpectedly missing despite .enabled(if:) gate")

    await Gemma4AssistantRegistration.register()
    let factory = MTPDrafterModelFactory.shared

    let container = try await factory.loadContainer(
        from: snapshot, using: NoOpTokenizerLoader()
    )
    let isDrafter = await container.perform { ctx in
        ctx.model is Gemma4AssistantDraftModel
    }
    #expect(isDrafter)
}

// MARK: - Helpers

/// A `TokenizerLoader` placeholder. `MTPDrafterModelFactory` ignores the
/// loader (drafters borrow their target's tokenizer), but the protocol
/// requires a non-optional argument.
private final class NoOpTokenizerLoader: TokenizerLoader {
    func load(from url: URL) async throws -> any Tokenizer {
        fatalError("MTPDrafterModelFactory must not call tokenizer load")
    }
}

import Foundation
import MLXLMCommon
import MLXLMTestSupport
import Testing

@Suite struct ResolveTests {

    @Test func nilTokenizerSourceUsesModelDirectory() async throws {
        let downloader = RecordingDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "abc123", tokenizerSource: nil)

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // Only one download call — the model itself.
        #expect(downloader.calls.count == 1)
        #expect(downloader.calls[0].id == "org/model")
        #expect(downloader.calls[0].revision == "abc123")
        #expect(downloader.calls[0].patterns.contains("*.jinja"))

        // No separate tokenizer download, so both point to the model directory.
        #expect(resolved.modelDirectory == resolved.tokenizerDirectory)
    }

    @Test func tokenizerSourceIDWithoutRevisionPassesNil() async throws {
        let downloader = RecordingDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "abc123",
            tokenizerSource: .id("org/tokenizer"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.count == 2)

        // Model download uses model revision.
        #expect(downloader.calls[0].id == "org/model")
        #expect(downloader.calls[0].revision == "abc123")

        // Tokenizer download uses nil revision (provider default).
        #expect(downloader.calls[1].id == "org/tokenizer")
        #expect(downloader.calls[1].revision == nil)
        #expect(downloader.calls[1].patterns.contains("*.jinja"))

        // Model and tokenizer come from different repos, so directories differ.
        #expect(resolved.modelDirectory != resolved.tokenizerDirectory)
    }

    @Test func tokenizerSourceIDWithExplicitRevision() async throws {
        let downloader = RecordingDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "v1.0",
            tokenizerSource: .id("org/tokenizer", revision: "tok-v2"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.count == 2)

        #expect(downloader.calls[0].id == "org/model")
        #expect(downloader.calls[0].revision == "v1.0")

        #expect(downloader.calls[1].id == "org/tokenizer")
        #expect(downloader.calls[1].revision == "tok-v2")
        #expect(downloader.calls[1].patterns.contains("*.jinja"))

        // Model and tokenizer come from different repos, so directories differ.
        #expect(resolved.modelDirectory != resolved.tokenizerDirectory)
    }

    @Test func localDirectorySkipsDownloader() async throws {
        let downloader = RecordingDownloader()
        let localDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(directory: localDir)

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // No downloads should occur for a local directory.
        #expect(downloader.calls.isEmpty)

        // Both directories point to the local path.
        #expect(resolved.modelDirectory == localDir)
        #expect(resolved.tokenizerDirectory == localDir)
    }

    @Test func remoteModelSubdirectoryScopesDownloadAndResolvedDirectory() async throws {
        let downloader = RecordingDownloader()
        let config = ModelConfiguration(
            id: "org/multi-variant-model", modelSubdirectory: "4bit")

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        let call = try #require(downloader.calls.first)
        #expect(call.patterns == ["4bit/*.safetensors", "4bit/*.json", "4bit/*.jinja"])
        #expect(resolved.modelDirectory.lastPathComponent == "4bit")
        #expect(resolved.tokenizerDirectory == resolved.modelDirectory)
    }

    @Test func localModelSubdirectoryResolvesWithoutDownload() async throws {
        let downloader = RecordingDownloader()
        let root = URL(filePath: "/local/aggregate")
        let config = ModelConfiguration(directory: root, modelSubdirectory: "mlx/4bit")

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.isEmpty)
        #expect(resolved.modelDirectory.path == "/local/aggregate/mlx/4bit")
        #expect(try config.modelDirectory == resolved.modelDirectory)
    }

    @Test(
        arguments: [
            "", "/absolute", "../bf16", "./bf16", "variants/../bf16",
            "variants/./bf16", "variants//bf16", "variants/", "\\\\server\\share",
            "variants\\bf16", "*", "variants/?", "variants/[ab]", "variants/{a,b}",
            "variants\n4bit", "variants\u{0000}4bit",
        ])
    func unsafeModelSubdirectoryIsRejectedBeforeDownload(_ subdirectory: String) async {
        let downloader = RecordingDownloader()
        let config = ModelConfiguration(id: "org/model", modelSubdirectory: subdirectory)

        await #expect(
            throws: ModelConfiguration.DirectoryError.invalidModelSubdirectory(subdirectory)
        ) {
            try await resolve(
                configuration: config, from: downloader,
                useLatest: false, progressHandler: { _ in })
        }
        #expect(downloader.calls.isEmpty)
    }

    @Test(
        arguments: [
            ("4bit", "/local/aggregate/4bit"),
            ("mlx/4bit", "/local/aggregate/mlx/4bit"),
            ("variants/model v1", "/local/aggregate/variants/model v1"),
            ("variants/.internal", "/local/aggregate/variants/.internal"),
            ("variants/%2e%2e", "/local/aggregate/variants/%2e%2e"),
        ])
    func validModelSubdirectoryRemainsLiteral(
        _ subdirectory: String, _ expectedPath: String
    ) throws {
        let root = URL(filePath: "/local/aggregate", directoryHint: .isDirectory)
        let config = ModelConfiguration(directory: root, modelSubdirectory: subdirectory)

        #expect(try config.normalizedModelSubdirectory == subdirectory)
        #expect(try config.modelDirectory.path == expectedPath)
    }

    @Test func localDirectoryWithRemoteTokenizerSource() async throws {
        let downloader = RecordingDownloader()
        let localDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(
            directory: localDir,
            tokenizerSource: .id("org/tokenizer", revision: "v3"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // Only the tokenizer is downloaded; the model directory is local.
        #expect(downloader.calls.count == 1)
        #expect(downloader.calls[0].id == "org/tokenizer")
        #expect(downloader.calls[0].revision == "v3")
        #expect(downloader.calls[0].patterns.contains("*.jinja"))

        #expect(resolved.modelDirectory == localDir)
        #expect(resolved.tokenizerDirectory != localDir)
    }

    @Test func localConfigurationExposesResolvedDirectories() throws {
        let modelDir = URL(filePath: "/local/org/model")
        let tokenizerDir = URL(filePath: "/local/org/tokenizer")
        let config = ModelConfiguration(
            directory: modelDir,
            tokenizerSource: .directory(tokenizerDir))

        #expect(try config.modelDirectory == modelDir)
        #expect(try config.tokenizerDirectory == tokenizerDir)
    }

    @Test func tokenizerDirectoryFallsBackToModelDirectory() throws {
        let modelDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(directory: modelDir)

        #expect(try config.modelDirectory == modelDir)
        #expect(try config.tokenizerDirectory == modelDir)
    }

    @Test func unresolvedRemoteConfigurationThrowsForDirectories() {
        let config = ModelConfiguration(
            id: "org/model",
            tokenizerSource: .id("org/tokenizer"))

        do {
            _ = try config.modelDirectory
            Issue.record("Expected modelDirectory to throw for unresolved remote config")
        } catch let error as ModelConfiguration.DirectoryError {
            #expect(error == .unresolvedModelDirectory("org/model"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        do {
            _ = try config.tokenizerDirectory
            Issue.record("Expected tokenizerDirectory to throw for unresolved remote tokenizer")
        } catch let error as ModelConfiguration.DirectoryError {
            #expect(error == .unresolvedTokenizerDirectory("org/tokenizer"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

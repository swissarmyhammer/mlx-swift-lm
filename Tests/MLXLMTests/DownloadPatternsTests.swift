// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXLMTestSupport
import Testing

/// The file patterns that a download of a model fetches: the configuration's own, those of its
/// registry entry, or the defaults.
@Suite struct DownloadPatternsTests {

    /// The patterns of a drafter that ships one weight file beside its target's.
    private static let headPatterns = ["mtp.safetensors", "config.json"]

    /// The default patterns, which fetch the weights, the configuration and the tokenizer.
    private static let defaultPatterns = ["*.safetensors", "*.json", "*.jinja"]

    /// The repository of the registered drafter.
    private static let drafterID = "org/drafter"

    /// The revision a caller pins.
    private static let pinnedRevision = "0123abcd"

    /// The configuration of `id` with `patterns`.
    private static func configuration(
        _ id: String = drafterID, revision: String = "main", modelSubdirectory: String? = nil,
        patterns: [String]?
    ) -> ModelConfiguration {
        var configuration = ModelConfiguration(
            id: id, revision: revision, modelSubdirectory: modelSubdirectory)
        configuration.downloadPatterns = patterns
        return configuration
    }

    /// The patterns of the one download that a factory load of `configuration` requests, with a
    /// registry that holds `registered`.
    ///
    /// The downloader downloads nothing, thus the load fails after the download, when it reads
    /// `config.json`.
    private static func requestedPatterns(
        loading configuration: ModelConfiguration, registered: [ModelConfiguration]
    ) async throws -> [String] {
        let downloader = RecordingDownloader()
        let factory = MTPDrafterModelFactory(
            typeRegistry: ModelTypeRegistry(),
            modelRegistry: AbstractModelRegistry(modelConfigurations: registered))
        await #expect(throws: ModelFactoryError.self) {
            _ = try await factory.loadContainer(
                from: downloader, using: UnusedTokenizerLoader(), configuration: configuration)
        }
        let call = try #require(downloader.calls.first)
        #expect(downloader.calls.count == 1)
        #expect(call.id == configuration.name)
        return call.patterns
    }

    @Test func configurationPatternsReplaceTheDefaults() async throws {
        let downloader = RecordingDownloader()

        _ = try await resolve(
            configuration: Self.configuration(patterns: Self.headPatterns), from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.map(\.patterns) == [Self.headPatterns])
    }

    @Test func configurationPatternsTakeTheModelSubdirectory() async throws {
        let downloader = RecordingDownloader()

        _ = try await resolve(
            configuration: Self.configuration(
                modelSubdirectory: "4bit", patterns: Self.headPatterns),
            from: downloader, useLatest: false, progressHandler: { _ in })

        #expect(
            downloader.calls.map(\.patterns) == [["4bit/mtp.safetensors", "4bit/config.json"]])
    }

    @Test func noPatternsFetchTheDefaults() async throws {
        let downloader = RecordingDownloader()

        _ = try await resolve(
            configuration: Self.configuration(patterns: nil), from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.map(\.patterns) == [Self.defaultPatterns])
    }

    @Test func patternsTakePartInEquality() {
        #expect(
            Self.configuration(patterns: Self.headPatterns)
                != Self.configuration(patterns: nil))
        #expect(
            Self.configuration(patterns: Self.headPatterns)
                == Self.configuration(patterns: Self.headPatterns))
    }

    @Test func pinnedConfigurationTakesThePatternsOfItsRegistryEntry() async throws {
        let patterns = try await Self.requestedPatterns(
            loading: Self.configuration(revision: Self.pinnedRevision, patterns: nil),
            registered: [Self.configuration(patterns: Self.headPatterns)])

        #expect(patterns == Self.headPatterns)
    }

    @Test func configurationPatternsWinOverTheRegistryEntry() async throws {
        let own = ["*.safetensors", "config.json"]
        let patterns = try await Self.requestedPatterns(
            loading: Self.configuration(patterns: own),
            registered: [Self.configuration(patterns: Self.headPatterns)])

        #expect(patterns == own)
    }

    @Test func registryEntryOfAnotherSubdirectoryKeepsTheDefaults() async throws {
        let patterns = try await Self.requestedPatterns(
            loading: Self.configuration(modelSubdirectory: "bf16", patterns: nil),
            registered: [Self.configuration(patterns: Self.headPatterns)])

        #expect(patterns == Self.defaultPatterns.map { "bf16/\($0)" })
    }

    @Test func registryEntryWithoutPatternsKeepsTheDefaults() async throws {
        let patterns = try await Self.requestedPatterns(
            loading: Self.configuration(revision: Self.pinnedRevision, patterns: nil),
            registered: [Self.configuration(patterns: nil)])

        #expect(patterns == Self.defaultPatterns)
    }

    @Test func unregisteredConfigurationKeepsTheDefaults() async throws {
        let patterns = try await Self.requestedPatterns(
            loading: Self.configuration("org/other", patterns: nil),
            registered: [Self.configuration(patterns: Self.headPatterns)])

        #expect(patterns == Self.defaultPatterns)
    }
}

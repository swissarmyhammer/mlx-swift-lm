// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration

import Testing
import Foundation
import MLXLMCommon
import MLX
import FoundationModels
@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Multi-model correctness sweep.
///
/// Runs guided generation round-trip tests against multiple model families
/// to validate vocabulary extraction correctness across tokenizer
/// implementations.
@Suite(.serialized, .timeLimit(.minutes(15)))
struct MultiModelGuidedGenerationTests {

    /// Models to test. Each is downloaded on first run (~100-500MB each).
    static let modelIDs = [
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        TestFixtures.gemmaModelID,
    ]

    // MARK: - Int Round-Trip Per Model

    /// `intRoundTrip`'s own model list, excluding `gemma-3-270m-it-4bit` --
    /// see `intRoundTripGemma3270m` immediately below for why that specific
    /// model gets its own, currently-`.disabled`, copy of this test instead
    /// of running as a third parameterized case here. Deliberately NOT
    /// filtering the shared `modelIDs` array itself: `stringRoundTrip`,
    /// `boolRoundTrip`, and `nestedCountConstrainedAcrossModels` all
    /// parameterize over `modelIDs` too and are unaffected by this bug
    /// (their schemas don't leave gemma-3-270m thousands of unbiased tokens
    /// to loop in), so they should keep covering all three models.
    static let intRoundTripModelIDs = modelIDs.filter { $0 != TestFixtures.gemmaModelID }

    @Test(arguments: intRoundTripModelIDs)
    func intRoundTrip(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(modelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("What is 2+2? Reply with just the number."),
            schema: Int.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(\(modelID) Int)")

        let decoded = try JSONDecoder().decode(Int.self, from: Data(trimmed.utf8))
        _ = decoded
        print("[\(modelID)] Int round-trip: \(trimmed)")
    }

    /// `gemma-3-270m-it-4bit`'s int round-trip, split out of `intRoundTrip`
    /// above because of a REAL, root-caused adapter bug (kanban `t3nynaj`,
    /// full id 01KXEF8DHN3RTNMK4MAT3NYNAJ): `Executor.ConstraintSetup
    /// .reserves(forMaxTokens:)` used to compute `completionReserve` as
    /// `max(structuralReserve * 3, maxTokens / 4)`. For a bare `Int` schema,
    /// `CompletionReserve.estimate` synthesizes the minimal JSON `"0"`, so
    /// `structuralReserve` was tiny (1-3 tokens) and the `maxTokens / 4`
    /// floor always won -- with the Executor's `defaultMaxTokens = 4096`,
    /// that floor was 1024, so `GuidedGenerationLoop`'s soft-zone completion
    /// bias didn't engage until token 3072, 75% of the entire budget in.
    /// gemma-3-270m-it-4bit's own greedy repetition-degeneration preference
    /// ran unchecked that whole time, producing 3072 copies of the digit
    /// `"2"` -- a JSON-valid but astronomically large integer literal that
    /// overflowed Swift's `Int` and failed `JSONDecoder` with
    /// `.dataCorrupted`. Fixed by sizing the normal (unbiased) zone off
    /// `structuralReserve` (the schema's actual minimal-content need,
    /// `structuralReserve * normalZoneMultiplier` capped at half of
    /// `maxTokens`) instead of an unconditional fraction of `maxTokens` --
    /// see `ConstraintSetup.reserves(forMaxTokens:)`'s doc comment for the
    /// full policy. Kept as its own test function (rather than folded back
    /// into `intRoundTrip`'s parameterized `modelIDs`) to keep this
    /// regression's coverage clearly attributable to the bug it guards
    /// against.
    @Test
    func intRoundTripGemma3270m() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let modelID = TestFixtures.gemmaModelID
        let model = makeTestModel(modelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("What is 2+2? Reply with just the number."),
            schema: Int.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(\(modelID) Int)")

        let decoded = try JSONDecoder().decode(Int.self, from: Data(trimmed.utf8))
        _ = decoded
        print("[\(modelID)] Int round-trip: \(trimmed)")
    }

    // MARK: - String Round-Trip Per Model

    @Test(arguments: modelIDs)
    func stringRoundTrip(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(modelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("Name a color."),
            schema: String.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(\(modelID) String)")
        let decoded = try JSONDecoder().decode(String.self, from: Data(trimmed.utf8))
        #expect(!decoded.isEmpty, "\(modelID) should produce non-empty string")
        print("[\(modelID)] String round-trip: \(trimmed)")
    }

    // MARK: - Bool Round-Trip Per Model

    @Test(arguments: modelIDs)
    func boolRoundTrip(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(modelID)
        let executor = try makeMLXExecutor(for: model)

        let request = makeExecutorRequest(
            transcript: transcript("Is the sky blue? Reply true or false."),
            schema: Bool.generationSchema
        )

        let raw = try await collectText(from: executor, request: request, model: model)
        let trimmed = try assertValidJSON(raw, label: "(\(modelID) Bool)")

        let decoded = try JSONDecoder().decode(Bool.self, from: Data(trimmed.utf8))
        _ = decoded
        print("[\(modelID)] Bool round-trip: \(trimmed)")
    }

    // MARK: - Nested Count-Constrained Schema Per Model

    @Test("Nested object with count constraints across models", arguments: modelIDs)
    func nestedCountConstrainedAcrossModels(modelID: String) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let container = try await loadTestModelContainer(id: modelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "maxLength": 30 },
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": { "type": "string", "enum": ["a", "b"] },
                                "value": { "type": "string", "maxLength": 20 }
                            },
                            "required": ["kind", "value"],
                            "additionalProperties": false
                        },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["name", "entries"],
                "additionalProperties": false
            }
            """

        let raw: String = try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXgTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )

            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": "List two entries. Respond as JSON."]
            ]
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            let input = LMInput(tokens: MLXArray(tokens))

            let closingBias = ClosingTokenBias.compute(
                tokenizer: context.tokenizer,
                eosTokenID: context.tokenizer.eosTokenId
            )
            let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: context.tokenizer
            )
            let reserve = CompletionReserve.estimate(
                schemaJSON: schema,
                tokenizer: context.tokenizer
            )

            var collected = ""
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: 1024,
                vocabSize: Int(xgTokenizer.vocabSize),
                completionReserve: reserve,
                closingBias: closingBias,
                whitespaceBias: whitespaceBias,
                whitespaceTokenIDs: whitespaceTokenIDs
            ) { text in
                collected += text
                return true
            }
            return collected
        }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip control characters (< 0x20) that some tokenizers insert.
        let sanitized = String(trimmed.unicodeScalars.filter { $0.value >= 0x20 })
        let data = Data(sanitized.utf8)
        let obj = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "[\(modelID)] Should produce valid JSON dict, got: \(trimmed.prefix(200))"
        )
        let entries = try #require(
            obj["entries"] as? [[String: Any]],
            "[\(modelID)] Should have 'entries' array"
        )
        #expect(
            entries.count == 2,
            "[\(modelID)] Should have exactly 2 entries, got \(entries.count)"
        )
    }

    // MARK: - Itinerary-Shaped Schema (3 days x 3 activities)

    static let gemmaModelID = TestFixtures.gemmaModelID

    @Test("Itinerary-shaped schema (3 days x 3 activities) on Gemma")
    func itineraryShapedSchemaOnGemma() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let modelID = Self.gemmaModelID
        let container = try await loadTestModelContainer(id: modelID)

        let schema = """
            {
                "type": "object",
                "properties": {
                    "title": { "type": "string", "maxLength": 50 },
                    "destinationName": {
                        "type": "string",
                        "enum": ["Mount Fuji", "Grand Canyon", "Great Barrier Reef"]
                    },
                    "description": { "type": "string", "maxLength": 100 },
                    "rationale": { "type": "string", "maxLength": 100 },
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string", "maxLength": 40 },
                                "subtitle": { "type": "string", "maxLength": 60 },
                                "destination": { "type": "string", "maxLength": 30 },
                                "activities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                                            },
                                            "title": { "type": "string", "maxLength": 40 },
                                            "description": { "type": "string", "maxLength": 80 }
                                        },
                                        "required": ["type", "title", "description"],
                                        "additionalProperties": false
                                    },
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "required": ["title", "subtitle", "destination", "activities"],
                            "additionalProperties": false
                        },
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "required": ["title", "destinationName", "description", "rationale", "days"],
                "additionalProperties": false
            }
            """

        let raw: String = try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXgTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )

            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": TestFixtures.itineraryPrompt]
            ]
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            let input = LMInput(tokens: MLXArray(tokens))

            let closingBias = ClosingTokenBias.compute(
                tokenizer: context.tokenizer,
                eosTokenID: context.tokenizer.eosTokenId
            )
            let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: context.tokenizer
            )
            let reserve = CompletionReserve.estimate(
                schemaJSON: schema,
                tokenizer: context.tokenizer
            )

            print("[itinerary-test] CompletionReserve: \(reserve) tokens")

            var collected = ""
            var tokenCount = 0
            try GuidedGenerationLoop.run(
                input: input,
                context: context,
                constraint: constraint,
                maxTokens: 4096,
                vocabSize: Int(xgTokenizer.vocabSize),
                completionReserve: reserve,
                closingBias: closingBias,
                whitespaceBias: whitespaceBias,
                whitespaceTokenIDs: whitespaceTokenIDs
            ) { text in
                collected += text
                tokenCount += 1
                return true
            }
            print(
                "[itinerary-test] Generated \(tokenCount) token callbacks, \(collected.count) chars"
            )
            return collected
        }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let sanitized = String(trimmed.unicodeScalars.filter { $0.value >= 0x20 })
        print(
            "[itinerary-test] Raw output (\(sanitized.count) chars): \(sanitized.prefix(500))")

        let data = Data(sanitized.utf8)
        let obj = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "Should produce valid JSON dict, got: \(sanitized.prefix(300))"
        )

        #expect(obj["title"] is String, "Should have 'title' string")
        #expect(obj["destinationName"] is String, "Should have 'destinationName' string")
        #expect(obj["description"] is String, "Should have 'description' string")
        #expect(obj["rationale"] is String, "Should have 'rationale' string")

        let days = try #require(
            obj["days"] as? [[String: Any]],
            "Should have 'days' array"
        )
        #expect(days.count == 3, "Should have exactly 3 days, got \(days.count)")

        for (di, day) in days.enumerated() {
            #expect(day["title"] is String, "Day \(di) should have 'title'")
            #expect(day["subtitle"] is String, "Day \(di) should have 'subtitle'")
            #expect(day["destination"] is String, "Day \(di) should have 'destination'")

            let activities = try #require(
                day["activities"] as? [[String: Any]],
                "Day \(di) should have 'activities' array"
            )
            #expect(
                activities.count == 3,
                "Day \(di) should have exactly 3 activities, got \(activities.count)"
            )

            for (ai, activity) in activities.enumerated() {
                let actType = try #require(
                    activity["type"] as? String,
                    "Day \(di) Activity \(ai) should have 'type'"
                )
                #expect(
                    ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"].contains(
                        actType),
                    "Day \(di) Activity \(ai) type '\(actType)' should be valid enum"
                )
                #expect(
                    activity["title"] is String, "Day \(di) Activity \(ai) should have 'title'")
                #expect(
                    activity["description"] is String,
                    "Day \(di) Activity \(ai) should have 'description'")
            }
        }

        let nestingDepth = measureJSONDepth(sanitized)
        print("[itinerary-test] JSON nesting depth: \(nestingDepth)")
        #expect(
            nestingDepth <= 10,
            "Nesting depth \(nestingDepth) should be reasonable (expected ~5)")
    }

    // MARK: - Helpers

    /// Measures the maximum nesting depth of a JSON string by counting bracket/brace depth.
    private func measureJSONDepth(_ json: String) -> Int {
        var maxDepth = 0
        var current = 0
        var inString = false
        var escaped = false
        for ch in json {
            if escaped {
                escaped = false
                continue
            }
            if ch == "\\" && inString {
                escaped = true
                continue
            }
            if ch == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if ch == "{" || ch == "[" {
                current += 1
                maxDepth = max(maxDepth, current)
            } else if ch == "}" || ch == "]" {
                current -= 1
            }
        }
        return maxDepth
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collectText(
        from executor: MLXLanguageModel.Executor,
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) async throws -> String {
        let stream = try await executeResponse(executor, request: request, model: model)
        var text = ""
        for try await event in stream {
            if let response = reflectedChannelPayload(
                of: event, caseLabel: "response",
                as: LanguageModelExecutorGenerationChannel.Response.self),
                let delta = reflectedChannelPayload(
                    of: response.action, caseLabel: "appendText",
                    as: LanguageModelExecutorGenerationChannel.TextFragment.self)
            {
                text += delta.content
            }
        }
        return text
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func transcript(_ prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: prompt))
                    ], responseFormat: nil))
        ])
    }

    @discardableResult
    private func assertValidJSON(_ raw: String, label: String = "") throws -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!trimmed.isEmpty, "Output should be non-empty \(label)")

        let data = try #require(trimmed.data(using: .utf8), "UTF-8 encoding failed \(label)")
        let parsed = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
        #expect(parsed != nil, "Output should be valid JSON \(label): \(trimmed)")
        return trimmed
    }

    @Test("Constraint init with @Generable-sized schema")
    func constraintInitWithLargeSchema() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let schema = TestFixtures.itinerarySchemaProduction
        let modelID = Self.gemmaModelID
        let container = try await loadTestModelContainer(id: modelID)
        try await container.perform { context in
            let xgTokenizer = try await MLXLanguageModel.makeXgTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try GrammarConstraint(
                tokenizer: xgTokenizer,
                jsonSchema: schema,
                fastForward: true,
                hostTokenizer: context.tokenizer
            )
            let mask = try constraint.computeMask()
            #expect(!mask.isTerminated, "Constraint should not immediately stop")
        }
    }

    // MARK: - GPU Memory Cleanup

    @Test("Cleanup: release multi-model GPU resources")
    func releaseGPUResources() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let before = GPU.snapshot()
        await releaseAllGPUMemory()
        let after = GPU.snapshot()
        let freed = before.activeMemory - after.activeMemory
        print(
            "[MultiModelCleanup] freed \(freed / (1024 * 1024))MB active, "
                + "\(before.cacheMemory / (1024 * 1024))MB cache")
    }
}

#endif

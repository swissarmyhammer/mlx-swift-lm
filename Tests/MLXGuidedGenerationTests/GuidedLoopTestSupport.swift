// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// A processor the guided-loop tests never call: each test supplies the prepared input.
struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}

---
assignees:
- claude-code
position_column: todo
position_ordinal: '8580'
title: Preserve image attachments in tool-output segments instead of silently dropping them in TranscriptConverter
---
`TranscriptConverter.extractToolOutputText` / the shared `extractConcatenatedText` helper (Libraries/MLXFoundationModels/TranscriptConverter.swift) only handle `.text` and `.structure` segments. A `.toolOutput` entry whose segments include an `.attachment` (e.g. a tool that returns a photo) has that image silently dropped — `Chat.Message.tool(...)` only ever receives the concatenated text/structure content, never the image.

Found while comparing against Anthropic's own `ClaudeForFoundationModels` provider (https://github.com/anthropics/ClaudeForFoundationModels), whose `RequestBuilder.contentBlocks(from:)` explicitly preserves `.attachment` segments in tool-result replay with the comment "Block content, not flattened text — tool results may carry images."

Fix: extend tool-output handling so an image attachment segment in a `.toolOutput` entry is preserved and forwarded the same way `.prompt`/`.instructions` image attachments already are via `extractImages` (`Chat.Message.tool` may need a signature change to accept images, mirroring `Chat.Message.system`/`.user`, or an equivalent path that doesn't drop the image). Scope to whatever MLXLMCommon's `Chat.Message` type actually supports for the `.tool` role — if it has no image-carrying tool-message variant, that's itself worth surfacing rather than silently dropping the content.

Add a regression test in Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift that builds a `.toolOutput` entry with an image attachment segment and asserts the image is not silently discarded.
---
assignees:
- claude-code
position_column: done
position_ordinal: ffffdc80
title: 'ResolvedModelConfiguration: document its public properties'
---
Found by the ^4z5df0d review, on properties that change did not touch (Libraries/MLXLMCommon/Downloader.swift, `ResolvedModelConfiguration`):

- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'defaultPrompt'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'extraEOSTokens'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'stopStrings'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'eosTokenIds'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'toolCallFormat'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'reasoningConfig'.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'messageGenerator'.

Fix the cause across the whole type: every public property of `ResolvedModelConfiguration` (also `modelDirectory`, `tokenizerDirectory`, `name`, `generationConfig`) and its public `init` get a doc comment, pointing at the matching `ModelConfiguration` property where there is one.

#inkling

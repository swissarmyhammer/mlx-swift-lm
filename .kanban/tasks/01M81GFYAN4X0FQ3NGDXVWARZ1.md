---
assignees:
- claude-code
position_column: todo
position_ordinal: '8a80'
title: 'MTPDrafterRegistry: document the checkpoint constants and justify @unchecked Sendable'
---
Found by the ^6e0mtqt review, on code that change did not touch (Libraries/MLXLMCommon/MTPDrafterModelFactory.swift, `MTPDrafterRegistry`):

- `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe. (`public class MTPDrafterRegistry: AbstractModelRegistry, @unchecked Sendable`)
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'shared '.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'gemma4_26B_assistant_bf16 '.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'gemma4_31B_assistant_bf16 '.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'qwen3_8_27b_mtp_4bit '.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'lfm2_1_2B_instruct_dspark '.
- `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'lfm2_2_6B_dspark '.

Fix the cause across the whole type: every public constant of `MTPDrafterRegistry` (also `lfm2_8B_A1B_dspark`) gets a doc comment, and the `@unchecked Sendable` either goes or carries the swiftlint disable line with the invariant (check how `AbstractModelRegistry` guards its state).

#inkling

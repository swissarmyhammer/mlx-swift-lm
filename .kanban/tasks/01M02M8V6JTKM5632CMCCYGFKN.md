---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m02pcn711nt3cr8qp25jmg31
  text: |-
    Picked up the card and moved it to `doing`.

    Research: I enumerated every public declaration of `Libraries/MLXLLM/Models/DeepSeekV4.swift` before I wrote anything. There are 12:

    1. `public final class DeepSeekV4ModelInner`
    2. `public final class DeepSeekV4Model`
    3. `public let kvHeads`
    4. `public let model`
    5. `public init(_:)`
    6. `public func callAsFunction(_:cache:)`
    7. `public func newCache(parameters:)`
    8. `public var loraLayers`
    9. `public var missingChatTemplateRefusal`
    10. `public func promptTokenizer(wrapping:)`
    11. `public func sanitize(weights:)`
    12. `public var toolCallFormat` -- the one the finding names

    Eleven of the twelve already carried a documentation comment. `toolCallFormat` was the only one without, thus the sweep of the whole file is one declaration wide. Every other declaration of the file is internal or private, and each of those carries a documentation comment as well.

    Why swiftlint reports that one line and no other member: the shipped `missing-docs-swift` rule sets `excludes_inherited_types: true`. `DeepSeekV4Model` declares inherited types, thus swiftlint passes over its members. The `extension DeepSeekV4Model` block declares no inherited type, thus the member inside it IS reported.

    The `//` block above the extension is now the documentation comment of `toolCallFormat`. It names the syntax -- DSML -- and states the two elements a DSML call carries, the encoder that renders it, and the parser that reads it back. The paragraph about the absent `reasoningConfig` moved with it. I followed the shape `GPTOSS.swift` uses for the same declaration.

    Second pass: the confirming review gave one `swift/documentation` finding on `kvHeads` -- a first-sentence summary followed by elaboration with no blank `///` line between. That is a documentation-comment defect, thus it is inside the comments-only scope of this card. A finding gives one example of a cause, so I removed the cause from the whole file. Four blocks carried it: `ResidualStreamAxis.copy`, `kvHeads`, `projectionTensors` and `scoreCorrectionBiasPath`.

    The third review gave one `complexity/cognitive-complexity` finding on `stackRoutedExperts`. That asks for a code change, which this card forbids. I did NOT resolve it. Task `^zyhhbfx` holds it.
  timestamp: 2026-08-15T12:29:47.617021+00:00
- actor: claude-code
  id: 01m02pcvdrz3m1dryb1d36jcrj
  text: |-
    ### implement — changed
    - evidence: 1 file — Libraries/MLXLLM/Models/DeepSeekV4.swift (26 insertions, 16 deletions, comments only; `git diff` shows no code line). `review file` gives 0 `missing-docs-swift` findings and 0 `swift/documentation` findings (9 validators attempted, 0 failed, 0 skipped). `swift build` exits 0. `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test` exits 0 with 1030 tests (0 + 140 + 69 + 7 + 814) in 109 suites and 0 failures; the 3 warnings are the SwiftPM package warnings that stand before this change. `swift format lint --strict --configuration .swift-format` exits 0, and `swift format` gives the file back unchanged (empty diff).
    - next: /review. The `complexity/cognitive-complexity` finding on `stackRoutedExperts` is open on task `^zyhhbfx`.
  timestamp: 2026-08-15T12:29:53.976282+00:00
position_column: doing
position_ordinal: '8480'
title: Give the DeepSeekV4Model chat-convention declaration a documentation comment
---
## What

The review of task `^dhv1ave` swept the diffs of commits `507d5fa` and `465085d` and gave one finding on a line that neither commit touched. The standing rule from task `^ag7ant0` makes such a finding a record only, thus that review left it open. This task holds it, so that it is not lost.

`git log -L 656,658:Libraries/MLXLLM/Models/DeepSeekV4.swift` gives commit `bd15e30` ("feat(mlx-lm): register MiniMax-M3 and declare the chat conventions") for the line.

## Finding

- [x] `Libraries/MLXLLM/Models/DeepSeekV4.swift:657` `code-hygiene/missing-docs-swift` — public declarations should be documented.

The line the finding names:

```swift
extension DeepSeekV4Model {
    public var toolCallFormat: ToolCallFormat? { .dsml }
}
```

The block above the extension carries `//` comments, which are not documentation comments. The declaration itself carries none.

A finding gives one example of a cause. Remove the cause from the whole file: give every public declaration of `Libraries/MLXLLM/Models/DeepSeekV4.swift` a documentation comment, and start with each declaration that commit `bd15e30` added.

## Acceptance Criteria

- [x] `public var toolCallFormat` carries a documentation comment that says which syntax the model writes its tool calls in.
- [x] Each other public declaration of the file carries a documentation comment.
- [x] `review file Libraries/MLXLLM/Models/DeepSeekV4.swift` gives no `missing-docs-swift` finding.
- [x] `swift build` and `swift test` are green, with no new warning.
- [x] `swift format --configuration .swift-format` gives the file back unchanged.

## Out of scope, recorded elsewhere

The review that confirmed this work gave one `complexity/cognitive-complexity` finding on `stackRoutedExperts`, which is a code change this comments-only task must not make. Task `^zyhhbfx` holds it.

#deepseek-v4
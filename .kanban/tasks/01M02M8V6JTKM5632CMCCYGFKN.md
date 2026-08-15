---
assignees:
- claude-code
position_column: todo
position_ordinal: 9b80
title: Give the DeepSeekV4Model chat-convention declaration a documentation comment
---
## What

The review of task `^dhv1ave` swept the diffs of commits `507d5fa` and `465085d` and gave one finding on a line that neither commit touched. The standing rule from task `^ag7ant0` makes such a finding a record only, thus that review left it open. This task holds it, so that it is not lost.

`git log -L 656,658:Libraries/MLXLLM/Models/DeepSeekV4.swift` gives commit `bd15e30` ("feat(mlx-lm): register MiniMax-M3 and declare the chat conventions") for the line.

## Finding

- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:657` `code-hygiene/missing-docs-swift` — public declarations should be documented.

The line the finding names:

```swift
extension DeepSeekV4Model {
    public var toolCallFormat: ToolCallFormat? { .dsml }
}
```

The block above the extension carries `//` comments, which are not documentation comments. The declaration itself carries none.

A finding gives one example of a cause. Remove the cause from the whole file: give every public declaration of `Libraries/MLXLLM/Models/DeepSeekV4.swift` a documentation comment, and start with each declaration that commit `bd15e30` added.

## Acceptance Criteria

- [ ] `public var toolCallFormat` carries a documentation comment that says which syntax the model writes its tool calls in.
- [ ] Each other public declaration of the file carries a documentation comment.
- [ ] `review file Libraries/MLXLLM/Models/DeepSeekV4.swift` gives no `missing-docs-swift` finding.
- [ ] `swift build` and `swift test` are green, with no new warning.
- [ ] `swift format --configuration .swift-format` gives the file back unchanged.

#deepseek-v4
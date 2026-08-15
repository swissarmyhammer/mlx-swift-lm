---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzq739ahn4rh5rs196kq3bww
  text: |-
    ### The work this card asks for landed under `^wkv5j6f`

    A later run was dispatched to close the round-3 finding on `^wkv5j6f` itself, thus `Libraries/MLXLMCommon/Load.swift` now holds the fix and `Tests/MLXLMTests/LoadWeightsTests.swift` holds five new tests. Each acceptance criterion of this card is met:

    - An entry that holds `..` is rejected, wherever the `..` sits. A new private function, `weightFileURL(forIndexEntry:in:)`, splits the entry on `/` and examines the components, thus `shards/../../outside.safetensors` is rejected as well as `../outside.safetensors`.
    - An entry that gives an absolute path is rejected.
    - A bad entry is reported. `WeightLoadingError` is extended with `weightFileOutsideModelDirectory(entry:modelDirectory:)` and `modelDirectoryIsNotAFileURL(URL)`, each with its own `errorDescription`.
    - A URL whose scheme is not `file` is rejected by `guard modelDirectory.isFileURL` at the top of `safetensorWeightURLs`, before both branches.
    - The good path did not change. A plain entry and an entry in a subdirectory keep the URL they had, and the whole test suite is green.

    Tests: a `../` entry, a `..` entry from a subdirectory, an absolute-path entry, a `https:` URL, and an entry in a subdirectory for the good path. The four rejection tests each failed with "did not throw an error" before the fix.

    `swift test --filter LoadWeightsTests` 9/9; `Gemma4KVSharedLoadTests` 2/2, `GLM4LmHeadTiedLoadTests` 6/6, `BaseConfigurationTests` 2/2, `MiniMaxM3Tests` 47/47; full `swift test` exit 0.

    This card needs a decision from the user: close it as done by that change, or keep it for a wider examination of `Load.swift`. No agent closed it.
  timestamp: 2026-08-11T01:30:53.137525+00:00
- actor: claude-code
  id: 01kzq910yqpa0jjpperm21pmc5
  text: |
    ## Closed — the work landed under ^wkv5j6f

    This card was made to hold the path-traversal fix in `Load.swift`. The finding
    that made it was recorded on ^wkv5j6f, and the `finish` skill states that a
    review finding belongs to the task that holds it. The work thus landed on
    ^wkv5j6f and not here.

    **What is done**, in commit `638dde5`:

    - A new private `weightFileURL(forIndexEntry:in:)` reads each entry of
      `model.safetensors.index.json` before it maps the entry onto the model
      directory. It rejects an empty entry, an entry that starts with `/`, and an
      entry that holds a `..` component. It splits on `/` and reads the components,
      thus `shards/../../outside.safetensors` is rejected and not only
      `../outside.safetensors`.
    - A `guard modelDirectory.isFileURL` sits above both branches. `FileManager`
      reads only the path of a URL and ignores the scheme, thus an `https:` URL
      walked the root of the local file system before this guard.
    - `WeightLoadingError` gained `modelDirectoryIsNotAFileURL` and
      `weightFileOutsideModelDirectory(entry:modelDirectory:)`. Nothing is dropped
      without a word.
    - `Tests/MLXLMTests/LoadWeightsTests.swift` holds five new tests. The four
      rejection tests each went red first.

    **The review verified it.** The reviewer of ^wkv5j6f round 4 tried 28 attack
    strings, and none went out of the model directory: percent-encoding (which
    cannot work, because `appendingPathComponent` encodes and never decodes),
    backslash, UNC, `C:\`, Unicode near-copies of `..`, `~`, a leading `./`, and
    nested `..`. `Gemma4KVSharedLoadTests`, `GLM4LmHeadTiedLoadTests`,
    `BaseConfigurationTests` and `MiniMaxM3Tests` all stay green, thus no other
    model changed how it loads.

    Each acceptance criterion of this card is met by that commit.

    **One item stays open, and it is deliberate.** A symlink inside the model
    directory that points outside is NOT rejected. The check reads the text of the
    entry alone. No code on the load path makes a symlink, and a check that followed
    symlinks would break a download layout that uses one. A person who wants that
    must open a new card.
  timestamp: 2026-08-11T02:04:36.183458+00:00
position_column: done
position_ordinal: e080
title: Validate safetensors index paths against path traversal in Load.swift
---
## What

`Libraries/MLXLMCommon/Load.swift:49` maps the values of
`model.safetensors.index.json` onto the model directory and does not check them:

```swift
return Set(index.weightMap.values)
    .sorted()
    .map { modelDirectory.appendingPathComponent($0) }
```

A `model.safetensors.index.json` file comes inside a model repository that a
person downloads. It is input from outside, and a person must not trust it.
`appendingPathComponent` accepts a value that holds `../`, thus an entry in the
index can go out of the model directory, and the loader then opens that file.

A review of task ^wkv5j6f found this. The file was in the commit under review,
but this line was not part of the change.

## A second face of the same cause

A probe in the same review showed that `FileManager` reads only the path of a
URL and ignores the scheme and the host. The URL `https://example.com/` gave an
enumerator that walked the root of the local file system (`file:///home`,
`file:///usr/`, `file:///usr/bin/`).

Both problems are an unchecked path that reaches `FileManager`.

## Why this is a separate task

`Load.swift` is on the load path of every model in this repository. A change
here changes how each model loads. Task ^wkv5j6f is about the quantization plan
of DeepSeek-V4, and it must not grow to hold a security change to shared code.

## Acceptance Criteria

- [ ] An index entry that holds `..` does not go out of the model directory.
- [ ] An index entry that gives an absolute path does not go out of the model
      directory.
- [ ] The function reports a bad entry. It does not drop the entry without a
      word.
- [ ] A URL with a scheme that is not `file` does not walk the local file
      system.
- [ ] The good path does not change. Each model still loads.

## Tests

- [ ] A test for an index entry that holds `../`.
- [ ] A test for an index entry that gives an absolute path.
- [ ] `swift test --filter LoadWeightsTests`
- [ ] `Gemma4KVSharedLoadTests`, `GLM4LmHeadTiedLoadTests`,
      `BaseConfigurationTests` and `MiniMaxM3Tests` stay green.
- [ ] Full `swift test` stays green.

#deepseek-v4

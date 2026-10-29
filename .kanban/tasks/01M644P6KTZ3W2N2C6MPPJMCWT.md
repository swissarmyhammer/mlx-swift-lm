---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'Docs: scripts/verify-docs.sh fails on Xcode 27.1 for MLXGuidedGeneration/MLXHuggingFace (picojson ''algorithm'' not found in ExtractAPI)'
---
Pre-existing, not caused by mtp: reproduces in clean worktrees at 08d789b2 (old stable), 75fb5184 and 268dab08 with Xcode 27.1 / Swift 6.4. `swift package generate-documentation --target MLXGuidedGeneration|MLXHuggingFace --warnings-as-errors` fails while ExtractAPI builds the MLXCXGrammar symbol graph: `Libraries/MLXCXGrammar/xgrammar/3rdparty/picojson/picojson.h:44:10: fatal error: 'algorithm' file not found` — ExtractAPI passes the C++ headers to clang as `-x objective-c-header`, so libc++ is not on the include path. Logs: session scratchpad docs-268dab08.log, docs-08d789b2.log, docs-75fb5184.log. An earlier "All documentation builds passed" at 75fb5184 had produced empty symbolgraphs dirs, so it never extracted these targets.

- [ ] Make the MLXCXGrammar module's public headers not include C++ headers when parsed as Objective-C (umbrella/module map exposing only a C/ObjC-compatible API, or exclude the C++ headers from the symbol graph), or skip symbol extraction for the C++ module in verify-docs if it has no documented public API.
- [ ] verify-docs.sh passes on Xcode 27.1 in a clean worktree.

#docs
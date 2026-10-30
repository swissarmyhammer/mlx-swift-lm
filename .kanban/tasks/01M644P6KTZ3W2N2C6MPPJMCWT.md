---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66by53q43m3cd30st7xcb09
  text: |-
    Research + fix.

    Root cause: the public module was already C-only (include/module.modulemap exposes just xgrammar_c.h). The C++ headers came in another way: Swift Build hands EVERY header file under a C-family target's directory (public + project headers) to `clang -extract-api -x objective-c-header ...`. The failing command line listed xgrammar_c.h followed by dlpack.h, picojson.h, all xgrammar/cpp/**/*.h and xgrammar/include/xgrammar/*.h. So changing the umbrella/modulemap alone would not help.

    Fix (Package.swift): the MLXCXGrammar target now excludes every vendored `.h` under Libraries/MLXCXGrammar/xgrammar/ (`exclude: [grammar_functor.cc] + VendoredXGrammar.headers`). A small `VendoredXGrammar` enum in the manifest enumerates them from disk (FileManager.subpaths, filtered on the `h` extension, sorted), so a refresh by scripts/sync-xgrammar-source.sh needs no manifest edit. Excluding a header only drops it from the target's header list: the `.cc` sources still find the headers through the existing headerSearchPath settings, so the library builds unchanged. Symbol extraction still runs for MLXCXGrammar; its symbol graph now holds the 32 symbols from xgrammar_c.h only. No target is skipped in verify-docs.sh, so every Swift target stays fully checked.

    Second failure found on the way: in a clean worktree at bc7c7949, verify-docs.sh also failed for MLXLMCommon with a real DocC error — a stale symbol link in Libraries/MLXLMCommon/ChatSession.swift (`init(mtpDrafterBytes:blockSize:minimumDraftProbability:memoryPolicy:loadMTPDrafter:)` no longer exists since ed863693 added `adaptivePolicy:`). Fixed the link to `init(mtpDrafterBytes:blockSize:minimumDraftProbability:memoryPolicy:adaptivePolicy:loadMTPDrafter:)`. The card's "verify-docs.sh passes" needed it.

    Did not work / not tried: modulemap/umbrella changes (would not affect the header list ExtractAPI receives); skipping MLXCXGrammar in the script (unnecessary once headers are excluded, and would hide its C API).

    Note: `warning: missing creator for mutated node (mlx-swift_Cmlx.bundle)` is a pre-existing Swift Build notice from mlx-swift's Cmlx bundle; it appears in the before logs at bc7c7949, 268dab08 and 08d789b2 too.
  timestamp: 2026-10-30T12:18:15.287665+00:00
- actor: claude-code
  id: 01m66byd7x80g73210rtx0vssx
  text: |-
    ### implement — changed
    - evidence: 2 files — Package.swift (MLXCXGrammar excludes the vendored xgrammar .h files via VendoredXGrammar.headers), Libraries/MLXLMCommon/ChatSession.swift (stale DocC link now names adaptivePolicy:). Fresh detached worktree at bc7c7949: scripts/verify-docs.sh BEFORE exit 1 (MLXGuidedGeneration + MLXHuggingFace: picojson.h 'algorithm' not found in ExtractAPI; MLXLMCommon: link to init(mtpDrafterBytes:...:loadMTPDrafter:) does not exist). Same worktree with the 2 files copied, after rm -rf .build: exit 0, 10/10 archives, "All documentation builds passed."; MLXCXGrammar.symbols.json holds 32 symbols, all from xgrammar_c.h. swift build --build-tests exit 0, 0 compiler warnings (only the pre-existing Swift Build notice "missing creator for mutated node" for mlx-swift_Cmlx.bundle, also present at HEAD). xcrun xctest: CXGrammarTests 7, MLXGuidedGenerationTests 72, MLXHuggingFaceMacrosTests 7 XCTest, MLXFoundationModelsTests 457, MLXLMTests 920 XCTest + 2065 Swift Testing — all exit 0, 0 failures, 0 skipped. Logs: session scratchpad ppj-verify-before.log, ppj-verify-after-clean.log, ppj-build.log, ppj-test-*.log. Not committed.
    - next: /review
  timestamp: 2026-10-30T12:18:23.613404+00:00
position_column: doing
position_ordinal: '8380'
title: 'Docs: scripts/verify-docs.sh fails on Xcode 27.1 for MLXGuidedGeneration/MLXHuggingFace (picojson ''algorithm'' not found in ExtractAPI)'
---
Pre-existing, not caused by mtp: reproduces in clean worktrees at 08d789b2 (old stable), 75fb5184 and 268dab08 with Xcode 27.1 / Swift 6.4. `swift package generate-documentation --target MLXGuidedGeneration|MLXHuggingFace --warnings-as-errors` fails while ExtractAPI builds the MLXCXGrammar symbol graph: `Libraries/MLXCXGrammar/xgrammar/3rdparty/picojson/picojson.h:44:10: fatal error: 'algorithm' file not found` — ExtractAPI passes the C++ headers to clang as `-x objective-c-header`, so libc++ is not on the include path. Logs: session scratchpad docs-268dab08.log, docs-08d789b2.log, docs-75fb5184.log. An earlier "All documentation builds passed" at 75fb5184 had produced empty symbolgraphs dirs, so it never extracted these targets.

- [x] Make the MLXCXGrammar module's public headers not include C++ headers when parsed as Objective-C (umbrella/module map exposing only a C/ObjC-compatible API, or exclude the C++ headers from the symbol graph), or skip symbol extraction for the C++ module in verify-docs if it has no documented public API.
- [x] verify-docs.sh passes on Xcode 27.1 in a clean worktree.

#docs
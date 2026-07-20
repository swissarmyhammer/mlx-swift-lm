// Copyright © 2026 Apple Inc.
//
// Swift wrappers over the CXGrammar C shim. These are the guided-
// generation surface the library exposes to callers: `GrammarTokenizer`,
// `GrammarConstraint`, `GrammarError`, `MaskResult`, and `CommitResult`.
// `GrammarConstraint` owns three C handles (`XGGrammarCompiler`,
// `XGCompiledGrammar`, `XGMatcher`) and frees them in
// construction-reverse order in `deinit`.

import Foundation
import MLXCXGrammar
import MLXLMCommon

// MARK: - Errors

/// Errors thrown by the xgrammar bridge across the constraint lifecycle.
///
/// Covers tokenizer construction, grammar compilation, mask computation,
/// and matcher state transitions (commit, rollback, fork). Every case
/// carries a human-readable detail string — xgrammar's `what()` text
/// captured from the shim's thread-local error buffer when available,
/// otherwise a call-site fallback naming the failing primitive.
public enum GrammarError: Error {
    /// `xg_tokenizer_info_new` returned a non-OK status.
    ///
    /// The string is the thread-local `xg_last_error_message()` captured
    /// at the failure site, or a fallback if no message surfaced.
    case tokenizerCreationFailed(String)
    /// A `GrammarConstraint.init` step failed without a more specific case.
    ///
    /// Covers compiler creation, schema compilation, and matcher
    /// construction. The string is the best-available error message:
    /// xgrammar's `what()` via `xg_last_error_message()` when present,
    /// otherwise a call-site fallback naming the failing primitive.
    case constraintCompilationFailed(String)
    /// Schema source failed xgrammar's JSON-Schema validation.
    ///
    /// Either the text is not valid JSON (`XG_ERR_INVALID_JSON`) or it
    /// parses as JSON but is rejected as a JSON Schema
    /// (`XG_ERR_INVALID_JSON_SCHEMA`, e.g. `{"type": 42}`). The
    /// string carries xgrammar's `what()` text via the shim's
    /// thread-local error buffer. The discriminated case lets callers
    /// recognize user-schema errors separately from internal shim
    /// failures.
    case invalidJSONSchema(String)
    /// `xg_matcher_fill_next_token_bitmask` returned a non-OK status.
    case maskComputationFailed(String)
    /// `xg_matcher_accept_token` returned a non-OK status.
    ///
    /// Most commonly `XG_ERR_INVALID_ARG` when the grammar rejects the
    /// token; the string describes the specific failure.
    case commitFailed(String)
    /// `xg_matcher_rollback` returned a non-OK status.
    ///
    /// Also covers the case where the Swift-side stub is still in place.
    /// The string carries xgrammar's `what()` text via the thread-local
    /// error buffer when available.
    case rollbackFailed(String)
    /// `xg_matcher_fork` returned a non-OK status.
    ///
    /// The string carries xgrammar's `what()` text via the thread-local
    /// error buffer when available, or a call-site fallback otherwise.
    case forkFailed(String)
}

// MARK: - Shim error capture

/// Compose a human-readable error detail for shim failures.
///
/// xgrammar's `what()` arrives via the thread-local
/// `xg_last_error_message()` buffer. When the buffer is empty
/// (e.g. when the status was synthesized by a shim-level fast-fail
/// path like a NULL argument check), fall back to naming the
/// primitive that failed plus the numeric status so the error
/// surfaces something actionable.
///
/// File-scope (rather than a member of one wrapper type) because the
/// thread-local buffer is shared by every shim call in this file —
/// `GrammarTokenizer` and `GrammarConstraint` both route their
/// failure details through it.
private func captureShimError(status: XGStatus, fallback: String) -> String {
    if let cstr = xg_last_error_message() {
        return String(cString: cstr)
    }
    return "\(fallback) returned status \(status)"
}

// MARK: - GrammarTokenizer

/// Swift wrapper around `XGTokenizerInfo*`.
///
/// Manages C pointer lifetime via `deinit`.
///
/// Construction copies the vocab strings into xgrammar's internal
/// tables (xgrammar's `TokenizerInfo` owns its decoded/sorted vocab),
/// so the caller does not need to retain the `[String]` it passed in.
///
/// `@unchecked Sendable`: tokenizers are cached on the model cache
/// actor and handed across actors. The underlying `XGTokenizerInfo*`
/// is read-only after construction and xgrammar does not mutate it.
public final class GrammarTokenizer: @unchecked Sendable {
    let pointer: OpaquePointer
    /// Number of entries in the vocab this tokenizer was constructed with.
    ///
    /// The length of the `vocab` array passed to `init`, and the width
    /// grammar bitmasks are sized against (see `xg_bitmask_size`).
    public let vocabSize: Int
    /// The stop-token ids registered on the xgrammar TokenizerInfo.
    ///
    /// The ids the grammar gates to accept states (see
    /// ``init(vocab:vocabType:stopTokenIDs:)``).
    public let stopTokenIDs: [Int32]

    /// Construct a tokenizer from a pre-decoded vocab.
    ///
    /// - Parameters:
    ///   - vocab: Per-token strings in canonical `convertIdToToken`
    ///     form (raw SentencePiece piece or GPT-2 BPE piece — the
    ///     `vocabType` selects xgrammar's decoder).
    ///   - vocabType: Selects xgrammar's token-decoding path.
    ///     `.raw` treats each string as literal UTF-8 bytes;
    ///     `.byteFallback` applies SentencePiece `<0xNN>` + `▁`
    ///     decoding; `.byteLevel` applies GPT-2 `bytes_to_unicode`
    ///     decoding.
    ///   - eosTokenID: End-of-sequence token ID, registered as a stop
    ///     token on the xgrammar TokenizerInfo.
    public convenience init(vocab: [String], vocabType: VocabType, eosTokenID: Int32) throws {
        try self.init(vocab: vocab, vocabType: vocabType, stopTokenIDs: [eosTokenID])
    }

    /// Construct a tokenizer from a pre-decoded vocab, registering the full stop-token set.
    ///
    /// Every token id a model may legitimately end its turn with (primary
    /// EOS *and* secondary turn-enders like GLM's `<|user|>`/
    /// `<|observation|>`) must be registered here: xgrammar masks
    /// registered stop tokens out of ordinary content matching and only
    /// permits them at grammar-accept states. An unregistered turn-ender
    /// whose literal bytes are valid JSON string content is otherwise a
    /// perfectly legal *content* continuation, and constrained decode can
    /// sample it mid-string — the live GLM-4.7-Flash truncated-envelope
    /// failure (kanban y4s0w2j).
    ///
    /// - Parameters:
    ///   - vocab: Per-token strings, as in `init(vocab:vocabType:eosTokenID:)`.
    ///   - vocabType: Selects xgrammar's token-decoding path.
    ///   - stopTokenIDs: All stop-token ids to register on the xgrammar
    ///     TokenizerInfo.
    public init(vocab: [String], vocabType: VocabType, stopTokenIDs: [Int32]) throws {
        self.vocabSize = vocab.count
        self.stopTokenIDs = stopTokenIDs

        var info: OpaquePointer?
        let stopTokens: [Int32] = stopTokenIDs

        let status: XGStatus = vocab.withCStringPointers { ptrs in
            stopTokens.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    ptrs.baseAddress,
                    ptrs.count,
                    vocabType.xgVocabType,
                    stopBuf.baseAddress,
                    stopBuf.count,
                    &info
                )
            }
        }

        guard status == XG_OK, let ptr = info else {
            throw GrammarError.tokenizerCreationFailed(
                captureShimError(status: status, fallback: "xg_tokenizer_info_new")
            )
        }
        self.pointer = ptr
    }

    deinit {
        xg_tokenizer_info_free(pointer)
    }
}

// MARK: - MaskResult

/// Result of a mask computation step.
///
/// The `mask` array is an LSB-first int32 bitmask over the tokenizer's
/// vocab: bit `i` of word `w` is token `w * 32 + i`. The array is
/// caller-owned — xgrammar does not alias a mask pointer into its own
/// memory, so `MaskResult.mask` stays valid independently of subsequent
/// calls on the same constraint.
///
/// `isTerminated` mirrors `xgrammar::GrammarMatcher::IsTerminated()`:
/// true iff the matcher has accepted a stop token. The rename reflects
/// xgrammar's own terminology and disambiguates from the
/// `GuidedGenerationLoop`'s streaming "stop" concept.
///
/// `needsApply` tracks whether at least one token is excluded by the
/// grammar; when false, callers can skip applying the mask.
public struct MaskResult {
    /// LSB-first int32 bitmask over the tokenizer's vocab.
    ///
    /// Bit `i` of word `w` is token `w * 32 + i`. Caller-owned (see the
    /// type doc).
    public let mask: [Int32]
    /// True iff the matcher has accepted a stop token.
    ///
    /// Mirrors `xgrammar::GrammarMatcher::IsTerminated()`.
    public let isTerminated: Bool
    /// True iff the grammar excludes at least one token.
    ///
    /// When `false`, callers can skip applying the mask.
    public let needsApply: Bool
}

// MARK: - CommitResult

/// Result of committing a token to advance grammar state.
///
/// `tokens` carries the fast-forward token ids emitted by xgrammar's
/// `FindJumpForwardString` path, in the order they advanced the
/// matcher. Empty when `fastForward` is disabled on the owning
/// `GrammarConstraint`, when xgrammar returned no forced suffix, or when
/// mid-FF tokenization disagreement stopped emission before any token
/// was accepted. See `GrammarConstraint.commitToken` for the
/// mid-FF-rejection policy.
///
/// `isTerminated` matches `MaskResult.isTerminated`: true iff the
/// matcher has accepted a stop token. Reflects the state *after* any
/// FF advancement, so a FF sequence that lands on the stop token
/// surfaces here as `isTerminated = true`.
public struct CommitResult {
    /// Fast-forward token ids emitted by xgrammar's jump-forward path.
    ///
    /// Ordered as they advanced the matcher; empty when fast-forward is
    /// disabled or produced nothing (see the type doc).
    public let tokens: [Int32]
    /// True iff the matcher has accepted a stop token.
    ///
    /// Reflects the state after any fast-forward advancement.
    public let isTerminated: Bool
}

// MARK: - GrammarConstraint

/// Swift wrapper around a compiled xgrammar constraint plus its associated matcher.
///
/// Manages the lifetime of three C handles — the `XGGrammarCompiler`,
/// the `XGCompiledGrammar`, and the `XGMatcher` — freed in
/// construction-reverse order in `deinit`.
///
/// The `tokenizer` reference is retained so the underlying
/// `XGTokenizerInfo` outlives the matcher (xgrammar uses shared
/// ownership internally, but we still keep the Swift reference alive
/// as defense-in-depth against upstream changes).
///
/// Single-owner semantics: a single matcher must only be touched from
/// one logical caller at a time. `ModelCache` already enforces this in
/// production by handing each session its own constraint. For defense
/// in depth against future routing bugs or multi-threaded sampling
/// loops, an `NSLock` inside the bridge serializes every public C-side
/// operation (`computeMask`, `commitToken`) so concurrent Swift callers
/// see a consistent matcher state rather than the undefined behavior
/// that would come from racing `xgrammar::GrammarMatcher` PIMPL state.
///
/// `@unchecked Sendable`: the wrapper is shared across actors via the
/// model cache, but the underlying matcher is not thread-safe. Callers
/// serialize access through their session's isolation domain (e.g. a
/// `ModelContainer.perform` closure).
public final class GrammarConstraint: @unchecked Sendable {
    private let tokenizer: GrammarTokenizer
    private let compiler: OpaquePointer
    private let compiled: OpaquePointer
    private let matcher: OpaquePointer
    private let vocabSize: Int32
    private let bitmaskWords: Int
    /// Whether this constraint owns `compiler` and `compiled` and must release them in `deinit`.
    ///
    /// The root constructor sets this to `true`; the fork path sets it to
    /// `false` and pins `forkParent` to the constraint whose init
    /// created those handles. xgrammar's PIMPL + `shared_ptr` layout
    /// lets the forked matcher keep the underlying C++ compiled
    /// grammar alive independently, so the Swift-side parent retain is
    /// defensive rather than strictly required, but it makes the
    /// ownership contract explicit.
    private let ownsCompiledResources: Bool
    /// Strong reference to the forked-from constraint, held only on fork paths.
    ///
    /// Keeps the parent's `deinit` (and thus the `xg_*_free` calls on the
    /// shared handles) from running while this fork is alive. `nil` on
    /// root constraints.
    private let forkParent: GrammarConstraint?
    /// Fast-forward emission toggle.
    ///
    /// When `true`, every successful `commitToken` queries xgrammar's
    /// `FindJumpForwardString`, encodes it through `hostTokenizer`,
    /// advances the matcher once per resulting token, and returns those
    /// ids. When `false` or when `hostTokenizer` is `nil`, no FF emission
    /// happens and `CommitResult.tokens` is empty.
    private let fastForward: Bool
    /// Host-side tokenizer used to encode FF strings into token ids.
    /// Optional because not every caller needs FF; required when
    /// `fastForward` is `true` or FF silently degrades to empty.
    private let hostTokenizer: (any Tokenizer)?
    /// Serializes every call into the xgrammar matcher.
    ///
    /// xgrammar's `GrammarMatcher` mutates PIMPL state on both
    /// `FillNextTokenBitmask` and `AcceptToken`; without this lock, two
    /// Swift callers touching the same constraint would produce undefined
    /// behavior at the C++ layer. Placed here rather than at the
    /// ModelContainer-perform layer so the safety guarantee holds even if
    /// a future refactor changes how constraints are routed.
    private let lock = NSLock()
    /// Running count of mid-FF tokenization disagreements for this constraint's lifetime.
    ///
    /// Incremented once per `xg_matcher_accept_token` rejection inside
    /// the FF emission loop — i.e. each place where the host tokenizer's
    /// encoding of the xgrammar FF string crossed a grammar-forced
    /// boundary and the matcher refused the re-encoded id. Stays at zero
    /// when FF is disabled, when xgrammar has no FF suffix, or when every
    /// FF token re-encodes cleanly. Reads and writes are serialized
    /// through `lock`; observers go through `fastForwardDisagreementCount`.
    private var _fastForwardDisagreementCount: Int = 0

    /// Compile a JSON Schema string into a grammar matcher.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer the grammar binds to. Must outlive
    ///     this constraint; a Swift reference is retained here.
    ///   - jsonSchema: A standard JSON Schema source string.
    ///   - fastForward: When `true`, `commitToken` emits the tokens
    ///     produced by xgrammar's `FindJumpForwardString` on every
    ///     successful commit (requires `hostTokenizer`). Defaults to
    ///     `false` so callers that don't need fast-forward see no FF
    ///     emission.
    ///   - hostTokenizer: The HuggingFace-side tokenizer used to encode
    ///     FF strings back into token ids. Must be the same tokenizer
    ///     whose vocab built `tokenizer`. Ignored when `fastForward`
    ///     is `false`.
    public convenience init(
        tokenizer: GrammarTokenizer,
        jsonSchema: String,
        fastForward: Bool = false,
        hostTokenizer: (any Tokenizer)? = nil
    ) throws {
        try self.init(
            tokenizer: tokenizer, fastForward: fastForward, hostTokenizer: hostTokenizer,
            compile: Self.jsonSourceCompile(
                jsonSchema, using: xg_compile_json_schema,
                fallback: "xg_compile_json_schema")
        )
    }

    /// Compile an EBNF (GBNF) grammar source string into a matcher.
    ///
    /// Mirrors the `jsonSchema:` initializer but routes through
    /// xgrammar's `Grammar::FromEBNF(...)` + `CompileGrammar(...)` path
    /// rather than the JSON-schema compile path. Used by the Qwen
    /// tool-calling pipeline, which expresses the wrapped-vs-bare
    /// `<tool_call>...</tool_call>` envelope as an explicit grammar
    /// rather than as a JSON schema — schemas can't represent the
    /// wrapper text.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer the grammar binds to. Must outlive
    ///     this constraint; a Swift reference is retained here.
    ///   - grammar: The EBNF/GBNF source. Anything xgrammar's
    ///     `Grammar::FromEBNF` rejects (including Lark syntax) surfaces
    ///     as `GrammarError.constraintCompilationFailed` with the parser's
    ///     line/column message in the payload.
    ///   - rootRule: The name of the top-level production. Pass `nil`
    ///     to use xgrammar's default of `"root"`. The tool-calling
    ///     grammar uses `"start"`, matching the existing Lark shape.
    ///   - fastForward: Same semantics as the `jsonSchema:` init.
    ///   - hostTokenizer: Same semantics as the `jsonSchema:` init.
    public convenience init(
        tokenizer: GrammarTokenizer,
        grammar: String,
        rootRule: String? = nil,
        fastForward: Bool = false,
        hostTokenizer: (any Tokenizer)? = nil
    ) throws {
        try self.init(
            tokenizer: tokenizer, fastForward: fastForward, hostTokenizer: hostTokenizer
        ) { compilerHandle in
            var compiledPtr: OpaquePointer?
            let status: XGStatus = grammar.withCString { grammarPtr in
                if let rootRule {
                    return rootRule.withCString { rootPtr in
                        xg_compile_grammar_from_ebnf(
                            compilerHandle, grammarPtr, rootPtr, &compiledPtr)
                    }
                }
                return xg_compile_grammar_from_ebnf(
                    compilerHandle, grammarPtr, nil, &compiledPtr)
            }
            guard status == XG_OK, let compiledHandle = compiledPtr else {
                // EBNF rejections are never JSON-input errors, so this
                // path skips `compilationError`'s discrimination and
                // always surfaces the generic compilation case.
                throw GrammarError.constraintCompilationFailed(
                    captureShimError(
                        status: status, fallback: "xg_compile_grammar_from_ebnf")
                )
            }
            return compiledHandle
        }
    }

    /// Compile a structural-tag JSON source into a matcher.
    ///
    /// Routes through xgrammar's
    /// `Grammar::FromStructuralTag(json, nullopt)` + `CompileGrammar`
    /// path. Structural tag is xgrammar's first-class format for
    /// multi-format tool-calling dispatch — an `or` / `sequence` /
    /// `tag` / `json_schema` / `const_string` body lets callers express
    /// a wrapped-or-bare JSON envelope (the Qwen tool-calling shape)
    /// without hand-compiling a JSON schema into GBNF. The underlying
    /// JSON-schema-to-grammar compile that xgrammar does internally is
    /// the same one `jsonSchema:` reuses directly.
    ///
    /// The structural-tag bodies used here reference only
    /// `const_string` and `json_schema` formats, so the shim passes
    /// `std::nullopt` for `tokenizer_info`. A future caller that wants
    /// to use `token` / `token_dispatch` / `token_triggered_tags` in
    /// the body will need a variant of this init that threads the
    /// bound `GrammarTokenizer` through to
    /// `Grammar::FromStructuralTag`'s second argument.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer the grammar binds to. Must outlive
    ///     this constraint; a Swift reference is retained here.
    ///   - structuralTag: The structural-tag JSON source. Malformed
    ///     input surfaces either as `GrammarError.invalidJSONSchema` (bad
    ///     JSON or bad embedded schema) or as
    ///     `GrammarError.constraintCompilationFailed` (structural-tag-level
    ///     rejection or any other shim failure); both carry xgrammar's
    ///     `what()` text in the payload.
    ///   - fastForward: Same semantics as the `jsonSchema:` init.
    ///   - hostTokenizer: Same semantics as the `jsonSchema:` init.
    public convenience init(
        tokenizer: GrammarTokenizer,
        structuralTag: String,
        fastForward: Bool = false,
        hostTokenizer: (any Tokenizer)? = nil
    ) throws {
        try self.init(
            tokenizer: tokenizer, fastForward: fastForward, hostTokenizer: hostTokenizer,
            compile: Self.jsonSourceCompile(
                structuralTag, using: xg_compile_structural_tag,
                fallback: "xg_compile_structural_tag")
        )
    }

    /// Designated initializer shared by the three public compile paths.
    ///
    /// Owns everything `jsonSchema:`, `grammar:`, and `structuralTag:`
    /// have in common — property setup, compiler creation, matcher
    /// construction, and handle cleanup on every failure path — while
    /// `compile` encodes the one step that differs: the `xg_compile_*`
    /// call and its error mapping. The two JSON-source paths build
    /// their closure with `jsonSourceCompile`; the EBNF path supplies
    /// its own closure inline.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer the grammar binds to. Must outlive
    ///     this constraint; a Swift reference is retained here.
    ///   - fastForward: See the public initializers.
    ///   - hostTokenizer: See the public initializers.
    ///   - compile: Compiles the grammar source against the freshly
    ///     created compiler handle and returns the compiled-grammar
    ///     handle, throwing a `GrammarError` on failure. On a throw, this
    ///     initializer frees the compiler handle — the closure must not
    ///     free it itself.
    /// - Throws: Whatever `compile` throws, or
    ///   `GrammarError.constraintCompilationFailed` when compiler or
    ///   matcher creation fails.
    private init(
        tokenizer: GrammarTokenizer,
        fastForward: Bool,
        hostTokenizer: (any Tokenizer)?,
        compile: (OpaquePointer) throws -> OpaquePointer
    ) throws {
        self.tokenizer = tokenizer
        self.vocabSize = Int32(tokenizer.vocabSize)
        let words = Int(xg_bitmask_size(self.vocabSize))
        self.bitmaskWords = max(0, words)
        self.fastForward = fastForward
        self.hostTokenizer = hostTokenizer

        var compilerPtr: OpaquePointer?
        let compilerStatus = xg_grammar_compiler_new(tokenizer.pointer, &compilerPtr)
        guard compilerStatus == XG_OK, let compilerHandle = compilerPtr else {
            throw GrammarError.constraintCompilationFailed(
                captureShimError(
                    status: compilerStatus, fallback: "xg_grammar_compiler_new")
            )
        }

        let compiledHandle: OpaquePointer
        do {
            compiledHandle = try compile(compilerHandle)
        } catch {
            xg_grammar_compiler_free(compilerHandle)
            throw error
        }

        var matcherPtr: OpaquePointer?
        let matcherStatus = xg_matcher_new(compiledHandle, &matcherPtr)
        guard matcherStatus == XG_OK, let matcherHandle = matcherPtr else {
            xg_compiled_grammar_free(compiledHandle)
            xg_grammar_compiler_free(compilerHandle)
            throw GrammarError.constraintCompilationFailed(
                captureShimError(status: matcherStatus, fallback: "xg_matcher_new")
            )
        }

        self.compiler = compilerHandle
        self.compiled = compiledHandle
        self.matcher = matcherHandle
        self.ownsCompiledResources = true
        self.forkParent = nil
    }

    /// Maps a failed `xg_compile_*` status to the matching `GrammarError`.
    ///
    /// Discriminates user-input errors from generic compile failures:
    /// xgrammar's typed exceptions map 1:1 to `XG_ERR_INVALID_JSON` /
    /// `XG_ERR_INVALID_JSON_SCHEMA`, and both indicate bad input rather
    /// than an internal shim problem, so callers can pattern-match on the
    /// discriminated `invalidJSONSchema` case. Everything else stays on
    /// `constraintCompilationFailed`.
    ///
    /// - Parameters:
    ///   - status: The non-OK status the compile primitive returned.
    ///   - fallback: The failing primitive's name, used when no shim
    ///     error message is available (see `captureShimError`).
    /// - Returns: The error for the compile closure to throw.
    private static func compilationError(status: XGStatus, fallback: String) -> GrammarError {
        let message = captureShimError(status: status, fallback: fallback)
        if status == XG_ERR_INVALID_JSON_SCHEMA || status == XG_ERR_INVALID_JSON {
            return .invalidJSONSchema(message)
        }
        return .constraintCompilationFailed(message)
    }

    /// Builds the `compile` closure shared by the two JSON-source compile paths.
    ///
    /// Both `jsonSchema:` and `structuralTag:` route through shim entry
    /// points with the same `(compiler, source, out)` shape and the same
    /// status-to-error mapping — `compilationError` collapses
    /// bad JSON and bad embedded schemas to `invalidJSONSchema`, while
    /// everything else (including structural-tag-level rejections such
    /// as a malformed top-level shape or unknown format types) stays on
    /// `constraintCompilationFailed` — so the closure body, including
    /// its error-handling path, exists exactly once here. Each public
    /// initializer contributes only its source string, shim compile
    /// function, and fallback primitive name.
    ///
    /// - Parameters:
    ///   - source: The JSON source to compile (a JSON Schema or a
    ///     structural-tag body).
    ///   - compileFunction: The shim compile entry point to invoke.
    ///   - fallback: The failing primitive's name, used by
    ///     `compilationError` when no shim error message is available.
    /// - Returns: A closure for the designated initializer's `compile`
    ///   parameter.
    private static func jsonSourceCompile(
        _ source: String,
        using compileFunction: @escaping (
            OpaquePointer?, UnsafePointer<CChar>?, UnsafeMutablePointer<OpaquePointer?>?
        ) -> XGStatus,
        fallback: String
    ) -> (OpaquePointer) throws -> OpaquePointer {
        return { compilerHandle in
            var compiledPtr: OpaquePointer?
            let status = source.withCString { sourcePtr in
                compileFunction(compilerHandle, sourcePtr, &compiledPtr)
            }
            guard status == XG_OK, let compiledHandle = compiledPtr else {
                throw GrammarConstraint.compilationError(status: status, fallback: fallback)
            }
            return compiledHandle
        }
    }

    /// Private initializer used by `clone()`.
    ///
    /// Adopts the already-forked matcher handle and records that this
    /// constraint is *not* responsible for freeing the shared `compiler`
    /// / `compiled` handles — those belong to `forkParent`, which is
    /// retained here so its `deinit` is deferred past this fork's own
    /// lifetime.
    private init(
        fromFork matcherHandle: OpaquePointer,
        parent: GrammarConstraint
    ) {
        self.tokenizer = parent.tokenizer
        self.compiler = parent.compiler
        self.compiled = parent.compiled
        self.matcher = matcherHandle
        self.vocabSize = parent.vocabSize
        self.bitmaskWords = parent.bitmaskWords
        self.fastForward = parent.fastForward
        self.hostTokenizer = parent.hostTokenizer
        self.ownsCompiledResources = false
        self.forkParent = parent
    }

    deinit {
        xg_matcher_free(matcher)
        if ownsCompiledResources {
            xg_compiled_grammar_free(compiled)
            xg_grammar_compiler_free(compiler)
        }
    }

    /// Compute the bitmask of grammar-accepted next tokens at the matcher's current state.
    public func computeMask() throws -> MaskResult {
        lock.lock()
        defer { lock.unlock() }
        var mask = [Int32](repeating: 0, count: bitmaskWords)
        var needsApplyFlag: Int32 = 0
        let status = mask.withUnsafeMutableBufferPointer { buf in
            xg_matcher_fill_next_token_bitmask(
                matcher,
                buf.baseAddress,
                buf.count,
                vocabSize,
                &needsApplyFlag
            )
        }
        guard status == XG_OK else {
            throw GrammarError.maskComputationFailed(
                captureShimError(
                    status: status, fallback: "xg_matcher_fill_next_token_bitmask")
            )
        }
        return MaskResult(
            mask: mask,
            isTerminated: isMatcherTerminatedLocked(),
            needsApply: needsApplyFlag != 0
        )
    }

    /// Commit a sampled token to advance grammar state.
    ///
    /// Throws `GrammarError.commitFailed` if the token is not in the most
    /// recent mask (xgrammar returns `XG_ERR_INVALID_ARG` in that
    /// case). Matcher state is unchanged on rejection.
    ///
    /// When `fastForward` is on and a `hostTokenizer` is bound, the
    /// successful accept is followed by a jump-forward pass: xgrammar
    /// surfaces the longest currently-forced suffix via
    /// `FindJumpForwardString`, the host tokenizer encodes that
    /// suffix, and the matcher accepts each resulting token id in
    /// turn. The accepted ids are returned in `CommitResult.tokens`
    /// in the order they advanced the matcher, and `isTerminated`
    /// reflects the final post-FF state. If a mid-FF `AcceptToken`
    /// is rejected (tokenization disagreement — the encoded tokens
    /// cross the FF-valid boundary), emission stops at that point
    /// and the already-accepted prefix is returned; the matcher's
    /// state reflects exactly those accepts.
    public func commitToken(_ tokenID: Int32) throws -> CommitResult {
        lock.lock()
        defer { lock.unlock() }
        let status = xg_matcher_accept_token(matcher, tokenID)
        guard status == XG_OK else {
            throw GrammarError.commitFailed(
                captureShimError(
                    status: status, fallback: "xg_matcher_accept_token token=\(tokenID)")
            )
        }

        var terminated = isMatcherTerminatedLocked()
        let fastForwardTokens: [Int32]
        if !terminated, fastForward, let hostTokenizer {
            fastForwardTokens = try emitFastForwardLocked(via: hostTokenizer)
            terminated = isMatcherTerminatedLocked()
        } else {
            fastForwardTokens = []
        }

        return CommitResult(tokens: fastForwardTokens, isTerminated: terminated)
    }

    /// Query xgrammar's jump-forward string and replay it through the matcher.
    ///
    /// Feeds the forced string back token-by-token. Caller must already
    /// hold `lock`. Returns the accepted token ids in the order they were
    /// accepted. See `commitToken` for the tokenization-disagreement
    /// semantics.
    ///
    /// Tokenization-boundary safety: xgrammar's `FindJumpForwardString`
    /// returns the raw grammar-forced byte suffix. Naively encoding
    /// that suffix through the host tokenizer and accepting every
    /// token overshoots — the final token tends to straddle the
    /// FF-forced boundary and the unforced continuation, and greedy
    /// BPE would have picked a different boundary token once the
    /// unforced bytes arrive. We emit only tokens whose cumulative
    /// decoded byte length is strictly less than the FF string's byte
    /// length; the last token (which closes the boundary) is dropped
    /// and left to the sampler.
    private func emitFastForwardLocked(via hostTokenizer: any Tokenizer) throws -> [Int32] {
        var ptr: UnsafePointer<CChar>? = nil
        var length: Int = 0
        let status = xg_matcher_find_jump_forward_string(matcher, &ptr, &length)
        guard status == XG_OK else {
            throw GrammarError.commitFailed(
                captureShimError(
                    status: status, fallback: "xg_matcher_find_jump_forward_string")
            )
        }
        guard length > 0, let base = ptr else { return [] }

        // xgrammar owns the bytes through a thread-local std::string.
        // Copy into Swift memory immediately so any later shim call
        // that reuses the buffer (including the xg_matcher_accept_token
        // calls below, which don't touch g_jump_forward_buffer today
        // but could via future exception paths) can't invalidate the
        // slice we're encoding.
        let data = Data(bytes: UnsafeRawPointer(base), count: length)
        guard let fastForwardString = String(data: data, encoding: .utf8) else {
            // Non-UTF-8 FF string: surface as "no FF" rather than
            // failing.
            return []
        }
        let fastForwardByteLength = fastForwardString.utf8.count

        let encoded = hostTokenizer.encode(text: fastForwardString, addSpecialTokens: false)
        guard !encoded.isEmpty else { return [] }

        // Walk the encoding from the front, retaining tokens whose
        // cumulative decoded byte length is strictly less than
        // `fastForwardByteLength`. Stops at the first token whose inclusion
        // would reach or cross the FF boundary — that token is the
        // merge-able one and belongs to the sampler.
        var safeCount = 0
        for i in 1 ... encoded.count {
            let prefixDecoded = hostTokenizer.decode(tokenIDs: Array(encoded[0 ..< i]))
            if prefixDecoded.utf8.count < fastForwardByteLength {
                safeCount = i
            } else {
                break
            }
        }
        guard safeCount > 0 else { return [] }

        var accepted: [Int32] = []
        accepted.reserveCapacity(safeCount)
        for id in encoded.prefix(safeCount) {
            let tokenID = Int32(id)
            let acceptStatus = xg_matcher_accept_token(matcher, tokenID)
            if acceptStatus != XG_OK {
                // Mid-FF rejection: the host tokenizer re-encoded the
                // FF bytes into a token whose boundaries don't line up
                // with the grammar's forced region. The matcher refuses
                // the id; we bail out of the accept loop with the
                // already-accepted prefix intact. Tick the counter so
                // loop-level observability can page on sustained
                // disagreement; `_fastForwardDisagreementCount` is
                // lock-protected via the caller's pre-held `lock`.
                _fastForwardDisagreementCount += 1
                break
            }
            accepted.append(tokenID)
            if isMatcherTerminatedLocked() { break }
        }
        return accepted
    }

    /// xgrammar does not accumulate a log stream, so this always returns `nil`.
    ///
    /// Retained as a no-op so the diagnostic path in
    /// `GuidedGenerationLoop` stays shaped around an optional log
    /// string without needing a trait on the loop itself.
    public func flushLogs() -> String? {
        return nil
    }

    /// Observability counter for mid-fast-forward tokenization disagreements.
    ///
    /// The number of times `emitFastForwardLocked` saw the host tokenizer
    /// re-encode xgrammar's FF string into a token the matcher then
    /// rejected. See `_fastForwardDisagreementCount` for the rule about
    /// when this ticks. Surfaced as `var` (not `let`) so the loop can
    /// publish it through `GuidedGenerationLoop` telemetry. Read-locked
    /// so concurrent mask/commit callers see a consistent value rather
    /// than a half-torn Int (defense-in-depth on platforms that lack
    /// native atomic word loads).
    public var fastForwardDisagreementCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _fastForwardDisagreementCount
    }

    /// Roll back the most recently accepted `n` tokens.
    ///
    /// Restores the matcher to the state it held before those commits. A
    /// subsequent `computeMask()` must return a bit-identical mask to the
    /// one observed at that prior state.
    ///
    /// `n` counts actual xgrammar acceptances, not Swift commit calls:
    /// a fast-forward-emitting commit accepts `1 + result.tokens.count`
    /// tokens, and the caller must pass the same count to rollback.
    public func rollback(_ n: Int32) throws {
        lock.lock()
        defer { lock.unlock() }
        let status = xg_matcher_rollback(matcher, n)
        guard status == XG_OK else {
            throw GrammarError.rollbackFailed(
                captureShimError(status: status, fallback: "xg_matcher_rollback n=\(n)")
            )
        }
    }

    /// Fork the matcher, returning a `GrammarConstraint` with independent matcher state.
    ///
    /// The fork shares the compiler and compiled-grammar handles with
    /// this constraint but carries an independent `GrammarMatcher` state.
    /// Mirrors xgrammar's `GrammarMatcher::Fork()` contract: deep-copy of
    /// per-session state, shared immutable compiled grammar and
    /// tokenizer. Commits on one side do not affect the other.
    ///
    /// Ownership: the fork does not own the shared compiler/compiled
    /// handles; only the originating constraint is responsible for
    /// freeing them. The fork retains a Swift-level reference to the
    /// parent to prevent the parent's `deinit` from running (and
    /// invalidating the shared handles) while the fork is still alive.
    /// The fork owns its own matcher handle and frees it on deinit.
    public func clone() throws -> GrammarConstraint {
        lock.lock()
        defer { lock.unlock() }

        var forkedMatcher: OpaquePointer?
        let status = xg_matcher_fork(matcher, &forkedMatcher)
        guard status == XG_OK, let forkedHandle = forkedMatcher else {
            throw GrammarError.forkFailed(
                captureShimError(status: status, fallback: "xg_matcher_fork")
            )
        }
        return GrammarConstraint(fromFork: forkedHandle, parent: self)
    }

    /// Query termination while already holding `lock`.
    ///
    /// Named `Locked` as the convention for "caller must hold the lock";
    /// this avoids re-entrancy with `NSLock` (which is not reentrant).
    private func isMatcherTerminatedLocked() -> Bool {
        var result: Int32 = 0
        let status = xg_matcher_is_terminated(matcher, &result)
        return status == XG_OK && result != 0
    }
}

// MARK: - Vocab encoding helpers

extension Array where Element == String {
    /// Call `body` with a buffer of per-string NUL-terminated C pointers.
    ///
    /// Each pointer in the `[UnsafePointer<CChar>?]` buffer is the
    /// NUL-terminated UTF-8 encoding of the corresponding string. The
    /// backing byte storage and pointer buffer remain valid for the
    /// duration of `body` and are freed immediately after.
    ///
    /// Bridges `[String]` → xgrammar's `const char *const *` vocab
    /// contract without the "capture `baseAddress` outside the
    /// closure" pattern. UTF-8 bytes for all strings are packed into
    /// a single contiguous `[CChar]` buffer; each per-token pointer
    /// is an offset into that buffer. Lifetime is enforced by the
    /// nested `withUnsafeBufferPointer` scopes — no dangling pointers
    /// escape.
    ///
    /// Used by `GrammarTokenizer` and shared by any other path that needs
    /// the same `[String]` -> C bridge.
    func withCStringPointers<R>(
        _ body: (UnsafeBufferPointer<UnsafePointer<CChar>?>) throws -> R
    ) rethrows -> R {
        var offsets: [Int] = []
        offsets.reserveCapacity(count)
        var bytes: [CChar] = []
        for string in self {
            offsets.append(bytes.count)
            for codeUnit in string.utf8 {
                bytes.append(CChar(bitPattern: codeUnit))
            }
            bytes.append(0)  // NUL terminator
        }

        return try bytes.withUnsafeBufferPointer { bytesBuf in
            // `bytes` is empty when `self` is empty; in that case
            // `baseAddress` may be nil. xgrammar tolerates a NULL
            // vocab pointer when vocab_count is 0 (the shim's
            // fast-fail guard only rejects NULL with non-zero count),
            // so we pass through either way.
            var pointers: [UnsafePointer<CChar>?] = []
            pointers.reserveCapacity(offsets.count)
            if let base = bytesBuf.baseAddress {
                for off in offsets {
                    pointers.append(base.advanced(by: off))
                }
            }
            return try pointers.withUnsafeBufferPointer { ptrsBuf in
                try body(ptrsBuf)
            }
        }
    }
}

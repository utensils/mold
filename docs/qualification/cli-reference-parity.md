# CLI reference transport parity audit

## Scope and evidence

This audit covers H3 reference preparation and submission, reference batching,
mesh uploads, capability error handling, remote license acceptance, and generated
agent guidance. It does not qualify every CLI command or perform a production
render.

Plato answered unauthenticated `GET /api/capabilities` with
`reference_uploads.available: false` and `requires_api_key: true`; its status
reported version 0.34.0. These are compatible: API authentication is disabled,
while upload sessions require a credential identity. The server accepts bounded
inline reference media when authentication is explicitly disabled.

The CLI previously required `MoldClient::has_api_key` before opening Ref2VA files
and always bound their descriptors through upload sessions. This contradicted
the server's inline path and native client transport selection. It was unrelated
to model license acceptance.

## Reviewed adjacent paths

- H3 uploads: preserve authenticated one-use sessions and descriptor integrity;
  use bounded inline references when uploads cannot be selected. A server's
  authentication or policy refusal remains authoritative.
- H3 batching: upload-session limits must apply to the selected transport,
  rather than rejecting every reference batch before selection. Inline references
  share a 32 MiB decoded budget across all batch siblings before payload copies
  are made; this bounds memory and leaves room for JSON/base64 framing.
- Mesh workflows: capability lookup failures must not silently become inline
  fallback. Only a missing legacy endpoint (404/405) is compatibility fallback;
  authentication, rate limits, server failures and malformed replies surface.
- License commands and pulls already read and record acceptance on the target
  server. Remote generation does not require copying that state onto the client.
- Generic generation's local fallback is limited to connection failure;
  authentication/policy errors do not authorize another execution path.
- Identity and transparency preflights must preserve actual server failures,
  rather than misreporting them as unsupported features.
- Canonical prompting and skill sources are checked across all renderer profiles;
  generated documents are regenerated, not edited independently.

## Remaining catalog compatibility boundary

The CLI resolves model identity/family/defaults from its local catalog before
reference routing. A server-only custom model or a newer server's capability can
therefore require a matching CLI catalog/version. Current known-model reference
profiles share the same core authority; this audit found no same-version
mismatch for those recipes. Fixing only the reference flag would leave model,
prompt and default resolution inconsistent. A future remote-catalog authoring
change needs dedicated ambiguous-model/prompt, offline fallback and version-skew
routing tests; this transport fix does not claim to implement it.

## Prompt guidance research

The official MiniMax H3 prompt-writing repository was refreshed to
[`d21241f0a4b3acbb34c97dae47fa417b7065e438`](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438).
Its [Ref2VA guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/ref-en.txt)
lines 24–88 distinguish reusable subjects from source assets, lines 121–147
specify task prefixes and distinguish motion references from actual video edits,
and lines 246–278 distinguish target dialogue from borrowed voice timbre. The
canonical Mold leaf now includes those distinctions within the expander budget.

Mold retains its own conditioner's ordered media labels: a supplied video with
an audio track consumes an audio label. Upstream's optional-track wrapper prose
is not substituted for this wire behavior. Likewise upstream text-only mode
descriptions do not imply that Mold's released FL2VA model accepts text-only
requests; first and/or last boundary frames remain required.

The pinned model license at revision `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`
was fetched and re-read; its SHA-256 remains
`59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`, matching
`model_policy.rs`. This change leaves the existing reviewed model policy and
H3 governance unchanged; transport credentials are not model license acceptance.

## Validation

- The new H3 and mesh regressions failed before their corresponding fixes.
- All 927 CLI binary tests pass, including all skill renderer profiles and
  documented-command parsing. The first whole-suite run exposed an existing
  macOS `/var` versus `/private/var` temporary-path assertion in an unrelated
  cleanup test; canonicalizing the test environment's `TMPDIR` resolved it.
- All 14 core prompting tests pass; canonical generator `--check` passes.
- Independent review identified the inline batch amplification risk; the
  pre-allocation combined byte guard and regression address it.
- Plato discovery was read-only. No production generation, model download,
  server restart, app launch or simulator was needed.

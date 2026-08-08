# MiniMax H3 authorization gate

Status: **blocked**

Mold does not currently activate, advertise, download, or execute MiniMax H3.
The reviewed H3 Community License revision limits the territory in which its
rights apply, and this repository has no written authorization record covering
the current development and distribution environment.

This is a product compliance boundary, not a location-detection feature. The
gate is enforced from model identity and resolved family at the shared catalog,
request-validation, nested generation artifact, prompt-expansion, download,
cloud-provisioning, durable-sequence, and frozen-engine boundaries. Weight
presence, an environment variable, an HTTP header, a client flag, or a
geographic guess must never bypass it.

Artifact checks are relative to the configured trusted models root. Storage
ancestors are placement, not model identity: an operator may safely use a UAT
home such as `/Volumes/ExternalStorage/mold-uat/minimax-h3`, while an H3-named
model, sidecar directory, ControlNet, LoRA, upscaler, or nested component below
that root remains gated.

## Reviewed sources

- [MiniMax H3 Community License, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
- [MiniMax license Q&A and authorization process, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/docs/QA-about-License.md)
- [Official implementation, pinned revision](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea)
- [Authorization tracking issue](https://github.com/utensils/mold/issues/831)

## Activation record

No authorization evidence has been accepted as of 2026-08-06.

| Field | Current record |
|---|---|
| Decision | Unavailable; fail closed |
| Policy owner | `utensils/mold` maintainers through issue #831 |
| Last review | 2026-08-06 |
| Authorization evidence | None accepted |
| Next mandatory review | Any upstream license/Q&A revision, proposed H3 artifact, or release touching H3 |

Activation requires all of the following in a reviewed change:

1. Written authorization whose scope explicitly covers implementation, local
   inference, automated tests and fixtures, distribution, and any hosted use.
2. A durable repository record of the allowed territories, products, users,
   attribution, downstream terms, generated-content duties, and expiry or
   revocation conditions.
3. Tests proving every disallowed path remains fail-closed and every newly
   allowed path follows that exact record.
4. Named ownership for recurring license review and immediate revocation.

If authorization expires, is narrowed, or is revoked, the compile-time gate is
restored before any release or hosted deployment proceeds.

## Release checklist

- [ ] Compare the pinned H3 license and Q&A revisions with their current
  upstream versions. Any change, or any proposed H3-specific artifact, blocks
  the release until the authorization record and policy tests are reviewed by
  the named compliance owner in issue #831.

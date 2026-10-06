---
paths:
  - "crates/mold-relay/**"
  - "scripts/relay/**"
  - "relay/aws/**"
  - "apps/macos/Sources/Mold/Settings/Remote*.swift"
  - "apps/macos/Sources/Mold/Settings/ManagedRelay*.swift"
  - "apps/macos/Sources/Mold/Settings/PhonePairingSheet.swift"
  - "apps/macos/rust/mold-macos-ffi/**"
---

# Optional remote relay

The relay is a GPU-independent Rust transport. `mold relay` forwards to this
crate; standalone `mold-relay` is the host connector artifact.
AWS production uses Regional REST streaming, frontend/router Lambda, API Gateway
WebSockets, DynamoDB epoch/admission leases and private S3 staged objects.
The direct Rust gateway remains a local development option. Preserve HTTP byte stream
semantics (SSE, Range/HEAD, uploads, keepalive and directional EOF), bounded
admission/backpressure, one-use session-owned stream IDs and cancellation.
Never replay mutations after reconnect. Bound stream inactivity using application
bytes in either direction; WebSocket heartbeats never extend the deadline.
The direct transport defaults to 3,600 seconds. AWS sessions are bounded by
Lambda duration; reopen SSE read streams and reconcile, never replay mutations.
Generation admits once through the durable queue and reconstructs saved results
with journaled response headers; reject no-save before admission.
Frames use v2 sequence/ACK credit windows, bounded gap/retry and epoch fencing.
Request EOF is logical: never half-close TCP before the host writes its response.
Stage requests above 2 MiB, verify complete size/hash before forwarding, and consume
grants once. Select finite-response staging before writing viewer headers.
Signed S3 URLs are pinned to the Mold bucket/account/region and reserved prefix;
never forward Mold credentials or follow redirects when using those URLs.
Namespaced object paths remain under `/_mold/objects/<host_id>/`; decoded dot
segments must be rejected before fetching, including percent-encoded traversal.

Tokens are separate from Mold API keys, owner-only file/env inputs, never URLs
or argument values or logs. Verify WSS except explicit localhost development.
Only numeric loopback targets are legal. Probe anonymous `/api/status` on the exact TCP connection to be forwarded;
consume its bounded Content-Length-framed 401 response and require keepalive
before forwarding. Reject redirects, close responses and ambiguous framing. Never inject an operator key or publish an auth-disabled host.

TLS terminates at the trusted cloud proxy; do not claim E2E encryption. Legacy
gateway access exposes one explicitly hosted server. Managed enrollment adds
isolated host namespaces with dedicated HTTPS root origins; never put host IDs
in path prefixes or permit connection frames/grants/jobs to cross namespaces.
Public proxy must block metrics, remove spoofed forwarding headers, use
HTTP/1.1 upstream and disable response buffering. AWS resources belong to
URandom Terraform; code/deployment artifacts belong to Mold. Never put tokens
in Terraform state/user-data or SSM command logs.

Native macOS Settings ▸ Remote Access keeps its existing grouped Form and
places **Pair your phone** in the first section. Saved machines reuse
PairingStore/PairingSheet. This Mac prepares managed remote access on that
explicit click through RemotePairingStore: keep the authenticated numeric
loopback listener, persist owner enrollment in SecretStore, start the embedded
outbound connector, publish its validated HTTPS origin, and verify a fresh
instance-bound pairing proof through that origin before displaying the QR.
Repeated clicks share preparation; errors/retry remain in the sheet, and stale
host sessions are cleared. Closing the sheet does not disable remote access.
Stop Remote Access cancels the connector and withdraws its advertised origin;
quit/engine death also stop it. Reuse saved enrollment when opted in on restart.
Never advertise LAN/Tailscale interfaces a loopback listener does not serve.

Managed enrollment uses POST `/_mold/relay/enroll` at the trusted configured
origin (`https://mold-link.urandom.io` for the native app). It is disabled unless
the gateway has a trusted enrollment base origin, wildcard host domain and
verified WSS endpoint. The bundled native app requires
`MANAGED_HOST_DOMAIN=mold-link.urandom.io`, matching its trusted enrollment host.
Production wildcard DNS/TLS and runtime/config deployment
are prerequisites, never inferred from a successful app build. Host IDs are
random 128-bit lowercase hex; owner tokens are separate random 256-bit bearers
stored as SHA-256 verifiers by the gateway. Capacity, leases and source quotas
are bounded; renew/delete require that namespace's owner token. Never trust
X-Forwarded-For for source quotas. Host, stream, transfer and media-job authority
must remain scoped to the enrolled namespace. Preserve legacy connector access.

Connection advertisements come from the actual bound listener and its interfaces,
plus explicit `MOLD_PUBLIC_URL`; never request Host/Forwarded headers. Routes are
bounded, exact credential-free origins. Existing clients learn only through an
already authenticated origin; alternative routes require the instance-bound,
nonce- and kind-bound credential-free proof. This detects accidental address
reuse, not HTTP forwarding MITM. Keyless and legacy servers retain the original
route. Saved route metadata must preserve host UUID and secret-store identity,
respect edits/removal/rekey, and never replay uncertain mutations.

Anonymous API route proofs accept only server-minted random paired grants;
arbitrary operator API keys remain explicit-address only and receive empty
authenticated route catalogs. Gate clients before computing/sending a hash tag,
including cached catalogs. Pairing-token proofs remain bounded to active minted
tokens; probes never consume or extend them. A hash tag is stable and would
permit offline guessing of weak secrets; never opt arbitrary operator strings
in based on length or apparent randomness. Normal operator authentication is
unchanged. Plain HTTP still trusts the network.

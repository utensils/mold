---
paths:
  - "crates/mold-relay/**"
  - "scripts/relay/**"
  - "relay/aws/**"
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

Tokens are separate from Mold API keys, owner-only file/env inputs, never URLs
or argument values or logs. Verify WSS except explicit localhost development.
Only numeric loopback targets are legal. Probe anonymous `/api/status` on the exact TCP connection to be forwarded;
consume its bounded Content-Length-framed 401 response and require keepalive
before forwarding. Reject redirects, close responses and ambiguous framing. Never inject an operator key or publish an auth-disabled host.

TLS terminates at the trusted cloud proxy; do not claim E2E encryption. One
gateway exposes one explicitly hosted server. Native This Mac remains private.
Public proxy must block metrics, remove spoofed forwarding headers, use
HTTP/1.1 upstream and disable response buffering. AWS resources belong to
URandom Terraform; code/deployment artifacts belong to Mold. Never put tokens
in Terraform state/user-data or SSM command logs.

Native macOS Settings ▸ Remote Access reuses PairingStore and PairingSheet
for inline expiring QR codes and device revocation. The Settings scene must
inject the shared pairing store. Codes carry the selected saved host address;
never substitute a gateway URL without explicitly adding that machine. The
private This Mac host does not offer pairing and this pane never starts a tunnel.

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

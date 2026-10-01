---
paths:
  - "crates/mold-relay/**"
  - "scripts/relay/**"
---

# Optional remote relay

The relay is a GPU-independent Rust transport. `mold relay` forwards to this
crate; standalone `mold-relay` is the cloud artifact. Preserve HTTP byte stream
semantics (SSE, Range/HEAD, uploads, keepalive and directional EOF), bounded
admission/backpressure, one-use session-owned stream IDs and cancellation.
Never replay mutations after reconnect.

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

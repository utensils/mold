---
paths:
  - "crates/mold-relay/**"
  - "crates/mold-secure/**"
  - "crates/mold-secure-client/**"
  - "crates/mold-server/src/secure_pairing/**"
  - "scripts/relay/**"
  - "relay/aws/**"
  - "apps/macos/Sources/Mold/Settings/Remote*.swift"
  - "apps/macos/Sources/Mold/Settings/ManagedRelay*.swift"
  - "apps/macos/Sources/Mold/Settings/PhonePairingSheet.swift"
  - "apps/macos/rust/mold-macos-ffi/**"
---

# Secure remote and LAN transport

Protocol 3 is a hard cutover. Cloud and non-loopback LAN access use authenticated
end-to-end encrypted WebSockets. Legacy cloud HTTP APIs, media, uploads and
version 2 upgrades return 426; there is no compatibility switch. Plain HTTP API
access remains only on the actual numeric loopback socket for the local engine
and connector. Host/Forwarded headers never establish local authority.

AWS production stays on Lambda, API Gateway WebSockets and DynamoDB admission
leases. The frontend returns public discovery metadata, a redirect to the
independently deployed trusted Studio, or 426. It never proxies plaintext content,
serves executable client code or receives an API bearer. Bulk transport may stage
only existing Noise ciphertext under `ciphertext/v3/`; follow
`relay/aws/OPAQUE-OBJECTS.md` for grants, bounds, checksum and cleanup.
AWS resources belong to URandom Terraform; runtime artifacts belong to Mold.
Never place credentials in Terraform state, command arguments, URLs or logs.

The canonical wire contract is `relay/aws/SECURE-PROTOCOL.md`. Use the shared
`mold-secure` implementation and cross-language fixtures: standard
Noise_XX_25519_ChaChaPoly_SHA256, distinct Ed25519 admission and X25519 Noise keys,
and host-signed bindings. Host namespace is the first 16 bytes of SHA256(host
Ed25519 public key). First trust comes from QR/fingerprint approval, never cloud
metadata or a silently accepted replacement key. Canonical signatures use exact
domain-separated binary bytes; never sign a JSON serialization.

The checked-in browser WASM is generated on `x86_64-unknown-linux-gnu` with
Rust 1.98.0 and the locked wasm-bindgen CLI. Other hosts consume that artifact;
native Rust/Apple builds remain cross-platform. Keep the exact-byte CI check.
See `crates/mold-secure/README.md` for regeneration and why compiler-host identity
is part of the reproducible build contract.

The server alone creates `$MOLD_HOME/secure/host.identity` (64 bytes). Connector
and clients use shared private-file helpers: owner-only POSIX permissions or a
protected current-user-only Windows DACL, bounded reads, no symlinks/reparse
points or extra hard links. A corrupt, permissive or missing connector identity
fails closed; never replace the server pin on an error. Keep key buffers zeroized.
Private journals must commit before a paid provider mutation: flush file contents
and POSIX directory entries, or use a Windows write-through same-directory move.
An uncertain commit fails closed and retains the journal for reconciliation.

Pairing QR v2 carries the host key binding, namespace, endpoints, optional cloud WebSocket URL,
a short-lived invitation certificate and its private invitation seed. The seed
never leaves clients/host except inside the trusted QR. Cloud verifies signed
one-use admission, but only the host may consume the invitation and issue a
device grant. The claim proof binds invitation ID, Noise handshake hash and both
device public keys inside Noise. A distinct per-device API key is returned only
inside that encrypted channel and is bound to the device/grant in the local DB.
Operators' API keys are never injected into tunnels.

The connector signs fresh local request context covering the authenticated peer,
certificate, handshake hash, stream ID, method/path and nonce. Server middleware
accepts this header only from an actual loopback peer, verifies/replay-checks it,
and requires the matching device credential. Invited peers can only POST the
pairing claim. Live device certificates renew inside Noise; revocation removes
both the credential and grant, and closes active channels within the 1-second
policy polling interval. Cloud epoch updates revoke all certificates; individual
device revocation must not unintentionally revoke other devices.

Preserve HTTP semantics: SSE, Range/HEAD, uploads, streaming response bodies and
request cancellation. Request bodies are bounded to 64 MiB. Encrypted application
records multiplex at most 32 streams with fair credit. Application paths, headers,
status, stream IDs, content and logical EOF/cancellation stay inside Noise. Outer
version 3 frames expose only routing IDs and bounded reliable sequence/ACK data.
Reorder and deduplicate ciphertext before decrypting; retransmit identical bytes
at the same sequence, never encrypt new bytes with a reused nonce. Request EOF
is logical, not a TCP half-close. Interrupting an active response must report a
body error; only authenticated logical End may produce successful EOF. Heartbeats
do not count as application progress.
A reconnect starts a fresh Noise session; never replay uncertain mutations.

Cloud host admission additionally requires an enabled operator-provisioned public
key/principal registration. Runtime IAM may read but never write registration
rows; public enrollment is metadata-only. Admission has bounded pending source/global slots and per-host/device limits.
API Gateway socket source IP is trusted; X-Forwarded-For is not. DynamoDB TTL is
cleanup, never permission. Every frame checks namespace, current host session,
lease, epoch and membership. A PostToConnection 410 must be confirmed by
GetConnection before evicting a peer; retry a live peer once with exact bytes.
Diagnostics report scoped aggregate reasons and fleet metrics without raw logs,
message bodies, keys or credentials.

Route selection favors verified LAN, then Tailscale, then cloud. HTTPS browser
origins must not pre-filter encrypted LAN WebSockets: the browser enforces its
local-network permission and mixed-content rules; refusals fall back to cloud. Start lower
priority probes after bounded delays; the first valid best-priority route wins
without waiting for same-priority peers. Cancel losers and fence stale completions
after route replacement/removal. Advertise only interfaces the actual secure
listener serves. Validate bounded credential-free origins and preserve saved host
identity across route changes; encrypted host pins establish authority on every
route, independent of IP or DNS reuse.

Native macOS Settings ▸ Remote Access keeps its grouped Form and **Pair your
phone** first. Repeated pairing clicks share preparation and errors remain in the
sheet. Closing the sheet does not stop remote access. Explicit stop, quit or
engine death cancels the connector and withdraws its advertisement. LAN listener
lifetime follows local secure access and must not depend on cloud availability.
Production DNS/TLS, runtime/config deployment and cross-surface tests are separate
proof obligations; a successful app build does not establish live relay health.

The public pairing handoff is `website/public/pair/index.html`, copied verbatim
and checked by the website build. It must remain outside VitePress, without
analytics or third-party code, with its restrictive CSP and no-referrer policy.
The one-time invitation is cleared from history and kept in memory, never an
HTML attribute. Documentation navigation into pairing or Studio must load a new
document. No Mold documentation page may load third-party scripts: all pages on
the utensils.io origin share access to Studio key storage, so path exclusions do
not isolate keys. The website build scans generated HTML/JS for retired tracking
loaders as well as checking the standalone pairing artifact.

Owner-controlled provider creation may pass `MOLD_SECURE_BOOTSTRAP` only in the
authenticated provider control-plane request. The CLI consumes and removes it
before starting runtime threads. It requires operator authentication and an
empty authority/media store; signed host/device binding, a 30-day-or-shorter
grant, API credential hash and instance UUID are imported in one DB transaction.
An owner-only intent file fences identity-file crash recovery. Exact replay
verifies the committed bootstrap identity without restoring revoked clients or
overwriting renewed grants. Never replace an existing volume's identity or log
the bootstrap envelope. Reuse its original paired profile on volume resume.

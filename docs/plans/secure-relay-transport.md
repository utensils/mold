# Secure transport, relay reliability, and future user access

Status: implementation plan, 2026-10-11. The current production relay is a
trusted TLS proxy, **not end-to-end encrypted**. Reliability fixes do not change
that security property. This plan replaces the retired trusted-proxy design;
the HTTP semantics and all client surfaces below are the compatibility contract.

## Verified problem and immediate repair

- Halcyon's installed native engine listens only on `127.0.0.1:7680`; an iPhone
  on the same Wi-Fi cannot reach that socket. Advertising a nearby name does
  not establish reachability. Discovery must describe actual listeners.
- The paired HTTPS origin returned HTTP 503 while the local engine immediately
  returned its expected anonymous 401. A Library read surfaced a generic
  mutation-reconciliation diagnostic to the user.
- CloudWatch recorded 738 frontend throttles over the examined 24 hours. Its
  reserved concurrency is 40; router concurrency was below its 128 cap and had
  no throttles. Frontend p95 duration was 5.587 seconds. These are measured
  observations, not a steady-state performance or cost guarantee.
- Native and browser route selection waited for every candidate, including a
  slow relay after LAN was already verified. Native Bonjour resolution had no
  timeout/cancellation, and new discovery did not recover an offline paired
  machine using its persisted identity.

Repair with bounded, priority-aware route probing; authenticated same-instance
discovery recovery; truthful errors; proven connection liveness before lease
replacement/removal; bounded control/data scheduling; fewer redundant ACKs; and
headless managed enrollment. Keep regression tests and live diagnostics.
Do not replay a mutation on another route after an uncertain result.

The native Remote Access opt-in may open a separate authenticated LAN listener
using the exact engine router and identity. Its private loopback socket stays
stable. Publish the actual port, close existing streams on Stop, and withdraw
Bonjour/catalog entries on shutdown. The listener exposes the same pinned encrypted session as the cloud path;
ordinary remote HTTP API requests refuse with an upgrade instruction.

## Security boundary

The trusted endpoints are the host process and the user's installed client.
The relay, its administrator, Lambda, DynamoDB, S3, and the intervening network
must not learn API keys, request URLs, prompts, results, source media, headers,
or decrypted response content in secure mode. TLS remains enabled on public
connections but is not the E2EE boundary.

The relay necessarily observes routing identifiers, connection timing, IP
addresses, ciphertext lengths, and usage. E2EE does not hide this metadata or
prevent denial of service. Endpoint compromise remains out of scope. Do not
claim that deleting a grant erases content a previously authorized device saved.

## Identity and authorization

Keep five concepts separate:

| Concept | Authority and purpose |
| --- | --- |
| Host identity | Persistent host public/private key; independent of hostname, IP, cloud namespace and process restart |
| Relay enrollment | Permission to occupy a cloud routing namespace; never an API or decryption credential |
| Device identity | Distinct key per client installation; private key in platform secure storage or owner-only headless file |
| Principal | Host-local user/account identity; initially the existing owner, later separate users |
| Grant | Host-issued mapping of device to principal, scopes and revocation generation |

Existing per-device paired credentials already have separate revocation. Extend
that model rather than replacing every client with one shared encryption key.
Transport accepts a verified device identity and delivers an authenticated
principal/grant context to the existing API authorization layer. A device key
is authentication, not permission to every API. Never inject an operator API
key on behalf of an arbitrary secure tunnel.

Prepare versioned grant records with `device_id`, `principal_id`, key fingerprint,
scope set, creation/expiry, revoked state and monotonic authorization generation.
The first implementation can map existing access to the owner principal while
keeping pairing/grant administration restricted to the operator. Implementing
an account service or inventing roles before product requirements is unnecessary.
Future roles must be enforced at host APIs, including reads/media and mutations.

Revocation closes the affected secure sessions, rejects new handshakes and
prevents new object grants. Check current authorization before dispatch and
before releasing protected media. Other devices keep working. Identity reset
requires explicit re-pairing; an IP/port change never silently replaces a pin.

## Protocol and implementation choice

Use a specified authenticated key-exchange protocol with an established library,
not a copy of a home-grown handshake. Use Noise XX with X25519,
ChaCha20-Poly1305 and SHA-256, fresh ephemeral keys for each session, explicit
key confirmation, and no early application data. Pin the host key from the
pairing QR or separately verified fingerprint; the host approves the device key
through a one-use, short-lived invitation inside the authenticated channel.

Bind protocol name/version, host identity, device identity, namespace and session
nonce to the handshake transcript. Authenticate the selected protocol version;
once paired in secure mode, never fall back to trusted-proxy/plain HTTP after
failure. A raw discovery record or relay response cannot downgrade that choice.

Prefer one Rust secure-channel implementation built natively, through a narrow
C ABI for Swift, and as WebAssembly for browser clients. Prove iOS, Android,
Windows and WASM build/runtime support before selecting the library. The pinned `snow` 0.10.0 implementation is
not an audited guarantee: its upstream README explicitly says it
has not received a formal audit. Review the exact pinned implementation and
protocol integration, use published vectors, and add cross-language fixtures.
Do not call an independent code review a cryptographic audit.

Secure records sit above the existing ordered/reliable byte transport. Reorder
and deduplicate bounded wire frames before feeding an ordered Noise session;
never skip missing bytes. Authenticate stream IDs, record type, method/target,
response status/headers, sequence, EOF and cancellation within the channel.
Use separate directional keys, monotonically increasing counters, bounded
record sizes and specified rekey/session-lifetime limits. Retransmit the same
ciphertext within a session; never encrypt different data with a reused nonce.
Reconnect creates fresh session keys and terminates uncertain requests.

## Lambda and low cost

Secure clients keep one bounded WebSocket session to their host and multiplex
HTTP-equivalent requests over it. Lambda only authenticates relay admission and
routes opaque records. DynamoDB stores bounded membership/epoch/quota state;
A private S3 bucket carries bounded batches of already-encrypted Noise bytes
under random host/session paths. Admission and short-lived object grants remain
scoped to live sessions; neither plaintext nor decryption keys enter S3. There
is no content parsing or payload storage in DynamoDB, no long-lived frontend Lambda per API/SSE
stream, no EC2, and no provisioned concurrency. The HTTP frontend serves only
public discovery, a redirect to the independently deployed Studio, and explicit
426 refusals for retired plaintext endpoints.

Bound sessions and concurrent streams per host/device/principal, handshake
rate, object count/bytes/lifetime, frame size, reorder memory and work queues.
Apply fair scheduling so uploads cannot starve status, cancellation or media.
Cloud admission is a cost/abuse control; the host remains the permission authority.
Use strongly consistent membership and epoch-conditional cleanup; AWS
`$disconnect` is best effort. Do not widen concurrency blindly to mask leaked
requests. Measure latency, throttles, bytes, frame/ACK counts and spend before
and after. Reserved concurrency has no standing compute charge; active work
still costs money and byte-heavy WebSocket forwarding is not automatically cheap.

## Complete API and media parity

Transport carries method, exact encoded target, request/response headers and
body as encrypted application records. Preserve status codes, binary bytes,
repeated queries, multipart bodies, HEAD, Range/206/416, content disposition,
incremental SSE, cancellation and logical directional EOF. Keep durable
operation IDs and accepted-job reconciliation; connection retry is never
permission to repeat POST/DELETE/PATCH.

Uploads and downloads stream through encrypted WebSocket records. Each direction
has a 1 MiB window per HTTP stream, up to 32 streams per device session, with
16 KiB data chunks and consumption-driven credits. This removes the old Lambda
buffered ingress and streaming response size ceilings without introducing a
second cryptographic file format. Existing host request admission remains
64 MiB. Large responses are streamed without an application-wide body limit.
Range and HEAD are ordinary authenticated encrypted HTTP requests. Browser
media uses a pull-driven service worker bridge from the independently trusted
Studio; native AVFoundation uses a resource loader. Neither adapter downloads
a whole video before playback. The measured WebSocket-only baseline was about
$1.8/GiB and 0.6 MiB/s sustained, so this cutover includes ciphertext object
batching below Noise. Original sequence/credit semantics, authenticated HTTP
headers, Range, HEAD and EOF are unchanged. Object downloads are bounded and
validated before ordered Noise consumption, with ACK only after acceptance. Remove old temporary plaintext staged
objects before describing the deployed system as E2EE.

| Surface | Required adapter and acceptance |
| --- | --- |
| Headless host | CLI enrollment, persistent host identity, noninteractive owner-only state, service restart/reconnect, no GUI dependency |
| CLI/MCP/Discord | Shared Rust request/SSE/upload/download adapter; verify every raw HTTP bypass |
| Native iOS/macOS | Shared transport plus stream/upload/download paths; secure Keychain/storage; AVPlayer resource-loader or bounded local decrypted-file adapter with seeking |
| Web/Tauri desktop/mobile | Shared API/event/media adapter, WASM secure channel, encrypted uploads, image/GLB decoding and seekable video support |
| Generic curl/HTTP consumers | Existing authenticated host loopback API; remote callers use the secure Mold CLI or client library |

Ordinary curl cannot decrypt a remote secure session. Local operator HTTP remains
loopback-only; no general-purpose remote plaintext compatibility bridge is shipped.
Native media adapters use a scoped local bridge or resource loader and cannot
silently send the encrypted object URL directly to AVPlayer and claim parity.

## Browser trust and migration

A browser bundle served by the relay can be replaced by that relay and steal
keys. To protect against a compromised relay, distribute the secure browser
client from an independently trusted origin, or as an installed verified app.
State that origin and the app update/signing channel as trust dependencies.
CSP/SRI controlled by the same relay does not solve active code substitution.
Do not promise E2EE against a server that also controls the client code.

Owner decision: **hard cutover at the end of this delivery**, with no legacy
proxy mode retained. Build and verify all supported host/client adapters first,
then deploy the coordinated E2EE-only release. Old remote HTTP API/transport
versions explicitly refuse with an upgrade/pairing instruction. Delete the
plaintext forwarding implementation and old secret-based transport admission;
do not leave a compatibility flag, fallback or hidden old endpoint.

Existing TLS pairings cannot silently establish the new secure host pin: the
old relay sees their bearer credentials and can substitute advertised keys.
Require a fresh QR/fingerprint verified directly on the host (or another
independently trusted channel). The cloud capability response is discovery,
never the trust anchor. The secure browser distribution must have a separate
deployment/signing authority from the relay, not merely another DNS name under
the same compromised administrative boundary.

Revocation closes the affected active sessions within the one-second host policy
refresh interval and blocks further dispatch immediately at the API authority.
Already delivered plaintext cannot be clawed back. Short-lived S3 URLs expose
only existing Noise ciphertext; session keys are never exported. Active object
I/O closes with its session; exact-object cleanup and lifecycle expiry bound
storage. A revoked device cannot start new application requests.
Production remains labeled as TLS proxy until the coordinated secure release
passes every gate. No encryption claim based merely on a version field or
unused capability scaffold.

## PR sequence and proof

1. Reproduce reliability/LAN/CLI failures with regressions and diagnostics; keep
   the existing production deployment running while the coordinated replacement
   is built and reviewed. Do not deploy an intermediate compatibility mode.
2. Protocol ADR, threat model, grant/migration design, exact dependency selection
   and cross-platform build spike. Independent security-focused review before
   wiring a cryptographic protocol into production.
3. Host/Rust secure transport and headless lifecycle with malicious-relay tests:
   wrong host/device, replay, reorder, drops, epoch crossover, downgrade,
   revocation, expired invitations, nonce exhaustion and reconnect ambiguity.
4. Opaque Lambda admission/routing with bounded quotas and no plaintext staging.
   Inspect frame/log fixtures with recognizable secrets to prove application
   plaintext is absent from cloud components.
5. Swift and browser/Tauri adapters, all HTTP bypasses, native video seek and
   background/foreground recovery, multiple devices and two different scopes.
6. Exact-head CI, independent review, staged live deployment with verified v3-only recovery,
   LAN→cellular→LAN and offline/restart proof, then the authorized hard cutover and explicit device re-pairing.

Acceptance matrix must include at least two simultaneous clients; one revoked
while the other streams; two principals with different access; same-LAN direct
selection; remote fallback; a headless host with no app running; large upload
and download above Lambda ingress/response thresholds; Range/HEAD and seek;
long SSE reconnect; and a lost mutation response without duplicate execution.
Record p50/p95 latency, first-byte and media throughput, cloud invocation/byte
costs and resource ceilings. Simulator evidence and physical-device evidence
are separate. Passing unit tests alone does not prove deployed parity.

All Mold AWS assets stay inside the Terraform `mold-relay` module. Every
taggable resource, including CloudFormation children, carries `Project=mold`,
`Service=mold-relay`, `Environment`, `ManagedBy`, and `BillingGroup`. Verify
these tags on live resources and activate their AWS cost allocation keys.
Document untaggable children and shared Route53 billing separately rather than
claiming tags can divide a shared resource's bill. CLI clients and headless
hosts are explicit release gates alongside browser, Tauri and native apps.

## Sources and reference boundaries

- Current source: `crates/mold-server/src/auth.rs`, `connections.rs`,
  `crates/mold-relay/src/secure.rs`, `relay/aws/frontend.mjs`, `secure-router.mjs`,
  shared Rust/Swift/TypeScript transports and `scripts/relay/diagnose.py`.
- Zephra inspected at `daefdc067c6903c0f3c184354c76544ee8bb1734`:
  `Relay/link/README.md` describes opaque routing and distinct guests;
  `Packages/ZephraLink/Sources/ZephraLinkProtocol/Security/HandshakeKeys.swift`
  uses a custom Noise-style four-DH/HKDF schedule; `SecureChannel.swift` and
  `SecureChannel+Opening.swift` implement directional AES-GCM counters/windows.
  Reuse the separation of admission and encryption, not an unsupported claim
  that its custom handshake is the standard Noise protocol.
- [Noise specification](https://noiseprotocol.org/noise.html)
- [Snow implementation and audit caveat](https://github.com/mcginty/snow)
- [AWS disconnect behavior](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-websocket-api-route-keys-connect-disconnect.html)
- [AWS streaming constraints](https://docs.aws.amazon.com/apigateway/latest/developerguide/response-transfer-mode.html)
- [Lambda concurrency](https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html)
- [Lambda pricing](https://aws.amazon.com/lambda/pricing/)

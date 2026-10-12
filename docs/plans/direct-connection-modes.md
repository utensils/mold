# Explicit secure and regular connections

Secure remains the default for a newly added machine. Regular is an explicit,
direct HTTP(S) choice, including WAN addresses. A client keeps one selected mode
for each saved machine; different clients may use either mode concurrently.
Changing mode requires its own pairing or explicitly entered direct credential.
A failed secure request never falls back to regular HTTP. The managed proxy
accepts only the pinned Noise protocol; its plaintext API remains unavailable.

## Pairing payloads and endpoints

The existing secure version 2 QR payload is unchanged. An absent `mode` means
`secure`; an explicit `mode: "secure"` is equivalent. Regular payloads are a
disjoint union with these fields:

```json
{
  "type": "mold.mobile-pairing",
  "version": 2,
  "mode": "regular",
  "base_url": "https://direct-host.example:7680",
  "instance_id": "host-instance-uuid",
  "name": "Host display name",
  "pairing_token": "64 lowercase hexadecimal characters",
  "expires_at": 1800000000
}
```

Regular payloads contain no relay endpoints, WebSocket URL, Noise identity,
secure grant, or invitation signing key. `base_url` is a credential-free HTTP(S)
origin with no path, query, or fragment. Public WAN names are supported; private
address ranges are not an allowlist. Managed proxy names are rejected using
parsed hostname label boundaries, not substring matching.

An authenticated owner creates the regular invitation with
`POST /api/pairing/sessions` and JSON
`{"mode":"regular","direct_url":"https://direct-host.example:7680"}`.
An empty body preserves secure invitation creation. The token contains 256 bits
of OS randomness, expires after 120 seconds, is stored only as a hash in a
bounded in-memory invitation set, and is single use. Restart invalidates
unclaimed tokens. Invitation material must remain out of URLs and logs.

A regular client sends directly to `POST /api/pairing/regular/claim`:

```json
{"token":"the QR pairing_token","client_name":"My phone","client_kind":"ios"}
```

The response is:

```json
{
  "mode": "regular",
  "api_key": "new per-device bearer",
  "instance_id": "host-instance-uuid",
  "hostname": "Host display name",
  "base_url": "https://direct-host.example:7680"
}
```

The client verifies the expected instance and direct origin before saving the
credential. The response and invitation use `Cache-Control: no-store`. The same
client may instead explicitly enter a direct URL and API key, retaining generic
REST compatibility. Empty keys are usable only when the direct server has
explicitly disabled authentication. Secure credentials never authorize ordinary
HTTP, and regular credentials never authorize a Noise session.

## Host configuration and lifecycle

`MOLD_ALLOW_DIRECT_API=1`, `mold serve --allow-direct-api`, or Nix
`services.mold.allowDirectApi = true` enables regular API access on the configured
listener. The default is false. Ordinary numeric loopback engine access is
unchanged. Existing endpoint semantics, API-key enforcement, streaming, uploads,
Range/HEAD, SSE, cancellation, and individually revocable paired credentials are
preserved in regular mode. No proxy setting enables ordinary cloud HTTP.

Embedded shells use the typed `ConnectionAddresses` live regular-access setter
and bounded async readiness API. A separate regular listener publishes its
actual direct endpoints only after binding; disabling it closes its active
connections without ending secure LAN/cloud sessions. Host identity and paired
credentials survive mode changes, app restarts, and shell handoff for the same
`MOLD_HOME`.

Durable authority explicitly records regular versus secure credentials. Existing
pre-secure direct pairings may be migrated once; a missing secure device binding
after migration cannot turn its credential into regular authority. Mode
selection, credentials and transport routing are captured together for each
request, including media and queue mutations.

Admission recovery is bounded independently of HTTP delivery. A typed temporary
gateway-pressure notice before admission may start up to three handshakes with
short jittered backoff within the existing 15-second connection budget.
Authentication/protocol failures and any error after HTTP dispatch never repeat
the application request. Established carrier recovery keeps only original
ciphertext sequence records; it never resubmits a mutation.

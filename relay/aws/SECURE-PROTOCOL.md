# Mold cloud relay protocol 3

The Lambda router forwards opaque reliable byte records. It cannot decrypt
Noise payloads or authorize a Mold API operation. All application paths,
headers, request bodies, request IDs, statuses, events, errors and cancellation
live inside the authenticated encrypted channel. There is no HTTP proxy or
plaintext S3 content staging endpoint. Old upgrades fail with 426; old HTTPS
API requests fail with 426 and a secure-client-required error.

## Public identity and canonical signatures

All hex is lowercase with no prefix. Ed25519 and X25519 public keys are 32 bytes;
signatures are 64 bytes. The namespace is the first 16 bytes of SHA-256(host Ed25519
public key), rendered as 32 hex characters. Never derive identity trust from
cloud metadata: the first host pin comes from a trusted QR/fingerprint.

The binding JSON is `{signingKey, noiseKey, signature}`. Ed25519 signs
`ASCII("MOLD3-BIND\0") || Ed25519-public 32 || X25519-public 32`.

Integers below are unsigned 64-bit big endian, limited to JavaScript safe integers.
Strings use a two-byte unsigned big endian UTF-8 byte length followed by UTF-8.
No JSON serialization is signed. Deterministic shared test vectors are in
[`test/fixtures/admission-v3.json`](test/fixtures/admission-v3.json).

- JOIN: `"MOLD3-JOIN\0" || byte(3) || namespace 16 || role-byte || LP(connectionId) || nonce 32 || joining-device-Ed 32 || expiresAt-u64`.
- Grant: `"MOLD3-GRANT\0" || namespace 16 || device-Ed 32 || device-Noise 32 || grantId 16 || expiresAt-u64 || epoch-u64`.
- Invite: `"MOLD3-INVITE\0" || namespace 16 || inviteId 16 || invite-Ed 32 || expiresAt-u64 || epoch-u64`.

Roles are host=1, device=2, invite=3. Grants and invites are signed by the host's
Ed25519 identity. Grant JSON is `{id, expiresAt, epoch, signature}`; invite JSON
adds `signingKey` (the invitation's Ed25519 public key). The namespace and device
keys for grant verification come from the authenticated join context, never
from an overriding certificate field.

## Upgrade and admission

Cloud owners require an operator-provisioned registration at
`v3-registration#namespace`: `{namespace, signingKey, principalId, enabled}`.
There is no TTL; the namespace/key must match the verified host identity and
`enabled` must be true. Only operator IAM may write these rows. Public enrollment
returns metadata and cannot create authorization. Every active frame rechecks
registration and principal; disable/reassignment fences existing sessions.

Connect to the advertised WSS endpoint with `?v=3`; do not send bearer headers
or put any credential in URLs. Send:

```json
{
  "a": "hello",
  "v": 3,
  "namespace": "<hex 32>",
  "hostKey": "<host Ed hex 64>",
  "role": "device"
}
```

The router sends `{a:"challenge", v:3, connectionId, nonce, expiresAt}`. The
challenge is fresh, single-use, connection-bound and expires within 60 seconds
of upgrade. Reply `{a:"join", v:3, binding, signature, grant}`. For role host,
`signature` uses the host Ed key and `grant` is absent; include the persisted
nondecreasing `epoch` (initial 0). For device, use the device Ed key and a valid
host-signed grant. For invite, replace `grant` with `invite` and sign JOIN with
the invitation Ed key, while `binding` remains the joining device's own identity.

An accepted device gets `{a:"ready",v:3,sid,rid,role,epoch}`. The host receives
`{a:"open",v:3,sid,rid,role,binding,grant}` (or `invite`). The host validates that
public admission data again and sends an outer `accept` before Noise starts.
Clients must handle ready/accept arrival ordering. `sid` identifies this host
connection epoch; `rid` is this device's gateway connection ID. Both are pinned
into the Noise prologue. Host reconnect closes every old peer; no tunnel resumes.

The host's `policy` frame `{a:"policy",v:3,epoch}` increases the epoch, revokes
all old admission certificates and closes their sockets. The application host
is the final authority: it also revokes principals/grants locally and cancels
active decrypted operations. Cloud policy alone cannot provide authorization.

## Pairing

The QR contains the host Ed/Noise binding, namespace and WSS URL, plus a
short-lived signed invite and its private Ed25519 seed. The seed is never sent
to the cloud. An invite expires within 10 minutes and permits one cloud join;
reconnect requires a fresh invite until the device has its persistent grant.

After Noise verifies the host pin, the device must prove invitation possession
**inside** the encrypted channel as well; otherwise a malicious router could
impersonate an invited peer. The host verifies a claim signature over
`"MOLD3-CLAIM\0" || namespace 16 || inviteId 16 || Noise-handshake-hash 32 || device-Ed 32 || device-Noise 32`,
using the invitation public key. The host consumes its own local invite record
and applies the configured approval policy before issuing an application grant
and admission certificate inside that channel. Cloud membership does not grant
any application permission.

## Reliable ciphertext envelopes

The v2 sequence/ACK mechanics remain, with version 3 envelopes. `data` has
`{a:"data",v:3,sid,rid,seq,d}` where d is canonical base64 of at most 16384 bytes.
For optional bulk transport, data carries `o` instead of `d` and preserves all
original sequence slots; see [OPAQUE-OBJECTS.md](OPAQUE-OBJECTS.md).
`ack` has `{a:"ack",v:3,sid,rid,next,credit,ack_seq}`; credit is 0–64. `eof` has
`seq`; `accept` and `cancel` have no additional fields. Unexpected fields are
not forwarded. Send `{a:"heartbeat",v:3}` within each 90-second lease; the reply
includes sid. An invite/grant expiration can shorten that lease.

Buffer, reorder and deduplicate outer frames **before** passing bytes into
Noise. Retries resend the identical sequence and ciphertext. Never re-encrypt
with a reused nonce; never drop a gap to continue decrypting. The application
stream IDs, logical EOF and per-request cancellation are inside ciphertext.
An outer cancel terminates a device channel. Reconnect creates fresh Noise
state; only safe reads/SSE reconciliation may be reissued. Uncertain mutations
must not be replayed.

Pending upgrades are bounded to 4 per trusted source and 128 globally; reservations
last the 10-minute gateway idle limit, even for clients that never send hello.
Authenticated capacity is 32 hosts, 32 peers per host, at most 2 invitation peers.
Admission fields and opaque envelopes are bounded before cryptographic work.
No request payload or signature/private key is logged.

## REST and client surfaces

`GET /_mold/relay/info` and `POST /_mold/relay/enroll` return protocol 3 discovery
metadata (`websocket_url`, `host_domain`, `studio_url`) without an owner bearer.
Self-certifying host identity and signed WS admission replace cloud enrollment.
The cloud root redirects to the independently deployed trusted Studio origin.
LAN endpoints use the same pinned identity and encrypted application channel;
LAN discovery is only an address hint and cannot replace a key pin. The LAN
socket and client adapter are implemented by the host, outside this AWS router.

Host quota is a 90-second namespace lease, capped at 32 operator-registered
namespaces. The host row is the sole socket authority. Reconnects reuse the
namespace slot; disconnect does not release it prematurely across DynamoDB rows.
An unused namespace slot returns when its lease expires, at most 90 seconds after
its last renewal. This fences simultaneous JOIN and delayed-disconnect races.

# Encrypted managed host routing

Protocol 3 is a hard cutover. Cloud HTTP application/media/upload endpoints and
legacy WebSocket upgrades return 426. Cloud routes bounded opaque ciphertext;
Noise and host-owned device authorization protect application traffic. See
[SECURE-PROTOCOL.md](SECURE-PROTOCOL.md) for exact canonical bytes and envelope limits.

## Deployment and owner registration

URandom Terraform owns Lambda, API Gateway, DynamoDB, DNS/TLS and least-privilege
roles. Mold owns runtime ZIPs through `scripts/relay/deploy.py`. The frontend has
log permission only and returns public metadata or redirects to independently
trusted HTTPS Studio. There is no SSM bearer, plaintext API bridge or cloud-served
executable shell. The router brokers short-lived S3 grants only for already
encrypted Noise bytes under `ciphertext/v3/`; see [OPAQUE-OBJECTS.md](OPAQUE-OBJECTS.md).
It has no access to old plaintext prefixes.

A host's namespace derives from its persistent Ed25519 public identity. Operator
IAM must register that public key and an explicit stable owner/principal ID:

```sh
python3 scripts/relay/deploy.py provision --host-key <host-public-ed25519-hex> --principal <owner-id>
```

The `v3-registration#namespace` row has no TTL and can be written only by operator
IAM. A self-signed unknown host cannot allocate a live slot. Cloud enrollment is
public discovery, never authority. Repeated matching provisioning is idempotent;
reassignment is refused. `--disable` fences active traffic/heartbeats. Registration
does not grant API access: only the host can issue and revoke paired device grants.

## Headless host and clients

Start an authenticated Mold server. It alone creates owner-private
`$MOLD_HOME/secure/host.identity`; the connector reads that exact identity and
refuses missing, corrupt, permissive, linked or reparse-point files. POSIX and
Windows use the shared native private-file implementation. Never copy private
host identity into a client or into deployment metadata.

The host connector uses the advertised WSS URL and a numeric authenticated
loopback target. The standalone `mold-relay` and `mold relay` expose the same
connection commands. Host settings must advertise the secure WebSocket URL so
remote pairing can find the gateway.

Create an invitation on the owner machine:

```sh
mold pairing create --output /private/path/invitation.json
```

This calls local `POST /api/pairing/sessions` with operator authentication from
`MOLD_API_KEY` or its private-file form. Output is private, short lived and one use.
Transfer it by a trusted channel or use its QR; never publish the invitation seed.
On the client:

```sh
mold pair --invite-file /private/path/invitation.json --name workstation
```

Device Noise verifies the QR host pin before the claim. The claim proof binds
invitation, handshake and device public keys. The response issues a distinct
API credential and host-signed device grant only inside Noise. The client stores
private identity and grant securely, reconnects as that device and renews before
expiry. Revoking that device removes its local authority and closes its active
operations within the one-second policy update interval.

## Public metadata

`GET /_mold/relay/info` and `POST /_mold/relay/enroll` return HTTP 200 with only:

```json
{
  "protocol": 3,
  "websocket_url": "wss://<gateway>.execute-api.<region>.amazonaws.com/production",
  "host_domain": "mold-link.urandom.io",
  "studio_url": "https://utensils.io/mold/studio/"
}
```

Metadata is not a trust anchor and contains no owner/device credential. The host
binding comes from trusted pairing. Send no bearer headers or credentials in
WebSocket URLs. Signed challenge admission is described in the protocol document.

## Bounds and recovery

There are at most 32 live hosts, 32 peers per host and two invitation peers per
host. Unauthenticated pending upgrades are bounded per trusted socket source
and globally. Leases are checked synchronously; DynamoDB TTL is only cleanup.
Host reconnect advances its session and closes old peers. Capacity release is
fenced to the owning connection, so a delayed old disconnect cannot free a new
host's reservation.

GetConnection confirms PostToConnection 410 before eviction. A verified live
socket receives one retry of exactly the same ciphertext frame; clients reorder
and deduplicate before decryption. Reconnect creates a new Noise channel and
never replays uncertain mutations. Data/ACK buffers and multiplex stream credit
remain bounded; Range/HEAD, uploads, SSE and streaming responses share the same
secure transport.

Run local contracts with `node --test relay/aws/test/*.test.mjs`, the Rust secure
transport suites and private-file tests. Windows ACL/reparse tests must execute
on Windows CI; cross-compilation alone is insufficient. Production proof includes
live AWS routing, all client surfaces, revocation, failure recovery and bulk
transfer/cost measurements. Old plaintext temporary S3 objects must be verified
expired/removed before claiming complete E2EE cutover.

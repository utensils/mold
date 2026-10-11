# Encrypted remote connections

Mold connects paired clients to a host through the same encrypted protocol on
LAN and through the optional Lambda relay. The relay carries ciphertext; HTTP
paths, API credentials, prompts, uploads, progress events and media stay inside
a Noise XX session between the client and the pinned host. The cloud still sees
connection identities, IP addresses, timing and traffic sizes.

This is a hard protocol cutover. Upgrade the host and every client, then pair
again. Old bearer-only proxy credentials and raw remote HTTP clients are not
accepted. Plain HTTP remains available only on the host's private loopback
engine socket. A shared-network address is an address hint, not a trust source.

## Pair in the apps

On the host, open **Settings → Remote Access** and create a pairing code. Scan
or paste it on the client. The code contains the host's public key and a short
one-use invitation; transfer it privately. Each client receives its own identity
and independently revocable grant. Forgetting a client locally removes its local
access; revoke it on the host to invalidate every retained copy of that grant.

In hosted browser Studio, **Settings → Machines → Switch machine** opens the
connection screen while retaining saved pairings. The selected machine's menu
and detail page offer **Disconnect** and **Forget** too. Forget removes the local
pairing and returns to that screen; if removal fails, retry remains available.
Choose another saved machine or paste a fresh pairing link. Another browser tab
changing its selected machine also returns this tab to the connection screen.

Native clients prefer an advertised LAN address and retain the same pinned
identity when moving to the relay. An HTTPS browser may need the WSS relay
because browsers restrict unencrypted WebSocket URLs from HTTPS pages. Such
browser restrictions do not weaken the encryption or change the host pin.
Stopping Remote Access closes its listener and existing encrypted sessions.

## Headless host

Start an upgraded authenticated server first. Keep the operator key in the
service environment or an owner-only key file, never command arguments. The
server creates `$MOLD_HOME/secure/host.identity`; the connector only reads it.
The identity uses mode 600 with a private parent on Unix, and a protected owner
ACL on Windows. A missing, linked or insecure identity fails closed.

Configure the server's `MOLD_PUBLIC_URL` to its assigned HTTPS namespace origin
and `MOLD_SECURE_WEBSOCKET_URL` to the deployment's credential-free WSS endpoint.
The namespace is the first 16 bytes of SHA-256 of the host Ed25519 public key,
encoded as lowercase hexadecimal. The public signing key can be read from the
host's invitation export; it is safe to share with the relay operator.

The relay operator authorizes the host once through the deployment's AWS IAM
credentials:

```bash
python3 scripts/relay/deploy.py provision \
  --host-key PUBLIC_ED25519_KEY_HEX --principal OWNER_ID
```

With `MOLD_SECURE_WEBSOCKET_URL` configured, `mold serve` owns and stops its
outbound connector automatically. For a separately supervised connector instead:

```bash
mold relay connect --relay-url wss://API.execute-api.REGION.amazonaws.com/production \
  --target 127.0.0.1:7680
```

The standalone equivalent is `mold-relay connect`; build it with
`cargo build --release -p mold-ai-relay --features cli`. Use `--identity-file`
when the connector runs with a different `MOLD_HOME`. Pass the same WSS endpoint
to `mold pairing create --relay-url WSS_ENDPOINT` when exporting an invitation
for a separately supervised connector. Only numeric loopback
HTTP targets are accepted. Before forwarding any request, the connector probes
and reuses the exact socket to verify that anonymous requests are rejected.

Create a client invitation on the host:

```bash
mold pairing create --output "$MOLD_HOME/secure/invitation.json"
```

This uses the operator `MOLD_API_KEY` (also supports `@/private/key-file`), writes
a new owner-only file, and prints only its filename and the host public key.
The invitation expires after two minutes and may be claimed once. Transfer it
privately to the client, then run:

```bash
mold pair --invite-file /private/invitation.json --name "My laptop"
```

Set `MOLD_HOST` to the paired origin printed by the command. Existing commands
such as `mold server status`, `mold library`, `mold queue` and `mold run` use the
pinned encrypted transport automatically. Pairing state lives in
`$MOLD_HOME/secure/clients`; do not copy it between unrelated users or devices.
Expired/revoked grants require pairing again; active clients renew grants over
the encrypted connection. No operator key is injected into client requests.

## Provider-created hosts

The CLI and desktop provider flow can provision a new empty host automatically.
The owner-authenticated provider API carries `MOLD_SECURE_BOOTSTRAP`, a private
one-time envelope containing the host identity and a signed grant for the
creating client, plus a separate operator credential. This is trusted host
provisioning; the provider runs the GPU host and must be trusted with its keys.
Neither discovery nor the Lambda relay can create this authorization.

The server removes the envelope from its process environment before starting
runtime threads. It accepts an initial import only when the host identity,
authority database and media store are empty. A restart with the identical
bootstrap verifies the original identity without restoring revoked grants or
overwriting renewals. Resume an existing network volume with its original
paired profile; a newly generated bootstrap cannot overwrite that volume.
Keep provider credentials, bootstrap envelopes and client recovery files private.

## Streaming, media and limits

The protocol multiplexes up to 32 HTTP streams per client, with 1 MiB of body
credit per stream. Uploads are bounded to 64 MiB per request. Responses stream
with backpressure and do not have a total media-size ceiling. Range, HEAD,
conditional media reads, multipart uploads and server-sent progress use the
same encrypted HTTP records. Cancellation closes the corresponding upstream
request. A lost mutation is reported as an unknown outcome and is never replayed
automatically.

Native video uses the encrypted client directly. Browsers use a same-origin
ServiceWorker media bridge; Tauri uses a scoped loopback capability URL with
GET/HEAD, explicit media paths, Range support and window-lifetime cleanup.
Neither bridge gives the cloud plaintext media or creates an unrestricted
local proxy. Generic remote `curl`/AVPlayer HTTP URLs cannot speak the encrypted
protocol; use the official clients or a deliberately scoped local adapter.

## Lambda operations and costs

The production relay remains API Gateway WebSocket + short Lambda invocations +
DynamoDB. It has no always-running gateway. Larger groups of existing Noise
ciphertext can travel through short-lived, quota-bounded S3 objects; the
application keys remain on the client and host. Small requests and progress
remain inline on the WebSocket. Object addresses must match an independently
configured trusted bucket origin, and failed object allocation can fall back
only to the same original ciphertext. There is no plaintext S3 transfer path.
HTTP metadata endpoints cannot proxy the Mold API: legacy routes return an
upgrade-required response. WebSocket messages and acknowledgements incur
per-message costs, so bulk transfers are not free even though long-running
HTTP Lambda invocations have been removed.

See [the deployment scripts](https://github.com/utensils/mold/tree/main/scripts/relay)
for provisioning, bounded diagnostics and deployment validation, and
[the protocol specification](https://github.com/utensils/mold/blob/main/relay/aws/SECURE-PROTOCOL.md)
for canonical signatures and wire bounds. Diagnostics must not log pairing
seeds, API keys, decrypted requests or response bodies. TLS and the host key pin
are separate checks; never replace a pin with a key learned from the relay.

## Developer transport verification

The `secure_fixture` and `probe` examples use synthetic identities and media.
Never authorize their fixed keys on a production relay. Their optional inputs
are `MOLD_SECURE_FIXTURE_BIND` (encrypted listener address, default
`127.0.0.1:0`), `MOLD_SECURE_FIXTURE_MEDIA` (local test MP4 path, default
`/tmp/mold-secure-uat/fixture.mp4`), and `MOLD_SECURE_FIXTURE_RELAY` (isolated
staging WSS endpoint; omitted for LAN-only tests). `MOLD_SECURE_PROBE_BYTES`
selects a bounded download byte count instead of the standard matrix;
`MOLD_SECURE_PROBE_UPLOAD_BYTES` overrides that matrix's 1 MiB upload size.
`MOLD_SECURE_TRACE_CARRIER=1` prints bounded batch sizes and broker/PUT/GET
timings for transport diagnosis; it never prints signed object URLs, ciphertext,
credentials, host keys, or application paths. Leave it unset during normal use.

Native Rust clients and connectors accept `MOLD_SECURE_OBJECT_ORIGIN` only as
operator configuration of an independently trusted HTTPS object bucket origin.
It defaults to the production ciphertext bucket; isolated staging uses its own
exact origin. A relay advertisement cannot change this trust setting. Keep
fixture profiles and invitations private even though their data is synthetic.

# Secure and Regular connections

Secure connections use the same encrypted protocol on
LAN and through the optional Lambda relay. The relay carries ciphertext; HTTP
paths, API credentials, prompts, uploads, progress events and media stay inside
a Noise XX session between the client and the pinned host. The cloud still sees
connection identities, IP addresses, timing and traffic sizes.

Secure is the default. Upgrade and pair again when moving from the old proxy
protocol; old proxy bearer credentials are not accepted. Direct hosts can also
enable explicit Regular HTTP or HTTPS access alongside Secure.

| Mode               | Routes                                 | Protection                                                                     |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------ |
| Secure (default)   | LAN, Tailscale, WAN, or the Mold proxy | Pinned end-to-end encryption between client and host                           |
| Regular (explicit) | Direct LAN, Tailscale, or WAN HTTP(S)  | HTTPS or Tailscale encrypts transport when used; plain HTTP adds no encryption |

The Mold proxy always requires Secure, including HTTPS proxy URLs. A failed
Secure connection never switches to Regular. Each mode uses its own device
credentials; never reuse a Secure or operator key for Regular access.
A shared-network address is an address hint, not a trust source.

## Pair in the apps

On the host, open **Settings → Remote Access**, choose Secure or Regular, and
create a pairing code. Scan or paste it on the client and confirm its mode before
pairing. Secure codes contain the host's public key; both modes carry a short
one-use invitation. Transfer it privately. Each client receives its own
independently revocable grant. Forgetting a client locally removes its local
access; revoke it on the host to invalidate every retained copy of that grant.

In hosted browser Studio, **Settings → Machines → Switch machine** opens the
connection screen while retaining saved pairings. The selected machine's menu
and detail page offer **Disconnect** and **Forget** too. Forget removes the local
pairing and returns to that screen; if removal fails, retry remains available.
Choose another saved machine or paste a fresh pairing link. Selection changes,
including changes from another tab, reload the workspace before another machine
can use it. This clears old model and download state while keeping durable jobs
bound to the machine that accepted them.

Native clients prefer an advertised LAN address and retain the same pinned
identity when moving to the relay. An HTTPS browser may need the WSS relay
because browsers restrict unencrypted WebSocket URLs from HTTPS pages. Such
browser restrictions do not weaken the encryption or change the host pin.
Stopping Remote Access closes its listener and existing encrypted sessions.
Regular direct access is a separate host choice. Use the address advertised by
its invitation; an embedded app's Regular listener may use a different port from
the private engine or Secure listener. Pairing the same host again updates its
saved connection instead of adding a duplicate machine.

## A persistent key for your REST app

Keep authentication enabled. Enable Regular direct access on the host with
`mold serve --allow-direct-api`, `MOLD_ALLOW_DIRECT_API=1`, or
`services.mold.allowDirectApi = true` in NixOS. Secure remains available alongside
it. Choose the actual direct origin reachable by your app, such as a LAN or
Tailscale address; do not use a Mold proxy URL.

On the authenticated host, export a new private invitation:

```bash
mold pairing create --mode regular --direct-url http://HOST:7680 \
  --output "$MOLD_HOME/secure/custom-app-invitation.json"
```

Run this as the host owner with its actual `MOLD_HOME`. On NixOS, explicitly
select the service home and the operator key file configured by `apiKeyFile`:

```bash
sudo -u 'mold' env MOLD_HOME=/var/lib/mold \
  MOLD_API_KEY=@/run/agenix/mold-api-key mold pairing create --mode regular \
  --direct-url http://HOST:7680 \
  --output /var/lib/mold/secure/custom-app-invitation.json
```

Replace the home and key-file paths with your actual configuration. A custom
host-local wrapper can supply these settings if you configured one; the upstream
Mold module does not install a `mold-local` command.
Transfer the owner-only invitation privately and claim it within two minutes.
Use a separate private client home for each app:

```bash
mkdir -m 700 "$HOME/.mold-custom-app"
MOLD_HOME="$HOME/.mold-custom-app" mold pair \
  --invite-file /private/custom-app-invitation.json --name "Custom app"
```

The invitation is single use. The resulting device key persists across server
restarts until the host revokes that client. Its owner-only profile is
`$HOME/.mold-custom-app/secure/clients/regular-<SHA256(origin)>.json`; read its
`api_key` for your app and its `origin` for the direct URL. Do not commit or log
the key, and do not copy the host's operator key into the app.

REST requests use **`X-API-Key`**; `Authorization: Bearer` alone is not accepted.
For example, a Node app can read its private profile without putting the key in
command arguments:

```js
import { readFile } from 'node:fs/promises'

const { origin, api_key } = JSON.parse(
  await readFile(process.env.MOLD_APP_PROFILE, 'utf8')
)
const response = await fetch(`${origin}/api/status`, {
  headers: { 'X-API-Key': api_key },
})
if (!response.ok) throw new Error(`Mold returned ${response.status}`)
const status = await response.json()
```

Create another invitation to issue an independent key for another app. This is
a runtime operation: no Nix rebuild or authentication disable is needed.
Revoke only the intended client on the host when retiring its key. Forgetting
the local profile alone does not revoke a retained credential.

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

## Switching host apps

Native, Tauri, and headless hosts share identity, operator authority, and paired-device
grants when they use the same `MOLD_HOME`. Stop the current host before starting
another app with that home. Remote Access enablement is app-specific: Native
enrollment does not automatically enable Tauri’s cloud connector. Configure the
Tauri process with the host’s public `MOLD_PUBLIC_URL` and production
`MOLD_SECURE_WEBSOCKET_URL` before starting its engine. These public addresses do
not replace the home’s private identity or device grants. Keep the variables
process-specific rather than setting them globally for every app.

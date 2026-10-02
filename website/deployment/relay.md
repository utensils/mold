# Remote HTTPS relay

The optional relay gives an authenticated Mold machine a normal HTTPS address
without opening an inbound port on that machine. Direct LAN and VPN addresses
continue to work. One gateway publishes one machine; adding another machine
requires another gateway and address.

The host opens outbound WebSockets to the gateway. Clients continue using the
normal HTTP API, so streaming progress, uploads, gallery media tickets, video
ranges and pairing work through the same URL. The gateway terminates TLS and
can see API keys and media: use a gateway you trust. This is not end-to-end
encryption like Zephra's application relay.

## Connect a machine

Start a normal authenticated server. Keep its operator keys in an owner-only
file and point `MOLD_API_KEY` at that file:

```bash
MOLD_API_KEY=@/absolute/path/operator-keys.txt mold serve --bind 127.0.0.1
```

Your gateway operator supplies a separate random enrollment token. Save it in
an owner-only file, then run the connector alongside the server:

```bash
chmod 600 /absolute/path/relay-token
MOLD_RELAY_TOKEN_FILE=/absolute/path/relay-token mold relay connect \
  --transport aws \
  --relay-url wss://uwijdhg05d.execute-api.us-west-2.amazonaws.com/production \
  --target 127.0.0.1:7680
```

On Windows, where this connector cannot verify file ACLs, supply the token in
`MOLD_RELAY_TOKEN` through your service environment instead. Never pass its
value as a CLI argument. On Linux/macOS prefer the owner-only token file.

The enrollment token admits the **host** to the gateway; it is not a Mold API
key. Clients never need it. The connector accepts only a numeric loopback
server address and refuses a server whose anonymous `/api/status` does not
return 401. It verifies TLS and does not follow redirects. Do not put tokens in
URLs, command arguments, repository files or deployment logs.

`mold relay connect` stays running and reconnects after a network interruption.
In-flight requests fail when their connection drops; the relay never replays a
mutation. Accepted generation jobs remain owned by the host's durable queue.
Set `MOLD_RELAY_DIAGNOSTICS=1` for static connection/reconnect categories during
troubleshooting; diagnostics never include tokens, headers, URLs or raw frames.
Stop the connector to withdraw remote access. A sleeping or offline host is
unavailable; the relay does not supply a cloud GPU fallback.

## Use any client

| Surface              | Connection                                                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| CLI, MCP and Discord | Set `MOLD_HOST=https://mold-link.urandom.io` and use the existing Mold API credential mechanism.                             |
| Browser              | Open the HTTPS address. Enter this machine's API key when asked; the credential stays in this browser tab's session storage. |
| Tauri desktop        | Add the HTTPS address in Machines and supply the normal host key or pair.                                                    |
| Tauri iOS/Android    | Add the HTTPS address manually or scan a pairing code naming that address. Keys stay in the native credential store.         |
| Native macOS         | Add the HTTPS address in Machines; normal remote-machine operations apply.                                                   |
| Native iPhone/iPad   | Add or pair with the HTTPS address; keys stay in Keychain.                                                                   |

When minting a pairing QR, choose the public HTTPS address as the reachable
server URL. A QR naming localhost or a private LAN address cannot work over a
cellular connection. Use the existing per-device pairing/revocation controls;
the relay does not change permissions.

Native macOS **This Mac** remains an app-private engine. To host remote access
on that Mac, explicitly start authenticated `mold serve` and its connector.
This option does not automatically expose the app's private engine. CLI-only
headless hosts can use the GPU-free standalone `mold-relay` connector binary
with the same `connect` arguments.

## Lambda gateway

Production uses a Regional API Gateway HTTPS endpoint and two Node.js 22 Lambda
functions. A separate API Gateway WebSocket endpoint connects the machine to
per-request frontend sessions; DynamoDB stores admission and connection leases.
The host connector runs on your machine, outside Lambda. URandom Terraform owns
these resources and DNS; Mold owns the runtime code and browser shell. Zephra
and existing GPU services remain independent.

API credentials still reach and are checked by the host. Anonymous host access
must return 401 before a connector forwards requests. API Gateway and Lambda
terminate TLS and can see credentials and media. Enrollment and internal bridge
tokens live in SSM SecureString, outside Terraform state. Public metrics are
blocked. One enrolled host owns an address; another requires its own deployment.

The shared clients stage requests larger than 2 MiB through private S3, with a 64 MiB relay request limit. Grants bind the original method, URL, headers,
body checksum, host session and credential, and can be consumed once. Finite
responses larger than 16 MiB, or with an unknown length, use private S3 objects;
clients restore their original status and headers. Media players resolve short
lived signed object URLs without sending Mold keys to S3. These URLs expire
within 15 minutes; resolve media again to renew access. Staged objects are capped
at 8 GiB as a storage guard; each transfer must still finish within 840 seconds.
Throughput and concurrent use can require a direct connection for large files.
Gallery imports above the relay request cap require a direct connection. Signed S3 requests omit credentials and support browser shells served from LAN origins. Media staging that never starts fails after 90 seconds; event streams cannot be staged. Static browser assets are served by the frontend from the private bucket.

A Lambda request is bounded by the platform's 15 minute limit. SSE connections
close before that limit and clients reopen their read stream and reconcile state.
Generation clients submit once to the durable queue, then read progress and saved
results. CLI model pulls submit once to the durable download API and read the
retained job until completion; a stream closing does not report success. Chain
read streams reconcile their manifest and reconnect after EOF.
Relay generation therefore requires gallery retention: `--no-save` and
gallery-disabled generation are refused before submission. Disconnects never
cause automatic mutation replay or a local generation fallback after an uncertain
admission. Native This Mac remains private unless you explicitly start an
authenticated server and connector.

Build and deploy reviewed runtime packages with the tooling under `scripts/relay/`.
Inspect the complete scoped Terraform plan before applying infrastructure changes.
Frontend capacity is 40 Lambda invocations with at most 32 active host tunnels, leaving headroom for control calls. These limits bound resource use; they do not authenticate callers or prevent every denial of service. `/health`, `/api/docs` and `/api/openapi.json` forward to the host rather than returning the offline browser shell.

Lambda usage includes the lifetime of streaming requests; this is optional remote
access, not a free cloud GPU service.

For local development only, `mold relay serve` provides the original direct
transport with loopback control/data listeners on ports 7681/7682. A trusted TLS
reverse proxy must route `/_mold/relay/*` to control, normal requests to data,
remove spoofed forwarding headers, block metrics, and stream HTTP/1.1 responses.
Use `--transport direct` on its connector. Plain WS requires the explicit
`--allow-insecure-loopback` flag. Direct streams default to a 3,600 second
application-byte inactivity limit, configurable with `--idle-timeout-secs`.

Native macOS **Settings ▸ Remote Access**, beside Machines, shows the
connection address, learned LAN/Tailscale/relay routes, an inline pairing QR
and paired-device controls. An existing pairing learns routes from a reachable
authenticated server and keeps one machine and key when the connection changes.
Configure `MOLD_PUBLIC_URL` on the server to advertise its public HTTPS relay
origin. The private built-in This Mac engine remains private.

## One pairing across networks

Set `MOLD_PUBLIC_URL=https://mold-link.urandom.io` on the authenticated server
actually attached to that gateway. Mold advertises the addresses its listener
serves, including Tailscale interfaces, plus this explicitly configured HTTPS
origin. The connector does not open a listener or change firewall rules.

New pairing codes carry these routes. Existing saved machines learn them during
an authenticated connection check, so let the updated app connect while the
machine is still reachable. The same saved machine and credential then use a
reachable direct route or the relay; another pairing is unnecessary. A machine
that has never advertised a relay cannot be reached through a guessed gateway.
Older servers and older codes retain their original address.

Automatic roaming requires a server-minted paired credential. Operator API keys
remain tied to the explicitly saved address; pair once to enable route learning.
Probes expose a stable digest tag, so arbitrary operator keys never participate
in anonymous route proofs. Plain HTTP still requires a trusted network.

Clients verify a fresh credential-free proof before using an alternate route,
prefer direct connections, and retain a healthy choice briefly to avoid
oscillation. Public routes require verified HTTPS. Plain HTTP retains the
existing trusted-network assumption; the proof detects accidental address reuse
and does not provide TLS channel binding against a forwarding attacker.
Keyless servers retain their original address. HTTPS browser pages may be unable
to probe HTTP LAN routes because of browser mixed-content restrictions; the
HTTPS relay remains usable. Changing routes never resubmits an interrupted
mutation: durable jobs are reconciled by their existing IDs and read streams
reconnect.

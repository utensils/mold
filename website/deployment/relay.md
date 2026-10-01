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
  --relay-url wss://mold-link.urandom.io --target 127.0.0.1:7680
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

## Host a gateway

For local development, `mold relay serve` has loopback-only control and data
listeners (ports 7681 and 7682). Put a trusted HTTPS reverse proxy in front:
`/_mold/relay/*` goes to the control listener; normal requests go to the data
listener. Use plaintext HTTP/1.1 upstreams, disable response buffering, remove
untrusted forwarding headers and block public `/metrics`. The standalone
`mold-relay serve` command avoids GPU dependencies on the cloud instance.

Production hosts require WSS. Plain WS is permitted only with the explicit
`--allow-insecure-loopback` development flag and a loopback relay address.
The gateway has bounded connection admission and attachment deadlines. Streams
close after 300 seconds without application bytes in either direction;
WebSocket heartbeats do not extend this deadline. Set `--idle-timeout-secs`
(1–86,400) on both gateway and connector to change it. A
second connector is rejected while the first owns the machine address. Token
rotation requires restarting the gateway and connector, ending old sessions.
Mold sees loopback proxy peers, so enabled per-IP rate limits aggregate remote
clients and connector authentication probes; do not trust forwarded headers to
circumvent these limits.

The URandom Terraform `modules/mold-relay` example owns a separate AWS instance,
DNS and restricted SSM management role. Tokens are provisioned outside Terraform
state. See its README for resource costs and deployment procedure. Build the
small static Linux artifact with `scripts/relay/build-linux.sh`; only reviewed
artifacts should be installed. Inspect the complete Terraform plan before an
apply. Existing Zephra and GPU services are independent of this gateway.

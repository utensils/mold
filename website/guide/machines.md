# Machines

The **Machines** workspace in the web app keeps this server and the remote Mold
hosts your browser can reach in one place. Each card reports availability,
resources, models-disk usage, and queue depth; selecting a card opens that
host's detail view.

## Add a machine

Choose **+ Add machine** and use either path:

- **Remote server** — enter an IP address, DNS name, HTTPS URL, or Tailscale
  MagicDNS name. A schemeless address defaults to HTTP and port `7680`.
- **Local network** — on current servers, the primary `mold serve` process
  browses `_mold._tcp` with DNS-SD and returns its cached peers to the web app.
  Select a result to test it and save it. If the peer advertises authentication,
  Mold asks for that peer's API key first.

Local network results reflect **the server's LAN, not necessarily the browser's
LAN**. For example, opening Mold over Tailscale may show machines visible to the
remote studio server. Discovery only returns addresses: generation and all
other requests go directly from the browser to the selected peer. VLAN or
firewall rules can therefore make a discovered address unreachable; Mold tests
`/api/status` before saving it.

API keys are stored per host and sent in the `X-Api-Key` header. They are never
placed in URLs. Hosts dedupe by the stable `instance_id` reported by
`/api/status`, so adding the same server by hostname and IP updates one row.

## Compatibility and configuration

The web app enables **Local network** only when the primary server reports
`discovery.can_browse: true` from `GET /api/capabilities`. Older builds and
servers compiled without the `mdns` feature keep the option disabled without
probing the discovery route.

DNS-SD browsing and advertising are enabled by default in release and Nix
builds. Disable both with `mold serve --no-mdns` or `MOLD_MDNS=0`. A
loopback-bound server is not advertised, but it can still browse its machine's
LAN for the web UI.

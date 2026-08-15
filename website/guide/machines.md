# Machines

The web shell keeps engine reachability visible from every workspace. Library
also exposes machine chips so a merged multi-host print feed can be narrowed to
one origin without changing or forgetting the active generation target.

The **Machines** workspace in the web app keeps this server and the remote Mold
hosts your browser can reach in one place. Each card reports availability,
resources, models-disk usage, and queue depth; selecting a card opens that
host's detail view.

Host detail aggregates telemetry across every GPU and shows CPU, system RAM,
models-disk storage, downloads, installed models, and the live queue. On
Apple-silicon hosts, whose GPU and system memory share one physical pool, the
VRAM and RAM bars collapse into a single Memory row; CUDA machines keep both.
Below the live instruments, hosts that support MiniMax H3 show an **H3 on this
machine** panel reporting that machine's runtime qualification (memory floors,
attention and quantization profiles, and component readiness). Current
servers expose capability-gated **Pause/Resume** and confirmed **Cancel all**
actions; GPU lane selection is available only while a job is still queued.
Saved remotes can be renamed or forgotten from the same page. Forgetting also
removes that host's saved API key and returns a pinned generation target to the
serving origin.

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

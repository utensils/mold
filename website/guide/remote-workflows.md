# Remote Workflows

Mold's CLI-native design keeps remote work as scriptable as local work: change
the target host without changing the generation command, whether the caller is
a person, shell script, CI job, or agent.

One of mold's best deployment patterns is simple:

- run `mold serve` on the GPU machine
- point `MOLD_HOST` at it from everywhere else

That gives you local-first ergonomics with remote GPU horsepower.

## Basic Laptop → GPU Server

On the GPU host:

```bash
mold serve --bind 0.0.0.0 --port 7680
```

From your laptop or devbox:

```bash
export MOLD_HOST=http://gpu-host:7680
mold run "a cinematic portrait"
```

## Finding Servers on Your Network

If you don't know the GPU host's address, let mold find it. Builds with the
`mdns` feature (release binaries and the Nix package) advertise every
`mold serve` on the local network via mDNS/DNS-SD, and `mold server discover`
browses for them:

```bash
mold server discover
# NAME          URL                       VERSION  AUTH  GPU
# hal9000-7680  http://192.168.1.10:7680  0.26.0   -     1xNVIDIA GeForce RTX 4090
#
# Connect: export MOLD_HOST=http://192.168.1.10:7680
```

Add `--probe` for a `/health` latency column, `--json` for machine-readable
output, or `--timeout-secs N` to browse longer on a busy network. The desktop
app shows the same list in its **Machines** workspace with a one-click **Add**
button. The web app also offers **Machines → Add machine → Local network** when
the primary server reports DNS-SD browsing support; these results reflect the
server's LAN and the browser connects to the selected address directly. The
iPhone app's **Machines → Discover nearby** uses Apple's native Bonjour browser
for the same service and asks for Local Network permission. It also accepts a
manual IP address, hostname, HTTPS URL, or Tailscale MagicDNS name.

Advertising and server-assisted browsing are on by default; a server can opt
out of both with `mold serve --no-mdns` or `MOLD_MDNS=0` (for example on a shared
or untrusted LAN). Loopback-only binds (`--bind 127.0.0.1`) are never advertised,
but can still browse peers for the web UI. See the [Machines guide](/guide/machines).

## iPhone over Tailscale

Join the iPhone and GPU host to the same tailnet, then add the host in Mold by
its MagicDNS name, such as `plato` or `plato.example-tailnet.ts.net`. A bare
name becomes `http://<name>:7680`; use an explicit HTTPS URL only when your own
reverse proxy provides TLS. Bonjour remains LAN-local, so a remote Tailscale
host is normally added by name rather than discovered.

The app uses Tailscale's existing network path; it does not embed a Tailscale
SDK or manage login, ACLs, MagicDNS, or certificates. See the
[iPhone guide](/guide/iphone) for the complete setup.

## Recommended Pattern

1. `mold pull` models on the GPU host, not from every client machine.
2. Keep the server running so models stay warm between requests.
3. Use `mold ps` to confirm the client can reach the server.
4. Set default `HF_TOKEN` / `CIVITAI_TOKEN` values on the server for gated
   models, or open that server's web Settings and save owner-only catalog
   credential overrides there. Clearing a saved override returns to the server
   default. For remote catalog searches and pulls, the desktop can also send its
   local token as a request-scoped fallback when the server has no working
   token; the server's own credential remains first.
   The iPhone has no separate token editor or forwarding store, so its selected
   remote host must have any required upstream credentials.

## OpenClaw and Discord

Remote workflows pair well with both:

- [OpenClaw](/guide/openclaw) when you want agent-driven generation
- [Discord Bot](/api/discord) when you want a chat interface

In both cases, the key variable is still `MOLD_HOST`.

## LM Studio MCP

`mold mcp` starts a stdio MCP server that exposes mold generation tools to
LM Studio and other MCP hosts. It talks to the normal `mold serve` HTTP API, so
keep the server running separately.

```bash
mold serve --bind 127.0.0.1 --port 7680
```

In LM Studio, open the Program tab, choose **Install → Edit mcp.json**, and add
an entry like this:

```json
{
  "mcpServers": {
    "mold": {
      "command": "/absolute/path/to/mold",
      "args": ["mcp", "--host", "http://localhost:7680"],
      "timeout": 300000
    }
  }
}
```

The MCP server exposes synchronous `generate_image` and `generate_mesh` (plus
`export_mesh`, which transcodes a stored `.glb` into OBJ, STL, or PLY),
timeout-friendly `generate_image_async` / `generation_status` /
`generation_retry`, gallery tools `list_gallery` / `get_gallery_image`,
`list_models`, `list_loras`, `server_status`, and the prompt-transform tools
`expand_prompt` / `remix_prompt`, which apply the target model's prompting guide
on the server. The guides themselves are readable as `mold://prompting/<path>`
resources (`mold://prompting/route/<model>` returns one model's full route).
Generation tools accept a `loras` array using ids or paths
returned by `list_loras`; object entries can omit `scale` to use `1.0`. Use the
async generation flow for cold model loads or slow generations so LM Studio does
not need to keep one tool call open until the image is finished. Set
`MOLD_API_KEY` in the MCP process environment when the mold server requires one.

Poll `generation_status` until the durable child settles. A held child advertises
whether it is retryable; after correcting the reported cause, call
`generation_retry` with that MCP job id exactly once, then resume status polling.
Status polling reconciles the exact durable batch and job authority, including a
retry whose HTTP response was interrupted. Both a single-job poll and the bounded
job-list poll advance that reconciliation; do not submit a second retry while its
outcome is still being confirmed.

## Remote Pulls vs Local Pulls

Behavior depends on where the command runs:

- `mold pull` against a reachable server downloads onto that server
- if no server is reachable, the CLI falls back to local pulling

That distinction matters if your laptop has little disk space or no GPU.

## Example Multi-Client Setup

| Machine        | Role                                          |
| -------------- | --------------------------------------------- |
| GPU host       | Runs `mold serve`, stores model files         |
| Laptop         | Runs `mold run`, `mold list`, `mold ps`       |
| iPhone         | Remote Create, Library, Models, and Machines  |
| Discord worker | Runs `mold discord` or `mold serve --discord` |
| OpenClaw host  | Uses mold via `MOLD_HOST`                     |

## Deployment Choices

- [Docker & RunPod](/deployment/docker) for cloud or containerized GPUs
- [NixOS](/deployment/nixos) for declarative infra
- [Deployment Overview](/deployment/) for the high-level options

## Remote Troubleshooting

If remote generation fails:

- verify `MOLD_HOST`
- check firewall and bind address
- run `mold ps`
- hit `/health` directly with `curl`

```bash
curl http://gpu-host:7680/health
curl http://gpu-host:7680/api/status
```

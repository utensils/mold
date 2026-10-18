# Remote Workflows

Mold's CLI-native design keeps remote work as scriptable as local work: change
the target host without changing the generation command, whether the caller is
a person, shell script, CI job, or agent.

One of mold's best deployment patterns is simple:

- run `mold serve` on the GPU machine
- pair each client with it once, then name it with `--host`

That gives you local-first ergonomics with remote GPU horsepower.

## Basic Laptop → GPU Server

On the GPU host, start the server and print a one-use pairing link:

```bash
mold serve --bind 0.0.0.0 --port 7680
mold host invite                   # in another shell; --qr draws it for a phone
```

On your laptop or devbox, run `mold host add` and paste the link at the prompt
(a link passed as an argument works too, but lands in shell history). The host
is saved under its own name:

```bash
mold host add
mold run "a cinematic portrait" --host workstation
mold host use workstation          # or make it the default for every command
```

Fetch the finished print's original file from the host:

```bash
mold library download --latest --host workstation
```

An unpaired, non-loopback host over plain HTTP refuses requests
(`426 SECURE_CONNECTION_REQUIRED`, "No secure pairing for this host"). For a
Regular direct connection instead, run the host with
`mold serve --allow-direct-api` and save it with
`mold host add URL --regular` (plus `--api-key-file KEY` when authentication is
on).

## Saved Hosts

Pairing saves each host under a name, so you rarely type an origin again.
`--host`, `MOLD_HOST`, and a default host all accept it, and a Tailscale IP the
host advertised finds the same pairing (mold says so on stderr; LAN addresses
are always used as typed):

```bash
mold host list                     # NAME  MODE  ORIGIN  DEFAULT
mold ps --host workstation
MOLD_HOST=workstation mold library list
mold host use workstation          # default for every remote command
mold host use --clear              # back to http://localhost:7680
```

`--host` wins over `MOLD_HOST`, which wins over the default host; with none of
them, mold talks to `http://localhost:7680`. See
[`mold host`](/guide/cli-reference#mold-host).

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
# Connect: mold host add http://192.168.1.10:7680 …
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
its MagicDNS name, such as `workstation` or `workstation.example-tailnet.ts.net`. A bare
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
- [Discord Bot](/api/discord) when you want a bounded chat interface for
  one-shot generation, prompt tools, attachment upscaling, read-only catalog
  search, and permission-gated host operations. Discord reads the remote
  server's advertised model capabilities; scripted chain jobs and durable
  mesh workflows remain CLI/API-only.

In both cases, pair the host once and name it with `--host NAME`;
`MOLD_HOST` accepts the same saved name where an environment variable fits
better.

## LM Studio MCP

`mold mcp` starts a stdio MCP server that exposes mold generation tools to
LM Studio and other MCP hosts. It talks to the normal `mold serve` HTTP API, so
keep the server running separately.

```bash
mold serve --bind 127.0.0.1 --port 7680
```

For a remote GPU host, pair it first (`mold host add`) and pass its saved name
instead of the loopback URL. In LM Studio, open the Program tab, choose **Install → Edit mcp.json**, and add
an entry like this:

```json
{
  "mcpServers": {
    "mold": {
      "command": "/absolute/path/to/mold",
      "args": ["mcp", "--host", "workstation"],
      "timeout": 300000
    }
  }
}
```

The MCP server exposes synchronous `generate_image` and `generate_mesh` (plus
`export_mesh`, which exports a stored `.glb` as OBJ, OBJ+PBR ZIP, STL, or PLY),
timeout-friendly `generate_image_async` / `generation_status` /
`generation_retry`, gallery tools `list_gallery` / `get_gallery_image`,
`list_models`, `list_loras`, `server_status`, and the prompt-transform tools
`expand_prompt` / `remix_prompt`, which apply the target model's prompting guide
on the server. The guides themselves are readable as `mold://prompting/<path>`
resources (`mold://prompting/route/<model>` returns one model's full route).
Generation tools accept a `loras` array using ids or paths
returned by `list_loras`; object entries can omit `scale` to use `1.0`. `generate_image` and `generate_image_async` take
`output_format` `png`, `jpeg` or `webp` (a still), ordered `reference_images`
(base64 PNG, JPEG or WebP, sent as `edit_images` to a model whose
`capabilities.reference_images` is adjustable; with no width or height, a
`canvas: last-reference` model sizes the canvas from the last one), and
`transparent_background: true` for a model advertising
`capabilities.transparency` (PNG or WebP only). Use the
async generation flow for cold model loads or slow generations so LM Studio does
not need to keep one tool call open until the image is finished. A saved
host uses its saved credential; set `MOLD_API_KEY` in the MCP process
environment only for a host with no saved credential.

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
- a remote host (`--host`, `MOLD_HOST` or the default host) that does not
  answer is an error; nothing is downloaded onto this machine
- with no host named and no local server running, the CLI pulls locally
- `mold pull --local` always downloads onto this machine

That distinction matters if your laptop has little disk space or no GPU.

## Example Multi-Client Setup

| Machine        | Role                                          |
| -------------- | --------------------------------------------- |
| GPU host       | Runs `mold serve`, stores model files         |
| Laptop         | Runs `mold run`, `mold list`, `mold ps`       |
| iPhone         | Remote Create, Library, Models, and Machines  |
| Discord worker | Runs `mold discord` or `mold serve --discord` |
| OpenClaw host  | Uses mold via `--host NAME`                   |

## Deployment Choices

- [Docker & RunPod](/deployment/docker) for cloud or containerized GPUs
- [NixOS](/deployment/nixos) for declarative infra
- [Deployment Overview](/deployment/) for the high-level options

## Remote Troubleshooting

If remote generation fails:

- run `mold host list` to confirm the host is saved and which one is the default
- run `mold ps --host NAME` against it
- on "No secure pairing for this host" (`426 SECURE_CONNECTION_REQUIRED`), pair
  with `mold host invite` on the host and `mold host add` here, or run the host
  with `--allow-direct-api` and save it with `mold host add URL --regular`
- check firewall and bind address
- hit `/health` directly with `curl`

```bash
mold host list
mold ps --host workstation
curl http://gpu-host:7680/health
```

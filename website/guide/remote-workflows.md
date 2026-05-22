# Remote Workflows

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

## Recommended Pattern

1. `mold pull` models on the GPU host, not from every client machine.
2. Keep the server running so models stay warm between requests.
3. Use `mold ps` to confirm the client can reach the server.
4. Set `HF_TOKEN` on the server if you use gated Hugging Face repos.

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

The MCP server exposes synchronous `generate_image`, timeout-friendly
`generate_image_async` / `generation_status`, gallery tools
`list_gallery` / `get_gallery_image`, `list_models`, `list_loras`, and
`server_status`. Generation tools accept a `loras` array using ids or paths
returned by `list_loras`; object entries can omit `scale` to use `1.0`. Use the
async generation flow for cold model loads or slow generations so LM Studio does
not need to keep one tool call open until the image is finished. Set
`MOLD_API_KEY` in the MCP process environment when the mold server requires one.

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

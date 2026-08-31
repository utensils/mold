# Mold CLI workflows

Read `mold <command> --help` before using an unfamiliar or high-impact option.
The installed CLI is authoritative over examples in this bundle.

## Generate and inspect

```bash
mold list
mold info flux2-klein:q8
mold run flux2-klein:q8 "A red fox in falling snow" --seed 42 --output fox.png
mold run qwen-image-edit-2511:q8 "Change the chair to red leather; preserve everything else" --image chair.png --output edited.png
mold upscale input.png --model real-esrgan-x4plus:fp16 --output output-4x.png
```

Use `mold info <model>` or `/api/models` before selecting dimensions, frame
counts, steps, guidance, conditioning, or audio. A catalog model can differ
from a built-in manifest profile.

## Local and remote execution

`mold run` first targets `MOLD_HOST` (default `http://localhost:7680`) and can
fall back to local inference when the server is unavailable. `--local` skips
the server. When a task must stay on a particular machine, set the host for
that invocation and confirm it with a read-only status call first.

```bash
mold server status
MOLD_HOST=http://gpu-host:7680 mold server status
mold run --local flux2-klein:q8 "A graphite sketch of a lighthouse"
```

Never paste an API key into a prompt or command example. Supply secrets through
the user's existing environment or approved secret store at execution time.

## Jobs and queues

```bash
mold jobs list
mold queue list
mold queue cancel job-abc123
mold queue cancel --batch batch-7
```

Accepted generation is asynchronous on queue-backed servers. Retain exact job
and batch IDs, poll them, and reconcile after transport errors. Queue semantics
and durability are capabilities of the selected host and request; inspect the
live capability and row data instead of promising replay after restart.

Cancellation is state-sensitive. A cancel request can race dispatch or
settlement. Re-read the affected row or batch and describe the observed final
state. `mold queue cancel --all --yes` is intentionally absent from routine
examples because it is a broad destructive action.

## Model and library management

```bash
mold pull flux2-klein:q8
mold library list
mold trash list
mold clean
```

`mold clean` is a dry run unless the user explicitly requests deletion and the
CLI confirms the force flag. Trash purge, model removal, cloud-volume deletion,
and pod termination are destructive; follow the safety reference.

## Server and MCP

```bash
mold serve --help
mold mcp --host http://localhost:7680
```

Starting, stopping, restarting, or reconfiguring a server changes external
state. Do so only when requested, and verify health plus the selected host
afterward. For MCP async generation, keep the returned job ID and poll the same
job; retry only when the status explicitly marks it retryable.

`generate_mesh` is the one MCP generate tool whose schema requires `image`
rather than `prompt`: the 3-D family has no text encoder, so there is nothing
for a prompt to do. It returns a rendered poster plus mesh statistics; the
glTF itself lands in the gallery and is fetched by filename.

For published CUDA images, use Mold's live distribution resolver rather than
guessing an architecture tag. Its current contract includes B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported. B200 support
is simulated until hardware-qualified.

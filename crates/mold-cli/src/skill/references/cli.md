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
mold server status --host gpu-host
MOLD_HOST=http://gpu-host:7680 mold server status
mold run --local flux2-klein:q8 "A graphite sketch of a lighthouse"
```

`mold server status` reports on the named host over HTTP; PID, port and log
path appear only for this machine's managed daemon. An unreachable host exits
non-zero rather than reporting on the local machine.

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

Some weights carry third-party terms mold will not accept on a user's behalf
(PuLID's InsightFace models, every Hunyuan3D tier). `mold licenses` lists them
and says which machine the answer is about — acceptance is recorded per Mold
data root, so it belongs to the host that runs the pull, not necessarily this
one. Never accept on the user's behalf: show the terms and let them choose.

```bash
mold licenses                                 # what is required, and on which host
mold licenses accept <ID>                     # agree WITHOUT downloading
mold pull <model> --accept-license <ID>       # agree and pull; repeat for several
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
for a prompt to do. `mold run hunyuan3d-mini-turbo --image chair.png` is
likewise a complete CLI request with no prompt at all. It returns a rendered
poster plus mesh statistics; the glTF itself lands in the gallery and is
fetched by filename. Its optional `octree` (the advertised allowlist; cost is
cubic), `threshold` (0–1 iso-level, ComfyUI `VoxelToMesh` scale), and
`target_faces` mirror `--octree`, `--mesh-threshold`, and `--target-faces`;
omit them for the recipe's defaults.

`export_mesh` converts one stored `.glb` into `obj`, `stl`, or `ply`; the CLI
equivalent is `mold library export <file> --format stl`. Both are transcodes of
geometry that already exists — the gallery file is never renamed or replaced —
and `-o` on a 3-D render must still name a `.glb`.

`expand_prompt` and `remix_prompt` on a Hunyuan3D model (or `mold expand` /
`mold remix --model hunyuan3d-mini-turbo`) do not call a language model: the
one result is the family guide's advice on preparing the source image,
because the profile advertises `prompt.mode: ignored`. Do not retry it with a
different backend; improve the image instead.

For published CUDA images, use Mold's live distribution resolver rather than
guessing an architecture tag. Its current contract includes B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported. B200 support
is simulated until hardware-qualified.

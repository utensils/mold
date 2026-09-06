# Mold CLI workflows

Current Mold versions are distributed through GitHub releases, Nix/FlakeHub,
Docker, AUR, and source builds. crates.io publishing is retired; registry
versions are historical and should not be recommended for installation.

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

`mold run --offload` carries the forced-offload preference to the GPU host,
including durable sequences. Without it, the request inherits the host policy.
Wan 1.3B and dense Wan 2.1 14B always run without residual caching because
cached output is not quality-qualified; setting `MOLD_WAN_STEP_CACHE` cannot
override that refusal. Wan 2.2 A14B remains qualified for the cache.

## Reference-image editing

`--reference PATH` sends one ordered reference image; repeat it in semantic
order, and name each image's role in the prompt ("the jacket from image 1 on
the model from image 2"). The recipe decides whether a model has the protocol
at all: `/api/models[].generation_profile` carries a
`capabilities.reference_images` block, and a model without one refuses
`--reference` by name rather than silently ignoring it.

```bash
mold run flux2-klein:bf16 "Put sunglasses on the person; keep the pose and background" --reference person.jpg
mold run flux2-klein-9b:q8 "The woman from image 1 wearing the eyeglasses from image 2" --reference person.jpg --reference glasses.jpg
```

The block also says what references do to the source image.
FLUX.2 [klein] renders from a source image OR from references, never both in
one pass, so `--reference` together with `--image` is refused. FLUX.2 [dev] and
Qwen-Image-Edit have no source image at all: there the ordered group IS
`--image`, repeated, and for Qwen-Image-Edit the first image is the thing being
edited. `--reference` also carries MiniMax H3's Ref2VA inputs, which is why it
additionally accepts `video=PATH` and `audio=PATH`; a bare path always means an
image.

## Local and remote execution

`mold run` first targets `MOLD_HOST` (default `http://localhost:7680`) and can
fall back to local inference when the server is unavailable. `--local` skips
the server. When a task must stay on a particular machine, set the host for
that invocation and confirm it with a read-only status call first.

MiniMax H3 `--local` accepts one FL2VA request with an owned, single-use
attempt. Local batches and chains are refused before preparation. Ref2VA
reference uploads use the server route. The same limits apply to local
fallback when the server is unavailable; never retry a refused local batch
by reusing its first request's prepared authority.

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
omit them for the recipe's defaults. The older `octree_resolution` and
`mesh_threshold` names are declared in the schema as deprecated aliases.

`export_mesh` converts one stored `.glb` into `obj`, `stl`, or `ply` (`glb`
returns the stored bytes unchanged); the CLI equivalent is
`mold library export <file> --format stl`. Both are transcodes of
geometry that already exists — the gallery file is never renamed or replaced —
and `-o` on a 3-D render must still name a `.glb`. `obj`, `stl`, and `ply`
additionally take `size_mm` (1–1000; CLI `--size-mm`), `up_axis` (`y` | `z`;
`--up-axis`), and `origin` (`center` | `floor`; `--origin`) to make the
export print-ready: the stored GLB is in Hunyuan3D's normalized unit-cube
space, which a slicer reads as a few millimetres and refuses. Omit any of
them for the format's own default (100 mm, Z-up, floor for STL and PLY;
unscaled, Y-up, floor for OBJ). All three are refused on `glb` and on a
turntable, and on a host that does not advertise
`capabilities.mesh.export_geometry`.

```bash
mold library export chair.glb --format stl --size-mm 120 --up-axis y --origin center
```

The same tool and command take `gif`, `apng`, or `webp` to render a
**turntable**: the gallery poster's
view spun a full turn around the mesh, the way to show a mesh anywhere a
`.glb` cannot open. Its optional `playback` (`loop` | `bounce`), `repeat`
(`forever` | `once`), `max_dimension` (240–2048, default 512), `frames`
(8–180, default 36) and `fps` (1–30, default 10) mirror `--playback`,
`--repeat`, `--max-dimension`, `--frames` and `--fps`; bounce and once are GIF
only, and the flags are refused on a geometry format. Only the formats the
host lists in `capabilities.mesh.export_formats` succeed (`webp` needs a build
with the `webp` feature).

`expand_prompt` and `remix_prompt` on a Hunyuan3D model (or `mold expand` /
`mold remix --model hunyuan3d-mini-turbo`) do not call a language model: the
one result is the family guide's advice on preparing the source image,
because the profile advertises `prompt.mode: ignored`. Do not retry it with a
different backend; improve the image instead.

For published CUDA images, use Mold's live distribution resolver rather than
guessing an architecture tag. Its current contract includes B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported. B200 support
is simulated until hardware-qualified.

## macOS Metal memory

`mold system metal-memory status [--json]` inspects this machine, ignoring
`MOLD_HOST`. Explicit root-only `set <MiB> [--persist]` and `reset [--persist]`
administer its system-wide limit; never run the server as root. Use
`mold gpu list --json` for the inference host's effective capacity and headroom.
Zero means automatic; increases may require restarting an idle inference process.

Model checksums are verified when files are downloaded. Complete installed models queue and switch without full checksum scans, including after restart. To check existing bytes explicitly, run `mold info MODEL --verify`. Normal loading still checks file sizes and formats; it does not guarantee detection of same-size corruption.

For Z-Image on Metal, whole-decode attempts finish inside the memory-error
recovery boundary: an OOM can retry with tiles, and the eager path can still
reload the VAE on CPU if GPU recovery is exhausted. A repeated cleanup OOM
does not prevent that retry; unrelated errors propagate. `MOLD_VAE_TILED`
controls this Metal recovery path. Bounded Candle convolution workspaces replace
the old proactive span cap. CPU/CUDA Z-Image decode policy is unchanged.

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

The prompt is OPTIONAL, not absent, on a video render that already carries
visual conditioning — a source image, keyframes, a clip to continue, or a
reference set. LTX-2, Wan and MiniMax H3 all answer this way: what the user
attached already decides the render, so a prompt refines it. Never invent one
to satisfy the CLI, and expect near-static micro-motion when there is none. A
text-to-video tier attaches nothing, so its prompt stays required, and an
audio-only render (`--pipeline t2a`) reads no pixels so an attached still does
not make its prompt optional. The recipe's advertised `prompt.mode` in
`/api/models` is the authority for any model; never a family list.

```bash
# The attached frame is the whole conditioning
mold run wan22-ti2v-5b --image still.png
mold run ltx-2-19b-distilled:fp8 --image chef.png
```

`--offload` carries the forced-offload preference to the GPU host, for a
one-shot and for a scripted sequence alike. Without it, the request inherits
the host policy.

```bash
mold run wan22-t2v-a14b:q8 "A kite over a harbour" --offload
```

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

## Image prompting (SD 1.5 and SDXL)

SD 1.5 and SDXL read a reference as an image PROMPT rather than an edit target:
IP-Adapter encodes it with a CLIP-Vision tower and injects it into every
cross-attention layer beside the text, so its appearance carries into the
render while the prompt still steers. Pull `ip-adapter-sd15` or
`ip-adapter-sdxl` first; the two share a vision tower, so the second costs only
its adapter.

```bash
mold run sdxl-base:fp16 "A sailboat on a calm lake at sunrise" --reference streetscape.png --reference-weight 0.7
```

`--reference-weight` is 0.0 to 2.0, default 1.0; 0.6-0.8 is the usable range
because 1.0 lets the picture dominate the prompt. Exactly 0.0 renders what the
same request with no reference renders, pixel for pixel, and downloads and
loads nothing.

One reference per render, and unlike every other reference family it rides WITH
everything else: `--image`/`--strength`, `--mask`, ControlNet and a LoRA all
stay live in the same pass, and batches work because the encoded picture
applies to every image in the batch.

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

## Scripted sequences

A sequence — several clips rendered as one continuous video — is authored as a
`mold.chain.v1` TOML script and submitted with `mold run --script`. It is a
CLI and API shape only: no app composes one. Each `[[stage]]` carries its own
`prompt`, `frames` and `transition` (`smooth`, the default motion-tail morph;
`cut`, a fresh latent; `fade`, a cut plus an RGB crossfade), and a stage may
name a `source_image_path` relative to the script file.

```bash
mold chain validate shot.toml
mold run --script shot.toml --dry-run
mold run --script shot.toml --output walk.mp4
```

Repeating `--prompt` is the sugar for the uniform case — same length, smooth
transitions throughout. Anything heterogeneous belongs in a script.

```bash
mold run ltx-2-19b-distilled:fp8 --prompt "a cat wakes on a windowsill" --prompt "it stretches and jumps down" --frames-per-clip 97
```

A remote `--script` run is a DURABLE chain job: it survives a dropped
connection, `mold jobs list` finds it afterwards, and the stitched print is
hydrated from the host's gallery rather than returned inline. A long one-shot
(`mold run --frames 200`) is auto-chained by the host when the model cannot
render that length in one pass; that job is not an authored sequence and does
not appear in `mold jobs list`.

```bash
mold jobs list
mold jobs show job-abc123 --json
mold jobs show job-abc123 --script
mold jobs resume job-abc123
mold jobs retake job-abc123 --stage 2 --mode splice --prompt "it lands on the rug"
mold jobs cancel job-abc123
mold jobs gc
```

`mold jobs retake` re-renders ONE stage. To change the shape of the sequence,
export its effective script with `mold jobs show <ID> --script` (`--json`
prints a different document, the job detail), edit it, and hand the whole list
back — amend replaces every stage, and the host keeps the cached
clips of the leading stages that did not change. The model, size and container
are not amendable; those need a new sequence. `--dry-run` prints what would be
sent without touching the job.

```bash
mold jobs show job-abc123 --script > edited.toml
mold jobs amend job-abc123 --script edited.toml --dry-run
mold jobs amend job-abc123 --script edited.toml --fps 30 --no-audio
mold jobs delete job-abc123 --yes
```

## Durable 3-D workflows

`mold run <3-D model>` is one render. A mesh WORKFLOW is the durable,
multi-stage form: each stage — the picture a text-to-3-D run starts from, its
matted and delighted copies, the shape, the paint — is admitted as its own
generation and keeps its own retained artifact, the job survives a server
restart, and a resume picks up at the first unfinished stage instead of
rerunning the whole thing.

Every verb is remote. The manifest, the stage artifacts and the queue rows
live in ONE machine's data root, so there is no local form; `--local` is
refused by name and the honest alternative is a one-shot `mold run`.

The mode is inferred from what you supply: a prompt renders a picture and
reconstructs it, a mesh with an appearance image textures that mesh, and a
mesh on its own is rebuilt through the 2.1 shape VAE. Name it with `--mode`
when a script would rather not depend on inference.

```bash
mold mesh-workflow create --prompt "a small ceramic fox" --texture --follow
mold mesh-workflow create --mesh chair.glb --image chair-albedo.png --texture-resolution 2048
mold mesh-workflow create --mesh chair.glb --mode mesh_roundtrip --octree 320 --target-faces 40000
mold mesh-workflow list --json
mold mesh-workflow show WORKFLOW-ID
mold mesh-workflow events WORKFLOW-ID
mold mesh-workflow resume WORKFLOW-ID
mold mesh-workflow cancel WORKFLOW-ID
mold mesh-workflow delete WORKFLOW-ID
```

`--octree`, `--threshold` (spelled `--mesh-threshold` on `mold run`, and
accepted under both names here), `--target-faces`, `--matting` and `--delight`
are the same controls a one-shot render takes, and this is the only surface
that exposes all of them on a durable workflow. Omit one and the recipe's own
default answers; do not restate a default here. `--seed` applies to every
stage, so one value reproduces the whole run. `delete` is settled-only —
cancel or wait first, and the server says so if you do not.

A supplied `--mesh` is a `.glb` or `.obj`. On a host that advertises
reference uploads and is reached with an API key, its bytes stream through a
request-bound upload lease instead of riding the request as base64.

## Jobs and queues

```bash
mold queue list
mold queue list --held --json
mold queue show job-abc123
mold queue cancel job-abc123
mold queue cancel --batch batch-7
mold queue retry job-abc123
mold queue move job-abc123 --to 0
mold queue pause job-abc123
mold queue resume
mold queue send job-abc123 --to http://gpu-host:7680
mold queue sweep
```

`mold queue pause` and `mold queue resume` with no job id hold and release
host-wide dispatch, which affects everyone using that machine. `mold queue
send` hands a HELD job to another running server with its request intact; the
destination credential is read from `MOLD_DESTINATION_API_KEY`, never spliced
into the command.

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
mold list
mold list --json
mold info flux2-klein:q8 --verify
mold stats --json
mold default flux2-klein:q8
mold ps
mold unload
mold gpu list --json
mold version
```

`mold info MODEL --verify` re-checksums the installed bytes. Normal loading
checks file sizes and formats only, so this is the explicit way to answer
"are these weights intact".

`mold search` queries the Hugging Face and Civitai catalogs. It runs on the
selected server when one answers, so it sees the credentials THAT machine
stored and its `installed` column answers about the machine that would do the
downloading; with no server reachable it runs locally against `HF_TOKEN` and
`CIVITAI_TOKEN`. A merged search whose one provider failed still returns the
other's rows and reports the failure on stderr. Install a result with
`mold pull` and its printed id.

```bash
mold search flux
mold search "anime style" --kind lora --sort recent
mold search sdxl --source civitai --page 2 --page-size 50 --no-nsfw --json
```

`mold downloads` is the server's own download queue — what is transferring,
what is waiting, and what recently finished. It is remote only, because a
queue belongs to the machine whose disk fills up. `mold downloads add` takes a
model name from `mold list`; a `cv:` or `hf:` catalog id belongs to
`mold pull`, which routes on the id itself, and is refused here by name.

```bash
mold downloads list
mold downloads list --json
mold downloads add flux-dev:q4
mold downloads add pulid-flux --accept-license insightface-antelopev2
mold downloads watch
mold downloads cancel DOWNLOAD-ID
```

`mold quantize` derives a smaller Hunyuan3D shape tier from an installed one
and registers it in THIS host's config; no other machine knows the name.

```bash
mold quantize hunyuan3d-2.1:fp16 --tier q4
```

Settings live in two surfaces — `config.toml` for paths, ports and
credentials, and the metadata database for user preferences — behind one view.
`mold config where` says which surface owns a key before you change it.

```bash
mold config list --json
mold config get expand.backend
mold config set expand.enabled true
mold config where models_dir
mold config path
```

The Library is one serving host's gallery, filtered and organized over HTTP.

```bash
mold library list --tag owl --favorite --limit 20 --json
mold library show mold-flux-dev-q4-1700000000000.png --json
mold library title mold-flux-dev-q4-1700000000000.png "Smurf village"
mold library favorite mold-flux-dev-q4-1700000000000.png
mold library unfavorite mold-flux-dev-q4-1700000000000.png
mold library tag list
mold library tag add mold-flux-dev-q4-1700000000000.png --tag owl --tag night
mold library tag remove mold-flux-dev-q4-1700000000000.png --tag night
mold library tag rename owl owls
mold library tag delete owls --yes
mold library collection list
mold library collection create "Night Owls" --description "Long-exposure studies"
mold library collection show night-owls --json
mold library collection add night-owls mold-flux-dev-q4-1700000000000.png
mold library collection remove night-owls mold-flux-dev-q4-1700000000000.png
mold library collection update night-owls --name "Owls at night"
mold library collection delete night-owls --yes
```

Deletion is two-stage. `mold library trash` moves live prints into the
recoverable trash, and `mold trash restore` brings them back. `mold trash
delete` is the permanent one: it acts on live and trashed prints alike and
there is nothing to restore afterwards, so confirm the exact filenames with a
listing first.

```bash
mold library trash mold-flux-dev-q4-1700000000000.png
mold trash list
mold trash restore mold-flux-dev-q4-1700000000000.png
mold trash sweep
mold trash delete mold-flux-dev-q4-1700000000000.png --yes
mold clean
```

The host is the only authority on what it kept of a print's conditioning
media, so ask it rather than inferring from the print's metadata.
`mold library source-media` lists what was retained and downloads one member
by its opaque id. A print that predates retention, or whose conditioning was
recorded only as text, answers `unavailable_legacy` — that is a fact about the
print, not damage.

```bash
mold library source-media mold-flux-dev-q4-1700000000000.png
mold library source-media mold-flux-dev-q4-1700000000000.png --json
mold library source-media mold-flux-dev-q4-1700000000000.png --member 6f1c1d2e --output recovered.png
```

A Library video can be upscaled durably, frame by frame, without holding the
connection open.

```bash
mold video-upscale create mold-ltx-2-1700000000000.mp4 --wait
mold video-upscale list
mold video-upscale status vu-abc123
mold video-upscale pause vu-abc123
mold video-upscale resume vu-abc123
mold video-upscale cancel vu-abc123
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
mold skill list
```

The MCP server exposes thirteen tools: `generate_image`, `generate_mesh`,
`export_mesh`, `generate_image_async`, `generation_status`,
`generation_retry`, `list_gallery`, `get_gallery_image`, `list_models`,
`list_loras`, `server_status`, `expand_prompt` and `remix_prompt`.
`generate_mesh` is a ONE-SHOT render, not the durable 3-D workflow. A
multi-stage workflow is `mold mesh-workflow` at the CLI and
`/api/mesh-workflows` over HTTP; no MCP tool wraps it.

Starting, stopping, restarting, or reconfiguring a server changes external
state. Do so only when requested, and verify health plus the selected host
afterward. For MCP async generation, keep the returned job ID and poll the same
job; retry only when the status explicitly marks it retryable.

`generate_mesh` is the one MCP generate tool whose schema requires image
conditioning rather than `prompt`: the 3-D family has no text encoder, so
there is nothing for a prompt to do. `mold run hunyuan3d-mini-turbo --image
chair.png --matting auto` is likewise a complete CLI request with no prompt
at all. A 2mv recipe instead accepts any non-empty semantic subset through
`--front`, `--left`, `--back`, and `--right`; never renumber a missing view.
`--matting auto` preserves useful alpha and removes opaque backgrounds, `on`
recomputes every supplied cutout, and `off` keeps the original pixels.
Use `--delight` only when the advertised mesh profile enables it; the host then
runs the fixed Hunyuan3D lighting and highlight removal stage after matting and
before shape or paint.
`hunyuan3d-2.1` uses the same single-image contract and requires the separate
`tencent-hunyuan3d-2.1` licence acceptance. It returns a rendered
poster plus mesh statistics; the glTF itself lands in the gallery and is
fetched by filename. Its optional `octree` (the advertised allowlist; cost is
cubic), `threshold` (0–1 iso-level, ComfyUI `VoxelToMesh` scale), and
`target_faces`, `matting`, and `delight` mirror `--octree`, `--mesh-threshold`,
`--target-faces`, `--matting`, and `--delight`;
omit them for the recipe's defaults. An omitted `target_faces` keeps the raw
surface on a geometry-only request and takes
`capabilities.mesh.target_faces_texture_default` when `texture` is true. The older `octree_resolution` and
`mesh_threshold` names are declared in the schema as deprecated aliases.

`export_mesh` converts one stored `.glb` into `obj`, `zip`, `stl`, or `ply` (`glb`
returns the stored bytes unchanged); the CLI equivalent is
`mold library export <file> --format stl`. Both are transcodes of
geometry that already exists — the gallery file is never renamed or replaced —
and `-o` on a 3-D render must still name a `.glb`. `obj`, `zip`, `stl`, and `ply`
additionally take `size_mm` (1–1000; CLI `--size-mm`), `up_axis` (`y` | `z`;
`--up-axis`), and `origin` (`center` | `floor`; `--origin`) to make the
export print-ready: the stored GLB is in Hunyuan3D's normalized unit-cube
space, which a slicer reads as a few millimetres and refuses. Omit any of
them for the format's own default (100 mm, Z-up, floor for STL and PLY;
unscaled, Y-up, floor for OBJ and ZIP). All three are refused on `glb` and on a
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
(8–180, default 36), `fps` (1–30, default 10) and `transparent` mirror
`--playback`, `--repeat`, `--max-dimension`, `--frames`, `--fps` and
`--transparent`; bounce and once are GIF only, and the flags are refused on a
geometry format. `transparent` renders the object over nothing instead of the
slate backdrop — APNG and WebP keep the antialiased edge, while a GIF's single
transparent palette index makes it a hard cut. Only the formats the
host lists in `capabilities.mesh.export_formats` succeed (`webp` needs a build
with the `webp` feature).

`mold expand` rewrites one prompt for a chosen model's style, and `mold remix`
returns alternatives from an existing prompt. Both are prompt work only —
neither generates an image — and both take the target model so the rewrite
matches that family's grammar.

```bash
mold expand "a cat on a windowsill" --model flux2-klein:q8
mold remix "a cat on a windowsill at dawn" --model flux2-klein:q8 --variations 3 --json
```

`expand_prompt` and `remix_prompt` on a Hunyuan3D model (or `mold expand` /
`mold remix --model hunyuan3d-mini-turbo`) do not call a language model: the
one result is the family guide's advice on preparing the source image,
because the profile advertises `prompt.mode: ignored`. Do not retry it with a
different backend; improve the image instead.

For published CUDA images, use Mold's live distribution resolver rather than
guessing an architecture tag. Its current contract includes B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported. B200 support
is simulated until hardware-qualified.

## macOS Metal memory

Wan is performance-qualified on Apple Metal for the 1.3B BF16 and 5B Q8/FP16
paths. Prefer the 5B Q8 tier for sustained 720p work; dense FP16 is supported
but can slow as unified-memory pressure and VAE decode cost accumulate.

`mold system metal-memory status` inspects THIS machine, ignoring `MOLD_HOST`.

```bash
mold system metal-memory status --json
```

Its root-only `set <MiB> [--persist]` and `reset [--persist]` administer the
system-wide limit; never run the server as root, and never run either without
being asked to. Use `mold gpu list --json` for the inference host's effective
capacity and headroom. Zero means automatic; increases may require restarting
an idle inference process.

Model checksums are verified when files are downloaded. Complete installed
models queue and switch without full checksum scans, including after restart.
Normal loading still checks file sizes and formats; it does not guarantee
detection of same-size corruption, which is what `mold info MODEL --verify`
above is for.

For Z-Image on Metal, whole-decode attempts finish inside the memory-error
recovery boundary: an OOM can retry with tiles, and the eager path can still
reload the VAE on CPU if GPU recovery is exhausted. A repeated cleanup OOM
does not prevent that retry; unrelated errors propagate. `MOLD_VAE_TILED`
controls this Metal recovery path. Bounded Candle convolution workspaces replace
the old proactive span cap. CPU/CUDA Z-Image decode policy is unchanged.

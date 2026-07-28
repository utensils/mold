# CUDA distribution qualification

The schema and runner define the deferred real RTX 3090 acceptance gate. No
passing hardware report is checked in, and the runner never provisions cloud
hardware.

The weight-free engine-family contract and arbitrary-N scheduler simulation
evidence are tracked separately in the
[multi-GPU family matrix](./multi-gpu-family-matrix.md). That matrix explicitly
distinguishes deterministic CI evidence from local-hardware and deferred
B200/MIG/12 GiB qualification.

Extract the sm86 binary from one exact stable release. Install one image model,
one video model, and prepare a valid
`mold.chain.v1` script for that video model, then run:

The runner does not override dimensions inside `--chain-script`. Its decoded
chain acceptance check requires exactly 256×256 output, so the script must set
both dimensions under `[chain]`:

```toml
schema = "mold.chain.v1"

[chain]
model = "ltx-video-0.9.6:bf16"
width = 256
height = 256
fps = 24
steps = 2
guidance = 3.0
strength = 1.0
motion_tail_frames = 0
output_format = "mp4"

[[stage]]
prompt = "mold CUDA qualification chain"
frames = 9
```

```bash
scripts/qualify-cuda-sm86.sh \
  --release-tag v0.20.2 \
  --sm86-binary ./mold-sm86 \
  --sm86-archive ./mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz \
  --sm89-binary ./mold-sm89 \
  --sm89-archive ./mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz \
  --image-model flux-dev:bf16 \
  --video-model ltx-video:q8 \
  --chain-script ./qualification-chain.toml \
  --report ./cuda-sm86-qualification.json
```

The runner downloads `mold-release-provenance.json` and `SHA256SUMS` from that
exact official GitHub release. Caller-supplied hashes are deliberately
unsupported. Release publication first generates artifact checksums, derives
provenance from that set, then appends the provenance digest to `SHA256SUMS`;
this one-way ordering binds both files without a circular checksum. The runner
opens and hashes the supplied sm86 and sm89 archives and binaries, requires each
archive to contain that exact binary, and binds both through provenance and the
checksum manifest.
Mold's Candle/cudarc kernels are driver-JITed from these strings; intermediate
cubins are not required to survive final linking.

The workload matrix performs two decoded 256×256 images with the sm86 binary.
For the PTX case, it extracts an exact PTX entry module from that executable,
records its hash, and loads those bytes through `cuModuleLoadData` before
running normal full-Mold generation. It deliberately does not set
`CUDA_FORCE_PTX_JIT`: that process-wide switch also forces NVIDIA runtime
libraries through PTX JIT and can fail during cuBLAS initialization before a
Mold kernel is reached. The matrix also performs video and chained-video
generation with sm86. Every workload pins an RTX 3090 UUID with
`CUDA_VISIBLE_DEVICES`, observes the exact generation PID and that UUID
together in NVIDIA's active compute process list, requires Mold's CUDA-device
log, and validates the output media rather than trusting process exit status.
Image runs use a non-GGUF FLUX model with block offloading so they must select
and log Mold's math attention backend.

PTX compatibility is one-way: code targeting a compute capability may JIT on
devices with an equal or greater compute capability, never a lower one. NVIDIA
states this directly in the
[CUDA C++ Programming Guide, PTX Compatibility](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#ptx-compatibility).
Consequently, the old proposed positive “sm89/JIT on RTX 3090” gate is
physically invalid. Its truthful replacement is two-part:

1. complete sm89 PTX extracted from an exact sm89 ELF pinned to the same
   source/provenance must be rejected by the CUDA Driver on the sm86 RTX 3090;
   this is an exact-ELF negative compatibility control, not runtime generation
   or hardware qualification;
2. the exact sm86 replacement artifact must pass the attention image smoke,
   load a hashed PTX module extracted from itself through the CUDA Driver API,
   and complete the second image smoke on that same UUID.

The sm89 result is an expected negative Driver regression, not a successful
generation or hardware qualification. Keep its binary/archive hashes, complete
embedded PTX target set, UUID-pinned command, and hashed Driver-probe JSON beside
the positive sm86 evidence. The probe command itself exits successfully only
when every exact sm89 candidate is rejected with a recognized incompatibility.

Outputs, logs, raw PID/GPU observation CSVs, PTX probes, release provenance, and
the checksum manifest remain under `<report>.d/`. The validator hermetically
walks the checked-in JSON Schema and fails closed on unsupported schema keyword
drift. It then opens and hashes every referenced evidence file and validates
the archive/binary/provenance, process/GPU, attention-log, and Driver-probe
relationships:

```bash
scripts/validate-cuda-qualification-report.py \
  ./cuda-sm86-qualification.json
```

A report with `hardware_qualified: false` is failure evidence, not acceptance.
The exact sm89 Driver rejection is mandatory for a passing two-part report but
is never itself called runtime or hardware qualification.
The schema is
[`cuda-sm86-report.schema.json`](./cuda-sm86-report.schema.json). This is a
reproducible evidence workflow, not a cryptographic attestation service.

## Local scheduler acceptance on two RTX 3090s

The distribution runner above proves one sm86 artifact on one selected device
at a time. It does not prove multi-GPU scheduling or lifecycle behavior.
`scripts/qualify-local-multi-gpu.py` is the separate candidate-binary gate for
the named `local-2x-rtx3090-sm86` development-host profile. A passing report
requires exactly two `NVIDIA GeForce RTX 3090` devices, exactly 24576 MiB per
device, compute capability 8.6, and the two expected NVIDIA UUIDs.

The runner rejects port `7680`, refuses an occupied alternate port, and never
stops, restarts, or connects to the normal service. It binds only loopback and creates an
isolated `MOLD_HOME`, `MOLD_DB_PATH`, `MOLD_OUTPUT_DIR`, gallery, and server
log tree beside the report. Every candidate invocation, including `version`
and `gpu list --json`, runs through the same Bubblewrap mount namespace with
the host root, real home, source tree, and `--models-dir` read-only; only its
private runtime tree and `/tmp` overlay are writable. Session-bus and agent
environment variables are not inherited. A before/after
path/type/mode/size/mtime manifest is an additional mutation tripwire.
Bubblewrap is therefore a required host tool. Do not point `--report` into a
temporary directory if the evidence needs to be retained.

`--timeout-seconds` is one absolute campaign deadline. The runner terminates
whole subprocess groups on expiry, always stops the private candidate, performs
the post-run model-tree comparison, and writes failure evidence when a campaign
step fails.

Prepare a request that is slow enough for the runner to observe both active
workers and disable one while it is busy. Keep `batch_size` at one; the runner
submits independent compatible jobs and assigns distinct prompts/seeds:

```json
{
  "prompt": "Mold real local multi-GPU acceptance",
  "model": "sd15:fp16",
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance": 7.5,
  "seed": 9042026,
  "batch_size": 1,
  "output_format": "png"
}
```

Run the exact final CUDA candidate on an unused alternate port:

```bash
scripts/qualify-local-multi-gpu.py \
  --binary ./target/release/mold \
  --models-dir /home/killswitch/.mold/models \
  --model-artifact /home/killswitch/.mold/models/sd15/model.safetensors \
  --model-artifact /home/killswitch/.mold/models/sd15/vae.safetensors \
  --request ./local-2x3090-request.json \
  --expected-gpu-uuid GPU-44f80ce5-23fc-a5dd-ac4e-133142952997 \
  --expected-gpu-uuid GPU-ba027fc5-7915-8d58-6738-b7eaafe427b4 \
  --port 17681 \
  --report ./local-2x3090-qualification.json
```

Repeat `--model-artifact` for every weight, encoder, tokenizer, VAE, LoRA, or
other component used by the selected request. Each file must be a regular file
below `--models-dir`; its exact path, size, and SHA-256 are bound into both the
report and typed evidence. Omission means the resulting campaign is not a
complete artifact qualification and must not be accepted.

The required evidence includes:

- exact `nvidia-smi` inventory and same-sample exact-candidate-PID observations
  on both UUIDs while distinct bound work IDs are active;
- `/api/devices`, legacy `/api/status`, `/api/resources`, `/api/queue`, and
  `mold gpu list --json` projections;
- decoded exact-dimension PNG outputs bound to unique work IDs, returned seeds,
  ordinals, and the corresponding exact GPU UUIDs;
- busy disable returning `202`/`draining`, plan-version advance away from that
  lane, drain completion, and re-enable;
- paused queued cancellation with a typed terminal SSE event, no matching
  active work, and no output-tree change; all-disabled maintenance rejection;
  exact old/new PID plus stable-ID/UUID persistence across restart; and
  legacy-mode mutation fencing;
- missing/empty/all/none/ordinal/stable-ID/NVIDIA-UUID/unmatched selector
  behavior plus stable-ID resolution under reversed `CUDA_VISIBLE_DEVICES`.

These two physical UUIDs share no non-empty prefix, so they cannot produce an
ambiguous real selector. The runner executes the deterministic ambiguous/
missing-prefix source test and records `hardware_claimed: false` for that
case. This is deliberate: synthetic ambiguity evidence must never be
misrepresented as a property observed on this hardware.

Validate retained evidence independently:

```bash
scripts/validate-local-multi-gpu-report.py \
  ./local-2x3090-qualification.json
```

The validator reopens and hashes the exact candidate, normalized request,
selected model artifacts, decoded outputs, and every typed evidence file. It
requires the exact named hardware profile and independently derives every gate
from the API/PID/work-ID/UUID/ordinal evidence; reported `status` strings are
not accepted as proof. Evidence paths must remain unique and confined to the
report's adjacent `.d` directory. This is reproducible, internally
tamper-evident evidence, not a remote cryptographic attestation.
Use `--allow-failure` only when inspecting a valid failed run; it does not
convert that run into hardware qualification.

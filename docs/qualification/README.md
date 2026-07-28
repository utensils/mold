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

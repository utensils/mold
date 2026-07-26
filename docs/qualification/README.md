# CUDA distribution qualification

The schema and runner define the deferred real RTX 3090 acceptance gate. No
passing hardware report is checked in, and the runner never provisions cloud
hardware.

Extract the sm86 binary from one exact stable release. Install one image model,
one video model, and prepare a valid
`mold.chain.v1` script for that video model, then run:

```bash
scripts/qualify-cuda-sm86.sh \
  --release-tag v0.20.2 \
  --sm86-binary ./mold-sm86 \
  --image-model flux-dev:bf16 \
  --video-model ltx-video:q8 \
  --chain-script ./qualification-chain.toml \
  --report ./cuda-sm86-qualification.json
```

The runner downloads `mold-release-provenance.json` from that exact official
GitHub release. Caller-supplied hashes are deliberately unsupported. It
requires exact sm86 PTX embedded in the final executable, a release binary
checksum derived from the published archive, and matching source identity.
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

1. an exact sm89 artifact pinned to the same source/provenance must fail on the
   sm86 RTX 3090 with a CUDA PTX/kernel incompatibility before producing media;
2. the exact sm86 replacement artifact must pass the attention image smoke,
   load a hashed PTX module extracted from itself through the CUDA Driver API,
   and complete the second image smoke on that same UUID.

The sm89 result is an expected negative regression, not a successful
generation or hardware qualification. Keep its binary hash, complete embedded
PTX target set, UUID-pinned command, nonzero exit, and CUDA error log beside the
positive sm86 evidence.

Outputs and logs remain under `<report>.d/`. After the run, validate both the
JSON Schema and cross-field relationships:

```bash
scripts/validate-cuda-qualification-report.py \
  ./cuda-sm86-qualification.json
```

A report with `hardware_qualified: false` is failure evidence, not acceptance.
The schema is
[`cuda-sm86-report.schema.json`](./cuda-sm86-report.schema.json). This is a
reproducible evidence workflow, not a cryptographic attestation service.

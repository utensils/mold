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
  --image-model flux-dev:q8 \
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

The workload matrix performs two decoded 256×256 images with the sm86 binary,
forcing `CUDA_FORCE_PTX_JIT=1` for one to exercise its PTX path without
pretending higher-target PTX can run backward on a 3090. It also performs video
and chained-video generation with sm86. Every workload pins an RTX 3090 UUID with
`CUDA_VISIBLE_DEVICES`, observes that UUID in NVIDIA's active compute process
list, requires Mold's CUDA-device log, and validates the output media rather
than trusting process exit status. Image runs must also select the math
attention backend.

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

# LTX-2.5 convolutional VAE spatial Auto

The 2026-10-10 Plato failure occurred after successful denoising at 1408x2560,
193 frames. CUDA allocation failed in VAE decode on GPU2 at 18:45:22 UTC and
again on GPU3 at 18:55:28 UTC. The successfully allocated peak was 35.4 GB;
the CUDA pool reserved 47.1 GB. Neither number proves the total memory needed.
The stage-2 transformer was dropped and synchronized before VAE loading.
The old temporal estimate selected no chunking, which also disabled spatial
Auto. Four separate GPUs do not provide a pooled allocation for this decode.

Auto now selects convolutional spatial tiles independently of that estimate.
The reference is Lightricks/LTX-2 commit
`9ec55f9f22798a3198d9c923856824821bc3317e`:

- `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:137-175` selects
  768px long-side tiles and 64px overlap.
- `packages/ltx-core/src/ltx_core/tiling.py:867-909` scales the short axis on the
  latent grid with ties-even rounding.

Mold retains its existing pairwise crossfade and therefore clamps the short
axis to at least four latent cells, rather than upstream's three, to prevent
triple overlaps. Explicit overrides retain square tiles and their existing
256px default overlap. The full-decode diagnostic disables spatial Auto;
explicit forced spatial tiles continue to take precedence as before.

This changes spatial geometry only. Temporal receptive context can still
span a whole 193-frame clip, and retained strips/full output consume memory.
Admission estimates are not a measured decode bound. CPU regression coverage
and synthetic decoder tests do not qualify the formerly failing GPU shape.

## GPU qualification

The ignored test `ltx25_spatial_auto_cuda_decode_qualification` loads a
standalone convolutional LTX-2.5 VAE checkpoint, creates fixed zero latents
for the failing shape, uses the production decode configuration and decoder,
and checks output shape and finite values using scalar reductions. It saves
no media and does not run the text encoder or transformer. The VRAM probe
prints successful-allocation peaks and reserved pool memory even on decode
failure. This is decoder memory qualification, not a visual quality oracle.

First reserve one GPU from the serving scheduler through an authorized
lifecycle operation. **An idle snapshot and `CUDA_VISIBLE_DEVICES` alone do
not reserve a GPU.** Do not run this against an unreserved production device.
Use an external timeout and the CUDA build environment documented by the
repository; keep the service binary and deployed environment unchanged.

```sh
CUDA_VISIBLE_DEVICES=<reserved-physical-gpu> \
LTX25_UAT_VAE_PATH=/path/to/standalone-convolutional-vae.safetensors \
timeout 20m cargo test -p mold-ai-inference --features cuda,cudnn --lib \
  ltx25_spatial_auto_cuda_decode_qualification -- --ignored --nocapture
```

Unset `MOLD_LTX2_SPATIAL_TILE`, `MOLD_LTX2_VAE_FORCE_FULL_DECODE`, and
`MOLD_LTX2_VAE_FORCE_FRAMEWISE` for this test. A passing synthetic decode does
not establish pixel parity, full-generation peak memory, or support for all
long clips. Compare real decoded outputs separately before making quality
or performance claims.

### Plato result — 2026-10-10

Passed on commit `0726e47b10cba1be917e6cc09ea4440950d653a0`, using one
NVIDIA L40S (physical GPU 3). The device was idle with no active or planned
work, disabled through `PATCH /api/devices/{id}`, and confirmed unavailable
to the scheduler before the test. The three other GPUs continued serving
existing work. The reserved device had 533 MiB used, 44,928 MiB free and
0% utilization across three baseline samples. Its only remaining context
belonged to the verified server process.

The checkpoint was `ltx-2.5-video-vae-conv-bf16.safetensors`, SHA-256
`685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d`.
The ignored test used zero BF16 latents and the production Auto configuration:
24 × 13 latent-cell tiles with two-cell overlap. It returned the expected
`[1, 3, 193, 2560, 1408]` output, with finite minimum `-0.6484375` and maximum
`0.076171875`.

The decode phase took 96.881 seconds. Its allocator probe reported a 12.7 GB
successful-allocation peak and 16.3 GB reserved pool peak; the full test passed
in 101.80 seconds. These rounded probe values describe this synthetic decoder
run, not a measured full-generation bound or a direct comparison with the
original failed generation. The test saved no generated media. Its restoration
handler re-enabled GPU 3 and confirmed the enabled state after the process
exited successfully.

Evidence was retained outside the repository in
`/tmp/mold-ltx25-qualification-evidence/` on the operator machine, including
build/decode logs, checkpoint hash, and device snapshots before reservation,
while reserved, and after restoration. Visual quality and full-generation
qualification remain separate.

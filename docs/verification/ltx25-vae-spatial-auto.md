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

## Optional GPU qualification (not yet executed)

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

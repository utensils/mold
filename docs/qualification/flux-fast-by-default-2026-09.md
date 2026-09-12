# FLUX fast-by-default campaign — qualification ledger (2026-09-11/12)

Branch `perf/flux-fast-by-default`, one PR. This record states what was measured
on hardware, what was reviewed, and what is claimed on unit tests alone. The
raw evidence (renders, server logs, `nvidia-smi dmon` samples, contact sheets,
the six-round peer review) lives outside the repository in
`tmp/perf-audit-flux-2026-09-11/` on plato; this file is the durable summary.

## Hardware and method

- plato: 4× NVIDIA L40S (46 GB usable each), 1.5 TB host RAM, NixOS, CUDA 12.8,
  shipping feature set (`h3-cuda,cudnn,preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid,mesh-texture,mesh-matting,mesh-delight`).
- Oracle for the audit: stable-diffusion.cpp on the same card and the same GGUF files.
- Every UAT render was opened and inspected. Same-seed pairs were compared by SHA-256.
- Production (`mold serve` on 7680, shared `/storage/mold`) ran throughout; scratch
  servers used the shared home only with storage v3 OFF, and a throwaway home for
  everything that changes on-disk formats or config.

## Targets and results (final binary, 1024², L40S)

| target                                | goal                           | result                                                                           | verdict                 |
| ------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------- | ----------------------- |
| `flux-dev:q8` denoise                 | ≤ 0.90 s/it                    | 0.78–0.80                                                                        | met                     |
| `flux-dev:q8` warm wall (20 steps)    | ≤ 25 s                         | 16.9 s                                                                           | met                     |
| `flux2-klein:q8` warm wall            | ≤ 8 s                          | 2.41 s                                                                           | met                     |
| `flux2-dev:q8` denoise                | ≤ 2.60 s/it                    | 2.92 (fp8 tier 2.21–2.24)                                                        | short on q8; met on fp8 |
| `flux2-dev:q8` transformer load       | ≤ 8 s, once per process        | 5.9–8.3 s; second render reuses it (no reload)                                   | met                     |
| fixed overhead per render             | ≤ 3 s                          | 0.40 s server / 0.04 s CLI                                                       | met                     |
| 24 GB simulation                      | no OOM; klein keeps; dev drops | 0 OOM; klein keeps; no [dev] tier fits 24 GB (refused at submit with the reason) | met as documented       |
| byte anchors (sd15, SDXL, qwen-image) | pixel-identical to baseline    | identical                                                                        | met                     |
| determinism                           | same seed → same bytes         | identical across cold/warm/batch/restart; `MOLD_ATTN=math` reproduces            | met                     |

Speedups against the pre-campaign baseline on the same card: `flux-dev:q8` 3.6×
warm, `flux-schnell:q8` 4.6×, img2img 4.6×, inpaint 3.8×, PuLID 2.9×,
`jibmix-flux:fp8` 3.0×, SDXL 2.3×, the Wan two-stage chain 1.4×.

## Defects the UAT found on the branch, and their state on the final tip

| #     | defect                                                                                                                                                                    | state                                                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | streamed Mistral3 prefetch raced the forward once the parked prefix was page-locked → every FLUX.2 [dev] tier non-finite at step 0                                        | fixed; park-on and park-off renders byte-identical                                                                                                                 |
| 2     | retained transformer wedged the queue (a resolver VRAM block erased by the plan pass; the planned-engine wrapper hid the retained bytes; the FLUX.1 engine reported none) | fixed; identical second render reuses the transformer on both families, zero blocked lines; a Klein load after a resident FLUX.1 releases it under a 22 GB reserve |
| 3     | tip did not compile with any GPU feature while CI stayed green                                                                                                            | fixed; PR-route `cuda-typecheck` job added                                                                                                                         |
| 4     | `downgrade` succeeded against a live writer, which then wrote v3 bytes under the v2 name                                                                                  | fixed; writer lease + version-checked commit path                                                                                                                  |
| 5     | LTX-2 IC-LoRA reference encode failed at the tier's default 1216×704 (odd ÷64 latent grid) — pre-existing on main                                                         | fixed; canvas snaps onto the reference grid like upstream's floor                                                                                                  |
| 6     | a 9B FLUX.2 adapter on the 4B tier crashed in the merge; the baseline had silently rendered without it; the 9B pairing applied 8 of 32 layers                             | fixed; width refusal by name; per-projection slab merge                                                                                                            |
| 7     | refusal messages said "0.0 GB short" against numbers the decision did not use                                                                                             | fixed; the ceiling is named                                                                                                                                        |
| 8     | a control render recorded and merged the caller's LoRA twice — pre-existing on main                                                                                       | fixed                                                                                                                                                              |
| 9     | the writer lease file made older binaries refuse to start                                                                                                                 | fixed; lease lives in the gallery root, removed on exit, stale-tolerant                                                                                            |
| 10    | a field added by the defect-7 fix missed a construction site inside the private H3 cfg arm; a `cuda`-only PR typecheck stayed green                                       | fixed; the PR typecheck now compiles the shipping feature set                                                                                                      |
| #1707 | FLUX.2 denoise priced with FLUX.1's per-pixel factor; failed runs inflated the learned envelope past the card                                                             | fixed; q8 50-step plan 41.6 GB vs 39.74 GB measured high water                                                                                                     |

## Final-binary confirmations (Parts F and G, `4b1ad97d` and `01090880`)

- Identical second `flux2-dev:q8` render: 120.9 s → 60.5 s, transformer and prompt cache hits, no reload, zero blocked lines.
- LTX-2 IC-LoRA control at the tier default: renders at 1152×640 with the guide driving the motion; one caller LoRA recorded once, after the control adapter.
- A 9B adapter on the 4B Klein: refused in 0.46 s by name before any weight loads; on the 9B tier all 32 layers apply with no skipped-merge warnings.
- 24 GB budget: `flux2-dev:fp8` refused at submit naming the ceiling and the single-file reason, GPU never left idle; `flux2-dev:q4` refused with the GGUF reason; FLUX.1 and Klein render.
- Writer lease: an older binary starts with the lease present; SIGTERM removes it; a stale lease from a killed process is reported and swept by the downgrade.
- The F4 sequence replayed on the last binary: a Klein request after a resident FLUX.1 transformer under a 22 GB reserve is admitted with the retained bytes credited and renders; a second identical FLUX.1 render skips the reload.

## What is claimed on unit tests alone

- The streamed (block-offload) FLUX.2 [dev] path: only the sharded `flux2-dev:bf16`
  layout can stream, it is gated and not installed on plato, so the streamed path is
  pinned by tests and was not rendered in this campaign.
- FlashAttention on sm86 and sm100: every kernel compiled for those capabilities; no
  such card was available. sm120 is deliberately left on math attention.
- Metal: the F32 materialization charge for dense checkpoints is pinned by tests; no
  Metal render was made in this campaign.

## Review

Six rounds of read-only peer review by a separate agent, each running the full gate
set and break-testing at least five fixes; the final verdict was GO. The rounds and
their findings are preserved in `peer-review.md` beside the UAT evidence.

# Multi-GPU family and simulation qualification matrix

This is the checked-in acceptance map for Scheduler V2 model compatibility and
weight-free arbitrary-device-count simulation. The production source of truth
is `mold_inference::production_family_capabilities()`. Server placement and
batch planning project that registry; they do not maintain a second family
match.

Run the deterministic slice in one command:

```bash
scripts/tests/multi-gpu-deterministic-acceptance.sh
```

That command loads no model weights and uses no CUDA, Metal, NVML, cloud, or
wall-clock timing. The simulated names `3090`, `B200`, `12GiB`, and `MIG`
describe immutable planner fixtures, not observed hardware.

## Evidence status

| Tier | Evidence in this repository | Current claim |
| --- | --- | --- |
| 0 | Family registry, frozen-factory alias construction, execution-plan projection, request normalization/fingerprints, scheduler and batch simulations | Deterministic CI evidence |
| 1 | `scripts/regression-matrix.sh`, family-specific smoke tests, and the separate sm86 runner | Local-hardware runner exists; a new complete multi-GPU family campaign is not recorded by this document |
| 2 | Concrete deep-path tests listed below | Deterministic contract coverage plus separately invoked hardware paths where stated |
| 3 | Real 8×B200, real MIG, and real 12 GiB | Deferred; not hardware-qualified |

The local 2×RTX 3090 gate is also separate. See
[`README.md`](./README.md). A synthetic 12 GiB snapshot can prove admission and
CPU/offload selection, but the process can still physically allocate against
the host GPU's full memory. It is not a real 12 GiB qualification.

## Authoritative Tier-0 family contract

`CPU placement` below means a request-controlled execution-plan placement, not
an unadvertised engine-owned OOM fallback. `Exact` means exact only for the same
normalized request, immutable artifacts, execution-equivalence fingerprint,
code, and supported backend contract. Every family uses CPU-seeded initial
noise transferred to the execution device.

| Canonical family | Factory aliases | CUDA / Metal / CPU | CPU placement | Block offload | Tiled VAE | Native batch / cancellation | Media and workflows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `flux` | — | supported / supported / supported | text encoder | yes | shared policy | `[1]` / cooperative | image; source, inpaint, LoRA |
| `flux2` | `flux.2`, `flux2-klein` | supported / supported / supported | text encoder, VAE | yes | shared policy | `[1]` / cooperative | image; source, inpaint, LoRA |
| `sd15` | `sd1.5`, `stable-diffusion-1.5` | supported / supported / supported | none | no | shared policy | `[1]` / cooperative | image; source, inpaint, ControlNet, LoRA |
| `sdxl` | — | supported / supported / supported | none | no | shared policy | `[1]` / cooperative | image; source, inpaint, LoRA |
| `sd3` | `sd3.5`, `stable-diffusion-3`, `stable-diffusion-3.5` | supported / supported / supported | none | yes | shared policy | `[1]` / cooperative | image; source, inpaint, LoRA |
| `z-image` | — | supported / supported / supported | none | yes | no | `[1]` / cooperative | image; source, inpaint, LoRA |
| `qwen-image` | `qwen_image` | supported / supported / supported | none | yes | native CUDA | `[1]` / cooperative | image; source, inpaint, LoRA |
| `qwen-image-edit` | — | supported / supported / supported | none | yes | native CUDA | `[1]` / cooperative | image; ordered edit references, LoRA |
| `ltx-video` | `ltx_video` | supported / supported / supported | none | no | no | `[1]` / cooperative | video; independent-clip chains; no source/audio/LoRA |
| `ltx2` | `ltx-2`, `ltx2.3` | supported / correctness-only / correctness-only | Gemma text encoder | yes | native temporal chunks | `[1]` / cooperative | video; source/keyframes/retake/LoRA/chain; generated audio is checkpoint-specific |
| `wan` | — | supported / unsupported / correctness-only | UMT5 text encoder | no | no | `[1]` / cooperative | video; text-to-video and single-image conditioning, no chain/audio/LoRA yet |
| `wuerstchen` | `wuerstchen-v2` | supported / supported / supported | none | no | no | `[1]` / cooperative | image; source and inpaint; no LoRA |

The registry intentionally does not claim generic tiled VAE for Z-Image.
Qwen-Image has a separate CUDA tiler, and LTX-2 uses native temporal/framewise
video-decode chunks. The shared tiling policy is wired through FLUX, Flux.2,
SD1.5, SDXL, and SD3.

## Tier-1 owner and runnable reference

| Family | Owner | Concrete reference |
| --- | --- | --- |
| `flux` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `flux2` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `sd15` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `sdxl` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `sd3` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `z-image` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source cases |
| `qwen-image` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source/LoRA cases |
| `qwen-image-edit` | runtime qualification | `scripts/qwen-edit-parity-smoke.sh`, explicit source-edit CUDA smoke |
| `ltx-video` | runtime qualification | `scripts/regression-matrix.sh`, video and independent-chain cases |
| `ltx2` | runtime qualification | `scripts/regression-matrix.sh`, video/source/audio/durable-chain cases |
| `wan` | runtime qualification | `scripts/regression-matrix.sh`, text-to-video, image-to-video, first/last-frame, and single-frame-still cases across every installed tier (no hardware campaign recorded yet) |
| `wuerstchen` | runtime qualification | `scripts/regression-matrix.sh`, installed-family base/source cases |

These are executable owners, not a claim that every case passed on the current
branch. A hardware report must identify the exact commit, binary, device UUID,
model artifact, command, output validation, and logs.

## Tier-2 deep-path owners

| Family/path | Owner | Concrete deterministic reference |
| --- | --- | --- |
| FLUX LoRA | `mold-inference` | `flux/lora.rs::build_patches_stacks_multiple_specs_on_same_tensor` and `pipeline.rs::offload_lora_registry_is_built_before_adaptive_planning_when_enabled` |
| FLUX block offload | `mold-inference` | `flux/pipeline.rs::forced_offload_uses_sequential_generation_path_for_bf16_flux` |
| FLUX encoder fallback | `mold-server` | `execution_plan.rs::auto_cpu_exists_only_under_pressure_for_supported_family` |
| Shared VAE tiling | `mold-inference` | `vae_tiling.rs::test_decode_tiled_single_offset_matches_full` and `test_decode_tiled_three_offset_smooths_seams` |
| FLUX source/variation | runtime qualification | `scripts/regression-matrix.sh` source case; exact seed/normalized-request identity is covered by `execution_equivalence` |
| Flux.2 offload/source | `mold-inference` | `flux2/pipeline.rs::flux2_selected_bf16_offload_reaches_runtime_loader` and `flux2_img2img_uses_minus_one_to_one_source_normalization` |
| SD1.5 single-file/LoRA | `mold-inference` | `sd15/pipeline.rs::from_single_file_real_shape_load_smoke` and `lora_stack_fingerprint_equality_drives_unet_drop` |
| SDXL single-file/LoRA | `mold-inference` | `sdxl/pipeline.rs::from_single_file_real_shape_load_smoke` and `lora_stack_fingerprint_equality_drives_unet_drop` |
| SD3 offload/source | `mold-inference` | `sd3/pipeline.rs::sd3_selected_bf16_offload_reaches_runtime_loader` and `sd3_img2img_uses_minus_one_to_one_source_normalization` |
| Z-Image offload/source/LoRA | `mold-inference` | `zimage/pipeline.rs::zimage_selected_bf16_offload_reaches_runtime_loader`, `zimage_img2img_source_decode_uses_vae_native_zero_to_one_range`, and `zimage_lora_requests_use_sequential_generation_path` |
| Qwen-Image tiling/source | `mold-inference` | `qwen_image/pipeline.rs::qwen_proactive_tiled_decode_skips_primary_full_decode` and `qwen_img2img_uses_minus_one_to_one_source_normalization` |
| Qwen-Image-Edit | `mold-inference` | `qwen_image/pipeline.rs::qwen_image_edit_accepts_quantized_text_with_bf16_vision_sidecar` and `qwen_quantized_edit_always_uses_split_cfg_on_high_vram_cuda` |
| LTX-Video video/chain | `mold-inference` | `ltx_video/pipeline.rs::decode_apng_round_trips_rgb_frames` and `chain/capability.rs::ltx_video_is_independent_clips_without_audio` |
| LTX-2 source image/video/keyframes | `mold-inference` | `ltx2/conditioning.rs::stage_conditioning_stages_source_image_as_frame_zero_replacement`, `stage_conditioning_keeps_audio_and_reference_video_paths`, and `stage_conditioning_preserves_keyframe_targets` |
| LTX-2 audio supported/unsupported | `mold-inference` | `ltx2/runtime.rs::runtime_prepare_tracks_audio_and_video_latent_shapes` and `ltx2/pipeline.rs::audio_request_is_rejected_before_runtime_for_video_only_checkpoint_assets` |
| LTX-2 external assets | `mold-inference` | `ltx2/pipeline.rs::from_transformer_only_single_file_preserves_external_vae_for_chains` |
| LTX-2 ConvRot | `mold-inference` | `ltx2/convrot.rs::dequantizes_and_unrotates_packed_rows` and `runtime.rs::ltx2_convrot_forces_streaming_for_reconstructed_bf16_weights` |
| LTX-2 chains/chained video | `mold-inference` + `mold-server` | `ltx2/pipeline.rs::render_chain_supports_one_stage_and_distilled_pipelines`, `chain_job_runner.rs::durable_execute_job_records_exactly_one_runner_gallery_row`, and `resumed_job_reuses_durable_companions_after_runtime_config_changes` |
| LTX-2 retake | `mold-inference` | `ltx2/runtime.rs::runtime_prepare_derives_retake_mask_from_request_range` and `retake_runtime_keeps_requested_full_resolution_shape` |
| Prepared/server batches | `mold-scheduler` + `mold-server` | `batch_partition_planner`, `batch_parent.rs::n_1_2_8_64_completion_parity_and_compaction`, and `multi_gpu_acceptance` |
| Wuerstchen source/load | `mold-inference` | `wuerstchen/pipeline.rs::wuerstchen_loads_vqgan_tensors_through_shared_pool` plus runtime qualification source case |

## Deterministic arbitrary-N simulation coverage

`crates/mold-scheduler/tests/multi_gpu_acceptance.rs` covers:

- inventories of 0, 1, 2, 8, 16, and 64 devices;
- homogeneous synthetic 2×RTX 3090 and 8×B200 fleets;
- singleton adaptive parents on 1, 2, 8, 16, and 64 devices;
- a synthetic 12 GiB capacity, aggregate 8+8 GiB admission against 12 GiB
  host headroom, and the older-8-versus-two-younger-4 priority case;
- heterogeneous backend, capability, speed, and VRAM edges;
- stable UUID assignment under input/visible-ordinal reordering;
- distinct MIG child IDs without parent/child collapse;
- disabled, draining, degraded, and unavailable devices with typed reasons;
- compatible work beyond rank 200 and a late specialist without loss of
  cardinality;
- exact planner equality under device/work/candidate permutation;
- 200-ready/8-device and 10,000-ready/64-device envelopes bounded by
  `operation_budget`, with no wall-clock assertion.

Real throughput, CUDA allocation behavior, thermal behavior, NVLink/topology,
and hardware faults cannot be established by these simulations.

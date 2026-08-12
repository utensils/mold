# MiniMax H3

MiniMax H3 is an audio-video generation family from
[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3). Mold can discover,
download, verify, repair, inventory, and remove two compact Comfy variants. The
files are downloaded directly from their pinned Hugging Face repositories;
Mold does not bundle or mirror the weights.

::: warning Downloadable does not mean runnable
Both compact variants can be downloaded on any Mold host, but H3 execution is
not generally enabled in ordinary release builds. FL2VA runs only when a server
authenticates the exact reviewed CUDA runtime, artifacts, device, and request
profile described below. Ref2VA, Metal, CPU, broader request shapes, and hosted
third-party inference are unsupported.
:::

## Compact variants

| Model                                 | Task                                     | Total pull | Runtime status                       |
| ------------------------------------- | ---------------------------------------- | ---------: | ------------------------------------ |
| `minimax-h3-fl2va:comfy-pruned-int8`  | First/last-frame conditioning with audio |  42.482 GB | Exact CUDA route is first-frame-only |
| `minimax-h3-ref2va:comfy-pruned-int8` | Reference media to video with audio      |  42.482 GB | Downloadable; execution unavailable  |

Pull a variant from the CLI, or install it from **Models → Discover** in Mold
Studio:

```bash
mold pull minimax-h3-fl2va:comfy-pruned-int8
mold pull minimax-h3-ref2va:comfy-pruned-int8
```

The files are revision-pinned and SHA-256 verified before Mold marks the model
complete. Raw repository IDs, custom manifests, configured aliases, and live
catalog recipes cannot substitute for either registered graph.

## Download size and sources

Each compact variant has the same component graph except for its task-specific
transformer:

| Component                                              |              Bytes |  Decimal size | Upstream source                                                                                                     |
| ------------------------------------------------------ | -----------------: | ------------: | ------------------------------------------------------------------------------------------------------------------- |
| Task transformer                                       |     20,970,379,616 |     20.970 GB | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/eb8a16107c595128b3a578f82d2ce2f75920c355) |
| Qwen3-VL NVFP4-AWQ text encoder                        |     15,687,142,551 |     15.687 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| FP16 video VAE                                         |      5,207,808,496 |      5.208 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| FP32 audio VAE                                         |        605,254,808 |      0.605 GB | `Comfy-Org/MiniMax-H3`                                                                                              |
| Tokenizer, processor, scheduler, and component configs |         11,504,847 |      0.012 GB | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/bfc8ed0353f5a9733be73e6b2c98ec0948195b86) |
| **One complete variant**                               | **42,482,090,318** | **42.482 GB** | Both pinned repositories                                                                                            |

The encoder, VAEs, and common support files are shared between the variants.
After one complete variant is installed, adding the other downloads its 20.970
GB transformer and 546-byte task config. Both variants together occupy 63.452
GB (63,452,470,480 bytes) of model payloads, excluding filesystem and Hugging
Face cache overhead.

Sizes above are decimal gigabytes (`1 GB = 1,000,000,000 bytes`) and describe
downloads and disk use, not peak VRAM. They come from Mold's registered,
full-file manifest identities rather than estimates from repository listings.

## Qualified FL2VA request

Even with all files installed, a request is accepted only when the server
advertises the authenticated FL2VA capability and the request matches this
entire envelope:

- CUDA on the exact qualified runtime and device identity; no generic
  “CUDA-compatible” promise
- `1344x768`, batch size 1
- exactly 124 frames at 24 fps
- exactly 21 terminal-inclusive sampler grid points (20 model evaluations)
- one required first-frame image and no last-frame endpoint
- MP4 output with synchronized generated audio

Mold rejects rather than resizing, rerouting, changing steps, dropping the
source image, or falling back to another backend. The Models screen may still
show a downloaded H3 checkpoint on a host that cannot execute it; Create and
request routing remain unavailable unless that host supplies the exact runtime
capability.

## License and support boundary

H3 uses the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE),
not Mold's MIT license. Review the upstream terms for your intended use. Mold's
[authorization decision](https://github.com/utensils/mold/blob/main/docs/architecture/minimax-h3-authorization.md)
permits these upstream-direct compact downloads and local storage; it does not
permit Mold-hosted weight redistribution or hosted third-party H3 inference.

The official BF16 checkpoints remain hidden qualification references. Their
much larger artifact graphs are not public Mold download options.

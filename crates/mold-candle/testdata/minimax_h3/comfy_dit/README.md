# MiniMax H3 pruned DiT golden header fixtures

Each `.header` file is the exact eight-byte little-endian safetensors
header-length prefix followed by the exact published JSON header bytes of one
pruned INT8 ConvRot transformer — no tensor payload. The SHA-256 of each file
is therefore that artifact's pinned header identity
(`H3ComfyPublishedArtifact::published_int8_header_pins`).
The golden tests in `crates/mold-candle/src/minimax_h3/comfy_dit.rs`
(`published_int8_golden_headers_*`) parse these bytes through the production
reader over a sparse file of the published length, require the pinned header
length, identity, tensor count, and metadata, and require the schema mold
derives for each artifact to equal the header's tensor inventory exactly.

| Fixture | Repository | Revision | Repo path | Fetched | `header_len` | Header identity (SHA-256) | Published LFS SHA-256 | File bytes | Tensors |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `comfy-org-fl2va-pruned-int8-convrot.header` | `Comfy-Org/MiniMax-H3` | `eb8a16107c595128b3a578f82d2ce2f75920c355` | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 2026-10-10 | 95,416 | `8232b3c428e54591c286b5eee3eaba840f03d2cd2cb359ec09dd2558974ddc74` | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` | 20,970,379,616 | 932 |
| `fasth3-v2-pruned-int8-convrot.header` | `FastVideo/FastVideo-FastH3-Comfy` | `ec1e3aa374a91c57b0b94a1623b7e657c0498cf2` | `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | 2026-10-10 | 114,256 | `9b750bb184553a8c94a0feef4f67f72972bb4492f323753bebaa9fc08a57cd2e` | `0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8` | 22,128,378,696 | 1,098 |

FastH3 V2 against the Comfy-Org FL2VA file, as the headers record it: every
shared tensor keeps its name, shape, and dtype except the eight token-refiner
linears, which are `I8` with an `F32 [out, 1]` `weight_scale` and a
`comfy_quant` sidecar (Comfy-Org keeps them `BF16`); fifty
`blocks.N.attn.to_gate_compress.{weight I8 [7168, 5376], weight_scale F32 [7168, 1], comfy_quant}`
VSA gates are added (unused by ComfyUI's dense attention,
`comfy/ldm/minimax/model.py:169-172`); `__metadata__` is `{"format": "pt"}`
instead of absent. The AdaLN curve is the same `F32 [1025, 8]` table and every
`comfy_quant` payload in both files (258 and 200 of them, checked by ranged
reads on 2026-10-10) is the identical
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.

Re-fetching either row is a straight rerun of the turbo header tool, e.g.:

```bash
python3 scripts/fetch-minimax-h3-turbo-header.py \
  --repo FastVideo/FastVideo-FastH3-Comfy \
  --revision ec1e3aa374a91c57b0b94a1623b7e657c0498cf2 \
  --path diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors \
  --out crates/mold-candle/testdata/minimax_h3/comfy_dit/fasth3-v2-pruned-int8-convrot.header
```

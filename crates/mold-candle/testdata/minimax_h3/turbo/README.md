# MiniMax H3 Turbo LoRA golden header fixtures

Each `.header` file is the exact eight-byte little-endian safetensors
header-length prefix followed by the exact published JSON header bytes of
one Turbo adapter — no tensor payload. `published_tier_pins_are_recomputed_from_the_checked_in_headers`
in `crates/mold-candle/src/minimax_h3/turbo_lora.rs` re-derives every
tier's `header_len`, `header_identity_sha256`, payload byte count, file
size, tensor count, and training metadata from these bytes; **only
`content_sha256` (the published file's own LFS SHA-256) is pinned as a
literal in code** — everything else a tier reports is recomputed from its
fixture by that golden test, so a fixture fetched from the wrong revision
fails the test rather than the download.

| Fixture | Repository | Revision | Repo path | Fetched | Tool | `header_len` | Header identity (SHA-256) | Published LFS SHA-256 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `fl2v-8step-v1.0.header` | `Comfy-Org/MiniMax-H3` | `dc559027db79c174125df4d827db55cd11178860` | `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2026-08-19 | ad hoc (pre-dates this script; #1172) | 73,632 | `eadcdb12138db967789252da26d2abe41905b2579e1cf07b866a573e88d298fd` | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| `fl2v-4step-768p-v1.0.header` | `Comfy-Org/MiniMax-H3` | `dc559027db79c174125df4d827db55cd11178860` | `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 2026-08-19 | ad hoc (pre-dates this script; #1172) | 73,624 | `3db9fe99ff46229525c43cbe6ba5bafc8d96bdeb22ee69949ef61d4d58d561d8` | `c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc` |
| `ref2v-4step-v0.1.header` | `Comfy-Org/MiniMax-H3` | `dc559027db79c174125df4d827db55cd11178860` | `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 2026-08-19 | ad hoc (pre-dates this script; #1172) | 73,632 | `53370bff715f074018793b9ebc71fa0ecd8bdfd8c5554a716ccf7bf5e6a6f745` | `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c` |
| `fl2v-4step-768p-v1.1.header` | `lightx2v/Minimax-h3-Turbo` | `05ef678438e84933c406131b59abbf86919b3aac` | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | 2026-09-01 | `scripts/fetch-minimax-h3-turbo-header.py` | 73,624 | `e7a5b995877b2997c0055cad77d1a1ef48a28bc8fd388f8b19be601249e7d27c` | `449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558` |
| `fl2v-8step-768p-v1.0.header` | `lightx2v/Minimax-h3-Turbo` | `05ef678438e84933c406131b59abbf86919b3aac` | `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 2026-09-01 | `scripts/fetch-minimax-h3-turbo-header.py` | 73,632 | `0541a8b7d525096f45df5f6e8d076f49173cb2d3d58ad233e37e04a63677d78d` | `08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980` |
| `fl2v-4step-768p-v1.0-r21.header` | `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` | `be8eb3ea3466cbb7def202ffec0d2fdc054256ac` | `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors` | 2026-09-02 | `scripts/fetch-minimax-h3-turbo-header.py` | 52,928 | `e9a8cf11d436ab25df9667896a02c9768aca800ed5b8e5d794e80b7cb866f539` | `1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115` |
| `fl2v-8step-v1.0-r21.header` | `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` | `be8eb3ea3466cbb7def202ffec0d2fdc054256ac` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors` | 2026-09-02 | `scripts/fetch-minimax-h3-turbo-header.py` | 52,944 | `f1bbb213d10d64aaf63d4e973d72887e43d356a3352ba73534e04aa317795f2a` | `a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840` |
| `ref2v-4step-v0.1-r21.header` | `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` | `be8eb3ea3466cbb7def202ffec0d2fdc054256ac` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors` | 2026-09-02 | `scripts/fetch-minimax-h3-turbo-header.py` | 52,952 | `3c1db66284973ee4eec4e9700e12b1fc587b1aa7f85af6a811daac0d15b4db6f` | `2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0` |

The last three rows are SVD-resized derivatives rather than published PEFT
exports: 416 tensors instead of 624 (no `alpha` scalars), a per-module rank
read from the header rather than one `training_rank` for the whole file, and a
numeric `__metadata__.baked_scale` recording the source `alpha / rank` that was
multiplied into `lora_B`. Each still declares `training_rank "128"` — the rank
it was resized FROM — and a `resized_from` naming the exact published adapter
it approximates, both of which the golden test welds to the source tier.
`drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` publishes further rank-28/64 resizes
that are deliberately NOT pinned here: their `baked_scale` is an English
sentence rather than a number, which the contract refuses by design.

Re-fetching any row is a straight rerun, e.g.:

```bash
python3 scripts/fetch-minimax-h3-turbo-header.py \
  --repo lightx2v/Minimax-h3-Turbo \
  --revision 05ef678438e84933c406131b59abbf86919b3aac \
  --path minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors \
  --out crates/mold-candle/testdata/minimax_h3/turbo/fl2v-4step-768p-v1.1.header
```

The tool refuses (rather than silently downloading gigabytes) if either
ranged GET answers HTTP 200 instead of 206, and it independently
cross-checks the header-derived file size against the HF repository tree
API's own recorded size for that exact path and revision **before** it
opens the output file — so a copy pulled from the wrong revision or a
mirror that ignores `Range` is caught before it reaches disk.

Every `.header` in this directory must be claimed by some
`H3TurboLoraTier::header_fixture_path()`;
`every_checked_in_header_fixture_is_claimed_by_a_tier` fails otherwise.
A fixture no tier names is validated by nothing, so a golden header for a
tier that has not landed belongs in the change that lands the tier, not
ahead of it.

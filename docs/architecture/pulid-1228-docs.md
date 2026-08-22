# PuLID for SDXL (#1228) — staged documentation deltas

**This file is a hand-off, not a doc.** It records every user- and
agent-facing documentation change PR #1301 implies, with the target file and
section for each, so one docs agent can apply them all in a single pass rather
than several PRs conflicting on the same paragraphs. Delete it once they land.

Nothing in `CLAUDE.md`, `crates/mold-cli/src/skill/SKILL.md`, `README.md`,
`THIRD_PARTY_NOTICES.md`, `website/`, or `docs/architecture/pulid.md` was
edited by this PR. `changelog.d/pulid-sdxl.md` IS in this PR, per
`changelog.d/README.md`.

---

## 0. The one-paragraph summary

PuLID face-identity conditioning now covers SDXL as well as FLUX. It is one
contract with two adapters: `mold_core::identity::IdentityFamily` decides which
checkpoints qualify and which adapter they condition with, the face extractor
is shared, and clients need no change because every surface already gates on
the server's advertised `/api/models[].supports_identity`.

---

## 1. `CLAUDE.md` (and its `AGENTS.md` symlink)

### 1a. Replace the "PuLID identity conditioning is one authority per layer" bullet

Current text says "the milestone-1 qualified checkpoints (`flux-dev:q4`,
`flux-dev:q8` …)" and "the hidden files-only `pulid-flux` bundle". Both are now
per-family. Suggested replacement, keeping the bullet's existing voice:

> **PuLID identity conditioning is one authority per layer, and weight zero is
> inert everywhere.** `mold_core::identity` owns the request contract, and
> `IdentityFamily` is its single answer to both halves of the question — which
> checkpoints qualify, and which adapter they condition with. FLUX takes
> PuLID-FLUX v0.9.1 (`flux-dev:q4`, `flux-dev:q8`; bare `flux-dev` resolves to
> `:q8`); SDXL takes PuLID v1.1 (`sdxl-base:fp16`, `juggernaut-xl:fp16`,
> `realvis-xl:fp16`, `dreamshaper-xl:fp16` — upstream's own
> `docs/pulid_v1.1.md` names the last three, and every other `family: "sdxl"`
> manifest entry is an enumerated refusal: turbo, Playground's EDM objective,
> and the two Pony derivatives). Being an SDXL checkpoint is necessary and
> never sufficient, which is why the list is enumerated rather than
> pattern-matched and why a new SDXL manifest entry fails a test rather than
> inheriting qualification. `supports_identity` on the generation profile and
> `/api/models[]` derive from `identity_family`; never add a second predicate.
> The `id_weight` / `id_start_step` ranges, the header-only bounded decode
> limits, the no-LoRA / no-img2img gate, and the hard refusal on a build
> without the `pulid` feature are family-blind and apply identically.
> `true_cfg` / `cfg_start_step` are FLUX-only (`TRUE_CFG_FLUX_ONLY`): they
> exist because FLUX [dev] is guidance-distilled, and SDXL's ordinary
> `guidance` already IS the classifier-free scale, so accepting them there
> would either mean nothing or double the guidance the render already
> performs. Assets are TWO hidden files-only bundles — `pulid-flux` and
> `pulid-sdxl` — sharing one `pulid` family, one `shared/pulid/` storage root,
> and the same four extraction artifacts, so they differ by exactly the adapter
> and a machine holding one pulls only the other's (984 MB for v1.1). The
> antelopev2 acceptance follows the FILES, so both bundles ask for it;
> removal's ref-counting keeps the shared downloads and the derived
> tower/parser when either is removed. An identity request materializes its
> family's bundle through ordinary dependency preparation
> (`mold-server/src/identity_dependencies.rs`, keyed on the resolved model
> family and cross-checked against the model's own): placement preview lists it
> under `pending_downloads` and fetches nothing, admission is the only refusal
> point for a missing acceptance, the five exact paths freeze into
> `FrozenEngineConfig.identity_assets` (which now carries the family), and the
> device overhead is charged per family —
> `IDENTITY_VRAM_OVERHEAD_BYTES` 1.25 GB for FLUX,
> `IDENTITY_SDXL_VRAM_OVERHEAD_BYTES` 850 MB for SDXL. Effective
> `id_weight == 0` or no identity fields must be byte-indistinguishable from a
> plain request at every layer, on both families. The forced-local CLI path
> resolves the bundle itself in `create_engine_with_pool`, choosing it with the
> same `identity_family`; `FrozenEngineConfig::resolve` deliberately stays
> `None`.

### 1b. Add a new bullet after the existing FLUX adapter bullet

> **PuLID's SDXL adapter is a UNet cross-attention injection, and its layer
> table is a permutation.** `crates/mold-inference/src/sdxl/pulid.rs` ports
> `pulid/attention_processor.py:275-422` under the module globals this
> checkpoint was trained with — `NUM_ZERO = 0`, `ORTHO = ORTHO_v2 = False`, so
> the shipped arithmetic is the plain additive branch at `:378`. Each of the 70
> `attn2` modules keeps its own query, projects the 32-token identity through a
> bias-free `id_to_k` / `id_to_v` pair, attends, and adds
> `id_scale * id_hidden` onto the text attention output BEFORE `to_out`, via
> the candle fork's `CrossAttentionHook` (so `crate::attention`'s Metal
> chunking and `MOLD_ATTN` rules apply). The correctness argument is the layer
> table: PuLID keys its weights by diffusers' `attn_processors` position, which
> walks `down -> up -> mid`, while candle's UNet forward — and therefore the
> hook index — walks `down -> mid -> up`, so reading
> `id_adapter_attn_layers.<hook_index>` silently conditions the mid block on the
> up blocks' weights. `plan_attn_layers` derives the permutation from the UNet
> config; the load checks the checkpoint's inventory in BOTH directions
> (an SD1.5-shaped plan's indices are a prefix of the SDXL file's, so a
> one-directional check loads 16 of its 70 modules); and at run time the hook
> refuses an index it was not planned for and a head count the UNet disagrees
> with. The identity rides the CFG batch: mold runs ONE `[uncond, cond]`
> forward where upstream runs two, so the embedding is concatenated on dim 0 in
> that order and the unconditional half is REQUIRED — a `[1, 32, 2048]`
> embedding would broadcast the conditional identity onto the negative branch
> and cancel most of the identity out of the guided result without failing.
> `request_needs_unconditional_identity` is the one authority the extraction
> and the engine both read. The IDFormer is shared:
> `identity::extraction::idformer_prefix` is the single family-specific fact
> (`id_adapter` versus `pulid_encoder`) because upstream instantiates the same
> `IDFormer()` class in both pipelines, and the extraction cache key takes the
> family's own adapter digest so one photograph never serves one family the
> other's embedding. `sdxl_unet_layout()` is mold's copy of SDXL's published
> `unet/config.json` because candle keeps `StableDiffusionConfig.unet` private;
> the fixture test and the two runtime guards are what keep it honest.

### 1c. `docs/design/mold-studio-spec.html` "Identity-photo invariant" bullet

The invariant already says the gate is the server's own
`capabilities.supports_identity` / `/api/models[].supports_identity` with
absence reading as NO, and that iPhone/TUI/Discord are a separate issue. Only
one sentence needs adding:

> The gate is family-blind by construction: SDXL checkpoints qualify exactly as
> FLUX ones do, so nothing on any browser surface changes — a qualified SDXL
> checkpoint simply starts advertising `supports_identity: true` and the well
> appears.

### 1d. Feature-flag paragraph

No change. `pulid` still gates both families identically, and
`crates/mold-inference/src/sdxl/pulid.rs` is deliberately NOT behind the cargo
feature, for the same reason `flux::pulid` is not: the gate lives once at the
request contract, and gating the adapter would take it out of the workspace
clippy run.

---

## 2. `crates/mold-cli/src/skill/SKILL.md`

Section **"Face-identity conditioning (PuLID-FLUX)"** (~line 747).

- Retitle to **"Face-identity conditioning (PuLID)"**.
- Replace the bundle line with the per-family pair:

  ```bash
  # FLUX
  mold pull pulid-flux --accept-license insightface-antelopev2
  mold run flux-dev:q4 "an astronaut in a diner" --id-image face.jpg

  # SDXL — a machine that already has pulid-flux pulls only the 984 MB adapter
  mold pull pulid-sdxl --accept-license insightface-antelopev2
  mold run sdxl-base:fp16 "an astronaut in a diner" --id-image face.jpg
  mold run juggernaut-xl:fp16 "a studio portrait" --id-image face.jpg --id-weight 0.8
  ```

- Add to the prose: qualified models are `flux-dev:q4`, `flux-dev:q8`,
  `sdxl-base:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, and
  `dreamshaper-xl:fp16`. Clients read the server's advertised
  `/api/models[].supports_identity` rather than this list.
- `--true-cfg` / `--cfg-start-step` row: add "**FLUX only.** SDXL's
  `--guidance` already is the classifier-free scale; naming these on an SDXL
  identity request is refused."
- `--negative-prompt` sentence: today it reads "does nothing on FLUX unless
  `--true-cfg` is set". Add that on SDXL it works normally, as it does for any
  CFG model, and that PuLID conditions the negative pass on the unconditional
  identity automatically.
- `mold rm pulid-flux` sentence: name both bundles and note that removing one
  keeps the four shared extraction artifacts while the other is installed.

---

## 3. `README.md`

Wherever PuLID is introduced, change "PuLID-FLUX" to "PuLID" and name both
families and both bundles. Keep the InsightFace non-commercial licence sentence
verbatim — it is unchanged and applies to both bundles, because the gated files
are the shared extractor's.

---

## 4. `THIRD_PARTY_NOTICES.md`

- The InsightFace antelopev2 entry is unchanged in substance; if it names
  `pulid-flux` as the manifest that requires it, change that to "the `pulid-flux`
  and `pulid-sdxl` bundles".
- Add `pulid_v1.1.safetensors` (`guozinan/PuLID`) beside the existing
  `pulid_flux_v0.9.1.safetensors` entry. Same project, same **Apache-2.0**
  licence, no acceptance gate of its own.
- No new third-party code, no new dependency: the SDXL adapter is a pure-Rust
  port and the candle fork bump adds only `CrossAttentionHook`.

---

## 5. `website/guide/identity.md`

The largest delta. Suggested structure:

1. **Intro** — PuLID conditions a render on a face; it works on FLUX and on
   SDXL, with a different adapter for each.
2. **Which models** — the six qualified names in a table with their bundle.
   State plainly that being an SDXL checkpoint is not enough and name the four
   refusals with their reasons (turbo/lightning distillation, Playground's EDM
   objective, the two Pony derivatives' retrained conditioning). Point readers
   at `/api/models[].supports_identity` as the live answer.
3. **Getting the assets** — `mold pull pulid-flux` / `mold pull pulid-sdxl`,
   both `--accept-license insightface-antelopev2`, and the "already have one?
   you pull one file" note with the real sizes (1.14 GB FLUX adapter, 984 MB
   SDXL adapter, ~1.19 GB of shared extractor).
4. **Using it** — the existing FLUX examples, plus SDXL ones. `--id-weight`
   and `--id-start-step` mean the same thing on both.
5. **What differs between the families** — a short table:

   | | FLUX | SDXL |
   | --- | --- | --- |
   | Adapter | `pulid_flux_v0.9.1.safetensors`, 20 modules between transformer blocks | `pulid_v1.1.safetensors`, 70 UNet cross-attention injections |
   | `--guidance` | distilled conditioning scalar | the classifier-free scale |
   | `--true-cfg` / `--cfg-start-step` | supported | refused |
   | `--negative-prompt` | only with `--true-cfg` | always |
   | Extra VRAM | ~1.25 GB | ~850 MB |

6. **Limits** — unchanged and family-blind: up to 4 photographs, no LoRA, no
   img2img, PNG/JPEG within 16 MiB / 8192 px / 32 MP.
7. **Removal** — `mold rm pulid-sdxl` deletes the v1.1 adapter and keeps the
   shared extractor while `pulid-flux` is installed, and vice versa.

## 6. `website/guide/generating.md`, `website/guide/tui.md`, `website/api/*`

Anywhere the identity control is described as FLUX-only, make it family-neutral
and point at `supports_identity`. The TUI, Discord, iPhone, and browser
surfaces need no behavioural change, so their pages need only the model list
updated.

## 7. `docs/architecture/pulid.md`

Retitle from "PuLID-FLUX: asset and encoder lifecycle" to "PuLID: asset and
encoder lifecycle", and:

- **"The bundle is four unrelated artifacts"** → five, and there are two
  bundles. Explain the sharing: same `pulid` family, same `shared/pulid/`
  root, `IdentityAdapter` is not a model-specific component, so
  `storage_path` lands the four extraction artifacts at identical paths by
  construction rather than by a special case.
- **"The encoders"** → note that `IDFormer` is the same class in both
  pipelines and that only its checkpoint prefix differs
  (`idformer_prefix`), with the measured SDXL parity numbers below.
- **"Parity coverage"** → add the SDXL goldens: their directory, the
  weight-gated invocation, and the measured errors (below).
- **"Memory"** → add `IDENTITY_SDXL_VRAM_OVERHEAD_BYTES` and its derivation.
- **"Removal"** → the ref-counting story for two bundles.
- **"Not yet built"** → drop SDXL from it if listed; add the two follow-ups in
  section 9.

---

## 8. Measured numbers, for whichever page quotes them

Captured on this branch, CPU, f32, against the pinned
`pulid_v1.1.safetensors` (SHA-256
`4cb8ceec1078e0165399b88332ab3c5971619111b8e1730e6bae64144aabae41`,
984,405,232 bytes, Apache-2.0):

| Golden | error, relative to the tensor's own peak |
| --- | --- |
| `idformer.single.output` | 4.286e-7 |
| `idformer.uncond.output` | 1.154e-6 |
| `attn1.id_hidden_states` (down_blocks.1, 640) | 2.268e-7 |
| `attn1.combined_s1p0` / `_s0p7` | 1.243e-7 / 1.100e-7 |
| `attn121.id_hidden_states` (mid_block, 1280) | 1.535e-7 |
| `attn121.combined_s1p0` / `_s0p7` | 7.520e-8 / 6.226e-8 |
| `attn49.id_hidden_states` (up_blocks.0, 1280) | 2.201e-7 |
| `attn49.combined_s1p0` / `_s0p7` | 1.032e-7 / 9.369e-8 |

Layer geometry: 70 `attn2` modules — 10 at 640 channels / 10 heads
(`down_blocks.1`, `up_blocks.1`) and 60 at 1280 / 20 heads (`down_blocks.2`,
`up_blocks.0`, `mid_block`); `dim_head` is 64 throughout;
`cross_attention_dim` 2048. Weights:
`2 x 2048 x (10 x 640 + 60 x 1280) = 340,787,200` elements = 681,574,400 bytes
at f16/bf16.

---

## 9. Follow-ups this PR deliberately does not take

1. **`StableDiffusionConfig` exposes no accessor for its `unet` field**, so
   `sdxl::pulid::sdxl_unet_layout()` is mold's own copy of the published SDXL
   `unet/config.json`. A one-line `pub fn unet(&self) -> &UNet2DConditionModelConfig`
   in the candle fork would remove the duplication; until then the fixture test
   and the two runtime guards are the defence.
2. **No PuLID checkpoint exists for SD1.5.** `plan_attn_layers` already handles
   its geometry (16 `attn2` modules, pinned against
   `attn_layer_map_sd15.json`), so if one is ever published the work is a
   manifest entry and a qualified-model list.
3. **The adapter is opened by pathname on both families.** A review raised
   the descriptor-retention rule for `SdxlPulidAdapter::load`. It was left as
   it is, matching `flux::pulid::PulidAdapter::load` and the IDFormer half of
   the extraction, and the reasoning is now recorded on the function: the rule
   protects a loader that has just authenticated bytes and would throw that
   away by reopening a name (the EVA02-CLIP and BiSeNet conversions), and the
   adapter has no such fresher authentication — its pin was verified at
   download time and admission freezes the exact path. If mold decides a
   manifest-pinned adapter should be re-authenticated at load, FLUX and SDXL
   must move together; doing it for one family would give one contract two
   answers.
4. **iPhone, TUI, and Discord** are untouched by design — they gate on
   `supports_identity` and get SDXL for free — but none of them has an SDXL
   identity UAT yet.

---

## 10. UAT recipe (for the orchestrator's Metal / CUDA run)

No GPU work or checkpoint download was performed on this branch. The recipe:

```bash
# 0. Build with the feature. `protoc` is in the devshell.
nix develop -c cargo build --release -p mold-ai --features metal,pulid   # macOS
nix develop -c cargo build --release -p mold-ai --features cuda,pulid    # Linux

# 1. Assets. A machine that already ran the FLUX UAT pulls ONE file (984 MB);
#    a fresh one pulls the whole ~2.2 GB bundle.
mold pull pulid-sdxl --accept-license insightface-antelopev2
mold pull sdxl-base:fp16

# 2. Forced-local, single reference. Compare against the same seed with no
#    identity: the face must change, nothing else about the composition should
#    move much.
mold run sdxl-base:fp16 "a studio portrait of a person, soft key light" \
  --local --seed 12345 --steps 25 --guidance 7.5 \
  --output /tmp/sdxl-plain.png
mold run sdxl-base:fp16 "a studio portrait of a person, soft key light" \
  --local --seed 12345 --steps 25 --guidance 7.5 \
  --id-image /Volumes/ExternalStorage/pulid-dev/uat/face.jpg \
  --output /tmp/sdxl-id.png

# 3. Weight sweep. 0.0 must be BYTE-IDENTICAL to the plain render above.
for w in 0.0 0.4 0.8 1.0 1.5; do
  mold run sdxl-base:fp16 "a studio portrait of a person, soft key light" \
    --local --seed 12345 --steps 25 --guidance 7.5 \
    --id-image /Volumes/ExternalStorage/pulid-dev/uat/face.jpg \
    --id-weight "$w" --output "/tmp/sdxl-id-w$w.png"
done
shasum -a 256 /tmp/sdxl-plain.png /tmp/sdxl-id-w0.0.png   # must match

# 4. Delayed start, and the other three qualified checkpoints.
mold run sdxl-base:fp16 "..." --local --id-image face.jpg --id-start-step 5
for m in juggernaut-xl:fp16 realvis-xl:fp16 dreamshaper-xl:fp16; do
  mold pull "$m" && mold run "$m" "a studio portrait" --local \
    --id-image face.jpg --output "/tmp/sdxl-id-${m%%:*}.png"
done

# 5. Multi-photograph averaging (up to 4).
mold run sdxl-base:fp16 "a hiker on a ridge" --local \
  --id-image front.jpg --id-image side.jpg --id-image smiling.jpg

# 6. Refusals. Each must fail with the named message, not render.
mold run sdxl-base:fp16 "..." --local --id-image face.jpg --true-cfg 2.0
mold run sdxl-turbo:fp16 "..." --local --id-image face.jpg
mold run playground-v2.5:fp16 "..." --local --id-image face.jpg
mold run sdxl-base:fp16 "..." --local --id-image face.jpg --lora some.safetensors
mold run sdxl-base:fp16 "..." --local --id-image face.jpg --image source.png

# 7. Served path. `supports_identity` must be true for the four qualified
#    entries and absent/false for the refusals.
mold serve &
curl -s localhost:7680/api/models | jq -r \
  '.[] | select(.name|startswith("sdxl")or startswith("juggernaut")or startswith("realvis")or startswith("dreamshaper")) | "\(.name) \(.supports_identity)"'

# 8. Identity similarity, the objective check. Reuse the FLUX UAT's cosine
#    script (scratchpad/q8-cosine.sh pattern): extract ArcFace embeddings from
#    the reference photograph and from each render, and report cosine
#    similarity. The plain render is the control — it should sit near zero
#    against the reference, and every id_weight >= 0.8 render well above it.
#    A sweep whose similarity does NOT rise with id_weight means the layer
#    table is wrong, which is the one failure the goldens cannot catch.

# 9. Residency. Run an identity render, then an unconditioned one on the same
#    engine, and confirm VRAM returns — the adapter must not survive the
#    second request or an unload/park.
```

**What would falsify the port**: a render that looks like a plausible face but
does not resemble the reference at any weight. That is the signature of a
mis-ordered layer table, and step 8's cosine sweep is what detects it.

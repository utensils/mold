# PuLID for SDXL (#1228) — staged documentation deltas

**Sections 1-8 have been applied** to `CLAUDE.md`, `docs/design/mold-studio-spec.html`,
`crates/mold-cli/src/skill/SKILL.md`, `README.md`, `THIRD_PARTY_NOTICES.md`,
`website/guide/identity.md`, `website/guide/generating.md`, `website/guide/tui.md`,
`website/api/index.md`, `website/api/discord.md`, and `docs/architecture/pulid.md`,
and removed from this file. What remains below — the follow-ups this PR
deliberately does not take, and the UAT recipe for the orchestrator's Metal /
CUDA run — is still live and has not been executed.

---

## 0. The one-paragraph summary

PuLID face-identity conditioning now covers SDXL as well as FLUX. It is one
contract with two adapters: `mold_core::identity::IdentityFamily` decides which
checkpoints qualify and which adapter they condition with, the face extractor
is shared, and clients need no change because every surface already gates on
the server's advertised `/api/models[].supports_identity`.

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
3. **PuLID weights are opened by pathname, on both families — file a
   cross-family hardening issue.** Two independent review passes raised the
   descriptor-retention rule against `SdxlPulidAdapter::load`. It was left as
   it is and the reasoning recorded on the function, because the change is
   genuinely cross-cutting rather than a line in this PR:

   - **Four sibling loaders do the same thing** — `sdxl::pulid` and
     `flux::pulid`'s adapters, and the `id_adapter.*` / `pulid_encoder.*`
     IDFormer loads in `identity::extraction`. Hardening one gives a single
     contract two answers.
   - **The rule's target is a different case.** It protects a loader that has
     just authenticated bytes and would throw that away by reopening a name —
     the EVA02-CLIP and BiSeNet conversions, whose
     `pickle_convert::AuthenticatedArtifact` hashes a private copy and
     publishes through `renameat`. A manifest adapter has no fresher
     authentication to discard: its pin was verified at download time, and
     admission freezes the exact path the planned factory proves local.
   - **The mechanism is not free.** `VarBuilder::from_mmaped_safetensors`
     takes paths. Authenticating means either a ~1 GB private read — which
     defeats the mmap the whole residency accounting is built on — or a candle
     API that accepts a retained descriptor. The second is the right answer and
     belongs with the `StableDiffusionConfig::unet()` accessor in follow-up 1.

   The residual risk is a group-writable model root where another member
   replaces a pinned adapter between download and load. Real, but identical
   for FLUX today, so it is an existing invariant gap this PR inherits rather
   than one it opens.
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

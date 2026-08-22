# #1226 documentation staging

Proposed deltas to the files this issue was told not to edit directly
(`CLAUDE.md` / `AGENTS.md`, `README.md`, `crates/mold-cli/src/skill/SKILL.md`,
`website/` navigation). The orchestrator consolidates these after merge.

Everything already landed in this branch: `website/guide/identity.md`,
`docs/architecture/pulid.md`, `crates/mold-inference/testdata/pulid/README.md`,
and `changelog.d/pulid-multi-image-cfg.md`.

---

## 1. `CLAUDE.md` / `AGENTS.md`

### 1a. Amend the identity-photo invariant paragraph

The paragraph beginning **"Identity-photo invariant:"** describes a single
photograph. Two clauses need adding. Suggested insertion after the sentence
ending "…never fitted, cropped, or resized to the canvas."

> A request may carry SEVERAL references of one person: `id_images` is the
> plural shape of `id_image` and supplying both is a hard validation error, never
> a precedence rule, because with averaging a silently dropped photograph changes
> the face that renders with nothing visible to the caller. The set is bounded by
> `ID_IMAGES_MAX` (4 — a latency budget, since extractions serialize), plus
> whole-set encoded-byte and decoded-pixel caps that sit deliberately BELOW the
> per-image limits times the count. Averaging is `cubiq/PuLID_ComfyUI`'s, and it
> is POST-IDFormer: each photograph runs the whole detector → ArcFace →
> EVA02-CLIP → IDFormer pipeline and the final `[1, 32, 2048]` token sets are
> meaned (`pulid.py:406,415-419`). Averaging the raw ArcFace vectors or the EVA
> hidden states first is a different, untrained composition and is never done —
> `ToTheBeginning/PuLID` handles only one image
> (`pulid/pipeline_flux.py:120-194`), so ComfyUI is the reference for the multi
> case. The mean is order-independent but the provenance is not: the frozen
> fingerprint hashes the source digests in request order. A photograph with no
> detectable face refuses the whole request and names its one-based position.
> Saved metadata carries the plural names/digests ONLY for the plural form, so a
> single-photograph print's metadata stays byte-identical to a pre-`id_images`
> build's.

### 1b. Add a true-CFG invariant

Suggested as a new paragraph immediately after the identity-photo invariant.

> **True-CFG invariant:** FLUX is guidance-distilled and runs one forward per
> step; PuLID's true CFG (`PuLID/flux/sampling.py:136-149`) restores a real
> negative branch, and `mold_core::identity` is the only authority for it.
> Additive `true_cfg` / `cfg_start_step` reuse the existing `negative_prompt`;
> `request_uses_true_cfg` is the single predicate every layer reads (admission,
> the memory estimate, the extractor, the engine), and it is false for an absent
> scale, for one within `TRUE_CFG_EPSILON` of `TRUE_CFG_OFF` — upstream's own
> `abs(true_cfg - 1.0) > 1e-2` comparison at `sampling.py:120` — and for a zero
> `id_weight`. True CFG is qualified ONLY alongside active identity conditioning,
> because the negative branch needs the unconditional identity embedding the
> extractor produces beside the real one; that gate is what keeps `id_weight: 0`
> byte-indistinguishable from a plain request. `1.0` is refused as a scale rather
> than run as an identity lerp: `neg + 1.0 * (pos - neg)` is `pos` arithmetically
> but not bit-identically. The unconditional identity is `IDFormer(zeros, zeros)`
> (`pulid/pipeline_flux.py:188-192`) — not a zero tensor, and a pure function of
> the adapter weights, so it is computed once at extraction and only when the
> branch actually runs. Both halves ride on one `FrozenIdentityEmbedding` and are
> installed and cleared by ONE call, because a leftover unconditional embedding
> would condition this render's negative branch on the previous person's absence.
> `FluxTransformer::denoise` takes `Option<&TrueCfgBranch>` and `None` leaves the
> loop byte-for-byte what it always was, pinned per transformer variant in
> `flux/pulid_variants.rs` on the same synthetic route the zero-weight tests use.
> A request that asked for the branch and was frozen without an unconditional
> identity is an explicit error, never a silent distilled render — `guidance` is
> chosen for whichever regime is running. Admission charges
> `TRUE_CFG_VRAM_OVERHEAD_BYTES` on top of the identity overhead so a branched
> render is never admitted on the plain estimate. Web, desktop, iPhone, TUI, and
> Discord are a separate issue.

### 1c. Amend the PuLID face-extraction bullet

The bullet beginning **"PuLID face extraction is a CPU `candle-onnx` path…"**
ends by listing deliberate deviations. Append:

> `extract_identity_embeddings` is the multi-photograph entry point and
> `extract_identity_embedding` is its one-element case; the composer builds the
> 609 MB tower ONCE and runs it per photograph before dropping it, then builds
> the IDFormer once, so N photographs cost N times the latency and ONE host peak
> — `EXTRACTION_RETAINED_BYTES_PER_IMAGE` is the only term that scales with N.
> The whole set is extracted inside a SINGLE `ExtractionSlot` permit.

---

## 2. `README.md`

The identity section's flag list gains two rows and one note. Suggested wording,
matching the existing table style:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--id-image` | — | Reference photograph. **Repeatable**, up to 4 — several references of one person are averaged into one identity. |
| `--true-cfg` | `1.0` | True classifier-free guidance scale, `1.0`–`10.0`. `1.0` is off. Requires `--id-image`; drop `--guidance` to `1.0` with it. |
| `--cfg-start-step` | `1` | First denoise step the true-CFG negative branch runs at. Requires `--true-cfg`. |

And in the identity feature blurb, one sentence:

> Give it several photographs of the same person (`--id-image` repeats) and mold
> averages them into one identity; `--true-cfg` restores a real negative branch
> so `--negative-prompt` works on an identity render.

---

## 3. `crates/mold-cli/src/skill/SKILL.md`

The identity quick-start needs the same three facts. Suggested additions to the
existing identity block:

```bash
# Several references of the same person, averaged (up to 4)
mold run flux-dev:q4 "a chef in a kitchen" \
  --id-image front.jpg --id-image side.jpg --id-image smiling.jpg

# A real negative branch (upstream advises --guidance 1.0 with it)
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg \
  --true-cfg 2.0 --guidance 1.0 --negative-prompt "blurry, cartoon"
```

Flag table rows:

| Flag | Default | Notes |
| --- | --- | --- |
| `--id-image PATH` | — | Repeatable up to 4. Set budgets: 32 MiB and 64 MP total. |
| `--true-cfg N` | `1.0` | 1.0–10.0, `1.0` = off. Requires `--id-image` and a non-zero `--id-weight`. ~2x denoise time. |
| `--cfg-start-step N` | `1` | Requires `--true-cfg`. Must be `< --steps`. |

Gotcha worth stating explicitly for an agent:

> `--negative-prompt` does nothing on FLUX unless `--true-cfg` is above 1.0 —
> FLUX.1-dev is guidance-distilled and has no negative branch without it.

---

## 4. `website/` navigation

No navigation change. Everything lands inside the existing
`website/guide/identity.md` page, which is already in the sidebar.

---

## 5. Surfaces deliberately out of scope, and what each would need

Named here rather than left implicit, since #1226 shipped server + core + CLI
only.

**Web and desktop** (`studio/lib/identityConditioning.ts`,
`studio/components/IdentityPhotoWell.vue`) — the shared policy answers one
photograph. It would need: a well that holds an ordered list with add/remove and
a visible 4-item cap, the whole-set byte/pixel budgets applied inline beside the
control (never a toast, per the existing invariant), the `id_images` /
`id_image_names` wire shape chosen only when the list has more than one entry,
and Reuse settings re-attaching each photograph from the content-addressed stash
by its own digest with per-photograph inline disclosure when one is missing. The
true-CFG controls belong in Advanced and must count toward its badge, stay
absent until touched, and hide whole when the selected checkpoint does not
advertise `supports_identity`. Both need the server capability gate to stay the
authority — an older host that does not know `id_images` must not be sent one,
so a capability signal (or a version probe) is a prerequisite rather than an
afterthought.

**iPhone** — same as above through `desktop/src/mobile`, plus the native photo
picker returning several assets and the 44pt/16px interaction rules.

**TUI** (`crates/mold-tui/src/ui/create_form.rs`) — the Identity rows are
single-valued. A photograph list needs a popup that validates through
`mold_core::identity::validate_id_images` before it can close, exactly as the
File-under rows validate through `mold_core::organization`.

**Discord** — `/identity` is at the 25-option cap, so multiple photographs need
either attachment-list handling or a second command.

---

## 6. Decisions recorded for review

- **`ID_IMAGES_MAX = 4`** — a latency budget, not a memory one. Extractions
  serialize, and `docs/architecture/pulid-face-extraction.md` measured p95
  1574.5 ms on the slowest qualified box, so four is ~6.3 s of admission latency.
- **Both shapes together is an error**, not a precedence rule.
- **`ID_IMAGES_TOTAL_ENCODED_BYTES_MAX = 32 MiB`** and
  **`ID_IMAGES_TOTAL_DECODED_PIXELS_MAX = 64 MP`**, both deliberately below the
  per-image limit times the count so they can actually bite.
- **A faceless photograph refuses the whole request**, naming its position;
  ComfyUI skips it.
- **Averaging is post-IDFormer.** The issue brief asserted ComfyUI averages the
  raw ArcFace + EVA features before the IDFormer. It does not — `pulid.py:406`
  appends `get_image_embeds(...)`, which IS the IDFormer, and `pulid.py:415-419`
  means over those outputs. Upstream was followed over the brief.
- **True CFG requires an active identity**, so `--true-cfg` on a plain FLUX
  render is refused rather than ignored. This preserves the zero-weight
  invariant; a general FLUX true-CFG control would be a separate decision.
- **`true_cfg` range is `[1.0, 10.0]`**, matching upstream's own slider
  (`app_flux.py:221`); `cfg_start_step` defaults to `1` (`app_flux.py:68`).
- **Single-photograph metadata is unchanged.** Plural provenance fields are
  written only for the plural form.

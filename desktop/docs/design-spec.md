# mold Desktop — Design Specification

## Design language: **SAFELIGHT**

---

## 1. Design concept & thesis

**Safelight** treats mold as a digital darkroom. The subject's world is literal: an image begins as pure noise and is _developed_ — step by step, under constrained chemistry (VRAM), from a fixed formula (seed, model, steps, guidance) — into a print. Everything a darkroom has, mold has: the amber safelight you work under (active process), the cold silver-halide latent that hasn't been developed yet (queued work), the paper rebate around every print (media chrome), the edge codes Kodak printed along film margins (seed/model/steps metadata), the developer bath that only runs one print at a time (the single-model queue), and the chemistry supply you watch (VRAM/GPU telemetry). Safelight renders the app as a warm, dark, physical workroom where images are _objects_ — prints with square corners on paper mats — not floating glass cards. It is the opposite of the existing web SPA (cool slate, indigo brand, radial-gradient glassmorphism, 1.25rem blobby cards): warm instead of cool, matte instead of glass, square instead of round, two functional temperatures instead of one brand hue.

**One-line thesis:** _cold things are latent, warm things are developing, and every image earns its edge code._

---

## 2. Token system

### 2.1 Color — 6 named values (dark-first)

| Token         | Hex       | Role                                                                                                                                        |
| ------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bath**      | `#141110` | App background. Warm black with a brown undertone — a lit darkroom, not a void. Never pure `#000`.                                          |
| **Bench**     | `#211B16` | Raised surfaces: panels, inspector, cards, popovers. One step warmer/lighter than Bath. Borders are `Rebate @ 8%`.                          |
| **Rebate**    | `#EFE7DA` | Primary text and the 1px paper border ("rebate") around every media object. Secondary text = Rebate @ 62%; tertiary @ 40%.                  |
| **Halide**    | `#93A7B0` | The cold state: queued jobs, undeveloped/latent placeholders, idle telemetry, disabled-but-explained controls, links to reference material. |
| **Safelight** | `#F08A24` | The warm state: anything actively developing — progress, running jobs, primary buttons, focus rings, selection. The room's lamp.            |
| **Stop**      | `#C94F3D` | Stop bath. Errors, destructive actions, OOM, cancel. Desaturated brick red, clearly distinct from Safelight amber.                          |

**Semantic temperature rule (load-bearing):** state color encodes development stage — Halide (cold) = queued/latent → Safelight (warm) = running/active → Rebate (paper) = done/developed → Stop = failed/killed. This rule is applied everywhere: queue chips, chain stage cards, download rows, telemetry.

**Light variant policy ("Lights on"):** chrome inverts to warm paper — Bath→`#F2EBDE`, Bench→`#FBF6EC`, Rebate→`#231D18`; Safelight darkens to `#C96A0A` for AA contrast; Halide darkens to `#5C737E`. **Media surfaces never invert**: the generate canvas, gallery tiles, lightbox, and chain filmstrip always sit on Bath — like a print-viewing booth in a lit office. Follows system appearance by default; overridable in Settings.

**Theme families:** **Mold** is the fresh-install default and translates the website/logo identity into product semantics without using gradients as UI decoration: cyan (`#67E8F9` dark / `#0E7490` light) is Halide for queued, latent, and reference states; magenta (`#E879F9` dark / `#A21CAF` light) is Safelight for active, selected, and focus states. Its dark surfaces are violet-black (`#12101D` / `#201B30`) and its light surfaces are neutral lavender-white (`#F7F5FA` / `#FFFFFF`). Safelight remains available as the original darkroom palette, and existing settings written before theme families migrate to Safelight instead of changing unexpectedly. Users choose a family and System/Dark/Light appearance independently; new users start on System. Generated media continues to use the fixed print surface in both families.

**Why this isn't an AI default:** it is dark but not near-black-plus-one-acid-accent — the base is warm brown-black, there are _two_ functional hues with opposing semantic temperatures plus paper-white as a third material, and nothing is neon. No cream+serif+terracotta (dark-first, grotesque display). No broadsheet hairlines (borders are soft 8% paper, radii exist, density is workbench not newspaper). And it is explicitly not the repo's own SPA (no indigo, no Inter, no glass, no gradient wash).

### 2.2 Typography — 3 roles, all OFL variable fonts, bundled in-app

| Role                    | Face                                        | Usage                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Display**             | **Bricolage Grotesque VF** (wght/opsz/wdth) | Screen titles, empty-state headlines, first-run. Used sparingly at 22–34px, wght 600–800, slightly narrow wdth 90. Its quirky ink-trap grotesque voice is the app's personality; never used below 18px.                           |
| **Body**                | **Schibsted Grotesk VF** (wght)             | Everything interactive: labels, inputs, buttons, lists. 13px base (macOS native default), 15px emphasized, 11px captions. wght 400/500/600.                                                                                       |
| **Utility / Edge code** | **Martian Mono VF** (wght/wdth)             | Seeds, dimensions, steps, VRAM bytes, model tags, timestamps, TOML. 10.5px for edge codes (wdth 87.5, tracking +2%), 12px for data tables. Always tabular. Its wide industrial letterforms read as machine-printed film markings. |

**Type scale:** 34 / 22 / 18 (Display) · 15 / 13 / 11 (Body) · 12 / 10.5 (Utility). Line heights 1.2 display, 1.45 body. No other sizes.

**The edge code** is a fixed typographic pattern (Martian Mono 10.5, Rebate @ 50%) printed along the bottom margin of every media object, exactly like film edge printing:

```
FLUX-DEV·Q8   S 4203968117   28/28   1024×1024   4.2s   M2 ULTRA
```

### 2.3 Spacing, radius, elevation

- **Spacing:** 4px base grid. Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48. Panel padding 16; card padding 12; control height 28 (compact, macOS-native), 36 for the primary Generate button.
- **Radius:** chrome 6px (panels, buttons, inputs), small controls 4px, **media 2px** — prints have near-square corners; this is a visible stance against the SPA's 20px blobs. Pills only for tag chips (full radius).
- **Elevation:** flat-first. Levels: 0 = Bath; 1 = Bench + 1px `Rebate/8%` border; 2 (popovers, palette) = Bench + border + `0 12px 32px rgba(0,0,0,.5)`. **No backdrop blur in content** — vibrancy is earned only by the native sidebar material (NSVisualEffectView sidebar) and the cmd-K palette.

### 2.4 The SIGNATURE element: **the Develop**

Every generation in the app is visualized as a _print developing_: a deterministic grain field, seeded from the job's actual seed, that resolves in lockstep with real `DenoiseStep` SSE events.

Mechanics (honest to the API — there is no server-side preview-frame stream, so this is client-side and truthful about what it knows):

1. **Latent (Queued):** the output frame renders a static, coarse, _cold_ monochrome grain (Halide-tinted), generated by a tiny WebGL/canvas hash-noise shader seeded with `seed` — every job's grain is unique and reproducible, like a negative's grain structure.
2. **Developing (Running):** on each `DenoiseStep {step, total}`, grain amplitude and correlation length ease down proportionally to `step/total` — the field visibly organizes from white-noise to smooth low-frequency blotches, and its tint warms from Halide toward Safelight. The step counter renders inside the frame's edge-code margin as `12/28`, ticking in Martian Mono.
3. **Fixing (Complete):** the final image crossfades in _under_ the last frame of grain over 450ms with a brief warm cast that neutralizes over 300ms more — a print settling in the fixer. The edge code stamps on (single 120ms fade), and the paper rebate border draws in.
4. **Everywhere, at every scale:** the same shader runs in the hero canvas, the queue-strip chips (48px), chain stage tiles (96px), the Jobs sidebar badge (16px), and a 1-bit variant in the dock icon badge. This scale-invariance is what makes it a signature instead of a gimmick.

Failed jobs "stop-bathe": grain freezes and desaturates to Stop-tinted, never disappears — a failed print is still an object you can inspect and retry.

**Restraint budget:** the Develop is the only animated/atmospheric element in the app. Everything else — buttons, lists, panels — is quiet, matte, and disciplined.

---

## 3. App shell & navigation

Native single window, minimum 1080×700 and default 1360×860. macOS uses `titleBarStyle: Overlay`, `hiddenTitle: true`, and traffic lights inset at (14, 19); Linux keeps native window-manager decorations and removes the traffic-light inset from the app toolbar.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ●●●  MOLD                                    ⌘K Search or run a command…   │  ← overlay titlebar (drag region)
├───────────────┬────────────────────────────────────────────────────────────┤
│  ▸ Generate ⌘1│                                                            │
│    Gallery  ⌘2│                                                            │
│    Chains   ⌘3│                    MAIN CONTENT                            │
│    Models   ⌘4│                    (per-screen, §4)                        │
│    History  ⌘5│                                                            │
│               │                                                            │
│  ── JOBS ──── │                                                            │
│  ▒▒ flux-dev  │  ← live Develop chips for running/queued jobs              │
│  ░░ ltx chain │     (click → focuses that job's screen)                    │
│               │                                                            │
│  ⚙ Settings ⌘,│                                                            │
├───────────────┴────────────────────────────────────────────────────────────┤
│ ● M2 Ultra  VRAM ▓▓▓▓▓▓░░░░ 38.2/64 GB   RAM 21 GB   QUEUE 2   ⌁ local     │  ← the Bench rail
└────────────────────────────────────────────────────────────────────────────┘
```

- **Sidebar** (208px default, drag-resizable 160–320px via its right-edge divider — width persists, double-click resets — native sidebar vibrancy, collapsible ⌘\): five destinations + Settings, plus a live **Hosts** section and a live **Jobs** section. It is navigation _and_ ambient status.
  - **Hosts:** every connected host (primary + extras) as a row — status dot (Safelight ready / Halide connecting / Stop error), label, live queue depth — plus mold servers detected on the network (mDNS) with a one-click `+` connect. Right-click an extra for Reconnect/Disconnect. Multi-host is client-side: jobs stream from and cancel against the host they queued on, and the Generate inspector's **Host** selector (visible with >1 live host) offers Auto (least busy, by live queue depth), **Most capable** (backend CUDA > Metal, then total VRAM, then queue), or an explicit sticky pick — the automatic modes prefer hosts that already have the selected model (per-host `/api/models`, the `hostModels` store), and the model picker shows the union of every host's installed models with a quiet availability tag for models absent from the primary.
  - **Jobs:** miniature Develop chips for every in-flight job (from `/api/queue` + SSE), labeled with their host when routed off the primary. The full queue console is the **Jobs view (⌘6)**: per-host sections showing the whole server queue (other clients' rows marked `OTHER CLIENT`), Pause/Resume + two-step Cancel all (feature-detected via `/api/capabilities.queue`), per-job cancel, and this session's finished prints with one-click reuse.
- **The Bench rail** (28px, bottom, Bath): always-on telemetry from `/api/resources/stream` (1 Hz) — GPU name, VRAM meter (Halide fill, warms to Safelight while a job runs, Stop at >92%), RAM, queue depth, and the engine mode chip (`⌁ local` embedded engine / `⇄ studio.local:7680` remote). While a routed job is live on another host, the rail follows that host — chip label (Halide-tinted), VRAM via its resources stream (falling back to the status poll's `gpu_info` on older servers), loaded models, and queue — and reverts to the primary once the last routed job settles; the embedded-engine recovery poll stays bound to the primary throughout. Clicking the VRAM meter opens a resources popover with per-GPU detail and loaded-model residency (Gpu / Parked / Unloaded).
- **Command palette (⌘K):** navigation, actions ("Pull flux-dev:q8", "Cancel all queued", "Switch to remote engine"), model search, and prompt-history search in one field. Elevation-2, the one blurred surface besides the sidebar.
- **Movement model:** sidebar click or ⌘1–⌘5; deep links everywhere (gallery item → "Reuse settings" → Generate prefilled; queue chip → owning screen; chain stage → Chains editor). Back/forward via ⌘[ / ⌘].

---

## 4. Screen-by-screen specification

### 4.1 Generate — the enlarger

**Composer visibility is invariant.** The preview fits both the width and the
height of the canvas region for every output aspect ratio. Portrait or extreme
custom dimensions may shrink the print, but may never push the prompt composer
or Generate action outside the window.

Three regions: **canvas** (center), **composer** (bottom of canvas), **inspector** (right, 320px default, drag-resizable 280–480px via its left-edge divider — width persists, double-click resets).

```
├──────────┬──────────────────────────────────────────────┬──────────────────┤
│ sidebar  │  ┌────────────────────────────────────────┐  │ MODEL            │
│          │  │                                        │  │ ┌──────────────┐ │
│          │  │        ▓▓▒▒░░  THE DEVELOP  ░░▒▒▓▓     │  │ │ flux-dev  q8 │ │
│          │  │        (grain resolving w/ steps)      │  │ │ 11.8 GB ▾    │ │
│          │  │                                        │  │ └──────────────┘ │
│          │  │  FLUX-DEV·Q8  S 4203…  12/28  1024²    │  │ SIZE  1024×1024 ⇄│
│          │  └────────────────────────────────────────┘  │ STEPS ────●── 28 │
│          │  ◉ ◎ ◎ ◎        1 of 4 (batch)              │ GUIDANCE ──●  3.5│
│          │ ┌──────────────────────────────────────────┐ │ SEED 4203968117⟳ │
│          │ │ a lighthouse at dusk, kodak portra…      │ │ BATCH  ◂ 1 ▸     │
│          │ │                                          │ │ FORMAT  png ▾    │
│          │ │ [⤢ Expand] [🎞 img2img ▾] [neg ▾]        │ │ ── LORAS ─────── │
│          │ │                    [ Generate  ⌘↩ ] ▓36px│ │ + Add LoRA       │
│          │ └──────────────────────────────────────────┘ │ detail-tweaker   │
│          │ ── QUEUE ────────────────────────────────────│   scale ──●─ 0.8 │
│          │ ▓▓12/28  ░░queued  ░░queued   (48px chips)   │ ── ADVANCED ▾ ── │
├──────────┴──────────────────────────────────────────────┴──────────────────┤
```

- **Prompt composer:** multiline, Schibsted 15px, grows to 6 lines. `⌘↩` generates from anywhere. **Expand** (`POST /api/expand`) runs inline: expanded text types into the field with a Halide underline; `⌘Z` restores the original (`original_prompt` preserved on the request). Negative prompt is a collapsible second field shown _only_ for CFG families (sd15/sdxl/sd3/wuerstchen) — capability-gated, not merely disabled.
- **Model picker:** popover with installed models grouped by family; each row shows a **source mark** (HF tile / Civitai diamond / local disk — neutral currentColor monograms, `SourceGlyph.vue`), the complete model name (wrapping rather than ellipsizing), multi-host availability on a secondary line when present, and a residency dot (Safelight = on GPU; parked/cold states and quant-tag chips remain aspirational). The selected-model control also wraps long names and grows vertically. Footer: **"Browse all models →"** deep-links to `/models`, where one searchable library places Installed above the live Catalog and filters the surface by All / Installed / Available.
- **Per-family parameter panels:** the inspector is generated from the capability matrix. Scheduler row exists only for sd15/sdxl; CFG++ toggle only for sd3/sdxl/sd15+DDIM (with hint "Lower guidance to 1.5–2.5"); frames/fps/audio/pipeline/keyframes appear only for ltx-video/ltx2; edit-images multi-drop only for qwen-image-edit (batch locked to 1 with an inline reason). Unsupported controls are absent, not grayed — the panel _is_ the family's contract. Width/height snap to /16, the ⇄ button swaps them, and the size block pairs model-native quick presets with a live proportion diagram labeled by aspect ratio and Square/Portrait/Landscape orientation. The Qwen lists mirror the core's recommended buckets, including native 1328×1328. A megapixel readout warns (`x-mold-dimension-warning` surfaced post-hoc too).
- **Empty canvas:** before the first job, the media surface shows a quiet print glyph, **No print yet**, and one sentence pointing the user to the model, prompt, and Generate controls. It must read as an intentional placeholder in every theme, not as a failed black preview.
- **LoRA stack:** vertical list of chips with per-adapter scale sliders, drag-to-reorder, ✕ to remove. "+ Add LoRA" opens a picker fed by `GET /api/loras?model=` (family-filtered); Civitai entries show `trained_words` as tappable chips that insert trigger phrases into the prompt.
- **img2img / mask / control:** a drop well under the composer's `img2img ▾` split-button. Dragging any image over the window dims the app and shows three labeled drop zones — **Source** / **Mask** / **Control** (Control zone only for sd15) — with paper-rebate outlines. Dropped source sets a strength slider (0.75 default) and previews as a small print clipped to the corner of the canvas. Qwen-edit mode swaps the well for an ordered multi-image tray ("First image is the edit target").
- **VRAM preflight:** on parameter change (debounced), `POST /api/generate/estimate` renders a one-line forecast above the Generate button: `Fits · est. 21.4 GB of 64 GB` (Halide) or `Tight — enable offload?` (Safelight) or `Won't fit on this GPU` (Stop, button still enabled — the queue may free memory).
- **Generation lifecycle:** click Generate → job chip appears in the queue strip (cold grain, `Queued #2`) → `/api/generate/stream` events drive the Develop in the canvas → complete → print fixes in, edge code stamps, result auto-saved to gallery, batch dots (◉◎◎◎) fill in as `base_seed+i` siblings finish. Toast: **"Generated — saved to Gallery."** Zombie reconciliation: `/api/queue` polled to dead-letter chips whose SSE dropped, shown as Stop-frozen grain with a Retry action.
- **Queue strip:** horizontal filmstrip of 48px Develop chips below the composer. Right-click a queued chip: _Cancel_, _Move to GPU 1_ (`PATCH /api/queue/:id`), _Duplicate_. The strip mirrors into the sidebar Jobs section.

### 4.2 Gallery — the print drawer

```
│ GALLERY      1,284 prints · 18.2 GB       [All ▾] [Images|Video] [⊞ size]  │
│ ┌────┐┌────────┐┌────┐┌──────┐┌────┐┌────────┐                            │
│ │    ││        ││    ││  ▶   ││    ││        │   justified rows,          │
│ └────┘└────────┘└────┘└──────┘└────┘└────────┘   virtualized,             │
│ ┌──────┐┌────┐┌────────┐┌────┐┌────┐              2px radius, 1px rebate  │
│ │      ││    ││        ││    ││    │              hover: edge code rises  │
│ └──────┘└────┘└────────┘└────┘└────┘              from bottom margin      │
```

- **Grid:** justified rows (aspect-correct, like contact sheets), virtualized against `GET /api/gallery`, thumbnails from the thumbnail endpoint. Videos wear a ▶ glyph and scrub their animated GIF preview sidecar on hover (`/api/gallery/preview/:filename`); reduced-motion shows the first-frame thumbnail only.
- **Selection:** click selects (Safelight 2px ring outside the rebate), ⌘-click multi, shift-click range, ⌘A all. Selection summons a bottom action bar: _Export…_, _Upscale_, _Reuse settings_, _Delete_ (n).
- **Quick Look:** **Space** opens/closes the lightbox instantly (macOS reflex). ←/→ navigate while open.
- **Lightbox / detail:**

```
│ ┌──────────────────────────────────────────────┬──────────────────────────┐
│ │                                              │ shot_042.png             │
│ │              PRINT ON BATH                   │ PROMPT  a lighthouse…    │
│ │        (video: native <video> w/ Range       │ MODEL flux-dev:q8        │
│ │         scrubbing, loop, ⎵ play/pause)       │ SEED 4203968117  ⧉       │
│ │                                              │ 1024×1024 · 28 st · g3.5 │
│ │  FLUX-DEV·Q8  S 4203…  28/28  1024²  4.2s    │ LORA detail-tw. 0.8      │
│ └──────────────────────────────────────────────┤ Jul 8 2026 · 2.1 MB      │
│      ← 42 / 1,284 →                            │ [ Reuse settings ]       │
│                                                │ [ Upscale ] [ Reveal ]   │
│                                                │ [ Delete ]               │
```

- Metadata panel reads the embedded `mold:parameters` (a `synthetic` badge when `metadata_synthetic`). **Reuse settings** is the headline action: jumps to Generate with every parameter restored (prompt, model, seed, LoRAs, size…) — seed restored _locked_, with a ⟳ to re-randomize. Any value is click-to-copy (⧉).
- **Video:** `<video>` against `/api/gallery/image/:filename` (Range/206 gives free scrubbing); fps/frames/audio badge in the edge code.
- **Delete flow:** ⌫ or Delete button → inline confirm on the button itself ("Delete print?" → **Delete** / Esc), never a modal for one item; multi-delete uses a sheet listing count. Toast: **"Deleted 3 prints"** (no undo offered — the API is destructive; the confirm carries the weight).
- **Drag-out:** any tile or the lightbox image drags to Finder/other apps as the real file (Tauri drag-out with the on-disk path; remote engines download to a temp file first, cursor shows a progress badge).
- **Clipboard:** right-clicking a still image offers **Copy image** and writes the full-resolution bitmap to the macOS clipboard. The same action is available on the completed Generate canvas; videos keep metadata-copy actions but disable bitmap copy.

### 4.3 Models — the chemistry shelf

Two tabs: **Installed** and **Catalog**, plus a persistent **Downloads** tray.

```
│ MODELS      [Installed 14 · 96 GB]  [Catalog]                 [Search ⌘F]  │
│ ── FLUX ─────────────────────────────────────────────────────────────────  │
│ ◉ flux-dev        q8    11.8 GB  ▓▓▓▓▓▓▓░░  ● on GPU     [Load][Info][✕]  │
│   flux-schnell    q4     6.4 GB  ▓▓▓░░░░░░               [Load][Info][✕]  │
│ ── LTX-2 ────────────────────────────────────────────────────────────────  │
│   ltx2-19b-dist   fp8   28.1 GB  ⚠ Requires CUDA — runs on Linux           │
│ ── SHARED COMPONENTS ────────────────────  t5-xxl 9.2 GB · clip-l 1.7 GB   │
```

- **Installed:** grouped by family; rows show quant chip, disk usage bar (proportional, Halide), residency dot, per-row actions. **Info** expands components (`/api/models/:model/components`) with per-component download/verify state and a "Verify checksums" action. Removing warns about shared components: "Keeps t5-xxl (used by 3 models)."
- **Catalog:** the Models view combines installed inventory and live catalog results under one search field, with All / Installed / Available filters and a Grid / Table toggle. Grid cards use a bounded preview height and responsive 260px minimum columns; Table rows favor names, source, family, popularity, size, and the Pull action. Source, Family, and NSFW filters refine paged `/api/catalog/search` results. Pulling with several ready hosts opens a labelled target dialog; a single-host setup starts immediately. Result cards keep **SIZE vs FETCH** honest.
- **Downloads tray:** slides up from the Bench rail while `/api/downloads/stream` has activity: up to two active downloads plus the remaining queue, per-download progress (bytes + %), and cancellation (✕ → `DELETE /api/downloads/:id`) that stays visibly pending until the engine confirms termination. Companion downloads remain grouped under their primary. Progress bars here are plain Safelight fills; the Develop is reserved for generation.
- Built-in catalog entries and `hf:`/`cv:` live entries are visually identical; the id chip (`cv:12345`) is copyable.

### 4.4 Chains — the editing bench (mold.chain.v1)

A horizontal **filmstrip timeline** where transitions are _splices_ — grounded in physical film editing, and structurally true: a chain is a sequence.

```
│ CHAINS    [+ New chain]  [Open .toml…]           JOBS: 2 running · 5 done  │
│ ┌ chain: ltx2-19b-distilled · 1280×720 · 24fps · seed 8841 ── [Edit ⚙] ─┐  │
│ │                                                                       │  │
│ │ ┌STAGE 1───┐    ┌STAGE 2───┐    ┌STAGE 3───┐                          │  │
│ │ │▓▓ done   │────│▒▒ 41/97  │ ─▷ │░░ latent │   + Add stage            │  │
│ │ │97f ·     │smooth 97f     │cut │49f ·     │                          │  │
│ │ │"dawn…"   │    │"the ship…│    │"storm…"  │                          │  │
│ │ └──────────┘    └──────────┘    └──────────┘                          │  │
│ │  ── total 243 frames · 10.1s @ 24fps · est 18.9 GB ✓ fits ──          │  │
│ │            [ Validate ]  [ Dry run ]  [ Render chain ]                │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
│ ── JOB HISTORY ── durable jobs list (resume / retake / cancel / gc) ──────  │
```

- **Stage cards** (96px tall tiles): live Develop grain per stage (cold→warm→print as `ChainJobEvent`s arrive; completed stages show the per-stage JPEG preview from `/api/chain-jobs/:id/stages/:idx/preview`). Card fields: prompt (truncated, click to edit in a popover with frames stepper, seed offset, per-stage model override, LoRAs, source-image well, negative prompt where allowed).
- **Splice marks between cards encode the transition:** `smooth` = an unbroken strip (cards visually touch, a small motion-tail tick labeled with `motion_tail_frames`); `cut` = a hard diagonal splice line; `fade` = a gradient wedge with a `fade_frames` stepper. Click a splice to cycle smooth → cut → fade.
- **Frame math is always visible:** per-stage frame count validates 8n+1 live (invalid → Stop underline + "Frames must be 8n+1 (9, 17, 25…)"); the footer totals frames/duration and runs the chain-limits + VRAM preflight (`/api/capabilities/chain-limits`, worst-case estimate) — `✓ fits` / `✗ needs 24.2 GB`.
- **Authoring parity:** _Open .toml…_ imports `mold.chain.v1`; _Edit as TOML_ flips the editor to a mono source view (two-way). _Validate_ and _Dry run_ mirror the CLI verbs and print results in a console strip.
- **Durable jobs list:** rows with state chips (Queued/Running/Interrupted/Failed/Done in temperature colors), _Resume_, _Cancel_, _Delete_, and **Retake** — retake opens the finished chain's filmstrip, you click the stage to redo, choose **Cascade** ("re-renders this stage and everything after") or **Splice** ("replaces this stage in place"), optionally new prompt/seed offset. Long-job progress: the sidebar chip shows `stage 2/3 · 41/97`, and the window's dock tile badges overall percent.

### 4.5 History — the notebook

```
│ HISTORY                              [Search prompts…        ]  [Clear…]   │
│ TODAY                                                                      │
│  "a lighthouse at dusk, kodak portra"        flux-dev:q8    14:22  [↩ Use] │
│  "the ship breaks through ice…"              ltx2-19b       13:40  [↩ Use] │
│ YESTERDAY                                                                  │
│  …                                                                         │
```

Two lenses behind a segmented toggle. **Runs** (default): every finished generation from the gallery DB as a day-grouped run log — 48px thumbnail, prompt, model, dimensions, seed, step count, time; click reuses the FULL settings (seed included); right-click offers Reuse settings / Copy prompt / Copy seed / Show in Gallery. **Prompts**: the flat, fast, keyboard-first list over `prompt_history` (recent/search) for prompts whose outputs are gone. ↩ Use fills the Generate composer (and switches model if still installed; otherwise offers the pull). ↑/↓ in the Generate composer also cycles history inline (TUI parity); ⌘K searches history too. _Clear…_ maps to `trim_to`/`clear` with a count confirm and only affects the prompt log.

### 4.6 Settings — the two stores, honestly

```
│ SETTINGS   [General] [Engine] [Generation] [Expansion] [Profiles] [Advanced]│
│ ENGINE                                                                     │
│  Mode        ◉ Built-in (this Mac)   ○ Remote server                       │
│  Remote host  http://studio.local:7680        [Test connection]            │
│  API key      ••••••••••••                    stored securely              │
│ GENERATION                                                                 │
│  Default size   1024 × 1024                                    ⌂ db        │
│  Default steps  28                                             ⌂ db        │
│  Models dir     ~/models/mold                                  ⛁ file      │
│  Embed metadata ON — overridden by MOLD_EMBED_METADATA=0       ⚿ env       │
```

- **Provenance is first-class:** every row carries a source tag — `⌂ db` (SQLite settings), `⛁ file` (config.toml), `⚿ env` (MOLD_* override). Env-overridden rows render locked with the exact variable name and value and the copy "Set by your environment — unset MOLD_EMBED_METADATA to edit here." _Reset_ on a db row drops the key and shows what it falls back to. This mirrors `mold config list/where/reset` exactly instead of pretending there's one store.
- **Profiles:** switcher at the top of the tab (`profile.active`), create/duplicate; a Halide banner notes "Settings marked ⌂ are per-profile."
- **Expansion tab:** backend (local qwen3-expand / OpenAI-compatible URL), model, temperature/top-p/max-tokens, thinking toggle, and per-family word-limit/style-notes overrides in an editable table; system/batch prompt editors in mono with placeholder chips (`{WORD_LIMIT}`, `{MODEL_NOTES}`).
- **Advanced:** device placement grid (component × auto/cpu/gpu:N — mirrors `DevicePlacement`, persisted via the placement endpoints), text-encoder variant pickers, offload/eager toggles, queue size, cache size, artifact TTL, "Open config.toml", "Open logs".
- Remote API keys live in the app's **owner-only secret file**
  (`secrets.json`, mode 0600, allowlisted names), never ordinary settings and
  never the macOS Keychain — Keychain access re-prompts on every ad-hoc dev
  rebuild and after signed updates, which users experience as nagging.

---

### 4.7 RunPod — the remote bench

The RunPod workspace keeps provisioning in the app: secure API setup, balance and active hourly spend, GPU stock, cloud/datacenter/storage choices, live pod status, console handoff, lifecycle controls, and **Use in Mold** to connect the engine to `https://<pod>-7680.proxy.runpod.net`. Network volumes can be created, selected, renamed/grown, and deleted in place; selection persists, volume-backed launches visibly lock to Secure Cloud and the volume datacenter, and destructive deletion names the permanent-data risk. Poll status every ten seconds while the screen is open without replacing button labels or flashing a loading state. Destructive delete uses an inline two-step confirmation. Keys go to the app's owner-only local secret store (no Keychain, no permission prompts). Existing CLI environment/config credentials continue to work and are identified as externally managed.

While a remote engine is selected, Gallery uses a standard two-tab location switch: **Remote** shows that engine's output and **This Mac** reads the configured local output directory through a restricted native media protocol. Switching gallery location never changes the generation engine.

## 5. States

- **First run / no models (the empty bench):** Generate shows a full-canvas invitation — Bricolage headline **"Develop your first print."**, one sentence ("mold runs models locally on your Mac's GPU. Pull one to start."), and three curated starter cards sized for Apple Silicon (flux-schnell q4 · 6.4 GB "fastest", flux-dev q8 · 11.8 GB "best quality", sdxl-turbo · 6.9 GB "classic") each with a single **Pull** button that starts the download and threads progress right there. A quiet "Browse all models" link below. Gallery/Chains/History empties are one line + one action, e.g. Gallery: "No prints yet — generate one." [Go to Generate].
- **Loading:** screens skeleton with static cold grain blocks (the latent motif), never spinners in content; the only spinner is the 12px one in the Bench rail chip during engine handshake.
- **Engine starting (embedded):** Bench rail chip `⌁ starting…`; canvas usable (composer accepts input, Generate queues). If startup fails: full-width Stop banner "The engine didn't start. [Show log] [Retry]".
- **Server down (remote mode):** banner "Can't reach studio.local:7680. [Retry] [Switch to built-in engine] [Edit host]" — direction, not mood; queued UI state preserved.
- **VRAM OOM:** job chip stop-bathes; inline error on the job: "Ran out of GPU memory (needed ~23.9 GB, had 18.2 GB). Try: enable Offload · lower resolution · use q4." with one-click **Retry with offload**.
- **CUDA-only on Mac (LTX-2):** model rows and pickers show the model normally but tagged `⚠ CUDA only` (Halide); selecting it explains "LTX-2 generates on CUDA GPUs only. Connect a remote Linux engine to use it." with [Set up remote engine]. Never silently hidden — parity demands visibility.
- **Long video / chain job:** sidebar chip with stage + frame counters; window can close to dock — job continues (engine is in-process but detached from the view); notification + dock bounce on completion; if the app was quit mid-durable-job, next launch shows an "Interrupted" row with **Resume**.
- **Queue full (503):** toast "Queue is full (600 jobs). Cancel something or wait." with [Open queue].

---

## 6. Motion & micro-interactions

**One orchestrated moment:** the generation lifecycle (the Develop, §2.4) — cold grain → warming resolve on each DenoiseStep → 450ms fixer crossfade → 300ms warm-cast neutralize → edge-code stamp (120ms) → 250ms chip flight from canvas to the sidebar Jobs list as it archives. Total choreography ≤ 1.2s beyond the actual compute; nothing else in the app animates ambiently.

Everything else is small and mechanical:

- Hover: media tiles raise their edge code from the bottom margin (80ms); buttons brighten 6%; no scale transforms on chrome.
- Press: 1px translate-down on buttons (native feel).
- Sliders: value scrubs in Martian Mono beside the thumb; width/height snap-to-16 gives a 60ms tick.
- Panel/popover: 140ms fade + 4px rise, `ease-out`.
- Toasts: slide 8px up from the Bench rail, 4s, stack max 3.
- Splice cycling in Chains: 180ms morph between splice glyphs.
- **Reduced motion (`prefers-reduced-motion`):** the Develop degrades to a _static_ grain frame whose opacity steps down at each DenoiseStep (no shimmer), crossfades become instant swaps, chip flights and rises are removed, video hover-scrub disabled. State is still fully legible from color temperature + counters — motion is reinforcement, never the only channel. Keyboard focus: 2px Safelight ring, 2px offset, on every focusable element, always.

---

## 7. macOS nativeness

**Keyboard map**

| Shortcut     | Action                                        |
| ------------ | --------------------------------------------- |
| ⌘1–⌘5 / ⌘,   | Screens / Settings                            |
| ⌘K           | Command palette                               |
| ⌘N           | New generation (focus composer, clear)        |
| ⌘↩           | Generate                                      |
| ⌘E           | Expand prompt                                 |
| ⌘D           | Duplicate last job (same seed) · ⇧⌘D new seed |
| ⌘R           | Randomize seed                                |
| Space        | Quick Look in Gallery · play/pause video      |
| ←/→, ⌘A, ⌫   | Gallery navigate / select all / delete        |
| ⌘F           | Filter/search current screen                  |
| ⌘.           | Cancel focused job                            |
| ⌘\           | Toggle sidebar                                |
| ⌘[ / ⌘]      | Back / forward                                |
| ⌘0 / ⌘+ / ⌘− | Interface size reset / larger / smaller       |
| ⇧⌘C          | Copy seed (lightbox) · ⌥⌘C copy prompt        |

**Menu bar:** **mold** (About, Check for Updates…, Settings… ⌘,, Quit) · **File** (New Generation ⌘N, New Chain, Open Chain Script…, Import Image for img2img…, Export Selection…, Reveal in Finder) · **Edit** (standard + Copy Seed, Copy Prompt) · **Generate** (Generate ⌘↩, Expand Prompt ⌘E, Randomize Seed ⌘R, Duplicate Last ⌘D, Cancel Job ⌘.) · **View** (screens ⌘1–5, Toggle Sidebar ⌘\, Appearance ▸, Actual Size/Zoom) · **Window** / **Help** (Shortcuts, API Reference → /api/docs, Open Logs).

- **Drag-in:** any image dropped anywhere targets the labeled Source/Mask/Control wells (§4.1); `.toml` files dropped open in Chains; multiple images onto qwen-edit fill the tray in drop order.
- **Drag-out:** gallery tiles/lightbox → Finder, Mail, Photoshop (real file paths).
- **Notifications:** "Generated — lighthouse at dusk" with thumbnail on job complete _when the app is backgrounded_; "Chain finished · 243 frames" ; "Pull complete — flux-dev:q8". Bundled macOS builds send through the native identity-image path so Notification Center shows Mold's app icon; Tauri's notification plugin remains the cross-platform/development fallback. Clicking focuses the relevant item.
- **Dock:** badge = THIS app's active job count, event-driven from the generation store (never the whole engine's queue depth — a shared or remote engine runs other clients' work, and badging those numbers reads as "my app is busy" when it isn't); cleared the moment the last job settles. A subtle determinate progress overlay during a running chain and a dock menu (New Generation, Pause Queue, recent prints) remain aspirational.
- Native services: window state restore, full-screen support, standard text editing (dictation/emoji work in the composer), system appearance sync, and an owner-only local secret store for API keys (deliberately not the Keychain — no permission prompts).

---

## 8. UX copy guidelines

**Rules:** active voice; the verb on the button is the noun in the toast (Generate → "Generated"); one consistent vocabulary — **print** (a generated image), **develop/developing** (generating), **pull** (download model), **chain** (multi-stage video), **stage**, **splice** (transition), **engine** (local server), **studio** never used; sentence case everywhere; errors say what happened + one concrete fix, never apologize, never "Oops"; empty states are invitations with exactly one primary action; numbers always with units in Martian Mono.

**Examples**

| Context        | Copy                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Primary button | `Generate` · running: `Developing… 12/28` · queued: `Queued #2`                                                        |
| Success toasts | `Generated — saved to Gallery` · `Pulled flux-dev:q8` · `Chain finished · 243 frames` · `Upscaled 4× — saved`          |
| Delete confirm | `Delete print? This can't be undone.` → `Delete`                                                                       |
| Server error   | `Can't reach studio.local:7680. Check the host or switch to the built-in engine.`                                      |
| OOM            | `Ran out of GPU memory (needed ~23.9 GB, had 18.2 GB). Enable Offload or lower the resolution.` → `Retry with offload` |
| Queue full     | `Queue is full. Cancel a job or wait for one to finish.`                                                               |
| CUDA gate      | `LTX-2 generates on CUDA GPUs only. Connect a remote Linux engine to use it.`                                          |
| Frames rule    | `Frames must be 8n+1 — try 97.`                                                                                        |
| Env lock       | `Set by MOLD_EMBED_METADATA in your environment. Unset it to edit here.`                                               |
| First-run      | `Develop your first print.` / `mold runs models locally on your Mac's GPU. Pull one to start.`                         |
| Empty gallery  | `No prints yet — generate one.` → `Go to Generate`                                                                     |
| Retake modes   | Cascade: `Re-renders this stage and everything after it.` · Splice: `Replaces this stage in place.`                    |

---

## Implementation notes for the build phase (summary)

- App = cargo root `desktop/src-tauri` (own Cargo.lock, added to mold's `[workspace] exclude`), embedding the engine in-process via `mold_server::run_server` and talking to it over localhost HTTP+SSE with `mold-core` wire types; remote mode reuses the same client path. Frontend: dedicated Vite app (not the web/ SPA), fonts bundled (Bricolage Grotesque VF, Schibsted Grotesk VF, Martian Mono VF — all OFL). Nix provides `desktop-dev`, `desktop-build`, `desktop-check`, `desktop-test`, `desktop-ui`, and `desktop-bun-lock`; the window uses `titleBarStyle: Overlay`.

### Critical Files for Implementation

- /Users/jamesbrink/Projects/utensils/mold/flake.nix — add desktop package + devshell commands (Aethon recipe)
- /Users/jamesbrink/Projects/utensils/mold/Cargo.toml — add `[workspace] exclude` for the Tauri cargo root
- /Users/jamesbrink/Projects/utensils/mold/crates/mold-server/src/lib.rs — `run_server` entry point the desktop backend embeds
- /Users/jamesbrink/Projects/utensils/mold/crates/mold-core/src/types.rs — wire types the frontend/backend model 1:1
- /Users/jamesbrink/Projects/utensils/aethon/src-tauri/tauri.conf.json — proven macOS window-chrome/signing template to copy

# Mold Studio — design guidelines

Distilled from `docs/design/mold-studio-spec.html` (v0.12), the design source of
truth for **mold** — a local, no-cloud AI image/video generation studio. One
design system across five surfaces (macOS, iOS, web, mobile web, terminal).

## Principles

- **Dead-simple by default; power on demand.** The primary path — describe,
  choose a look, generate — is one screen with no scrolling to essentials. If a
  screen feels empty, that is correct, not a gap to fill.
- **Progressive disclosure.** Advanced capability lives behind one
  clearly-labelled entry (Advanced, More options) and opens collapsed. Never
  surface a control a beginner can't act on.
- **Contained, not global.** Every overlay (drawer, sheet, lightbox, palette,
  modal) renders inside the frame it belongs to — never a page-level layer.
- **Local-first voice.** Terse, lowercase, technical, dry. Reinforce "runs on
  your machine" everywhere it's true. No cloud, no Python, no fuss.

## Color: role tokens only

Two theme families — **Mold** (cyan & magenta) and **Safelight** (warm darkroom
amber) — each in Dark and Light. Selected via root dataset attributes:
`data-theme-family="mold" | "safelight"` (absent = safelight) and
`data-theme="dark" | "light"` (absent = follow `prefers-color-scheme`). The four
combinations are the only supported palettes. **Never hard-code hex** — all
color goes through the custom properties defined in the shipped stylesheet:

| Token | Role |
| --- | --- |
| `--desk` | Behind the window / canvas (deepest plane) |
| `--bath` | Base surface (screens) |
| `--bench` | Elevated panel / card / bar |
| `--surface` | Nested / hover tier (3rd plane) |
| `--rebate` | Primary text / ink (family-tinted) |
| `--ink-2` / `--ink-3` | Secondary / tertiary text |
| `--edge` / `--ce` | Hairline border / control border |
| `--safelight` | **Primary accent** — action, focus, selection (warm) |
| `--halide` | **Info accent** — models, telemetry, links (cool) |
| `--success` / `--warning` / `--stop` | Semantic: done / in-progress / destructive |
| `--sel-ink` / `--sel-bg` / `--sel-border` / `--sel-fill` / `--sel-ring` | Selection set, derived from the accent — never a new hue |
| `--on-accent` | Text/icon on accent fills |
| `--grad` | Ink gradient — brand moments only (wordmark, hero) |
| `--card-hi` | 1px top inner-highlight on cards/tiles |
| `--print` / `--on-media` | Media bed — prints are always viewed on a dark bed, never inverts |

Surfaces step `desk → bath → bench → surface` with real tonal jumps. The two
accents have fixed jobs — safelight is warm/primary, halide is cool/informational
— a warm↔cool rhythm that avoids monotony without rainbow.

## Type, radius, motion, icons

- **Display** (titles, wordmark): Bricolage Grotesque 600–800 (`--f-display`)
- **Body / UI**: Schibsted Grotesk 400–700 (`--f-body`)
- **Mono** (data, model ids, labels, code): Martian Mono 300–700 (`--f-mono`)
- **Radii**: controls 6–10px (`--radius-control-sm/-control/-control-lg`),
  cards 12–16px (`--radius-card`, `--radius-card-lg`), pills/chips 20px
  (`--radius-pill`)
- **Motion**: `--dur-quick` 120ms · `--dur-base` 180ms · `--dur-slow` 240ms,
  all `--ease` `cubic-bezier(0.16,1,0.3,1)`. Shimmer 1.6s loop for
  "developing" placeholders (`.ms-shimmer`); entrance for overlays/results is
  `.ms-fade-up` (never on core chrome). No bounce, no parallax, no fade-in of
  core UI. Honor `prefers-reduced-motion`.
- **Icons**: line icons, 1.7–1.8 stroke, round caps, 24-unit grid, drawn with
  `currentColor`. Sizes 14/16/17/22. Never mix fills or weights (Lucide-
  compatible geometry). The shipped registry is `MoldStudio.ICONS`.

## Component vocabulary

New UI composes from this kit — do not invent a control when one fits:
Segmented control (2–4 exclusive options, accent-tinted active segment) ·
Chip (style presets; tapping an active chip deselects; 20px pill, accent fill
when on) · Nav item (icon + label; active = accent-tinted bg + accent icon) ·
Shape picker (aspect ratios drawn proportionally: 1:1 · 3:4 · 4:3 · 16:9 ·
9:16) · Resolution selector (megapixel-based human labels, resolved px below) ·
Slider / Stepper / Switch (accent thumb/track/knob) · Accordion section (the
disclosure primitive — collapsed by default; powers Advanced) · Card & Tile
(flat `--bench` card, hairline border; square image tiles, lift on hover, NEW
badge) · Progress ring (in-canvas generate) and bar (downloads, telemetry) ·
Drawer (right-side, wide surfaces) · Sheet (full-screen or bottom, phone) ·
Modal (centered focused task; stepped with 3-dot progress) · Command palette
(⌘K: search + grouped actions) · Empty state (dashed frame + one-line
what-to-do) · Badge (count/status pill) · Keycap (⌘↵ on primary Generate).

## Component states

Rest is the baseline; the rest are deltas:

- **Rest**: token surface + `--ce` border, `--ink-2` label.
- **Hover**: border strengthens / background lifts (rebate @ 5–7%); no movement on press.
- **Active/selected**: `--sel-bg` fill, `--sel-ink` label, `--sel-ring` ring; icon takes the accent.
- **Focus (keyboard)**: 2px accent ring, offset from the control, always visible.
- **Disabled**: 60% opacity, no hover, `not-allowed`; layout unchanged.
- **Loading**: shimmer (tiles) or ring (actions); label becomes present-tense status.
- **Error**: `--stop` border + one blunt line beneath; entered value preserved.
- **Empty**: dashed frame + one-line guidance; never a bare blank region.

## Information architecture

Four workspaces plus Settings — the entire surface area; do not re-expand it:

1. **Create** — compose & generate (home). Prompt, style, model, shape, resolution, result, activity.
2. **Library** — browse everything generated. Prints, filters, search, viewer.
3. **Models** — installed & discoverable weights. Rows, detail, pull.
4. **Machines** — local + remote GPUs. Host cards, telemetry, connect flow.
5. **Settings** — appearance & about.

Advanced (Create) is six collapsed accordion sections: Scheduler & sampling ·
Negative prompt · Source image · LoRA stack · Upscale after generate · Output &
seed — with an active-count badge and a Reset. On web at tablet width and
above, these render inline in the Create controls region (still collapsed);
phone surfaces keep the Advanced sheet.

## Content & voice

- Product name **mold** always lowercase; commands lowercase mono (`mold run`, `mold pull flux-dev:q4`).
- Section titles Title Case; table headers UPPERCASE; status values lowercase (`loaded`, `ready`, `developing`).
- Model ids as `family:variant`. Units tight: `6.9 GB`, `18.4 MiB/s`, `eta 6m 12s`.
- Feedback glyphs only: ✓ done · ★ loaded · • info. **No emoji** in product UI. Error copy is blunt and short.

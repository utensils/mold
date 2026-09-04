# Mold Studio — design system

The canonical design source for mold's graphical surfaces: the macOS / Linux /
Windows desktop app, the web studio, and the iPhone / Android companion. The
terminal TUI keeps its own mockup (`mold-tui-proposed.html`) and is out of this
package's scope. This directory is the September 2026 redesign package; the
earlier spec (`mold-studio-spec.html` v0.14) and the archived prototype it grew
from are gone — the reference implementations below are the source of truth,
and the shipped shared components are authoritative where a mock differs.

- **`mold-studio-desktop.dc.html`** — the desktop reference implementation:
  every view, overlay, and the six theme maps (`themeList()` in its script).
  Open it in a browser; fonts load from `../../ui/fonts/` through `fonts.css`.
- **`mold-studio-style-guide.dc.html`** — principles, the binding lexicon, the
  theme token map, shell anatomy, the component table, and the contrast rules.
- **`mold-studio-web.dc.html`**, **`mold-studio-iphone.dc.html`** — the web
  and phone mocks. Those surfaces still ship on the pre-redesign look through
  the legacy bridge in `ui/tokens.css`; these are what they move to.
- **`mold-tokens.css`** — the token vocabulary the mocks are written in.
  `ui/tokens.css` is the shipped copy, with one complete map per theme.
- **`mold-desktop.css`** — shell metrics, control heights, semantic surface
  and job-state aliases. Shipped verbatim as `ui/mold-desktop.css`.
- `support.js`, `doc-page.js`, `browser-window.jsx`, `ios-frame.jsx`,
  `assets/` — the mocks' runtime and imagery (gallery pictures are downsized
  copies; `assets/gallery/*.png` is what every mock image surface shows).
- `notes/` — design notes for shipped features; `prints/` — sample prints.

## 1 · Audience

The TUI is for people who already live in a terminal. The GUI is for people who
have never heard of a diffusion model. That single sentence decides most of the
design: jargon is demoted to secondary mono text, never removed, and every
progress state is written as a sentence a first-timer can act on.

Rule: **plain words in sans, technical truth in mono, on the same row.**

> Detail · `28 passes` — not "Steps: 28"
> Photoreal — best quality · `flux-dev:q4`
> Adding detail — about 12s left · `denoise 18/28 · 1.8 it/s`

## 2 · Lexicon (binding)

| Say                                                      | Never say                               | Where                |
| -------------------------------------------------------- | --------------------------------------- | -------------------- |
| New image · Queue · My images · Styles · Machines        | Create, Library, Models, Hosts (as nav) | Nav, titles, palette |
| Styles                                                   | Models, checkpoints                     | Nav, catalog         |
| Style engine / Photoreal — best quality                  | flux-dev, SDXL (as the primary label)   | Composer chip, rows  |
| Add-on looks                                             | LoRAs                                   | Inspector            |
| Detail · passes                                          | Steps, sampler steps                    | Inspector, presets   |
| Stick to my words                                        | Guidance, CFG                           | Inspector            |
| Repeat this look · Keep \| Surprise me                   | Seed, Fixed \| Random                   | Inspector, metadata  |
| Start from a photo · How much to change it               | img2img, denoise strength               | Inspector            |
| Generate                                                 | Submit, Add to queue, Render            | Primary action       |
| Write more for me                                        | Expand prompt                           | Composer chip (⌘E)   |
| Being made / Waiting / Finished / Needs a download first | active, queued, done, blocked           | Queue                |
| Machines · this mac · making images here                 | Hosts, target host                      | Nav, Machines        |
| Connect a machine · Rent a GPU · billing begins now      | Add host, Provision pod                 | Machines             |
| My images · albums · Favourites · Everything             | Library, collections, gallery, Prints   | Nav, My images       |
| Ready to use \| Browse more · Get it · ● ready           | Installed \| Discover, Pull, installed  | Styles               |
| Short clip · scenes · Length · Smoothness                | Sequence, clips, frames, fps (primary)  | Clip mode            |

Voice markers carried over from the CLI: terse, second person, directive.
Units stay tight and mono (`14.9 / 24 GB`, `eta 8m12s`, `$1.44/hr`).
Anything that costs money is stated in money, in `--mold-state-cost`.
`desktop/src/lib/lexicon.test.ts` pins the destination words on the router, the sidebar, the palette, and the native menu, and the inspector's seed labels.

## 3 · Shell anatomy

```
┌──────────────────────────────────────────────────────────────┐
│ unified toolbar            44px   ● ● ●  ⌫ ‹ ›  Title  ⌘K  🔔│
├───────────┬──────────────────────────────────┬───────────────┤
│ sidebar   │ view toolbar              40px   │ inspector     │
│ 270px     ├──────────────────────────────────┤ 300px         │
│           │                                  │ tabbed        │
│ nav       │ canvas (--mold-canvas)           │ Settings /    │
│ ·······   │ image gets the height            │ Starting pts /│
│ machine   │                                  │ Recent        │
│ card      │                                  │               │
│ ·······   ├──────────────────────────────────┤               │
│ QUEUE     │ composer (prompt + chips + CTA)  │               │
│ flex:1    │                                  │               │
├───────────┴──────────────────────────────────┴───────────────┤
│ status bar                 26px   host · vram · queue · keys │
└──────────────────────────────────────────────────────────────┘
```

Decisions worth keeping:

- **The queue lives in the sidebar, under the machines** — not docked under the
  canvas. Work-in-progress is context, not content; the image keeps the viewport.
- **Fixed-height chrome never shrinks** (`flex-shrink:0`); the canvas absorbs
  slack and clips (`overflow:hidden`), so nothing can escape its region.
- **No floating popovers for primary controls.** Templates / starting points
  are an inspector tab, not a hovering button.
- **The status bar answers "which machine, how full, how deep is the queue"** so
  those questions never require a view change. Key hints sit on the right, mono,
  accent-coloured keycaps.
- **Master/detail for infrastructure** (Machines): a 326px list plus a detail
  pane, both bordered. Settings is a 200px jump nav over always-open sections.

## 4 · Component vocabulary

| Component           | Anatomy                                                                                  | Tokens / kit                                                             |
| ------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Nav row             | 36px, icon 18px, label sans 13px, trailing count/dot                                     | selected: accent tint + `inset 0 0 0 1px accent`                         |
| Segmented control   | 2px padded track, items 26px; `inline` puts a mono count beside the label                | `ui/components/SegmentedControl.vue`                                     |
| Toolbar button      | 26px, 1px border, sans 12px; hover → `--mold-border-focus`                               | `.ms-toolbar-button` (`--on`, `--accent`, `--danger`, `--danger-hover`)  |
| Group / table label | mono `--mold-fs-micro`, tracked, dim; uppercase is the caller's call                     | `.ms-group-label`                                                        |
| Primary action      | 32px, accent fill, `--mold-on-accent` ink, mono shortcut. One word, `white-space:nowrap` | `--mold-radius-2`                                                        |
| Chip (filter/tag)   | 24px, 1px border, mono count at 70%                                                      | active: accent tint + inset ring                                         |
| Machine card        | dot · mono name · sentence · meter · two mono readouts                                   | target machine gets a 1px accent border                                  |
| Queue: active card  | 52px thumb, sentence status, meter + pause/stop, "What's this?"                          | `--mold-panel-raised` + inset accent ring                                |
| Queue: row          | 38px thumb, title, one-line status, ⋯                                                    | glyph placeholder for images that don't exist yet                        |
| Table row           | 52px, name+id stacked, mono values, ⋯                                                    | `desktop/src/components/models/ModelTableRow.vue`                        |
| Meter               | 5–8px, no radius, single fill                                                            | fill = `--mold-state-*` or accent                                        |
| Dialog              | 480–560px, header / body / footer, `--mold-radius-3`, scrim `--mold-scrim`               | `ui/components/ModalPanel.vue`, desktop `ConfirmDialog` / `RenameDialog` |
| Command palette     | 560px, group column (mono, 60px) + label + key                                           | selected row `--mold-surface-2`                                          |
| Toast               | 320px, glyph column, title + one line, one action; above the status bar                  | bordered in the state colour when urgent                                 |
| Explainer           | `•` + 2–3 sentences of plain English, opt-in                                             | `--mold-panel-raised`, never open by default                             |

## 5 · Imagery

Every image surface shows a real picture — the canvas result, the in-progress
thumbnail, queue rows, library tiles, starting points. A bordered empty box with
a filename is only correct for a job whose image does not exist yet, and then it
carries its queue position as a mono glyph.

## 6 · Motion

Mechanical, no bounce. Live status dots pulse with a 1s two-step opacity
(`.ms-pulse`); meters jump, they do not tween; view switches are instant.
`--mold-dur-quick` for hover colour only.

## 7 · Theme system

Six themes — Mocha (default), Safelight, Blueprint (light), Graphite, Porcelain
(light), Nebula — each a complete `--mold-*` map: colour, one sans and one mono
face, a 7-step type scale, and a theme-scoped radius scale. One `data-theme`
attribute on the root selects the map; **Match system** resolves a pick to its
light or dark partner through `THEME_PAIR` in `ui/theme.ts` before first paint.
No `prefers-color-scheme` in CSS. Text keeps WCAG AA on every plane and control
borders keep 3:1; `desktop/src/styles/tokens.contrast.test.ts` proves it for all
six. Radius is theme-scoped: components reference `--mold-radius-1/2/3`, never
a literal.

## Implementation map

| Design concept                                                                                                                                  | Code                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Tokens: six complete theme maps + the theme-invariant set; a fenced legacy bridge keeps the `--desk/--bath/…` names alive for web and the phone | `ui/tokens.css` (single source, consumed by `web/` and `desktop/`)                                                          |
| Shell metrics, control heights, semantic surfaces, `--mold-state-*`                                                                             | `ui/mold-desktop.css`                                                                                                       |
| Theme contract (`ThemeId`, `THEME_META`, `THEME_PAIR`, `migrateLegacyTheme`, `applyTheme`)                                                      | `ui/theme.ts`, re-exported by `desktop/src/lib/theme.ts` and consumed by `web/src/lib/theme.ts`                             |
| Desktop Tailwind layer (`bg-panel`, `text-fg-dim`, `rounded-control`, `text-micro`…)                                                            | `desktop/src/styles/tokens.css` + `base.css`; `tokens.legacy.test.ts` refuses the retired vocabulary                        |
| Shared kit: shimmer, pulse, toolbar button, group label                                                                                         | `ui/kit.css`                                                                                                                |
| Shared primitives                                                                                                                               | `ui/components/` (Vue, token-var styled)                                                                                    |
| Shell: unified toolbar · sidebar with the queue · status bar                                                                                    | `desktop/src/components/shell/{TitleBar,Sidebar,QueueRail,StatusBar}.vue`, `stores/hostStatus.ts`                           |
| Views                                                                                                                                           | `desktop/src/views/{GenerateView,QueueView,LibraryView,ModelsView,MachinesView,HostDetailView,RunPodView,SettingsView}.vue` |
| Fonts (one sans + one mono per theme, OFL)                                                                                                      | `ui/fonts/` (app-bundled; `fonts.legacy.css` carries only the Safelight pair for the embedded web bundle)                   |

Rules that gate any new UI: compose only from the shared kit, reference tokens
(never hard-coded hex), keep the default path one screen, render overlays
inside their owning frame, speak the lexicon, and keep copy terse and emoji-free.

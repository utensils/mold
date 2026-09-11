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
  and phone references. iOS was accepted complete after #1639; Android shares
  those redesigned screens. The five items the phone redesign deferred —
  per-screen large titles, the Queue running-job card, the Styles filter
  consolidation, Machines cards, and shared-panel touch density — shipped in
  #1685; see [the mobile redesign plan](mobile-redesign-plan.md). Web presentation and Android platform acceptance
  continue in [the web/Android completion plan](web-android-redesign-plan.md).
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
| Short clip · Length · Smoothness                         | Sequence, clips, frames, fps, timeline  | Clip toolbar         |

Voice markers carried over from the CLI: terse, second person, directive.
Units stay tight and mono (`14.9 / 24 GB`, `eta 8m12s`, `$1.44/hr`).
Anything that costs money is stated in money, in `--mold-state-cost`.
`desktop/src/lib/lexicon.test.ts` pins these words where a rename could leave one
surface behind: the destinations on the router, the sidebar, the palette and the
native menu; the File and Generate menu verbs; the finished-work toasts;
Settings' section and row labels; the inspector's seed, Detail, guidance, 3-D
and Add-on-looks labels; the composer's Generate and Write more for me; and the
retired words the Styles and Machines surfaces may never say again.

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
- **Where it runs is chrome, not a setting.** The routing chip (Auto · Most
  capable · a named machine) is the last item on the New image view toolbar,
  after the two doors, so the machine a print goes to is one glance away in
  every tab and every output kind. It began as the last row of the inspector's
  Settings list, where nobody found it.
- **A section only offers styles it can make.** Still picture | Short clip |
  3-D object partitions the installed styles, and the picker shows one part of
  it: a picture section never lists a clip or 3-D style, and no style belongs to
  nowhere. The style is what decides — a one-shot on a clip style IS Short clip,
  which is why the simple clip is not filed under Still picture. The menu names the section it is showing and its Browse more opens
  Styles filtered to the same kind. Every clip style is offered in the clip
  section; nothing is disabled for what it cannot compose.
- **A clip has one way of being made.** Short clip is a prompt, a clip style, a
  Length slider on the composer, Generate. There is no mode strip and no
  timeline: scene-by-scene authoring belongs to `mold run --script` and the
  chain API. A length past what the checkpoint renders in one pass is still
  honoured — the machine splits, carries and stitches it — but that is
  machinery, not a mode, and it is never surfaced as one. **Make** is a real
  batch count here, as on every other kind.
- **The three kinds have three names, used everywhere a kind is chosen or
  filtered.** Still picture · Short clip · 3-D object are the Create toolbar's
  words AND the Styles view's kind filter (All · Still picture · Short clip ·
  3-D object), from one label table, so a person learns the mapping once: what
  you pick in Create is what you filter by in Styles. The filter sorts by the
  same partition the sections do, and Browse more from every section — 3-D
  included — lands on it already filtered.
- **Each section remembers the style it was last used with, across a
  restart.** Still picture, Short clip and 3-D object each keep their own
  last-used style (`studio/stores/lastUsedStyles.ts`, localStorage like the
  draft): a section door opens onto the style the person was using there, and
  a fresh launch opens on the style — and so the section — they left, once the
  machine that has it has reported in. Names are remembered, never
  availability: a machine that lacks the style gets the section's first, and
  the name survives for the machine that has it. Desktop reads it today; web
  and the phone adopt the same store with their redesigns.
- **The view title names what is being made.** On New image the mono title
  follows the output kind — New image · New clip · New 3-D object — and the
  subtitle is the queue count alone. The sidebar's destination stays New image.
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
| Chip (slider)       | 28px composer chip: plain label, an 84px bare track, mono readout (`97f · 4.0s`)         | Length; same track ink as `SliderRow`, snapped to the family frame grid  |
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

Five themes — Mocha (default), Safelight, Blueprint, Graphite, Nebula — each in a
light and a dark tone (ten maps)
(light), Nebula — each a complete `--mold-*` map: colour, one sans and one mono
face, a 7-step type scale, and a theme-scoped radius scale. One `data-theme`
attribute on the root selects the map; **Match system** resolves a pick to its
tone before first paint. A theme's identity is its typography, radii, density and
accent hue; those hold across both of its tones, so **Match system** can never
substitute a different theme. The picker names the theme and a separate
**System · Light · Dark** control names the tone.
No `prefers-color-scheme` in CSS. Text keeps WCAG AA on every plane and control
borders keep 3:1; `desktop/src/styles/tokens.contrast.test.ts` proves it for all
six. Radius is theme-scoped: components reference `--mold-radius-1/2/3`, never
a literal.

## Implementation map

| Design concept                                                                                                                                  | Code                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Tokens: six complete theme maps + the theme-invariant set; a fenced legacy bridge keeps the `--desk/--bath/…` names alive for web and the phone | `ui/tokens.css` (single source, consumed by `web/` and `desktop/`)                                                          |
| Shell metrics, control heights, semantic surfaces, `--mold-state-*` (desktop and web; the phone reads the fallbacks)                             | `ui/mold-desktop.css`                                                                                                       |
| Theme contract (`ThemeId`, `THEME_FAMILY_META`, `toneChoice`, `migrateLegacyTheme`, `applyTheme`)                                               | `ui/theme.ts`, re-exported by `desktop/src/lib/theme.ts` and consumed by `web/src/lib/theme.ts`                             |
| Desktop Tailwind layer (`bg-panel`, `text-fg-dim`, `rounded-control`, `text-micro`…)                                                            | `desktop/src/styles/tokens.css` + `base.css`; `tokens.legacy.test.ts` refuses the retired vocabulary                        |
| Shared kit: shimmer, pulse, `.ms-toolbar-button`, `.ms-group-label`, `.ms-card-edge`, `.ms-lib-upscaled`                                        | `ui/kit.css`                                                                                                                |
| Shared primitives (`SegmentedControl` `inline`, `SliderRow` `low`/`high`, `ModalPanel` header + `#description`, `DrawerPanel`)                  | `ui/components/` (Vue, token-var styled)                                                                                    |
| Shell: unified toolbar · sidebar with the queue · status bar                                                                                    | `desktop/src/components/shell/{TitleBar,Sidebar,QueueRail,StatusBar}.vue`, `stores/hostStatus.ts`                           |
| Views                                                                                                                                           | `desktop/src/views/{GenerateView,QueueView,LibraryView,ModelsView,MachinesView,HostDetailView,RunPodView,SettingsView}.vue` |
| Queue: what a row is waiting on, and the only ETA source (`estimated_finish_unix_ms`)                                                           | `desktop/src/composables/useQueueRowContext.ts`, `lib/queueRows.ts`, `components/shell/QueueRowMenu.vue`                    |
| New image: the inspector's groups (no style field — see the row below)                                                                          | `desktop/src/components/create/InspectorPanel.vue`, `lib/qualityPresets.ts`, `lib/meshDetailLadder.ts`                      |
| The ONE style menu — family groups, plain name over mono `id · size · on GPU`, filter, ↑/↓, the not-here row, Browse more (shared by all surfaces) | `studio/components/StyleMenu.vue`, `studio/lib/{styleMenu,styleLabel,modelSource}.ts`                                       |
| New image: the composer's Style chip and the popover it opens upward around that menu                                                           | `desktop/src/components/create/{StylePicker,ModelPicker}.vue`, `composables/useStylePicker.ts`                              |
| Web Create: the Style card's chip and the teleported popover around that same menu                                                              | `web/src/components/create/CreateStylePicker.vue` (hosted by `ui/components/Popover.vue`)                                   |
| New image: the Starters and Recent tabs beside Settings                                                                                         | `desktop/src/components/create/{inspectorTabs.ts,StarterList.vue,RecentPrints.vue}`                                         |
| My images: scopes, the chip row, the trash banner, History as a column                                                                          | `desktop/src/components/library/{LibraryHeader,LibraryChipRow,CollectionsShelf,TrashBanner,BulkBar,HistoryDrawer}.vue`      |
| Styles: the one column axis (`--model-row-columns`) and the pinned download banner                                                              | `desktop/src/components/models/{InstalledTab,ModelTableRow,CatalogTab,DownloadsTray}.vue`                                   |
| Settings: the one schema, the jump-nav shell, the rows, controls, per-style disclosure and theme cards (web and desktop)                           | `studio/lib/settingsSchema.ts`, `studio/api/config.ts`, `studio/components/settings/`                                       |
| Context menu: one row, root list and submenu alike                                                                                              | `desktop/src/components/shell/{ContextMenu,ContextMenuItem}.vue`, `stores/contextMenu.ts`                                   |
| Fonts (one sans + one mono per theme, OFL)                                                                                                      | `ui/fonts/` (app-bundled; `fonts.legacy.css` carries only the Safelight pair for the embedded web bundle)                   |

Rules that gate any new UI: compose only from the shared kit, reference tokens
(never hard-coded hex), keep the default path one screen, render overlays
inside their owning frame, speak the lexicon, and keep copy terse and emoji-free.

### Deliberate divergences from the reference implementation

- **Settings section headers carry a summary sentence.** The mock's header is
  the bare mono group label; the app puts `SectionInfo.summary` on a second line
  beneath it in `text-micro text-fg-dim`. It is what the global search matches
  on, and it tells a reader what a section holds before they scroll into it.
- **No "Install updates automatically" toggle.** The mock has one, on by
  default. Updates are check-only by policy: an available update is announced
  and installed only by an explicit **Update and restart**, which is what the
  Updates & about copy already says.
- **The theme swatch band paints from a nested `data-theme`.** `ui/tokens.css`
  selects each theme map on any element, not only `:root`, so the Look picker
  can show a theme's own surfaces without repeating a hex in TypeScript. That
  band is the only themed island the app is allowed.
- **Smoothness rides the Clip card.** The lexicon row is `Short clip · Length · Smoothness`,
  so a clip's frame rate sits beside the length it pairs with.
  Advanced ▸ Video keeps the raw fps field; the card is the one people reach
  for, and the two must never disagree.
- **The bulk Delete says "Move N pictures to trash".** The lexicon's noun for a
  result is a picture, and the count is what makes a bulk action safe to
  confirm; "Delete" alone on a selection of forty reads as one thing.
- **An active download reads "Downloading", not "Being made".** The queue's
  vocabulary describes a picture arriving; a style arriving is a different
  event, and `DownloadsTray` names it plainly rather than borrowing the row
  sentence.
- **History is a column and keeps its Runs / Prompts tabs.** Moving it out of a
  modal drawer took the scrim away, not the lenses; the tabs render in the
  column body, and `?panel=history&tab=` still addresses them.
- **Refresh stays in My images.** The primary bucket is SSE-live, but a
  connected remote's gallery is polled, so the toolbar keeps one explicit way to
  ask every machine again.

- **Web Settings folds the jump nav into a chip strip below 900px.** A browser page
  has no fixed second pane, so the 200px nav column becomes a horizontally scrolling
  strip of section chips above the page; search, scroll-spy and `?section=` behave
  the same. The desktop keeps the column.
- **Per-style defaults are one collapsed row per style.** The mock shows eight curated
  rows and no raw key list; a machine reports `models.<style>.<field>` for every tuned
  style (104 rows on a thirteen-style box), so the section shows one disclosure per
  style — friendly name over mono id, `N overrides` — and the eight fields open inside
  it, each led by the inspector's word (Detail, Width, Words to avoid…) over the
  engine's field name. Every other engine key has a plain-words row; the raw
  "Server-provided configuration key." row exists only for a key newer than the client.

- **A print says how long it took, in one spelling everywhere.**
  `OutputMetadata.generation_time_ms` (additive; the gallery row fills it for
  a print made before the field) is read through `studio/lib/generationTime.ts`
  — `4.0s` under a minute, `1m 12s` above — on the canvas caption, the Recent
  row (`flux-dev:q8 · 4.0s`), and the Lightbox's **Took** fact; a print that
  does not know shows nothing, never `0.0s`. The Styles shelf's **Speed**
  column is derived, not served: `studio/lib/styleSpeed.ts` takes the median
  of the newest ten timed prints per style (`~20s`), and a style nobody has
  timed shows an empty cell rather than a guess.

- **A licence row leads with what it unlocks.** `GET /api/licenses` names the
  styles each licence gates with the registry's own description, and the row
  reads that in plain words ("FLUX.1 Dev Q4 — smaller/faster, good quality")
  over the licence's name and summary in mono; the id rides the tooltip.
  Acceptance stays one row per licence.
- **Save every result off means "straight to the trash", never "never
  saved".** `GenerateRequest.save_to_gallery: false` (additive) publishes the
  print exactly as any other — settlement, replay and provenance untouched —
  and then moves it to the trash while the publication writer is still held,
  so a throwaway never lists in My images yet stays recoverable until
  retention empties the trash. The inspector's toggle sits under Repeat this
  look with its hint. A requested Framewise upscale still runs: it is enqueued before the trash step
  and pins its source by hard link. Its own published print is a second,
  explicitly requested result and keeps saving.
- **Rent this GPU states RunPod's own hourly rate** for the chosen cloud
  (`gpuTypes.securePrice` / `communityPrice`), in the picker and in the
  "billing begins now" confirm; with none reported, the confirm keeps its
  by-the-minute sentence and invents nothing.
- **A machine's Storage tile says what the pictures take**
  (`/api/status.gallery_storage`: bytes and prints, live and trashed, summed
  from the host's own DB), never a client-side walk of the gallery.

### Named absences (no backend, not oversights)

Pausing the job that is already running: an in-flight denoise cannot be
suspended, so the active card's pause holds the queue after the current
image and says so.

## 3-D Studio workflows

The 3-D Studio is New image's anatomy, not a variation on it: the desktop
shell's 40px view toolbar, a full-height result canvas, a composer on the bottom
edge, and the standard inspector on the right (300px by default, resizable
280–480px with Generate's shared handle, width setting, keyboard controls, and
double-click reset). **From words**, **Rebuild**, and **Add texture** are the
shared segmented control on the toolbar; only workflows supported by the
connected machines appear.

**The description and Generate live on the composer**, with the ⌘↩ keycap, as
they do on New image. **3-D style** and **Picture style** are composer chips
beside them. On the desktop shell ⌘↩ generates HERE rather than leaving for New
image — every raiser of that intent (the native menu and the ⌘K palette alike)
stays on this route and lets the view consume it. The web SPA has no keyboard
Generate on any surface, so the keycap is the desktop composer's. Reuse
Generate's model picker with filtered candidates, including its search, family
groups, availability, and friendly labels; do not build a second selector, and
narrow the candidates through `outputKindForModel` rather than a second
predicate. The inspector holds the stage settings — texture and its size,
lighting removal, and for a supplied mesh its file wells and orientation — using
the shared switches, group labels and inspector typography. The result and its
stage progress occupy the canvas, without a separate landing-page heading or
oversized cards.

(This supersedes the original "workflow inputs belong together in the inspector,
with Generate at its foot". Putting the prompt where every other making-view
puts it is what makes the surface read as part of the app, and it gives ⌘↩ an
obvious home.)

**Reuse is a door, not a dropdown.** The desktop rail is **Settings | Recent**,
the peer of New image's tab strip, with a mono count. A Recent row says what
was made in the toolbar's own words, what it is doing in the queue's vocabulary
(the stage and `n/N` while it runs, `Finished` / `Stopped` / `Failed`
otherwise — never the raw wire state), and carries the accent **Use these
settings again** line. Clicking restores explicitly rather than through a
watcher, because the row you are most likely to click is the one already open.
A workflow's FILES cannot be restored, so a supplied-mesh run says which to
choose again rather than leaving Generate disabled for no visible reason, and
opening a run never clears an attachment you already made. **New workflow**
clears the authored work and keeps the machine and the styles in use.

**The draft belongs to a store, never to the view.** The router lazy-loads this
surface and nothing keeps it alive, so a component-local ref loses the
description, both styles, the attachments, the stage settings and the machine
pin on any trip to the Queue. `studio/stores/meshWorkflowDraft.ts` holds them,
exactly as `generateForm` does for New image. An attached file is deliberately
session-scoped: a browser cannot re-open a file handle from a previous session,
so the well comes back empty rather than promising bytes it cannot read.

The existing **Where it runs** chip is the last toolbar control. **Auto** chooses
the least-busy eligible machine; **Most capable** chooses its strongest eligible
GPU. Eligibility covers every selected stage and style on one machine, including
optional texture and lighting removal. A named machine is a pin, never a hint.
Routing is resolved when Generate is pressed, before source uploads. The
accepted job's machine remains visible beside its history selector; inspecting,
resuming, cancelling, and downloading the result stay attached to that owner.

Routine telemetry and polling must preserve the draft, focus, selected history
entry, and viewer camera. Only a changed URL or credential refreshes the browsing
context. Progress polling updates stage state in place, and a result's bytes are
loaded once per output identity. At narrow widths, toolbar controls may wrap;
the canvas and inspector retain independent scrolling.

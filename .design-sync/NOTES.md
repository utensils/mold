# design-sync notes — mold

- **This is a tokens-only sync by deliberate decision (2026-07-21).** The Mold
  Studio kit (`ui/`) is Vue 3; claude.ai/design renders React, so the 21 Vue
  primitives are NOT bundled. What ships: `tokens.css` + `kit.css` (as
  `_ds_bundle.css`), the three OFL variable fonts, `window.MoldStudio` =
  `icons.ts` + `theme.ts` (framework-agnostic), guidelines distilled from
  `docs/design/mold-studio-spec.html` (v0.12), and the conventions README
  header. The user chose this over aborting or Vue-in-React shims (shims
  rejected: Vue slots can't take React children).
- `ui/` has **no build and no dist** — `--entry`/`cfg.entry` points at
  `ui/icons.ts` (any real file works; its walk-up finds `ui/package.json` so
  PKG_DIR = ui/). No `buildCmd` needed.
- The repo is a **Bun workspace with no React**: converter deps AND
  `react`/`react-dom`/`playwright@1.61.0` are npm-installed into
  `.ds-sync/node_modules`, and `--node-modules .ds-sync/node_modules` keeps
  workspaceRoot = the mold repo (a scratch dir outside the repo would make
  cfgPath reject every `ui/` path).
- CSS routing: `kit.css` rides the esbuild graph via
  `.design-sync/ds-css.ts` (extraEntries); `tokens.css` rides `cfg.cssEntry`
  so the append path preserves its documentation comments verbatim.
  `cfg.cssEntry` being set is also what flips the adapter into tokens-only
  mode (zero components would otherwise be fatal).
- Fonts: `cfg.extraFonts: ["fonts/fonts.css"]` — the TTF url()s must stay OUT
  of the JS graph (no .ttf esbuild loader).
- Playwright: local cache has chromium build **1228** → playwright **1.61.0**
  (installed in `.ds-sync`). Render check trivially passes (0 previews).
- `[DTS_REACT]` warn on build is expected and harmless here (zero components,
  no .d.ts emitted).
- **Claude Design compiler feedback (2026-07-21) drove a tokens.css
  restructure**: its token registry only reads custom properties under plain
  attribute scopes, so the light media blocks' `:root:not([data-theme="dark"])`
  selectors were replaced with bare `:root` / `:root[data-theme-family="mold"]`
  plus explicit forced-dark blocks (`[data-theme="dark"]`), and 8 tokens got
  trailing `/* @kind ... */` comments (`--grad` color; `--f-*` font;
  `--dur-*`/`--ease` other). Behavior verified identical across 36
  states × 41 computed values (playwright matrix) and guarded by
  `ui/tokens.test.ts` (mirror invariants + cascade order + annotations).
  If the compiler still flags props inside `@media (prefers-color-scheme:
  light)` after this, the remaining option is attribute-only theming (theme.ts
  stamps the resolved appearance) — a cross-surface product refactor; get
  explicit sign-off first.

## Known render warns

(none — tokens-only, no component previews)

## Re-sync risks

- **Token/kit drift is the main risk**: any edit to `ui/tokens.css`,
  `ui/kit.css`, `ui/fonts/fonts.css`, `ui/icons.ts`, or `ui/theme.ts` needs a
  re-sync; nothing here regenerates automatically.
- `ui/tokens.css` must keep its design-tool constraints: no `:not()` scopes,
  and `@kind` annotations on non-inferable tokens — `ui/tokens.test.ts`
  enforces both, so keep it green rather than deleting it.
- The guidelines file (`.design-sync/guidelines/mold-studio-guidelines.md`)
  and `conventions.md` are **hand-distilled from the spec at v0.12** — when
  `docs/design/mold-studio-spec.html` bumps, re-validate both against the new
  spec (token names, component vocabulary, IA) before re-uploading.
- `conventions.md` names specific tokens/classes/icons — the validation pass
  (grep tokens/classes in `ds-bundle/_ds_bundle.css`, icon names against
  `MoldStudio.ICONS`) must be re-run on every re-sync.
- If the kit ever grows a React (or compiled-to-React) layer, revisit the
  tokens-only decision — the full component pipeline was skipped, not
  attempted and failed.
- Chromium cache build vs playwright pin can drift after a `playwright`
  update elsewhere on the machine — re-verify before trusting the render
  check.

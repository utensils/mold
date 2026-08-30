# Mold Studio — design system

The canonical design source for mold's UI surfaces (macOS/Linux/Windows
desktop, iOS, Android, web browser, mobile web, terminal TUI). Android ships
from the same remote-only Tauri crate and the same `desktop/src/mobile` Vue
surface as iOS ([Android guide](../../website/guide/android.md)). This replaces
the former `desktop/docs/design-spec.md` (Safelight spec); its still-relevant
content is carried forward as §10–§12 of the HTML spec.

- **`mold-studio-spec.html`** — the interface spec: principles, tokens,
  shared components, information architecture, flows, voice, motion values,
  and the open-items list. Open it directly in a browser (fonts and scripts
  are local to this directory).
- **`mold-studio-proposed-ui.html`** — the archived interactive exploration
  that established the four graphical shells. Use it for visual provenance,
  not current interaction behavior; the spec and shipped shared components are
  authoritative where the prototype differs.
- **`mold-tui-proposed.html`** — the interactive mockup of the terminal
  surface (spec §05 "Terminal surface" + gap G13). Partly implemented:
  `crates/mold-tui` already carries the five-workspace IA (`View` is exactly
  Create | Library | Models | Machines | Settings, `action.rs`) and the
  essentials + Advanced Create form (`ui/create_form.rs`); the terminal
  restyle in this mockup, plus the spec's recommended TUI stack, is what
  remains.

Implementation mapping:

| Spec concept | Code |
| --- | --- |
| Tokens (`--desk/--bath/--bench/--surface/…`, 2 families × dark/light) | `ui/tokens.css` (single source, consumed by `web/` and `desktop/`) |
| Theme switching contract | `desktop/src/lib/theme.ts` (desktop + iOS), `web/src/lib/theme.ts` (web) |
| Shared component kit | `ui/components/` (Vue primitives, token-var styled) |
| Fonts (Bricolage Grotesque, Schibsted Grotesk, Martian Mono — OFL) | `ui/fonts/` (app-bundled), `docs/design/fonts/` (spec-local copies) |

Rules that gate any new UI (see the spec's adherence checklist): compose only
from the shared kit, reference tokens (never hard-coded hex), keep the default
path one screen, use one surface-appropriate entry for depth without redundant
nested disclosure, render overlays inside their owning frame, and keep copy
terse, lowercase-technical, and emoji-free.

# Mold Studio conventions

This design system ships **tokens, fonts, kit classes, an icon registry, and
theme helpers — no prebuilt React components**. You compose every control
yourself from the token vocabulary below (the product's own primitives are Vue
and are deliberately not bundled). `guidelines/mold-studio-guidelines.md`
defines the component vocabulary, states, and voice to imitate — read it
before building.

## Setup

No provider is needed. Theming is driven by two attributes on the **root
element** (set them on `document.documentElement`, or call
`MoldStudio.applyTheme(theme, family)`):

- `data-theme-family`: `"mold"` (cyan/magenta) or `"safelight"` (warm amber — the default when absent)
- `data-theme`: `"dark"` or `"light"` (absent = follow `prefers-color-scheme`)

All four combinations work with zero component changes because every color is
a CSS custom property. Fonts (Bricolage Grotesque, Schibsted Grotesk, Martian
Mono) ship with the bundle and load via `styles.css`. Add the class `ms-kit`
to your app root so keyboard `:focus-visible` gets the standard 2px accent ring.

## Styling idiom: token custom properties, never hex

Style with `var(--*)` from the shipped stylesheet. The vocabulary
(full table in `guidelines/mold-studio-guidelines.md`):

- Surfaces, stepping upward: `--desk` → `--bath` → `--bench` (cards/panels) → `--surface` (nested/hover)
- Ink: `--rebate` (primary), `--ink-2`, `--ink-3`; borders `--edge` (hairline), `--ce` (controls)
- Accents with fixed jobs: `--safelight` = actions/focus/selection (warm); `--halide` = models/telemetry/links (cool)
- Selection (accent-derived): `--sel-ink`, `--sel-bg`, `--sel-border`, `--sel-fill`, `--sel-ring`
- Semantic: `--success`, `--warning`, `--stop`; on accent fills: `--on-accent`; brand gradient `--grad`
- Media bed: `--print`, `--on-media` (prints sit on a dark bed; never inverts)
- Type: `--f-display` (titles), `--f-body` (UI), `--f-mono` (data, model ids)
- Radius: `--radius-control-sm|control|control-lg` (6–10px), `--radius-card|card-lg` (12–16px), `--radius-pill` (20px)
- Motion: `--dur-quick|base|slow` (120/180/240ms) with `--ease`

Kit classes (the only global classes): `.ms-shimmer` (loading placeholder),
`.ms-fade-up` (overlay/result entrance), `.ms-focus` (explicit focus ring).

## Icons

`MoldStudio.ICONS` maps names (`create`, `library`, `models`, `machines`,
`settings`, `search`, `close`, `check`, `plus`, `play`, `download`, `trash`,
`image`, `video`, …all in `MoldStudio.ICON_NAMES`) to inner-SVG strings drawn
with `currentColor` on a 24-unit grid.

## Idiomatic build example

```jsx
const Icon = ({ name, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"
    strokeLinejoin="round"
    dangerouslySetInnerHTML={{ __html: MoldStudio.ICONS[name] }} />
);

const GenerateButton = () => (
  <button style={{
    background: "var(--safelight)", color: "var(--on-accent)",
    fontFamily: "var(--f-body)", fontWeight: 600,
    border: "none", borderRadius: "var(--radius-control)",
    padding: "10px 18px", display: "inline-flex", gap: 8,
    alignItems: "center", cursor: "pointer",
  }}>
    <Icon name="sparkle" /> Generate
  </button>
);
```

Voice: product name **mold** lowercase; status values lowercase (`loaded`,
`developing`); model ids as `family:variant` in `--f-mono`; glyphs ✓ ★ • only,
no emoji.

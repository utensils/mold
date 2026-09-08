/*
 * Mold Studio theme contract — shared by desktop, web, the phone and the TUI.
 *
 * A theme is an IDENTITY: its typography, its corner radii, its density and its
 * accent hue. Light-vs-dark is a TONE that identity is rendered in, not a second
 * theme. So there are five families (ui/tokens.css carries the maps), each in two
 * tones, and a ThemeId is simply `${family}-${tone}`.
 *
 * That makes tone DERIVABLE rather than tabulated. The predecessor of this file
 * paired each theme with an unrelated partner by hand — Nebula's daylight partner
 * was Porcelain — so "Match system" swapped the theme out from under the person
 * who chose it. The partner of a theme is now the same theme in the other tone,
 * and there is nothing left to keep in step.
 *
 * A surface persists ONE ThemeId plus a `matchSystem` flag. Pickers bind to
 * `toneChoice` / `applyToneChoice` and show a System · Light · Dark control; a
 * theme's own label never carries a tone. The Rust twin of this file is
 * desktop/src-tauri/src/settings.rs; the TUI's is crates/mold-tui/src/ui/theme.rs.
 */

export type ThemeFamilyId =
  "mocha" | "safelight" | "blueprint" | "graphite" | "nebula";
export type ThemeTone = "dark" | "light";
export type ThemeId = `${ThemeFamilyId}-${ThemeTone}`;

/** What a picker offers: the tone control's three positions. */
export type ToneChoice = "system" | ThemeTone;

export interface ThemeFamilyMeta {
  readonly id: ThemeFamilyId;
  readonly label: string;
  /** "Sans · Mono" pairing, for the picker's type line. */
  readonly type: string;
  /** One line about the identity — true of BOTH tones, never about a tone. */
  readonly blurb: string;
}

export const THEME_FAMILY_META: readonly ThemeFamilyMeta[] = [
  {
    id: "mocha",
    label: "Mocha",
    type: "Inter · JetBrains Mono",
    blurb:
      "Violet-leaning neutrals, one blue accent, nothing else raises its voice.",
  },
  {
    id: "safelight",
    label: "Safelight",
    type: "Schibsted Grotesk · Martian Mono",
    blurb:
      "The darkroom family: warm browns, amber for anything you press, softer corners.",
  },
  {
    id: "blueprint",
    label: "Blueprint",
    type: "Archivo · Azeret Mono",
    blurb: "Drafting-table blue, set one notch tighter and smaller.",
  },
  {
    id: "graphite",
    label: "Graphite",
    type: "IBM Plex Sans · IBM Plex Mono",
    blurb:
      "True neutral greys, hairline separators, one signal hue for anything live.",
  },
  {
    id: "nebula",
    label: "Nebula",
    type: "Georgia · Geist Mono",
    blurb:
      "Oxblood and hot crimson for actions, square corners, roomy leading.",
  },
];

export const THEME_FAMILIES = [
  "mocha",
  "safelight",
  "blueprint",
  "graphite",
  "nebula",
] as const satisfies readonly ThemeFamilyId[];

export const THEME_TONES = [
  "light",
  "dark",
] as const satisfies readonly ThemeTone[];

/** Every id, family-major. ui/tokens.css declares exactly this set. */
export const THEMES: readonly ThemeId[] = THEME_FAMILIES.flatMap((family) =>
  THEME_TONES.map((tone) => themeId(family, tone)),
);

export const DEFAULT_THEME: ThemeId = "mocha-dark";

export function themeId(family: ThemeFamilyId, tone: ThemeTone): ThemeId {
  return `${family}-${tone}`;
}

export function familyOf(id: ThemeId): ThemeFamilyId {
  return id.slice(0, id.lastIndexOf("-")) as ThemeFamilyId;
}

export function toneOf(id: ThemeId): ThemeTone {
  return id.endsWith("-light") ? "light" : "dark";
}

export function isThemeFamilyId(value: unknown): value is ThemeFamilyId {
  return (
    typeof value === "string" &&
    (THEME_FAMILIES as readonly string[]).includes(value)
  );
}

export function isThemeId(value: unknown): value is ThemeId {
  return (
    typeof value === "string" && (THEMES as readonly string[]).includes(value)
  );
}

export function themeFamilyMeta(id: ThemeId | ThemeFamilyId): ThemeFamilyMeta {
  const family = isThemeFamilyId(id) ? id : familyOf(id);
  return THEME_FAMILY_META.find((meta) => meta.id === family)!;
}

/** The same theme in the other tone. Never a different family. */
export function partnerTheme(id: ThemeId, tone: ThemeTone): ThemeId {
  return themeId(familyOf(id), tone);
}

/* ── The picker's two controls ─────────────────────────────────────────────
 * A theme entry carries only its name; one System · Light · Dark control sets
 * the tone. Both project onto the SAME persisted `{ theme, matchSystem }`, so
 * choosing a theme cannot change the tone and choosing a tone cannot change
 * the theme. */

export function toneChoice(saved: {
  theme: ThemeId;
  matchSystem: boolean;
}): ToneChoice {
  return saved.matchSystem ? "system" : toneOf(saved.theme);
}

/**
 * Apply a tone choice to the theme in hand. `system` keeps the stored id's own
 * tone as the fallback for a host that cannot read the OS appearance, which is
 * why the flag and the suffix are both persisted.
 */
export function applyToneChoice(
  choice: ToneChoice,
  theme: ThemeId,
): { theme: ThemeId; matchSystem: boolean } {
  if (choice === "system") return { theme, matchSystem: true };
  return { theme: partnerTheme(theme, choice), matchSystem: false };
}

/** Choosing a theme keeps whatever tone is in force. */
export function applyFamilyChoice(
  family: ThemeFamilyId,
  saved: { theme: ThemeId; matchSystem: boolean },
): { theme: ThemeId; matchSystem: boolean } {
  return {
    theme: themeId(family, toneOf(saved.theme)),
    matchSystem: saved.matchSystem,
  };
}

/** The concrete theme to paint for a pick, given the OS appearance. */
export function resolveTheme(
  theme: ThemeId,
  matchSystem: boolean,
  prefersLight: boolean,
): ThemeId {
  if (!matchSystem) return theme;
  return partnerTheme(theme, prefersLight ? "light" : "dark");
}

/**
 * Ids persisted before tone became a suffix. `porcelain` is the merge: Graphite
 * and Porcelain were one theme under two names, so Porcelain's palette lives on
 * as Graphite's light tone.
 */
const RENAMED_THEMES: Readonly<Record<string, ThemeId>> = {
  mocha: "mocha-dark",
  safelight: "safelight-dark",
  blueprint: "blueprint-light",
  graphite: "graphite-dark",
  porcelain: "graphite-light",
  nebula: "nebula-dark",
};

/**
 * Every saved value reaches a current ThemeId through this one function.
 *
 * Two generations precede the current shape: `theme: system|dark|light` beside
 * `themeFamily: safelight|mold`, and then the six single-word ids. Both land in
 * one hop, and an unreadable value falls back rather than failing — a theme this
 * build cannot parse must never cost the user their saved machines.
 */
export function migrateLegacyTheme(
  theme: unknown,
  family: unknown,
): { theme: ThemeId; matchSystem: boolean } {
  if (isThemeId(theme)) return { theme, matchSystem: false };
  if (typeof theme === "string" && theme in RENAMED_THEMES) {
    return { theme: RENAMED_THEMES[theme]!, matchSystem: false };
  }
  const dark: ThemeId = family === "mold" ? "mocha-dark" : "safelight-dark";
  if (theme === "light")
    return { theme: partnerTheme(dark, "light"), matchSystem: false };
  if (theme === "system") return { theme: dark, matchSystem: true };
  return { theme: dark, matchSystem: false };
}

export function systemPrefersLight(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
  );
}

/** Keep native browser/WebView chrome aligned with the active chrome tone. */
export function syncThemeColor(
  root: HTMLElement = document.documentElement,
  documentNode: Document = document,
): void {
  const meta = documentNode.querySelector<HTMLMetaElement>(
    'meta[name="theme-color"]',
  );
  if (!meta || typeof getComputedStyle !== "function") return;
  const chrome = getComputedStyle(root)
    .getPropertyValue("--mold-bg-deep")
    .trim();
  if (chrome) meta.content = chrome;
}

/** Stamp the resolved theme on a document root. Returns what was painted. */
export function applyTheme(
  theme: ThemeId,
  matchSystem: boolean,
  root: HTMLElement = document.documentElement,
  prefersLight: boolean = systemPrefersLight(),
): ThemeId {
  const resolved = resolveTheme(theme, matchSystem, prefersLight);
  root.dataset.theme = resolved;
  syncThemeColor(root, root.ownerDocument);
  return resolved;
}

/**
 * System appearance can change while the app is running. Re-resolve the
 * persisted pick on every flip; `read` keeps this store-agnostic.
 */
export function installSystemThemeSync(
  read: () => { theme: ThemeId; matchSystem: boolean },
): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function")
    return () => {};
  const query = window.matchMedia("(prefers-color-scheme: light)");
  const sync = () => {
    const { theme, matchSystem } = read();
    applyTheme(theme, matchSystem, document.documentElement, query.matches);
  };
  query.addEventListener?.("change", sync);
  return () => query.removeEventListener?.("change", sync);
}

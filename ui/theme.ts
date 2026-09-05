/*
 * Mold Studio theme contract — shared by desktop, web, and the phone.
 *
 * Six named themes (ui/tokens.css carries the maps). A surface persists ONE
 * ThemeId plus a `matchSystem` flag; when the flag is on, the OS appearance
 * picks between the chosen theme and its partner from THEME_PAIR, so the
 * stylesheet only ever sees a single `data-theme`. The Rust side of the
 * desktop app (desktop/src-tauri/src/settings.rs) mirrors THEME_PAIR's light
 * partners in `light_partner` and the legacy migration in `migrate_theme`.
 */

export type ThemeId =
  "mocha" | "safelight" | "blueprint" | "graphite" | "porcelain" | "nebula";
export type ThemeTone = "dark" | "light";

export interface ThemeMeta {
  readonly id: ThemeId;
  readonly label: string;
  readonly tone: ThemeTone;
  /** The tone as the picker says it: "Dark · the original". `tone` stays the
   *  machine value THEME_TONE and THEME_PAIR are keyed on. */
  readonly toneLabel: string;
  /** "Sans · Mono" pairing, for the picker's type line. */
  readonly type: string;
  readonly blurb: string;
}

export const THEME_META: readonly ThemeMeta[] = [
  {
    id: "mocha",
    label: "Mocha",
    tone: "dark",
    toneLabel: "Dark · the original",
    type: "Inter · JetBrains Mono",
    blurb:
      "Violet-leaning charcoal, one blue accent, nothing else raises its voice.",
  },
  {
    id: "safelight",
    label: "Safelight",
    tone: "dark",
    toneLabel: "Dark · the app's own",
    type: "Schibsted Grotesk · Martian Mono",
    blurb:
      "The darkroom family: warm browns, amber for anything you press, softer corners.",
  },
  {
    id: "blueprint",
    label: "Blueprint",
    tone: "light",
    toneLabel: "Light · drafting table",
    type: "Archivo · Azeret Mono",
    blurb:
      "Cool daylight and drafting-table blue, set one notch tighter and smaller.",
  },
  {
    id: "graphite",
    label: "Graphite",
    tone: "dark",
    toneLabel: "Dark · neutral, warm signal",
    type: "IBM Plex Sans · IBM Plex Mono",
    blurb:
      "True neutral greys, hairline separators, one amber signal for anything live.",
  },
  {
    id: "porcelain",
    label: "Porcelain",
    tone: "light",
    toneLabel: "Light · high-key, compact",
    type: "Manrope · IBM Plex Mono",
    blurb:
      "Near-white panels on a soft grey desk, deep teal for anything you can press.",
  },
  {
    id: "nebula",
    label: "Nebula",
    tone: "dark",
    toneLabel: "Dark · oxblood & crimson",
    type: "Georgia · Geist Mono",
    blurb:
      "Oxblood panels over near-black, hot crimson for actions, square corners.",
  },
];

export const THEMES = [
  "mocha",
  "safelight",
  "blueprint",
  "graphite",
  "porcelain",
  "nebula",
] as const satisfies readonly ThemeId[];

export const DEFAULT_THEME: ThemeId = "mocha";

export const THEME_TONE: Record<ThemeId, ThemeTone> = {
  mocha: "dark",
  safelight: "dark",
  blueprint: "light",
  graphite: "dark",
  porcelain: "light",
  nebula: "dark",
};

/** Which theme a pick becomes when the system appearance flips. */
export const THEME_PAIR: Record<ThemeId, { dark: ThemeId; light: ThemeId }> = {
  mocha: { dark: "mocha", light: "blueprint" },
  blueprint: { dark: "mocha", light: "blueprint" },
  graphite: { dark: "graphite", light: "porcelain" },
  porcelain: { dark: "graphite", light: "porcelain" },
  safelight: { dark: "safelight", light: "porcelain" },
  nebula: { dark: "nebula", light: "porcelain" },
};

export function isThemeId(value: unknown): value is ThemeId {
  return (
    typeof value === "string" && (THEMES as readonly string[]).includes(value)
  );
}

export function themeMeta(id: ThemeId): ThemeMeta {
  return THEME_META.find((meta) => meta.id === id)!;
}

/** The concrete theme to paint for a pick, given the OS appearance. */
export function resolveTheme(
  theme: ThemeId,
  matchSystem: boolean,
  prefersLight: boolean,
): ThemeId {
  if (!matchSystem) return theme;
  return prefersLight ? THEME_PAIR[theme].light : THEME_PAIR[theme].dark;
}

/**
 * The pre-redesign contract persisted `theme: system|dark|light` beside
 * `themeFamily: safelight|mold`. Every surface migrates a saved value through
 * this one table so an old install lands on the same theme everywhere.
 */
export function migrateLegacyTheme(
  theme: unknown,
  family: unknown,
): { theme: ThemeId; matchSystem: boolean } {
  if (isThemeId(theme)) return { theme, matchSystem: false };
  const dark: ThemeId = family === "mold" ? "mocha" : "safelight";
  if (theme === "light")
    return { theme: THEME_PAIR[dark].light, matchSystem: false };
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

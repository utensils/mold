export type Theme = "system" | "dark" | "light";
export type ThemeFamily = "safelight" | "mold";

export const THEMES = [
  "system",
  "dark",
  "light",
] as const satisfies readonly Theme[];
export const THEME_FAMILIES = [
  "mold",
  "safelight",
] as const satisfies readonly ThemeFamily[];

export function isTheme(value: unknown): value is Theme {
  return (
    typeof value === "string" && (THEMES as readonly string[]).includes(value)
  );
}

export function isThemeFamily(value: unknown): value is ThemeFamily {
  return (
    typeof value === "string" &&
    (THEME_FAMILIES as readonly string[]).includes(value)
  );
}

/**
 * Resolve the independent palette-family and light/dark root attributes.
 * A null appearance lets the system color-scheme media query drive the UI.
 */
export function resolveThemeAttributes(
  theme: Theme,
  family: ThemeFamily,
): { appearance: "light" | "dark" | null; family: ThemeFamily } {
  return {
    appearance: theme === "system" ? null : theme,
    family,
  };
}

/** Keep native browser/WebView chrome aligned with the active Bath token. */
export function syncThemeColor(
  root: HTMLElement = document.documentElement,
  documentNode: Document = document,
): void {
  const meta = documentNode.querySelector<HTMLMetaElement>(
    'meta[name="theme-color"]',
  );
  if (!meta || typeof getComputedStyle !== "function") return;
  const bath = getComputedStyle(root).getPropertyValue("--bath").trim();
  if (bath) meta.content = bath;
}

/** Apply the shared desktop/mobile theme contract to a document root. */
export function applyTheme(
  theme: Theme,
  family: ThemeFamily,
  root: HTMLElement = document.documentElement,
): void {
  const attrs = resolveThemeAttributes(theme, family);
  if (attrs.appearance === null) delete root.dataset.theme;
  else root.dataset.theme = attrs.appearance;
  root.dataset.themeFamily = attrs.family;
  syncThemeColor(root, root.ownerDocument);
}

/**
 * System appearance can change while the app is running. The CSS palette
 * follows automatically; this listener keeps the native status-bar color in
 * step as well.
 */
export function installSystemThemeColorSync(): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function")
    return () => {};
  const query = window.matchMedia("(prefers-color-scheme: light)");
  const sync = () => syncThemeColor();
  query.addEventListener?.("change", sync);
  return () => query.removeEventListener?.("change", sync);
}

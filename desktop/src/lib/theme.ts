/*
 * The theme contract lives in the shared Mold Studio design system (ui/) so
 * every surface — desktop, iOS, web, mobile web — applies palettes the same
 * way. This module re-exports it for the desktop/mobile import graph.
 */
export {
  THEMES,
  THEME_FAMILIES,
  isTheme,
  isThemeFamily,
  resolveThemeAttributes,
  syncThemeColor,
  applyTheme,
  installSystemThemeColorSync,
  type Theme,
  type ThemeFamily,
} from "@ui/theme";

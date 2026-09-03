/*
 * The theme contract lives in the shared Mold Studio design system (ui/) so
 * every surface — desktop, iOS, web, mobile web — applies themes the same
 * way. This module re-exports it for the desktop/mobile import graph.
 */
export {
  DEFAULT_THEME,
  THEMES,
  THEME_META,
  THEME_PAIR,
  THEME_TONE,
  applyTheme,
  installSystemThemeSync,
  isThemeId,
  migrateLegacyTheme,
  resolveTheme,
  syncThemeColor,
  themeMeta,
  type ThemeId,
  type ThemeMeta,
  type ThemeTone,
} from "@ui/theme";

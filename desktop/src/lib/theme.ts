/*
 * The theme contract lives in the shared Mold Studio design system (ui/) so
 * every surface — desktop, iOS, web, mobile web — applies themes the same
 * way. This module re-exports it for the desktop/mobile import graph.
 */
export {
  DEFAULT_THEME,
  THEMES,
  THEME_FAMILIES,
  THEME_FAMILY_META,
  THEME_TONES,
  applyFamilyChoice,
  applyTheme,
  applyToneChoice,
  familyOf,
  installSystemThemeSync,
  isThemeFamilyId,
  isThemeId,
  migrateLegacyTheme,
  partnerTheme,
  resolveTheme,
  syncThemeColor,
  themeFamilyMeta,
  themeId,
  toneChoice,
  toneOf,
  type ThemeFamilyId,
  type ThemeFamilyMeta,
  type ThemeId,
  type ThemeTone,
  type ToneChoice,
} from "@ui/theme";

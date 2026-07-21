import { ref, watch } from "vue";
import {
  applyTheme,
  installSystemThemeColorSync,
  isTheme,
  isThemeFamily,
  type Theme,
  type ThemeFamily,
} from "@ui/theme";

/*
 * Web-surface theme state on the shared Mold Studio contract (@ui/theme).
 * Fresh visitors default to the Safelight family — the same default as
 * desktop and iPhone — and appearance follows the system until the user
 * picks Dark or Light explicitly. A valid saved choice always wins.
 */

const STORAGE_KEY = "mold.web.theme.v1";

interface PersistedTheme {
  family: ThemeFamily;
  theme: Theme;
}

function load(): PersistedTheme {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        const candidate = parsed as Record<string, unknown>;
        return {
          family: isThemeFamily(candidate.family)
            ? candidate.family
            : "safelight",
          theme: isTheme(candidate.theme) ? candidate.theme : "system",
        };
      }
    }
  } catch {
    // Ignore storage failures — fall through to defaults.
  }
  return { family: "safelight", theme: "system" };
}

const initial = load();

export const themeFamily = ref<ThemeFamily>(initial.family);
export const theme = ref<Theme>(initial.theme);

let installed = false;

/** Apply the persisted theme before mount and keep it applied on change. */
export function installTheme(): void {
  if (installed) return;
  installed = true;
  applyTheme(theme.value, themeFamily.value);
  installSystemThemeColorSync();
  watch([theme, themeFamily], ([nextTheme, nextFamily]) => {
    applyTheme(nextTheme, nextFamily);
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          family: nextFamily,
          theme: nextTheme,
        } satisfies PersistedTheme),
      );
    } catch {
      // Storage may be unavailable (private mode); the session still themes.
    }
  });
}

export type { Theme, ThemeFamily };

import { ref, watch } from "vue";
import {
  applyTheme,
  installSystemThemeSync,
  isThemeId,
  migrateLegacyTheme,
  type ThemeId,
} from "@ui/theme";

/*
 * Web-surface theme state on the shared Mold Studio contract (@ui/theme).
 * Fresh visitors default to Safelight — the look the web studio shipped with —
 * until the web redesign lands; a valid saved choice always wins, and the
 * pre-redesign `{ family, theme }` shape migrates through the shared table.
 */

const STORAGE_KEY = "mold.web.theme.v1";

interface PersistedTheme {
  theme: ThemeId;
  matchSystem: boolean;
}

const DEFAULT: PersistedTheme = { theme: "safelight", matchSystem: false };

function load(): PersistedTheme {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        const candidate = parsed as Record<string, unknown>;
        if (isThemeId(candidate.theme)) {
          return {
            theme: candidate.theme,
            matchSystem: candidate.matchSystem === true,
          };
        }
        if (candidate.theme !== undefined || candidate.family !== undefined) {
          return migrateLegacyTheme(candidate.theme, candidate.family);
        }
      }
    }
  } catch {
    // Ignore storage failures — fall through to defaults.
  }
  return { ...DEFAULT };
}

const initial = load();

export const theme = ref<ThemeId>(initial.theme);
export const matchSystem = ref<boolean>(initial.matchSystem);

let installed = false;

/** Apply the persisted theme before mount and keep it applied on change. */
export function installTheme(): void {
  if (installed) return;
  installed = true;
  applyTheme(theme.value, matchSystem.value);
  installSystemThemeSync(() => ({
    theme: theme.value,
    matchSystem: matchSystem.value,
  }));
  watch([theme, matchSystem], ([nextTheme, nextMatch]) => {
    applyTheme(nextTheme, nextMatch);
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          theme: nextTheme,
          matchSystem: nextMatch,
        } satisfies PersistedTheme),
      );
    } catch {
      // Storage may be unavailable (private mode); the session still themes.
    }
  });
}

export type { ThemeId };

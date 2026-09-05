import { defineStore } from "pinia";
import { reactive, ref } from "vue";

/**
 * The style each Create section was last used with — Still picture, Short
 * clip, 3-D object — and which section was used last.
 *
 * ONE memory for every surface, kept beside the sequence draft rather than in
 * any view: switching sections restores the style the person was using in the
 * section they are returning to, and a fresh launch opens on the style and
 * section they left. Persistence is localStorage on every surface, like the
 * draft (the Tauri webviews already trust it for durable state).
 *
 * It remembers NAMES, never availability: a machine that lacks the remembered
 * style gets the section's first style through `pick`, and the name survives
 * for the machine that has it.
 */
export type StyleSection = "still" | "clip" | "mesh";

export const LAST_USED_STYLES_KEY = "mold.create.lastUsedStyles.v1";

/** Minimal storage surface — `localStorage` in browsers/webviews, an
 * injected stub in tests. */
export interface LastUsedStylesStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

let storageOverride: LastUsedStylesStorage | null = null;

/** Test hook: inject a storage stub (pass null to restore the browser default). */
export function setLastUsedStylesStorage(
  storage: LastUsedStylesStorage | null,
) {
  storageOverride = storage;
}

function storage(): LastUsedStylesStorage | null {
  if (storageOverride) return storageOverride;
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

const SECTIONS: readonly StyleSection[] = ["still", "clip", "mesh"];

function isSection(value: unknown): value is StyleSection {
  return (
    typeof value === "string" && (SECTIONS as readonly string[]).includes(value)
  );
}

interface PersistedV1 {
  version: 1;
  bySection: Partial<Record<StyleSection, string | null>>;
  lastSection: StyleSection | null;
}

function emptyBySection(): Record<StyleSection, string | null> {
  return { still: null, clip: null, mesh: null };
}

function load(): {
  bySection: Record<StyleSection, string | null>;
  lastSection: StyleSection | null;
} {
  const bySection = emptyBySection();
  let lastSection: StyleSection | null = null;
  try {
    const raw = storage()?.getItem(LAST_USED_STYLES_KEY);
    if (!raw) return { bySection, lastSection };
    const parsed = JSON.parse(raw) as Partial<PersistedV1> | null;
    if (!parsed || typeof parsed !== "object" || parsed.version !== 1) {
      return { bySection, lastSection };
    }
    for (const section of SECTIONS) {
      const name = parsed.bySection?.[section];
      if (typeof name === "string" && name.length > 0)
        bySection[section] = name;
    }
    if (isSection(parsed.lastSection)) lastSection = parsed.lastSection;
  } catch {
    // Unreadable storage or a corrupt record reads as nothing remembered.
  }
  return { bySection, lastSection };
}

export const useLastUsedStylesStore = defineStore("lastUsedStyles", () => {
  const initial = load();
  const bySection = reactive<Record<StyleSection, string | null>>(
    initial.bySection,
  );
  const lastSection = ref<StyleSection | null>(initial.lastSection);

  function persist() {
    const record: PersistedV1 = {
      version: 1,
      bySection: { ...bySection },
      lastSection: lastSection.value,
    };
    try {
      storage()?.setItem(LAST_USED_STYLES_KEY, JSON.stringify(record));
    } catch {
      // Storage refused (quota, private mode): the session still remembers.
    }
  }

  /** Record `model` as the style `section` is being used with. */
  function remember(section: StyleSection, model: string | null | undefined) {
    const name = model?.trim() ?? "";
    if (!name) return;
    bySection[section] = name;
    lastSection.value = section;
    persist();
  }

  function forget(section: StyleSection) {
    bySection[section] = null;
    persist();
  }

  /**
   * The style to land on when entering `section`, from what is on offer:
   * the remembered one when the list has it, otherwise the first. Never
   * something the list does not hold.
   */
  function pick<M extends { name: string }>(
    section: StyleSection,
    candidates: readonly M[],
  ): M | null {
    const name = bySection[section];
    if (name) {
      const remembered = candidates.find(
        (candidate) => candidate.name === name,
      );
      if (remembered) return remembered;
    }
    return candidates[0] ?? null;
  }

  return { bySection, lastSection, remember, forget, pick };
});

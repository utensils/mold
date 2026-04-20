import { computed, ref, watch } from "vue";

/*
 * Tweaks control two visual directions — Studio (refined dark) and Lab
 * (editorial duotone) — plus a few user-facing knobs (density, accent,
 * chip visibility). Defaults match where the original design chat landed:
 * Studio + compact + amber accent.
 *
 * Everything is persisted to localStorage so the user's pick survives
 * reloads, and published back to `document.documentElement` as CSS
 * custom properties so the stylesheet can adapt without branching.
 */

export type Direction = "studio" | "lab";
export type Density = "compact" | "cozy" | "roomy";
export type StudioAccent = "violet" | "blue" | "amber" | "rose" | "green";
export type LabAccent =
  | "teal-magenta"
  | "amber-violet"
  | "crimson-gold"
  | "ice-rose";

export interface Tweaks {
  direction: Direction;
  density: Density;
  accentStudio: StudioAccent;
  accentLab: LabAccent;
  showChips: boolean;
}

const DEFAULTS: Tweaks = {
  direction: "studio",
  density: "compact",
  accentStudio: "amber",
  accentLab: "crimson-gold",
  showChips: true,
};

const STUDIO_HUES: Record<StudioAccent, number> = {
  violet: 268,
  blue: 232,
  amber: 60,
  rose: 10,
  green: 145,
};

const LAB_HUES: Record<LabAccent, [number, number]> = {
  "teal-magenta": [200, 330],
  "amber-violet": [60, 290],
  "crimson-gold": [18, 75],
  "ice-rose": [215, 355],
};

const STORAGE_KEY = "mold.tweaks";

function loadTweaks(): Tweaks {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<Tweaks>;
    return { ...DEFAULTS, ...parsed };
  } catch {
    return { ...DEFAULTS };
  }
}

const state = ref<Tweaks>(loadTweaks());

function applyToRoot(t: Tweaks) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.style.setProperty(
    "--studio-accent-h",
    String(STUDIO_HUES[t.accentStudio]),
  );
  const [h1, h2] = LAB_HUES[t.accentLab];
  root.style.setProperty("--lab-accent-h", String(h1));
  root.style.setProperty("--lab-accent-2-h", String(h2));
}

applyToRoot(state.value);

watch(
  state,
  (t) => {
    applyToRoot(t);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(t));
    } catch {
      /* ignore */
    }
  },
  { deep: true },
);

export function useTweaks() {
  const tweaks = state;
  const dirClass = computed(() =>
    tweaks.value.direction === "lab" ? "dir-lab" : "dir-studio",
  );

  function update(patch: Partial<Tweaks>) {
    state.value = { ...state.value, ...patch };
  }

  return { tweaks, dirClass, update };
}

export const STUDIO_SWATCHES: { key: StudioAccent; color: string }[] = [
  { key: "violet", color: "oklch(0.68 0.19 268)" },
  { key: "blue", color: "oklch(0.68 0.19 232)" },
  { key: "amber", color: "oklch(0.78 0.17 60)" },
  { key: "rose", color: "oklch(0.70 0.19 10)" },
  { key: "green", color: "oklch(0.72 0.18 145)" },
];

export const LAB_SWATCHES: { key: LabAccent; bg: string }[] = [
  {
    key: "teal-magenta",
    bg: "linear-gradient(120deg, oklch(0.78 0.14 200), oklch(0.72 0.22 330))",
  },
  {
    key: "amber-violet",
    bg: "linear-gradient(120deg, oklch(0.8 0.17 60), oklch(0.7 0.2 290))",
  },
  {
    key: "crimson-gold",
    bg: "linear-gradient(120deg, oklch(0.7 0.22 18), oklch(0.8 0.15 75))",
  },
  {
    key: "ice-rose",
    bg: "linear-gradient(120deg, oklch(0.8 0.12 215), oklch(0.75 0.17 355))",
  },
];

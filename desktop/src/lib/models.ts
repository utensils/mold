/**
 * Pure helpers for the Models screen: grouping installed models by family with
 * a separate utility split, quant-tag parsing, and the max-disk figure the
 * proportional usage bars scale against.
 */
import type { ModelEntry } from "./api/types";

/** Families that aren't image/video generators — grouped separately at the
 * bottom of the Installed tab. Mirrors `stores/models.ts`'s exclusion set. */
export const UTILITY_FAMILIES = new Set(["real-esrgan", "upscaler", "qwen3-expand", "controlnet"]);

export function isUtilityModel(m: ModelEntry): boolean {
  return UTILITY_FAMILIES.has(m.family);
}

/** Quant tag = the part after the first colon (`flux-dev:q8` → `q8`). */
export function quantTag(name: string): string | null {
  const i = name.indexOf(":");
  return i >= 0 ? name.slice(i + 1) : null;
}

export function modelDiskBytes(m: ModelEntry): number {
  return m.disk_usage_bytes ?? 0;
}

export interface InstalledGroups {
  /** Generation families, family name sorted, each list name-sorted. */
  families: [string, ModelEntry[]][];
  /** Non-generation (upscalers, expanders, controlnet), name-sorted. */
  utility: ModelEntry[];
  /** Largest on-disk footprint across all rows — the usage-bar denominator. */
  maxDiskBytes: number;
}

/**
 * Group installed models by family, splitting utility families out. Pure and
 * order-stable so the view (and its test) get a deterministic layout.
 */
export function groupInstalledModels(models: ModelEntry[]): InstalledGroups {
  const byFamily = new Map<string, ModelEntry[]>();
  const utility: ModelEntry[] = [];
  let maxDiskBytes = 0;

  for (const m of models) {
    maxDiskBytes = Math.max(maxDiskBytes, modelDiskBytes(m));
    if (isUtilityModel(m)) {
      utility.push(m);
      continue;
    }
    const list = byFamily.get(m.family) ?? [];
    list.push(m);
    byFamily.set(m.family, list);
  }

  const byName = (a: ModelEntry, b: ModelEntry) => a.name.localeCompare(b.name);
  const families = [...byFamily.entries()]
    .map(([family, list]) => [family, list.slice().sort(byName)] as [string, ModelEntry[]])
    .sort((a, b) => a[0].localeCompare(b[0]));
  utility.sort(byName);

  return { families, utility, maxDiskBytes };
}

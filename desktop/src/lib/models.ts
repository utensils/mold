/**
 * Pure helpers for the Models screen: grouping installed models by family with
 * a separate utility split, quant-tag parsing, and the max-disk figure the
 * proportional usage bars scale against.
 */
import type { ModelEntry } from "./api/types";
import { formatGB } from "./format";

/** Families that aren't image/video generators — grouped separately at the
 * bottom of the Installed tab. Mirrors `stores/models.ts`'s exclusion set. */
export const UTILITY_FAMILIES = new Set(["real-esrgan", "upscaler", "qwen3-expand", "controlnet"]);

export function isUtilityModel(m: ModelEntry): boolean {
  return UTILITY_FAMILIES.has(m.family);
}

/** True for opaque catalog-install identifiers (`cv:<id>` / `hf:<repo>`). */
export function isCatalogModelId(name: string): boolean {
  return name.startsWith("cv:") || name.startsWith("hf:");
}

/** Quant tag = the part after the first colon (`flux-dev:q8` → `q8`).
 * Catalog ids (`cv:252914`) carry no quant variant — their colon is part
 * of the identifier, not a `base:tag` split. */
export function quantTag(name: string): string | null {
  if (isCatalogModelId(name)) return null;
  const i = name.indexOf(":");
  return i >= 0 ? name.slice(i + 1) : null;
}

/**
 * Human-readable label for a model row. Catalog installs are identified by
 * opaque `cv:`/`hf:` ids; the server sends the upstream title as an additive
 * `display_name`, and older servers still embed it in `description`. Display
 * only — selection, keys, and API calls must keep using `name`.
 */
export function modelDisplayName(
  m: Pick<ModelEntry, "name"> & Partial<Pick<ModelEntry, "display_name" | "description">>,
): string {
  if (m.display_name) return m.display_name;
  if (isCatalogModelId(m.name) && m.description) return m.description;
  return m.name;
}

export function modelDiskBytes(m: ModelEntry): number {
  return m.disk_usage_bytes ?? 0;
}

export interface ModelSizeLabels {
  /** Primary model weights, excluding shared encoders/VAEs. */
  weights: string | null;
  /** Full runtime footprint when materially larger than the primary weights. */
  runtime: string | null;
}

/**
 * Keep the two server size concepts explicit. `size_gb` is the primary model
 * weights; `disk_usage_bytes` includes every runtime file referenced by that
 * model and can therefore include large shared encoders and VAEs.
 */
export function modelSizeLabels(m: ModelEntry): ModelSizeLabels {
  const weightsBytes = m.size_gb > 0 ? m.size_gb * 1_000_000_000 : 0;
  const runtimeBytes = modelDiskBytes(m);
  const weights = weightsBytes > 0 ? `${m.size_gb.toFixed(1)} GB weights` : null;
  const differs = runtimeBytes > 0 && Math.abs(runtimeBytes - weightsBytes) >= 50_000_000;
  return {
    weights,
    runtime: differs ? `${formatGB(runtimeBytes)} on disk` : null,
  };
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

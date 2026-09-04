/**
 * Pure helpers for the Models screen: grouping installed models by family with
 * a separate utility split, quant-tag parsing, and the max-disk figure the
 * proportional usage bars scale against.
 */
import type { ModelEntry } from "./api/types";
import { formatGB } from "./format";
import {
  isCatalogModelId,
  isOpaqueModelId,
  modelDisplayName,
  modelDisplayNameForId,
  type DisplayableModel,
} from "@studio/lib/modelDisplay";

export {
  isCatalogModelId,
  isOpaqueModelId,
  modelDisplayName,
  modelDisplayNameForId,
  type DisplayableModel,
};

/** Families that aren't image/video generators — grouped separately at the
 * bottom of the Installed tab. Mirrors `stores/models.ts`'s exclusion set. */
export const UTILITY_FAMILIES = new Set(["real-esrgan", "upscaler", "qwen3-expand", "controlnet"]);

function hasText(value: string | null | undefined): value is string {
  return Boolean(value?.trim());
}

function isSyntheticDescription(
  description: string | null | undefined,
  title: string | null | undefined,
): boolean {
  if (!hasText(description) || !hasText(title)) return false;
  const normalized = description.trim().toLocaleLowerCase();
  const normalizedTitle = title.trim().toLocaleLowerCase();
  const bylinePrefix = `${normalizedTitle} by `;
  return (
    normalized === normalizedTitle ||
    (normalized.startsWith(bylinePrefix) && normalized.length > bylinePrefix.length)
  );
}

/**
 * Keep the first host's runtime/default fields while filling presentation
 * metadata that an older `/api/models` row omitted. Mature classification is
 * monotonic: one host reporting NSFW must label the all-host row as NSFW.
 */
export function mergeModelPresentationMetadata<T extends ModelEntry>(
  preferred: T,
  supplement: ModelEntry,
): T {
  const displayName = hasText(preferred.display_name)
    ? preferred.display_name
    : supplement.display_name;
  const title =
    (hasText(displayName) ? displayName : null) ??
    (!isCatalogModelId(preferred.name) ? preferred.name : null);
  const preferredDescriptionIsUseful =
    hasText(preferred.description) && !isSyntheticDescription(preferred.description, title);
  const supplementDescriptionIsUseful =
    hasText(supplement.description) && !isSyntheticDescription(supplement.description, title);
  const description = preferredDescriptionIsUseful
    ? preferred.description
    : supplementDescriptionIsUseful
      ? supplement.description
      : (preferred.description ?? supplement.description);

  return {
    ...preferred,
    display_name: displayName,
    kind: hasText(preferred.kind) ? preferred.kind : supplement.kind,
    modality: hasText(preferred.modality) ? preferred.modality : supplement.modality,
    description,
    nsfw:
      preferred.nsfw === true || supplement.nsfw === true
        ? true
        : (preferred.nsfw ?? supplement.nsfw),
  };
}

export function isUtilityModel(m: ModelEntry): boolean {
  return UTILITY_FAMILIES.has(m.family);
}

/** Quant tag = the part after the first colon (`flux-dev:q8` → `q8`).
 * Catalog ids (`cv:252914`) carry no quant variant — their colon is part
 * of the identifier, not a `base:tag` split. */

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

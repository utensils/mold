import { familyLabel } from "@studio/lib/modelFamily";
import { isGenerationModel } from "@studio/lib/generationModels";

import type { ModelInfoExtended } from "../types";

export type ModelFilterMode = "all" | "any" | "not";
export type DownloadedFilter = "any" | "downloaded" | "missing";
export type SizeBucket = "small" | "medium" | "large" | "xlarge";
export type ModelSortKey =
  "name" | "family" | "size" | "quantization" | "downloaded" | "variantRank";

export interface ModelFilterState {
  mode: ModelFilterMode;
  query: string;
  families: string[];
  quantizations: string[];
  sizeBuckets: SizeBucket[];
  downloaded: DownloadedFilter;
}

export interface ModelFamilyGroup {
  family: string;
  label: string;
  models: ModelInfoExtended[];
}

const FAMILY_ORDER = [
  "flux",
  "flux2",
  "flux-2",
  "sd15",
  "sd1.5",
  "sdxl",
  "sd3",
  "sd3.5",
  "z-image",
  "qwen-image",
  "qwen-image-edit",
  "wuerstchen",
  "ltx-video",
  "ltx2",
  "ltx-2",
  "wan",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
];

const VARIANT_RANKS: Record<string, number> = {
  bf16: 0,
  fp16: 1,
  q8: 2,
  fp8: 3,
  q6: 4,
  q5: 5,
  q4: 6,
};

/**
 * Whether a row is a style a person can pick.
 *
 * The list lives in `@studio/lib/generationModels` — this surface and desktop
 * each had their own and they had already drifted (this one knew `companion`
 * and `control-net`, desktop knew `real-esrgan`), which is how the 3-D Studio
 * came to offer the prompt expander as a picture style.
 */
export function isStandaloneGenerationModel(model: ModelInfoExtended): boolean {
  return isGenerationModel(model);
}

export function modelQuantization(model: ModelInfoExtended): string {
  const lower = model.name.toLowerCase();
  const colonVariant = lower.match(/:([a-z0-9.-]+)$/)?.[1];
  if (colonVariant) return normalizeQuantization(colonVariant);
  const suffixVariant = lower.match(/[-_](bf16|fp16|fp8|q[0-9])\b/)?.[1];
  return suffixVariant ? normalizeQuantization(suffixVariant) : "unknown";
}

function normalizeQuantization(value: string): string {
  const match = value.match(/^(bf16|fp16|fp8|q[0-9])/);
  return match?.[1] ?? value;
}

export function variantRank(model: ModelInfoExtended): number {
  return VARIANT_RANKS[modelQuantization(model)] ?? 999;
}

export function sizeBucket(model: ModelInfoExtended): SizeBucket {
  if (model.size_gb < 5) return "small";
  if (model.size_gb < 10) return "medium";
  if (model.size_gb < 20) return "large";
  return "xlarge";
}

export function applyModelFilters(
  models: ModelInfoExtended[],
  filters: ModelFilterState,
): ModelInfoExtended[] {
  const predicates = activePredicates(filters);
  if (predicates.length === 0) return models.slice();

  return models.filter((model) => {
    const results = predicates.map((predicate) => predicate(model));
    if (filters.mode === "any") return results.some(Boolean);
    if (filters.mode === "not") return results.every((matched) => !matched);
    return results.every(Boolean);
  });
}

function activePredicates(
  filters: ModelFilterState,
): Array<(model: ModelInfoExtended) => boolean> {
  const predicates: Array<(model: ModelInfoExtended) => boolean> = [];
  const query = filters.query.trim().toLowerCase();
  if (query) {
    predicates.push((model) =>
      [
        model.name,
        model.display_name ?? "",
        model.family,
        model.description,
        model.hf_repo,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }
  if (filters.families.length > 0) {
    const families = new Set(filters.families);
    predicates.push((model) => families.has(model.family));
  }
  if (filters.quantizations.length > 0) {
    const quantizations = new Set(filters.quantizations);
    predicates.push((model) => quantizations.has(modelQuantization(model)));
  }
  if (filters.sizeBuckets.length > 0) {
    const buckets = new Set(filters.sizeBuckets);
    predicates.push((model) => buckets.has(sizeBucket(model)));
  }
  if (filters.downloaded !== "any") {
    const wantDownloaded = filters.downloaded === "downloaded";
    predicates.push((model) => model.downloaded === wantDownloaded);
  }
  return predicates;
}

export function sortModels(
  models: ModelInfoExtended[],
  keys: ModelSortKey[],
): ModelInfoExtended[] {
  const indexed = models.map((model, index) => ({ model, index }));
  indexed.sort((a, b) => {
    for (const key of keys) {
      const cmp = compareByKey(a.model, b.model, key);
      if (cmp !== 0) return cmp;
    }
    return a.index - b.index;
  });
  return indexed.map((entry) => entry.model);
}

function compareByKey(
  a: ModelInfoExtended,
  b: ModelInfoExtended,
  key: ModelSortKey,
): number {
  if (key === "name") return a.name.localeCompare(b.name);
  if (key === "family") return compareFamilies(a.family, b.family);
  if (key === "size") return a.size_gb - b.size_gb;
  if (key === "quantization") {
    return modelQuantization(a).localeCompare(modelQuantization(b));
  }
  if (key === "downloaded") {
    return Number(b.downloaded) - Number(a.downloaded);
  }
  return variantRank(a) - variantRank(b);
}

export function groupModelsByFamily(
  models: ModelInfoExtended[],
): ModelFamilyGroup[] {
  const map = new Map<string, ModelInfoExtended[]>();
  for (const model of models) {
    const existing = map.get(model.family) ?? [];
    existing.push(model);
    map.set(model.family, existing);
  }

  return Array.from(map.entries())
    .sort(([a], [b]) => compareFamilies(a, b))
    .map(([family, familyModels]) => ({
      family,
      label: familyLabel(family),
      models: familyModels,
    }));
}

// The label table is shared with desktop and iPhone (#806); re-exported here
// so every existing `modelFilters` consumer keeps its import.
export { familyLabel };

function compareFamilies(a: string, b: string): number {
  const ai = FAMILY_ORDER.indexOf(a);
  const bi = FAMILY_ORDER.indexOf(b);
  if (ai >= 0 && bi >= 0) return ai - bi;
  if (ai >= 0) return -1;
  if (bi >= 0) return 1;
  return a.localeCompare(b);
}

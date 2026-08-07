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

const NON_GENERATION_FAMILIES = new Set([
  "upscaler",
  "qwen3-expand",
  "companion",
  "controlnet",
  "control-net",
]);

const SUPPORT_NAME_PATTERNS = [
  /\bupscal(e|er|ing)\b/i,
  /\breal[-_ ]?esrgan\b/i,
  /\bcontrol[-_ ]?net\b/i,
  /\b(qwen3|prompt)[-_ ]?expand/i,
  /\bclip[-_ ]?[lg]?\b/i,
  /\btext[-_ ]?encoder\b/i,
  /\btokenizer\b/i,
  /\bvae\b/i,
];

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

const FAMILY_LABELS: Record<string, string> = {
  flux: "FLUX",
  flux2: "Flux.2",
  "flux-2": "Flux.2",
  sd15: "SD 1.5",
  "sd1.5": "SD 1.5",
  sdxl: "SDXL",
  sd3: "SD3",
  "sd3.5": "SD3.5",
  "z-image": "Z-Image",
  "qwen-image": "Qwen Image",
  "qwen-image-edit": "Qwen Image Edit",
  wuerstchen: "Wuerstchen",
  "ltx-video": "LTX Video",
  ltx2: "LTX-2",
  "ltx-2": "LTX-2",
  wan: "Wan Video",
  "minimax-h3": "MiniMax H3",
  minimax_h3: "MiniMax H3",
  minimaxh3: "MiniMax H3",
};

const VARIANT_RANKS: Record<string, number> = {
  bf16: 0,
  fp16: 1,
  q8: 2,
  fp8: 3,
  q6: 4,
  q5: 5,
  q4: 6,
};

export function isStandaloneGenerationModel(model: ModelInfoExtended): boolean {
  const family = model.family.toLowerCase();
  if (NON_GENERATION_FAMILIES.has(family)) return false;

  const searchable = [model.name, model.description, model.hf_repo]
    .join(" ")
    .toLowerCase();
  return !SUPPORT_NAME_PATTERNS.some((pattern) => pattern.test(searchable));
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

export function familyLabel(family: string): string {
  return (
    FAMILY_LABELS[family] ??
    family
      .split(/[-_]/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

function compareFamilies(a: string, b: string): number {
  const ai = FAMILY_ORDER.indexOf(a);
  const bi = FAMILY_ORDER.indexOf(b);
  if (ai >= 0 && bi >= 0) return ai - bi;
  if (ai >= 0) return -1;
  if (bi >= 0) return 1;
  return a.localeCompare(b);
}

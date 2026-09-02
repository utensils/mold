export interface DisplayableModel {
  name: string;
  display_name?: string | null;
  description?: string | null;
}

/** True for opaque catalog-install identifiers (`cv:<id>` / `hf:<repo>`). */
export function isCatalogModelId(name: string): boolean {
  return name.startsWith("cv:") || name.startsWith("hf:");
}

/** Built-in H3 manifests predate the additive `display_name` wire field.
 * Keep their stable request ids while giving every Studio surface the same
 * task/layout label, including acquisition-only rows. */
const MINIMAX_H3_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  "minimax-h3-fl2va:official-bf16": "MiniMax H3 FL2VA · Official BF16",
  "minimax-h3-ref2va:official-bf16": "MiniMax H3 Ref2VA · Official BF16",
  "minimax-h3-fl2va:comfy-pruned-int8": "MiniMax H3 FL2VA",
  "minimax-h3-ref2va:comfy-pruned-int8": "MiniMax H3 Ref2VA",
  "minimax-h3-fl2va:comfy-pruned-nvfp4": "MiniMax H3 FL2VA · NVFP4",
  "minimax-h3-ref2va:comfy-pruned-nvfp4": "MiniMax H3 Ref2VA · NVFP4",
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step":
    "MiniMax H3 FL2VA Turbo 8-step",
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p":
    "MiniMax H3 FL2VA Turbo 4-step 768p",
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1":
    "MiniMax H3 FL2VA Turbo 4-step 768p v1.1",
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p":
    "MiniMax H3 FL2VA Turbo 8-step 768p",
  "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step":
    "MiniMax H3 Ref2VA Turbo 4-step",
};

/**
 * Catalog models use an opaque id as their runnable `name`. Keep that id in
 * form values and requests, but prefer the catalog-provided description in UI
 * labels so people see the actual model name.
 */
export function modelDisplayName(model: DisplayableModel): string {
  const displayName = model.display_name?.trim();
  if (displayName) return displayName;
  const builtInDisplayName = MINIMAX_H3_DISPLAY_NAMES[model.name.toLowerCase()];
  if (builtInDisplayName) return builtInDisplayName;
  if (isCatalogModelId(model.name)) {
    const description = model.description?.trim();
    if (description) return description;
  }
  if (model.name.startsWith("cv:")) {
    return `Civitai model #${model.name.slice(3)}`;
  }
  if (model.name.startsWith("hf:")) {
    const repo = model.name.slice(3).split("/").pop() ?? model.name.slice(3);
    return repo.replaceAll("-", " ").replaceAll("_", " ");
  }
  return model.name;
}

/** Resolve a model id carried by queue/status/download/history wire records. */
export function modelDisplayNameForId(
  name: string,
  models: readonly DisplayableModel[],
): string {
  return modelDisplayName(
    models.find((model) => model.name === name) ?? { name },
  );
}

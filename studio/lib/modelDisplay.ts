export interface DisplayableModel {
  name: string;
  display_name?: string | null;
  description?: string | null;
}

/** True for opaque catalog-install identifiers (`cv:<id>` / `hf:<repo>`). */
export function isCatalogModelId(name: string): boolean {
  return name.startsWith("cv:") || name.startsWith("hf:");
}

/**
 * Catalog models use an opaque id as their runnable `name`. Keep that id in
 * form values and requests, but prefer the catalog-provided description in UI
 * labels so people see the actual model name.
 */
export function modelDisplayName(model: DisplayableModel): string {
  const displayName = model.display_name?.trim();
  if (displayName) return displayName;
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

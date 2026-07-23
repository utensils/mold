export interface DisplayableModel {
  name: string;
  display_name?: string | null;
  description?: string | null;
}

/**
 * Catalog models use an opaque id as their runnable `name`. Keep that id in
 * form values and requests, but prefer the catalog-provided description in UI
 * labels so people see the actual model name.
 */
export function modelDisplayName(model: DisplayableModel): string {
  const displayName = model.display_name?.trim();
  if (displayName) return displayName;
  if (model.name.startsWith("cv:") || model.name.startsWith("hf:")) {
    const description = model.description?.trim();
    if (description) return description;
  }
  return model.name;
}

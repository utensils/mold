/**
 * Human-readable model titles. Catalog installs are identified by opaque
 * `cv:<id>` / `hf:<repo>` names; the server sends the upstream title as an
 * additive `display_name` field on `/api/models` rows, and older servers
 * still embed it in `description`. Display only — selection, keys, and API
 * calls must keep using `name`.
 */

/** True for opaque catalog-install identifiers (`cv:<id>` / `hf:<repo>`). */
export function isCatalogModelId(name: string): boolean {
  return name.startsWith("cv:") || name.startsWith("hf:");
}

/** Human-readable label for a model row. */
export function modelDisplayName(m: {
  name: string;
  display_name?: string | null;
  description?: string | null;
}): string {
  const displayName = m.display_name?.trim();
  if (displayName) return displayName;
  const description = m.description?.trim();
  if (isCatalogModelId(m.name) && description) return description;
  if (m.name.startsWith("cv:")) return `Civitai model #${m.name.slice(3)}`;
  if (m.name.startsWith("hf:")) {
    const repo = m.name.slice(3).split("/").pop() ?? m.name.slice(3);
    return repo.replaceAll("-", " ").replaceAll("_", " ");
  }
  return m.name;
}

export function modelDisplayNameForId(
  name: string,
  models: readonly {
    name: string;
    display_name?: string | null;
    description?: string | null;
  }[],
): string {
  return modelDisplayName(
    models.find((model) => model.name === name) ?? { name },
  );
}

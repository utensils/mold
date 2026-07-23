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
  description?: string;
}): string {
  if (m.display_name) return m.display_name;
  if (isCatalogModelId(m.name) && m.description) return m.description;
  return m.name;
}

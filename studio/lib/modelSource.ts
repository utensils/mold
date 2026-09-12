/**
 * Where a model came from, for the source glyph in pickers. Catalog installs
 * are named `cv:<versionId>` / `hf:<repoId>` in `GET /api/models`; built-in
 * manifest models carry their upstream repo in `hf_repo`.
 */
export type ModelSource = "hf" | "civitai" | "local";

export function modelSource(model: {
  name: string;
  hf_repo?: string | null;
}): ModelSource {
  if (model.name.startsWith("cv:")) return "civitai";
  if (model.name.startsWith("hf:")) return "hf";
  if (model.hf_repo) return "hf";
  return "local";
}

/**
 * Narrows a catalog entry's own `source` field to a glyph. `CatalogEntryWire`
 * types it `"hf" | "civitai"` on the wire, but the desktop/mobile
 * `CatalogEntry` union (which also carries an installed row already
 * classified through `modelSource` above, `"local"` included) types the same
 * field as a plain `string` — so a caller trusting either type without a
 * runtime guard has nothing stopping a value it did not expect. This is the
 * ONE guard for a wire/union `source` string, used everywhere one becomes a
 * glyph, so the two call sites cannot silently diverge.
 */
export function wireSourceGlyph(source: string): ModelSource {
  return source === "civitai" || source === "hf" ? source : "local";
}

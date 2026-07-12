/**
 * Where a model came from, for the source glyph in pickers. Catalog installs
 * are named `cv:<versionId>` / `hf:<repoId>` in `GET /api/models`; built-in
 * manifest models carry their upstream repo in `hf_repo`.
 */
export type ModelSource = "hf" | "civitai" | "local";

export function modelSource(model: { name: string; hf_repo?: string | null }): ModelSource {
  if (model.name.startsWith("cv:")) return "civitai";
  if (model.name.startsWith("hf:")) return "hf";
  if (model.hf_repo) return "hf";
  return "local";
}

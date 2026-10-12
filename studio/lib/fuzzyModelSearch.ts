/** Shared model visibility/ranking policy. Keep fixtures aligned with MoldClient. */
import { modelPresentation } from "./modelDisplay";
import { familyLabel } from "./modelFamily";
import type { ModelPresentationFields } from "./api/serverTypes";

import { fuzzyTextScore as fuzzyModelScore } from "@ui/lib/fuzzySearch";
export { fuzzyTextScore as fuzzyModelScore } from "@ui/lib/fuzzySearch";

export interface FuzzySearchModel extends ModelPresentationFields {
  name: string;
  family?: string | null;
  description?: string | null;
  hf_repo?: string | null;
  source_id?: string | null;
  author?: string | null;
  tags?: string[] | null;
}

export function modelSearchScore(
  model: FuzzySearchModel,
  query: string,
): number {
  const shown = modelPresentation(model);
  return fuzzyModelScore(
    query,
    [
      model.name,
      shown.displayName,
      shown.title,
      model.variant ?? "",
      model.family ?? "",
      model.family_label ?? familyLabel(model.family ?? ""),
      model.hf_repo ?? model.source_id ?? "",
      model.author ?? "",
    ],
    [shown.summary ?? "", model.description ?? "", ...(model.tags ?? [])],
  );
}

/** Stable relevance order; an empty query retains the host's existing order. */
export function rankModels<T extends FuzzySearchModel>(
  models: readonly T[],
  query: string,
): T[] {
  if (!query.trim()) return [...models];
  return models
    .map((model, index) => ({
      model,
      index,
      score: modelSearchScore(model, query),
    }))
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((row) => row.model);
}

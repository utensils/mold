import { outputKindForModel, type OutputKind } from "../composables/useCreateOutputKind";

export type MediaType = "all" | "image" | "video" | "mesh";

/**
 * Media-type filter for the unified Catalog view. Legacy deep links from the
 * split Models screen (`?tab=catalog`, `?availability=*`) are tolerated and
 * fall back to the unfiltered view.
 */
export function mediaTypeFromQuery(query: Record<string, unknown>): MediaType {
  const type = query.type;
  if (type === "image" || type === "video" || type === "mesh") return type;
  return "all";
}

/**
 * Each filter value is one Create section. The `?type=` values are the wire
 * the deep links already carry (`image` / `video`), so they stay, and the
 * partition behind them is `outputKindForModel` — the SAME one the Create
 * toolbar's Still picture | Short clip | 3-D object control sorts styles by —
 * so a style filters under exactly the kind it is offered under.
 */
export const MEDIA_TYPE_KIND: Readonly<Record<Exclude<MediaType, "all">, OutputKind>> = {
  image: "still",
  video: "clip",
  mesh: "mesh",
};

/** True when a style of `family` passes the active kind filter. */
export function mediaTypeMatches(type: MediaType, family: string): boolean {
  if (type === "all") return true;
  return outputKindForModel({ family }) === MEDIA_TYPE_KIND[type];
}

export type MediaType = "all" | "image" | "video";

/**
 * Media-type filter for the unified Catalog view. Legacy deep links from the
 * split Models screen (`?tab=catalog`, `?availability=*`) are tolerated and
 * fall back to the unfiltered view.
 */
export function mediaTypeFromQuery(query: Record<string, unknown>): MediaType {
  const type = query.type;
  if (type === "image" || type === "video") return type;
  return "all";
}

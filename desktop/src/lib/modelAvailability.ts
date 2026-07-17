export type ModelAvailability = "all" | "installed" | "available";

/** Preserve old catalog deep links while supporting the unified-library query. */
export function modelAvailabilityFromQuery(query: Record<string, unknown>): ModelAvailability {
  const availability = query.availability;
  if (availability === "installed" || availability === "available") return availability;
  if (query.tab === "catalog") return "available";
  return "all";
}

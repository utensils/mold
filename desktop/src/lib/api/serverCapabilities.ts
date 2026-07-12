import { apiJson } from "./client";
import type { ServerCapabilities } from "./types";

/**
 * `GET /api/capabilities` — feature toggles for the connected engine.
 * Callers must treat missing groups (e.g. `events` on older servers) as
 * "unavailable" rather than erroring.
 */
export async function fetchServerCapabilities(): Promise<ServerCapabilities> {
  return apiJson<ServerCapabilities>("/api/capabilities");
}

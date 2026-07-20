import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { ServerCapabilities } from "./types";

/**
 * `GET /api/capabilities` — feature toggles for the connected engine.
 * Missing expansion capability on an older server means unknown, not
 * unavailable. An explicit target preserves that host's API key.
 */
export async function fetchServerCapabilities(target?: ApiTarget): Promise<ServerCapabilities> {
  return target
    ? apiJsonTo<ServerCapabilities>(target, "/api/capabilities")
    : apiJson<ServerCapabilities>("/api/capabilities");
}

import { apiJson } from "./client";
import type { ExpandRequest, ExpandResponse } from "./types";

/**
 * Expand a short prompt into a detailed one. `model_family` steers the style;
 * the desktop only ever asks for a single variation. Throws `ApiError` with
 * status 404/503 when the expansion model isn't installed or the backend is
 * unavailable — callers surface that as a quiet toast.
 */
export function expandPrompt(prompt: string, modelFamily?: string): Promise<ExpandResponse> {
  const body: ExpandRequest = { prompt, variations: 1 };
  if (modelFamily) body.model_family = modelFamily;
  return apiJson<ExpandResponse>("/api/expand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

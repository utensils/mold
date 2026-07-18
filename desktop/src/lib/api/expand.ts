import { apiJson } from "./client";
import type { ExpandRequest, ExpandResponse } from "./types";

export interface ExpandPromptOptions {
  /** Steers the expansion style; omitted → the server's default family. */
  modelFamily?: string;
  /** Number of prompt candidates to return (server default 1). */
  variations?: number;
}

/**
 * Expand a short prompt into one or more detailed candidates. The one-shot
 * ⌘E path asks for a single variation; the chooser popover requests 1/3/5
 * and previews `expanded[]` before anything touches the prompt. Throws
 * `ApiError` with status 404/503 when the expansion model isn't installed or
 * the backend is unavailable — callers surface that as a quiet toast.
 */
export function expandPrompt(
  prompt: string,
  opts: ExpandPromptOptions = {},
): Promise<ExpandResponse> {
  const body: ExpandRequest = { prompt, variations: opts.variations ?? 1 };
  const family = opts.modelFamily?.trim();
  if (family) body.model_family = family;
  return apiJson<ExpandResponse>("/api/expand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

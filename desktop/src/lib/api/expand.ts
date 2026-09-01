import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { ExpandContext, ExpandRequest, ExpandResponse, ExpandTask } from "./types";

export interface ExpandPromptOptions {
  /** Steers the expansion style; omitted → the server's default family. */
  modelFamily?: string;
  /** Number of prompt candidates to return (server default 1). */
  variations?: number;
  /**
   * Natural-language style directive (see `styleHint`) the server appends to
   * the expander's system message so the look is woven into the rewrite —
   * never the literal preset suffix, and never appended to the prompt text.
   */
  style?: string;
  /** Resolved generation/conditioning policy. */
  task?: ExpandTask;
  /** Identity, canvas, clip, and ordered references of the target print. */
  context?: ExpandContext;
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
  target?: ApiTarget | null,
): Promise<ExpandResponse> {
  const body: ExpandRequest = { prompt, variations: opts.variations ?? 1 };
  const family = opts.modelFamily?.trim();
  if (family) body.model_family = family;
  const style = opts.style?.trim();
  if (style) body.style = style;
  if (opts.task) body.task = opts.task;
  if (opts.context) body.context = opts.context;
  const init = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  return target
    ? apiJsonTo<ExpandResponse>(target, "/api/expand", init)
    : apiJson<ExpandResponse>("/api/expand", init);
}

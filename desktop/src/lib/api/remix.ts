import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { RemixRequest, RemixResponse } from "./types";

export function remixPrompt(
  request: RemixRequest,
  target?: ApiTarget | null,
): Promise<RemixResponse> {
  const init = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  };
  return target
    ? apiJsonTo<RemixResponse>(target, "/api/remix", init)
    : apiJson<RemixResponse>("/api/remix", init);
}

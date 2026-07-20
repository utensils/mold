import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { LoraInfo } from "./types";

/**
 * Installed LoRA adapters, optionally filtered to the family compatible with
 * `model`. The server returns `[]` (not an error) for a model whose family
 * can't take LoRAs; an unknown model 400s.
 */
export function fetchLoras(model?: string, target?: ApiTarget | null): Promise<LoraInfo[]> {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  const path = `/api/loras${query}`;
  return target ? apiJsonTo<LoraInfo[]>(target, path) : apiJson<LoraInfo[]>(path);
}

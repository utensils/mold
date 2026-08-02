import { apiJsonTo, type ApiTarget } from "./client";
import type {
  ChainRequestWire,
  ChainValidationResponse,
} from "../lib/api/chainTypes";

/** Normalize and validate a sequence on its exact rendering host without
 * creating a durable job or starting downloads. */
export function validateChain(
  request: ChainRequestWire,
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<ChainValidationResponse> {
  return apiJsonTo<ChainValidationResponse>(
    target,
    "/api/generate/chain/validate",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      ...(signal ? { signal } : {}),
    },
  );
}

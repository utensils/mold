import type { GenerateRequestWire } from "../types";
import {
  GENERATION_MEDIA_AUTHORITY_FIELDS,
  requestCarriesGenerationMedia as sharedRequestCarriesGenerationMedia,
} from "@studio/lib/generationMedia";

/** Every GenerateRequest wire field that carries media bytes, a server-local
 * media path, or temporary media authority. Durable admission stores the
 * request JSON, so any present field here must stay on the legacy request
 * lifecycle until queue-owned encrypted staging exists. */
export const GENERATION_REQUEST_MEDIA_FIELDS =
  GENERATION_MEDIA_AUTHORITY_FIELDS;

/** Null/undefined are the normal wire representation for an unused optional
 * field. Any other value is conservatively media-bearing, including an empty
 * array or string supplied by a nonstandard caller. */
export function requestCarriesGenerationMedia(
  request: GenerateRequestWire,
): boolean {
  return sharedRequestCarriesGenerationMedia(request);
}

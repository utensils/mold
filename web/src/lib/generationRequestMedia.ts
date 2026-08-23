import type { GenerateRequestWire } from "../types";

/** Every GenerateRequest wire field that carries media bytes, a server-local
 * media path, or temporary media authority. Durable admission stores the
 * request JSON, so any present field here must stay on the legacy request
 * lifecycle until queue-owned encrypted staging exists. */
export const GENERATION_REQUEST_MEDIA_FIELDS = [
  "source_image",
  "edit_images",
  "references",
  "id_image",
  "mask_image",
  "control_image",
  "audio_file",
  "audio_file_path",
  "source_video",
  "source_video_path",
  "extend_video",
  "extend_video_path",
  "keyframes",
] as const satisfies readonly (keyof GenerateRequestWire)[];

/** Null/undefined are the normal wire representation for an unused optional
 * field. Any other value is conservatively media-bearing, including an empty
 * array or string supplied by a nonstandard caller. */
export function requestCarriesGenerationMedia(
  request: GenerateRequestWire,
): boolean {
  return GENERATION_REQUEST_MEDIA_FIELDS.some(
    (field) => request[field] !== undefined && request[field] !== null,
  );
}

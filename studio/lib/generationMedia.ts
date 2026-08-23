/**
 * Wire fields that carry media bytes, a server-local media path, or temporary
 * upload authority. Any non-null value keeps the request on the attached,
 * session-only lifecycle until encrypted durable media staging exists.
 */
export const GENERATION_MEDIA_AUTHORITY_FIELDS = [
  "source_image",
  "edit_images",
  "references",
  "id_image",
  "id_images",
  "mask_image",
  "control_image",
  "audio_file",
  "audio_file_path",
  "source_video",
  "source_video_path",
  "extend_video",
  "extend_video_path",
  "keyframes",
  "hdr_exr_dir",
] as const;

const GENERATION_PRIVATE_PERSISTENCE_FIELDS = new Set<string>([
  ...GENERATION_MEDIA_AUTHORITY_FIELDS,
  // Identity labels and controls describe a biometric input even without
  // carrying the photo bytes themselves.
  "id_image_name",
  "id_weight",
  "id_start_step",
]);

export function requestCarriesGenerationMedia(request: object): boolean {
  const record = request as Record<string, unknown>;
  return GENERATION_MEDIA_AUTHORITY_FIELDS.some(
    (field) => record[field] !== undefined && record[field] !== null,
  );
}

function redactRecord(
  record: Record<string, unknown>,
): Record<string, unknown> {
  const redacted: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (!GENERATION_PRIVATE_PERSISTENCE_FIELDS.has(key)) {
      redacted[key] = value;
    }
  }
  return redacted;
}

/**
 * Produce the byte-free presentation/recovery projection before JSON
 * serialization. Media values are dropped by key without inspecting them, so
 * work is proportional to ordinary request metadata rather than media bytes.
 */
export function redactGenerationMediaForPersistence<T extends object>(
  request: T,
): T {
  const redacted = redactRecord(request as Record<string, unknown>);
  if (Array.isArray(redacted.stages)) {
    redacted.stages = redacted.stages.map((stage) =>
      typeof stage === "object" && stage !== null && !Array.isArray(stage)
        ? redactRecord(stage as Record<string, unknown>)
        : stage,
    );
  }
  return redacted as T;
}

import { sha256 } from "@noble/hashes/sha2.js";
import type {
  OriginalVideo,
  VideoNormalizationCapabilities,
} from "../api/referenceNormalizationTypes";
/** Local authoring state, deliberately separate from canonical GenerationReference. */
export interface FreshVideoReference {
  readonly kind: "fresh_video";
  /** A pending candidate cannot accidentally carry a canonical authority. */
  media?: never;
  provenance?: never;
  mime_type?: never;
  readonly original: Readonly<OriginalVideo>;
  /** Absent after metadata-only restoration until IndexedDB or reattachment supplies original bytes. */
  readonly body?: Blob;
}
const MAX_FILE = 256 * 1024 * 1024;
const MIME_BY_EXTENSION: Readonly<Record<string, string>> = {
  mp4: "video/mp4",
  mov: "video/quicktime",
  m4v: "video/x-m4v",
  webm: "video/webm",
  mkv: "video/x-matroska",
};
export async function captureFreshVideoReference(
  file: File,
  capability: VideoNormalizationCapabilities | null | undefined,
  signal?: AbortSignal,
): Promise<FreshVideoReference> {
  if (
    !capability?.available ||
    capability.protocol_version !== 1 ||
    !capability.input_roles.includes("reference_video")
  )
    throw new Error(
      "Fresh video normalization is not negotiated for this machine.",
    );
  const limit = Math.min(MAX_FILE, capability.max_original_file_bytes);
  if (!Number.isSafeInteger(limit) || limit <= 0)
    throw new Error("The machine advertised an invalid video upload limit.");
  if (!file.size) throw new Error("The selected video is empty.");
  if (file.size > limit)
    throw new Error("The selected video exceeds this machine's upload limit.");
  const name = file.name;
  const extension = name.split(".").at(-1)?.toLowerCase() ?? "";
  const hint =
    file.type.trim().toLowerCase() ||
    (capability.accepted_extensions.includes(extension)
      ? MIME_BY_EXTENSION[extension]
      : undefined);
  if (!hint || !capability.accepted_mime_types.includes(hint))
    throw new Error(
      "This machine does not accept the selected video container.",
    );
  const hash = sha256.create();
  for (let offset = 0; offset < file.size; offset += 4 * 1024 * 1024) {
    signal?.throwIfAborted();
    hash.update(
      new Uint8Array(
        await file.slice(offset, offset + 4 * 1024 * 1024).arrayBuffer(),
      ),
    );
  }
  signal?.throwIfAborted();
  const digest = Array.from(hash.digest(), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return Object.freeze({
    kind: "fresh_video",
    original: Object.freeze({
      mime_type: hint,
      name,
      byte_length: file.size,
      sha256: digest,
    }),
    body: file,
  });
}

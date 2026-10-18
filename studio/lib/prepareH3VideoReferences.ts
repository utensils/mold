import { sha256 } from "@noble/hashes/sha2.js";
import { decodePaddedBase64 } from "./base64Digest";
import { prepareNormalizedReferenceUploads } from "../api/referenceNormalization";
import type {
  ReferenceUploadCapabilities,
  ReferenceUploadRequest,
} from "../api/referenceUploads";
import type { ReferenceCandidate } from "../api/referenceNormalizationTypes";
import type { ApiTarget } from "../api/client";
import type { MinimaxH3AuthoringState } from "./minimaxH3Authoring";
/** Build one exact scoped session from the frozen authoring snapshot. The live
 * draft retains original bytes; its canonical lease is never written back. */
export function prepareH3VideoReferences<
  T extends ReferenceUploadRequest,
>(options: {
  state: MinimaxH3AuthoringState;
  request: Omit<T, "references">;
  target: ApiTarget;
  expectedInstanceId: string;
  capabilities: ReferenceUploadCapabilities;
  current(): boolean;
  signal?: AbortSignal;
}) {
  const candidates: ReferenceCandidate[] = [],
    bodies = new Map<number, Blob>();
  for (const [offset, draft] of options.state.references.entries()) {
    const index = offset + 1,
      reference = draft.reference;
    if (reference.kind === "fresh_video") {
      if (!reference.body)
        throw new Error(
          `Reattach video reference ${index} before preparing it.`,
        );
      candidates.push({
        index,
        type: "fresh_video",
        kind: "video",
        original: { ...reference.original },
      });
      bodies.set(index, reference.body);
    } else if (reference.media.authority === "inline") {
      const encoded = reference.media.data;
      const decodedLength =
        (encoded.length / 4) * 3 -
        (encoded.endsWith("==") ? 2 : encoded.endsWith("=") ? 1 : 0);
      const limit = Math.min(
        256 * 1024 * 1024,
        options.capabilities.max_file_bytes,
        options.capabilities.video_normalization?.max_original_file_bytes ??
          256 * 1024 * 1024,
      );
      if (decodedLength > limit)
        throw new Error(
          `Reference ${index} exceeds the negotiated upload limit.`,
        );
      const bytes = decodePaddedBase64(encoded);
      const digest = Array.from(sha256(bytes), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("");
      if (
        reference.provenance?.sha256 &&
        reference.provenance.sha256 !== digest
      )
        throw new Error(
          `Reference ${index} does not match its original digest.`,
        );
      candidates.push({
        index,
        type: "canonical",
        upload: true,
        byte_length: bytes.length,
        reference: {
          ...reference,
          provenance: { ...reference.provenance, sha256: digest },
          media: { authority: "descriptor" },
        },
      });
      bodies.set(index, new Blob([bytes], { type: reference.mime_type }));
    } else if (reference.media.authority === "server_path")
      candidates.push({ index, type: "canonical", upload: false, reference });
    else
      throw new Error(
        `Reference ${index} requires its original bytes; a previous upload lease cannot be reused.`,
      );
  }
  return prepareNormalizedReferenceUploads<T>({
    ...options,
    candidates,
    uploadBodies: bodies,
  });
}

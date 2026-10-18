import type { GenerationReference } from "../lib/generationReferences";
/** Mirrors mold_core::video_normalization V1. V2 remains unchanged. */
export interface VideoNormalizationCapabilities {
  available: boolean;
  protocol_version: number;
  input_roles: string[];
  candidate_session_path: string;
  finalize_path: string;
  candidate_digest_header: string;
  max_original_file_bytes: number;
  max_canonical_file_bytes: number;
  max_session_bytes: number;
  accepted_mime_types: string[];
  accepted_extensions: string[];
  accepted_video_codecs: string[];
  canonical_mime_type: string;
}
export interface OriginalVideo {
  mime_type: string;
  name: string | null;
  byte_length: number;
  sha256: string;
}
export type ReferenceCandidate =
  | {
      index: number;
      type: "canonical";
      upload: boolean;
      reference: GenerationReference;
      byte_length?: number;
    }
  | {
      index: number;
      type: "fresh_video";
      kind: "video";
      original: OriginalVideo;
    };
export interface NormalizationReceipt {
  version: 1;
  kind: GenerationReference["kind"];
  original_sha256: string;
  original_byte_length: number;
  original_mime_type: string;
  canonical_sha256: string;
  canonical_byte_length: number;
  canonical_mime_type: string;
  converted: boolean;
}
export interface FinalizeReceipt {
  reference: number;
  kind: GenerationReference["kind"];
  original_sha256: string;
  canonical_sha256: string;
}

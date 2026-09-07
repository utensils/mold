/** Shared wire contract for MiniMax H3 ordered heterogeneous references. */

export type GenerationReferenceAuthority =
  | { authority: "inline"; data: string }
  | { authority: "upload"; handle: string }
  | { authority: "server_path"; path: string }
  /** Payload-free projection accepted only by placement preview. */
  | { authority: "descriptor" };

export interface GenerationReferenceProvenance {
  /** Display-only filename, never a client filesystem path. */
  name?: string | null;
  /** Exact content digest; required for upload and server-path authorities. */
  sha256?: string | null;
  /** The user crop applied client-side to an IMAGE reference before it was
   * digested and uploaded; the server validates the rect and keeps it as
   * provenance (it already received the cropped bytes). */
  crop?: GenerationReferenceCrop | null;
}

/** A client-side crop in SOURCE pixels of the original photograph. `width` /
 * `height` are the cropped reference's own dimensions; the `source_*` facts
 * let Reuse settings re-apply the crop to a reattached original exactly. */
export interface GenerationReferenceCrop {
  x: number;
  y: number;
  width: number;
  height: number;
  source_width: number;
  source_height: number;
  source_sha256: string;
}

interface GenerationReferenceBase {
  media: GenerationReferenceAuthority;
  provenance?: GenerationReferenceProvenance;
  mime_type: string;
}

export type GenerationReference =
  | (GenerationReferenceBase & {
      kind: "image";
      width: number;
      height: number;
    })
  | (GenerationReferenceBase & {
      kind: "named_image";
      role: "front" | "left" | "back" | "right";
      width: number;
      height: number;
    })
  | (GenerationReferenceBase & {
      kind: "video";
      width: number;
      height: number;
      /** Exact decoded source-frame count, supplied by canonical ingress. */
      frame_count?: number | null;
      duration_ms: number;
      fps: number;
      has_audio?: boolean;
      audio_duration_ms?: number | null;
      /** Exact decoded soundtrack samples per channel at audio_sample_rate. */
      audio_sample_count?: number | null;
      audio_sample_rate?: number | null;
      audio_channels?: number | null;
    })
  | (GenerationReferenceBase & {
      kind: "audio";
      duration_ms: number;
      sample_rate: number;
      channels: number;
      /** Exact decoded samples per channel, supplied by canonical ingress. */
      sample_count?: number | null;
    });

export interface GenerationReferenceMetadata {
  kind: GenerationReference["kind"];
  /** One-based position in the authoritative request order. */
  index: number;
  name?: string | null;
  sha256: string;
  mime_type: string;
  width?: number | null;
  height?: number | null;
  frame_count?: number | null;
  duration_ms?: number | null;
  fps?: number | null;
  has_audio?: boolean;
  audio_duration_ms?: number | null;
  audio_sample_count?: number | null;
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
  sample_rate?: number | null;
  channels?: number | null;
  sample_count?: number | null;
  prepared_shape?: GenerationReferencePreparedShape | null;
  /** The crop an image reference carried; absent for every other reference. */
  crop?: GenerationReferenceCrop | null;
}

/** Exact checked output of the versioned reference preprocessing policy. */
export interface GenerationReferencePreparedShape {
  version: number;
  normalized_width?: number | null;
  normalized_height?: number | null;
  normalized_video_frames?: number | null;
  video_frames?: number | null;
  qwen_video_frames?: number | null;
  audio_samples_per_channel?: number | null;
  visual_rows: number;
  audio_rows: number;
}

/** Keep planning descriptors and digests while removing every media authority
 * value that could contain bytes, a scoped handle, or a filesystem path. */
export function redactGenerationReference(
  reference: GenerationReference,
): GenerationReference {
  const media: GenerationReferenceAuthority = { authority: "descriptor" };
  return { ...reference, media };
}

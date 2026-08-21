import { titleSlug } from "@studio/lib/libraryOrganization";

/**
 * Mirror of the gallery filename the server writes:
 * `mold-{model}-{timestamp}[-{index}]~{slug}.{ext}`
 * (`mold_core::default_output_filename` + `default_output_filename_titled`).
 *
 * Used ONLY for the Create inspector's "files as …" preview, which is why the
 * timestamp is a caller-supplied `Date.now()` — the real name is stamped by
 * the host when the print lands, so the preview shows the grammar and the
 * title slug rather than promising an exact string. The `:` → `-` swap is the
 * server's own sanitizing (`flux-dev:q4` → `flux-dev-q4`).
 *
 * This is deliberately NOT `downloadFileName`: a download name is a label
 * (`{title}__{model}__s{seed}`) and the gallery filename is an identity.
 */
export interface PrintFilenameInput {
  /** Resolved model id, e.g. `z-image-turbo:bf16`. */
  model: string;
  /** Milliseconds since the epoch — the server stamps `timestamp_ms`. */
  timestamp: number;
  /** Extension without its dot, e.g. `png`. */
  ext: string;
  /** Print title; absent or unsluggable drops the `~slug` suffix. */
  title?: string | null;
  /** One-based-in-the-UI batch position; the wire index is zero-based and only
   * appears at all when the batch renders more than one print. */
  batchSize?: number;
  index?: number;
}

export function previewPrintFilename(input: PrintFilenameInput): string {
  const model = (input.model || "model").replaceAll(":", "-");
  const batch = input.batchSize ?? 1;
  const stem =
    batch > 1
      ? `mold-${model}-${input.timestamp}-${input.index ?? 0}`
      : `mold-${model}-${input.timestamp}`;
  const slug = input.title?.trim() ? titleSlug(input.title.trim()) : null;
  const ext = input.ext.trim().replace(/^\.+/, "").toLowerCase();
  const named = slug ? `${stem}~${slug}` : stem;
  return ext ? `${named}.${ext}` : named;
}

/** Stand-in for the job-id digest a durable sequence's name is built from —
 * it does not exist until the job is created, and inventing one would be a
 * promise the preview cannot keep. */
export const SEQUENCE_DIGEST_PLACEHOLDER = "…";

/**
 * A durable sequence's stitched print lands under a different grammar:
 * `mold-chain-{sha256(job_id)}-take-{n}[~{slug}].mp4`
 * (`chain_job_runner::chain_gallery_filename`), deliberately free of a wall
 * clock so a replay publishes the same name. Only the title slug is knowable
 * before the job exists, so the digest is elided.
 */
export function previewSequenceFilename(title?: string | null): string {
  const stem = `mold-chain-${SEQUENCE_DIGEST_PLACEHOLDER}-take-0`;
  const slug = title?.trim() ? titleSlug(title.trim()) : null;
  return slug ? `${stem}~${slug}.mp4` : `${stem}.mp4`;
}

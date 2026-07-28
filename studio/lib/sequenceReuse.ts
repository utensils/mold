/**
 * Sequence provenance helpers shared by every shell.
 *
 * A finished sequence is a print: a `/api/gallery` row carrying
 * `metadata.chain_job_id` and `metadata.chain`. These predicates are how a
 * surface gets from the durable job back to that print — the Create canvas's
 * settled-sequence result and History ▸ Sequences' "Show print" both need it.
 *
 * `OutputMetadataLike` is a structural minimum on purpose: desktop, web, and
 * iPhone each keep their own `OutputMetadata` interface, and unifying them is
 * a real but separate cleanup.
 */

export interface OutputMetadataLike {
  chain_job_id?: string | null;
}

/**
 * True when this gallery row is the stitched output of `jobId`.
 *
 * Filename or seed lookalikes are deliberately not accepted: ephemeral chain
 * outputs and legacy rows carry no job id at all, and guessing would hand the
 * canvas an unrelated video.
 */
export function isPrintOfChainJob(metadata: OutputMetadataLike, jobId: string): boolean {
  if (!jobId) return false;
  return metadata.chain_job_id === jobId;
}

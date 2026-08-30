export interface SubmittedActivityRow {
  createdAtMs: number;
}

export interface SubmittedQueueEntry {
  started_at_unix_ms: number;
}

/** Newest submissions render first. Returning zero for equal timestamps keeps
 * the source's stable sending order instead of inventing an ID-based order. */
export function compareNewestSubmitted(
  left: SubmittedActivityRow,
  right: SubmittedActivityRow,
): number {
  return right.createdAtMs - left.createdAtMs;
}

/** Queue display order only. Scheduler position remains authoritative for
 * dispatch and mutations; equal stamps retain the response's sending order. */
export function compareNewestQueueEntry(
  left: SubmittedQueueEntry,
  right: SubmittedQueueEntry,
): number {
  return right.started_at_unix_ms - left.started_at_unix_ms;
}

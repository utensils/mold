/**
 * One policy for reconciling this Create session's own generation rows with
 * the server-owned fleet activity rows that describe the same work.
 *
 * A local row and a shared row are the same print when the local job carries
 * the server's job id and both name the same host. Which one renders depends
 * on which is still true:
 *
 * - While this session is streaming the job, the local row is richer (live
 *   preview, cancel, step counter) and the shared row is a duplicate.
 * - Once the local row has SETTLED as a failure it may be stale: a durable
 *   host that retained the job across a restart keeps running it, and its
 *   shared row is the live truth. The settled local row loses — otherwise the
 *   resumed job renders as failed here and is invisible in the fleet list.
 *
 * Completed and cancelled local rows never reach the strip (the activity view
 * is present tense), so only a failed local row can be superseded.
 */

/** The subset of a local Create job this policy reads. */
export interface LocalActivityJob {
  /** Server-assigned job id, latched from the `queued` SSE event. */
  serverId: string | null;
  /** Registry id of the routed host; `null` means the serving origin. */
  hostId: string | null;
  state: string;
}

/** The subset of a server-owned fleet activity row this policy reads. */
export interface SharedActivityRow {
  kind: string;
  id: string;
  hostId: string;
}

function sameWork(
  job: LocalActivityJob,
  row: SharedActivityRow,
  originHostId: string,
): boolean {
  return (
    row.kind === "generation" &&
    !!job.serverId &&
    job.serverId === row.id &&
    (job.hostId ?? originHostId) === row.hostId
  );
}

/**
 * Whether a shared row duplicates a job this session is still streaming, and
 * should therefore be dropped from the shared list.
 */
export function sharedRowIsLocallyOwned(
  row: SharedActivityRow,
  jobs: readonly LocalActivityJob[],
  originHostId: string,
): boolean {
  return jobs.some(
    (job) => job.state === "running" && sameWork(job, row, originHostId),
  );
}

/**
 * Whether a settled local failure is superseded by a live shared row for the
 * same job, and should therefore be dropped from the local list.
 */
export function localFailureSupersededByShared(
  job: LocalActivityJob,
  rows: readonly SharedActivityRow[],
  originHostId: string,
): boolean {
  if (job.state !== "error") return false;
  return rows.some((row) => sameWork(job, row, originHostId));
}

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
  /** The host owns this job's fate — it was retained across a restart, or the
   * page was away while it ran. Settled `detached` is advisory, never a
   * failure. */
  detached?: boolean;
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
 * Whether a local row should be dropped from the activity strip.
 *
 * Two ways a settled local row stops being the truth:
 *
 * 1. It settled DETACHED — the host kept the job (retained across a restart)
 *    or ran it while the page was away. The strip models only `error` for a
 *    settled row and renders that as "Failed", which for this row is a lie
 *    that outlives the masking shared row: once the job finishes it leaves
 *    the host's active work, the shared row disappears, and the local row
 *    would resurface as a failure for a print sitting in the Library. It is
 *    retired instead; the fleet row covers it while it runs and the Library
 *    covers it afterwards.
 * 2. A live shared row reports the same job still running, so the server's
 *    view supersedes this one.
 *
 * A running row is never hidden — including a rehydrated detached one, which
 * is live work the reconciler has yet to rule on.
 */
export function localRowHiddenFromStrip(
  job: LocalActivityJob,
  rows: readonly SharedActivityRow[],
  originHostId: string,
): boolean {
  if (job.state === "running") return false;
  if (job.detached === true) return true;
  if (job.state !== "error") return false;
  return rows.some((row) => sameWork(job, row, originHostId));
}

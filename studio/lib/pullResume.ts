/*
 * Pull-and-resume: deciding when a missing-model download has settled.
 *
 * Desktop has offered "download it there and generate when it's ready" since
 * #439; web now does too, so the rule that decides *which* download job
 * settles the promise lives here instead of being restated per surface.
 *
 * Two things make it non-obvious. The job id is the precise thing to watch —
 * a stale completed pull of the same model sitting in history must never fire
 * a premature resume — but a server that reported no id (or a 409 "already
 * downloading") leaves only the model name, so the fallback matches by name
 * and ignores every terminal job that already existed when the watch was
 * armed.
 */

/** The download-job fields both surfaces' listings expose. */
export interface PullResumeJob {
  id: string;
  model: string;
  status: string;
  error?: string | null;
}

export interface PullResumeWatch {
  model: string;
  /** The enqueued pull's job id, or null when the server didn't report one. */
  jobId?: string | null;
  /** Terminal job ids that already existed when the watch was armed. */
  seenTerminal: readonly string[];
}

export type PullResumeOutcome =
  | { kind: "waiting" }
  | { kind: "ready"; job: PullResumeJob }
  | { kind: "failed"; job: PullResumeJob };

export function isTerminalPullJob(job: PullResumeJob): boolean {
  return (
    job.status === "completed" ||
    job.status === "failed" ||
    job.status === "cancelled"
  );
}

export function terminalPullJobIds(jobs: readonly PullResumeJob[]): string[] {
  return jobs.filter(isTerminalPullJob).map((job) => job.id);
}

export function resolvePullResumeOutcome(
  jobs: readonly PullResumeJob[],
  watch: PullResumeWatch,
): PullResumeOutcome {
  const fresh = jobs
    .filter(isTerminalPullJob)
    .filter((job) =>
      watch.jobId
        ? job.id === watch.jobId
        : job.model === watch.model && !watch.seenTerminal.includes(job.id),
    );
  const done = fresh.find((job) => job.status === "completed");
  if (done) return { kind: "ready", job: done };
  const failed = fresh.find((job) => job.status !== "completed");
  if (failed) return { kind: "failed", job: failed };
  return { kind: "waiting" };
}

/** One sentence for a pull that ended without producing the model. */
export function pullResumeFailureMessage(
  model: string,
  job: PullResumeJob,
): string {
  const verb = job.status === "cancelled" ? "was cancelled" : "failed";
  return `Download of ${model} ${verb}${job.error ? ` — ${job.error}` : ""}; generation not resumed.`;
}

import { readableBytes, userFacingError } from "./userFacingError";
import type { PullResumeJob } from "./pullResume";

export interface QueueDownloadState {
  phase:
    | "starting"
    | "license"
    | "queued"
    | "downloading"
    | "reconnecting"
    | "retrying"
    | "complete"
    | "failed"
    | "cancelled";
  message: string;
  fraction: number | null;
  busy: boolean;
}
export interface QueueDownloadJob extends PullResumeJob {
  bytes_done?: number;
  bytes_total?: number;
}
export function queueDownloadSettlement(
  ids: readonly string[],
  jobs: readonly PullResumeJob[],
): { kind: "waiting" | "ready" } | { kind: "failed"; message: string } {
  if (!ids.length) return { kind: "waiting" };
  const matching = [...new Set(ids)].map((id) =>
    jobs.find((job) => job.id === id),
  );
  const failed = matching.find(
    (job) => job?.status === "failed" || job?.status === "cancelled",
  );
  if (failed)
    return {
      kind: "failed",
      message: userFacingError(
        failed.error ||
          (failed.status === "cancelled"
            ? "Download cancelled. The job remains held."
            : "Download failed. The job remains held."),
      ),
    };
  return {
    kind: matching.every((job) => job?.status === "completed")
      ? "ready"
      : "waiting",
  };
}
export function queueDownloadProgress(
  jobs: readonly QueueDownloadJob[],
  host: string,
): QueueDownloadState {
  if (!jobs.length)
    return {
      phase: "reconnecting",
      message: "Waiting for the machine to confirm the download outcome.",
      fraction: null,
      busy: true,
    };
  const queued = jobs
    .filter((job) => job.status !== "completed")
    .every((job) => job.status === "queued");
  const known = jobs.every((job) => (job.bytes_total ?? 0) > 0);
  const total = jobs.reduce((sum, job) => sum + (job.bytes_total ?? 0), 0);
  const done = jobs.reduce((sum, job) => sum + (job.bytes_done ?? 0), 0);
  return {
    phase: queued ? "queued" : "downloading",
    message: queued
      ? `Download queued on ${host}.`
      : `Downloading on ${host}${known && total > 0 ? ` · ${readableBytes(done)} / ${readableBytes(total)}` : "…"}`,
    fraction:
      known && total > 0 ? Math.min(1, Math.max(0, done / total)) : null,
    busy: true,
  };
}

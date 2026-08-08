import type { DownloadJob } from "./api/types";
import { formatBytes, formatEta, percent } from "./format";
import { modelDisplayNameForId, type DisplayableModel } from "./models";

export type ExpansionPullPhase = "missing" | "connecting" | "starting";
export type ExpansionPullKind =
  ExpansionPullPhase | "queued" | "pulling" | "ready" | "failed" | "cancelled";

export interface ExpansionPullView {
  kind: ExpansionPullKind;
  job: DownloadJob | null;
  message?: string | null;
}

export interface ExpansionPullResolutionInput {
  model: string;
  phase: ExpansionPullPhase;
  jobId: string | null;
  observedJobId?: string | null;
  baselineJobIds: string[];
  allowExistingInFlight: boolean;
  activeJobs: DownloadJob[];
  queued: DownloadJob[];
  history: DownloadJob[];
  requestError?: string | null;
}

function downloadModelMatches(candidate: string | null | undefined, requested: string): boolean {
  if (!candidate) return false;
  if (candidate === requested) return true;
  // The server resolves bare manifest names before enqueueing, so the real
  // `qwen3-expand` request is streamed back as `qwen3-expand:q8`. Catalog ids
  // already contain a colon and must retain literal identity.
  return !requested.includes(":") && candidate.startsWith(`${requested}:`);
}

/** Exact or bare→canonical manifest identity from model/catalog download rows. */
export function expansionPullJobMatchesModel(job: DownloadJob, model: string): boolean {
  return downloadModelMatches(job.model, model) || downloadModelMatches(job.catalog_id, model);
}

/**
 * Reduce one exact host's download bucket into the compact Generate lifecycle.
 * A returned/observed job id is authoritative. Before an id exists, only a
 * newly-seen in-flight exact-model row may be adopted; old history can never
 * masquerade as a successful pull.
 */
export function resolveExpansionPullStatus(input: ExpansionPullResolutionInput): ExpansionPullView {
  if (input.requestError) return { kind: "failed", job: null, message: input.requestError };

  const inFlight = [...input.activeJobs, ...input.queued];
  const all = [...inFlight, ...input.history];
  const authoritativeId = input.jobId ?? input.observedJobId ?? null;
  if (authoritativeId) {
    const authoritativeJob = all.find((job) => job.id === authoritativeId) ?? null;
    return authoritativeJob ? viewForJob(authoritativeJob) : { kind: input.phase, job: null };
  }

  const baseline = new Set(input.baselineJobIds);
  const job =
    inFlight.find(
      (job) =>
        expansionPullJobMatchesModel(job, input.model) &&
        (input.allowExistingInFlight || !baseline.has(job.id)),
    ) ?? null;

  if (!job) return { kind: input.phase, job: null };
  return viewForJob(job);
}

/** Shared desktop/iPhone copy and progress semantics for the pull lifecycle. */
export function expansionPullPresentation(
  status: ExpansionPullView,
  model: string,
  hostLabel: string,
  missingError: string,
  etaSeconds: number | null,
  models: DisplayableModel[] = [],
) {
  const modelLabel = modelDisplayNameForId(model, models);
  const labels: Record<ExpansionPullKind, string> = {
    missing: missingError,
    connecting: `Connecting to ${hostLabel} to pull ${modelLabel}…`,
    starting: `Starting ${modelLabel} on ${hostLabel}…`,
    queued: `${modelLabel} queued on ${hostLabel}`,
    pulling: `Pulling ${modelLabel} on ${hostLabel}`,
    ready: `${modelLabel} ready on ${hostLabel}`,
    failed: `${modelLabel} pull failed on ${hostLabel}`,
    cancelled: `${model} pull cancelled on ${hostLabel}`,
  };
  return {
    label: labels[status.kind],
    modelLabel,
    busy: ["connecting", "starting", "queued", "pulling"].includes(status.kind),
    percent: status.job ? Math.round(percent(status.job.bytes_done, status.job.bytes_total)) : 0,
    bytes: status.job
      ? `${formatBytes(status.job.bytes_done)} / ${formatBytes(status.job.bytes_total)}`
      : "",
    files:
      status.job && status.job.files_total > 0
        ? `${status.job.files_done}/${status.job.files_total} files`
        : null,
    eta: formatEta(etaSeconds),
  };
}

function viewForJob(job: DownloadJob): ExpansionPullView {
  switch (job.status) {
    case "active":
      return { kind: job.bytes_total > 0 ? "pulling" : "starting", job };
    case "queued":
      return { kind: "queued", job };
    case "completed":
      return { kind: "ready", job };
    case "failed":
      return { kind: "failed", job, message: job.error ?? null };
    case "cancelled":
      return { kind: "cancelled", job };
  }
}

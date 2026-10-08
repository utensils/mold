import { reactive } from "vue";
import { ApiError, apiJsonTo, type ApiTarget } from "../api/client";
import { getGenerationBatch } from "../api/generationAdmission";
import {
  getQueueJob,
  retryQueueJobRecoveringAmbiguity,
  type QueueEntry,
  type QueueJobAuthority,
} from "../api/queuePlan";
import {
  runWithLicenseConsent,
  useLicenseAcceptance,
} from "./useLicenseAcceptance";
import {
  queueDownloadProgress,
  queueDownloadSettlement,
  type QueueDownloadJob,
  type QueueDownloadState,
} from "../lib/queueDownloadRecovery";
import { userFacingError } from "../lib/userFacingError";

// Recovery survives closing details/navigation, keyed by captured machine instance and job.
interface DownloadStartTicket {
  id?: string;
  primary_job_id?: string | null;
  companion_jobs?: { job_id: string }[];
}
const states = reactive<Record<string, QueueDownloadState>>({});
const controllers = new Map<string, AbortController>();
export function queueDownloadKey(
  target: ApiTarget,
  instance: string,
  job: string,
): string {
  return JSON.stringify([target.baseUrl, instance, job]);
}
export function queueDownloadState(
  target: ApiTarget | null | undefined,
  instance: string | null | undefined,
  job: string,
): QueueDownloadState | null {
  return target && instance
    ? (states[queueDownloadKey(target, instance, job)] ?? null)
    : null;
}
async function fencedJob(
  target: ApiTarget,
  instance: string,
  job: string,
  signal?: AbortSignal,
): Promise<QueueEntry> {
  const status = await apiJsonTo<{ instance_id?: string }>(
    target,
    "/api/status",
    signal ? { signal } : {},
  );
  if (status.instance_id !== instance)
    throw new Error(
      "This address now reaches a different Mold server. Reopen the job.",
    );
  const detail = await getQueueJob(target, job);
  const after = await apiJsonTo<{ instance_id?: string }>(
    target,
    "/api/status",
    signal ? { signal } : {},
  );
  if (after.instance_id !== instance || detail.job.id !== job)
    throw new Error("The job or machine changed. Reopen its details.");
  return detail.job;
}
export async function missingQueueModel(
  target: ApiTarget,
  instance: string,
  job: string,
): Promise<boolean> {
  const row = await fencedJob(target, instance, job);
  if (
    row.state !== "held" ||
    row.retryable !== true ||
    !row.batch_id ||
    !row.client_batch_id
  )
    return false;
  const lookup = await getGenerationBatch(target, row.batch_id);
  if (lookup.kind !== "found" || lookup.batch.instance_id !== instance)
    return false;
  const child = lookup.batch.children.find((child) => child.job_id === job);
  return (
    child?.state === "held" &&
    child.retryable !== false &&
    ["MODEL_NOT_FOUND", "UNKNOWN_MODEL"].includes(child.error_code ?? "")
  );
}

export function cancelQueueDownloadRecovery(
  target: ApiTarget,
  instance: string,
  job: string,
): void {
  const key = queueDownloadKey(target, instance, job);
  controllers.get(key)?.abort();
  const consent = useLicenseAcceptance();
  if (
    states[key]?.phase === "license" &&
    consent.pending.value?.target.baseUrl === target.baseUrl
  )
    consent.cancel();
  controllers.delete(key);
  states[key] = {
    phase: "cancelled",
    message: "Download recovery cancelled. The job remains held.",
    fraction: null,
    busy: false,
  };
}

export async function startQueueDownloadRecovery(
  target: ApiTarget,
  instance: string,
  job: string,
  host: string,
): Promise<void> {
  target = { ...target };
  const key = queueDownloadKey(target, instance, job);
  if (states[key]?.busy) return;
  const controller = new AbortController();
  controllers.set(key, controller);
  const signal = controller.signal;
  const set = (
    phase: QueueDownloadState["phase"],
    message: string,
    busy = true,
  ) => {
    if (controllers.get(key) === controller && !signal.aborted)
      states[key] = {
        phase,
        message,
        busy,
        fraction:
          phase === "reconnecting" ? (states[key]?.fraction ?? null) : null,
      };
  };
  set("starting", `Starting download on ${host}…`);
  try {
    if (!(await missingQueueModel(target, instance, job)))
      throw new Error(
        "This job no longer needs a model download. Refresh Queue.",
      );
    signal.throwIfAborted();
    const row = await fencedJob(target, instance, job, signal);
    const authority: QueueJobAuthority = {
      instanceId: instance,
      batchId: row.batch_id!,
      clientBatchId: row.client_batch_id!,
      jobId: job,
    };
    const current = async () => {
      const fresh = await fencedJob(target, instance, job, signal);
      if (
        fresh.state !== "held" ||
        fresh.retryable !== true ||
        fresh.model !== row.model ||
        fresh.batch_id !== row.batch_id ||
        fresh.client_batch_id !== row.client_batch_id
      )
        throw new Error(
          "This job changed or is no longer held. Download recovery stopped.",
        );
    };
    const acquisitionDeadline = Date.now() + 60 * 60 * 1000;
    let licenseTimer: ReturnType<typeof setInterval> | undefined;
    let checkingLicense = false;
    const monitor = new Promise<never>((_, reject) => {
      licenseTimer = setInterval(async () => {
        if (checkingLicense || states[key]?.phase !== "license") return;
        checkingLicense = true;
        try {
          if (Date.now() >= acquisitionDeadline)
            throw new Error("License review timed out. The job remains held.");
          await current();
        } catch (error) {
          if (
            error instanceof TypeError ||
            (error instanceof ApiError && error.status >= 500)
          )
            return;
          const consent = useLicenseAcceptance();
          if (
            consent.pending.value?.target.baseUrl === target.baseUrl &&
            consent.pending.value.requirements.some(
              (requirement) => requirement.installModel === row.model,
            )
          )
            consent.cancel();
          reject(error);
        } finally {
          checkingLicense = false;
        }
      }, 2000);
    });
    const acquire = runWithLicenseConsent({
      target,
      hostLabel: host,
      installModel: row.model,
      start: async () => {
        await current();
        const catalog =
          row.model.startsWith("hf:") || row.model.startsWith("cv:");
        try {
          return await apiJsonTo<{
            id?: string;
            primary_job_id?: string | null;
            companion_jobs?: { job_id: string }[];
          }>(
            target,
            catalog
              ? `/api/catalog/${encodeURIComponent(row.model)}/download`
              : "/api/downloads",
            {
              method: "POST",
              signal,
              headers: { "Content-Type": "application/json" },
              ...(catalog
                ? {}
                : { body: JSON.stringify({ model: row.model }) }),
            },
          );
        } catch (error) {
          if (
            error instanceof ApiError &&
            error.status === 409 &&
            error.body &&
            typeof error.body === "object" &&
            "id" in error.body
          )
            return error.body as DownloadStartTicket;
          if (error instanceof ApiError && error.status === 403)
            set("license", `Awaiting license acceptance on ${host}.`);
          throw error;
        }
      },
    });
    let outcome: Awaited<typeof acquire>;
    try {
      outcome = await Promise.race([acquire, monitor]);
    } finally {
      if (licenseTimer) clearInterval(licenseTimer);
    }
    signal.throwIfAborted();
    if (outcome.kind === "declined") {
      set(
        "cancelled",
        "License review cancelled. The job remains held.",
        false,
      );
      return;
    }
    {
      const ticket = outcome.kind === "ok" ? outcome.value : null;
      const ids = ticket
        ? [
            ticket.id,
            ticket.primary_job_id,
            ...(ticket.companion_jobs ?? []).map((job) => job.job_id),
          ].filter(
            (id): id is string => typeof id === "string" && id.length > 0,
          )
        : outcome.kind === "accepted"
          ? [...(outcome.jobIds ?? [])]
          : [];
      if (!ids.length)
        throw new Error(
          "The machine returned no download ticket. Refresh Models before retrying.",
        );
      const terminal = new Map<string, QueueDownloadJob>();
      const deadline = Date.now() + 60 * 60 * 1000;
      while (true) {
        signal.throwIfAborted();
        if (Date.now() >= deadline)
          throw new Error(
            "Download status could not be confirmed. Refresh the machine before retrying.",
          );
        try {
          await current();
          const listing = await apiJsonTo<{
            active_jobs?: QueueDownloadJob[];
            active?: QueueDownloadJob | null;
            queued: QueueDownloadJob[];
            history: QueueDownloadJob[];
          }>(target, "/api/downloads", signal ? { signal } : {});
          signal.throwIfAborted();
          const jobs = [
            ...(listing.active_jobs ??
              (listing.active ? [listing.active] : [])),
            ...listing.queued,
            ...listing.history,
          ];
          for (const download of jobs)
            if (
              ids.includes(download.id) &&
              ["completed", "failed", "cancelled"].includes(download.status)
            )
              terminal.set(download.id, download);
          const settlement = queueDownloadSettlement(ids, [
            ...terminal.values(),
          ]);
          if (settlement.kind === "failed") throw new Error(settlement.message);
          if (settlement.kind === "ready") break;
          if (controllers.get(key) === controller)
            states[key] = queueDownloadProgress(
              ids
                .map(
                  (id) => jobs.find((job) => job.id === id) ?? terminal.get(id),
                )
                .filter((job): job is QueueDownloadJob => !!job),
              host,
            );
        } catch (error) {
          if (
            signal.aborted ||
            (error instanceof ApiError
              ? error.status < 500
              : error instanceof Error && !(error instanceof TypeError))
          )
            throw error;
          set("reconnecting", "Reconnecting — download status unavailable.");
        }
        await new Promise<void>((resolve) => {
          const timer = setTimeout(done, 2000);
          function done() {
            signal.removeEventListener("abort", done);
            clearTimeout(timer);
            resolve();
          }
          signal.addEventListener("abort", done, { once: true });
        });
      }
    }
    await current();
    set("retrying", "Model ready — retrying job…");
    const retry = await retryQueueJobRecoveringAmbiguity(target, authority);
    signal.throwIfAborted();
    if (retry.kind === "uncertain") throw new Error(retry.error);
    if (retry.kind === "reconciled") {
      const child = retry.batch.children.find((child) => child.job_id === job);
      if (
        !child ||
        !["accepted", "queued", "paused", "running", "complete"].includes(
          child.state,
        )
      )
        throw new Error(
          child?.error ||
            "Retry was not confirmed. The job remains held; refresh Queue before retrying.",
        );
    }
    set("complete", `Retry accepted on ${host}.`, false);
  } catch (error) {
    if (!signal.aborted)
      set(
        "failed",
        userFacingError(error instanceof Error ? error.message : String(error)),
        false,
      );
  } finally {
    controller.abort();
    if (controllers.get(key) === controller) controllers.delete(key);
  }
}

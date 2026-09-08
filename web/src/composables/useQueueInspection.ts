import { computed, onBeforeUnmount, ref, shallowRef } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import { apiJsonTo, type ApiTarget } from "@studio/api/client";
import {
  cancelQueueJob,
  getQueueJob,
  retryQueueJobRecoveringAmbiguity,
  setQueueJobPaused,
  type QueueJobEntry,
  type QueueJobAuthority,
} from "@studio/api/queuePlan";
import {
  queueEntryDetailModel,
  type QueueDetailMetadata,
} from "@studio/lib/queueEntryDetail";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";
import type { Job } from "./useGenerateStream";
import type { GenerateRequestWire, ServerCapabilities } from "../types";
import { toast } from "../lib/toasts";

export interface LocalQueueInspection {
  job: Job;
  cancel: () => Promise<void>;
  retry: () => Promise<void>;
}

import type { HostRouting } from "./useHostRouting";

/** Queue inspection owns one captured host/job, never the current Auto target. */
export function useQueueInspection(
  routing: HostRouting,
  refresh: () => Promise<void>,
) {
  const selected = ref<FleetActiveWork | null>(null);
  const detail = ref<QueueJobEntry | null>(null);
  const error = ref<string | null>(null);
  const busy = ref(false);
  const pendingAction = ref<string | null>(null);
  const capabilities = ref<ServerCapabilities | null>(null);
  const local = shallowRef<LocalQueueInspection | null>(null);
  let opener: HTMLElement | null = null;
  let epoch = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function targetFor(row: FleetActiveWork): ApiTarget {
    const host = routing.hosts.value.find((host) => host.id === row.hostId);
    if (
      row.stale ||
      !row.instanceId ||
      !host ||
      host.status !== "ready" ||
      host.url !== row.routeUrl
    )
      throw new Error(
        "Reconnect to the original machine before changing this job.",
      );
    return { baseUrl: host.url, apiKey: host.apiKey ?? null };
  }
  async function inspect(
    row: FleetActiveWork,
    ticket: number,
  ): Promise<{
    target: ApiTarget;
    detail: QueueJobEntry;
    capabilities: ServerCapabilities | null;
  }> {
    if (row.kind !== "generation" || row.execution === "chain")
      throw new Error("Open this work from its machine details.");
    const target = targetFor(row);
    const status = await apiJsonTo<{ instance_id?: string }>(
      target,
      "/api/status",
    );
    if (status.instance_id !== row.instanceId)
      throw new Error(
        "This address now reaches a different Mold server. Reopen the job from its machine.",
      );
    const next = await getQueueJob(target, row.id);
    // Unknown capabilities disable capability-gated mutations, but still permit reading.
    const capabilities = await apiJsonTo<ServerCapabilities>(
      target,
      "/api/capabilities",
    ).catch(() => null);
    const after = await apiJsonTo<{ instance_id?: string }>(
      target,
      "/api/status",
    );
    if (after.instance_id !== row.instanceId)
      throw new Error(
        "This address now reaches a different Mold server. Reopen the job from its machine.",
      );
    const current = targetFor(row);
    if (
      ticket !== epoch ||
      current.baseUrl !== target.baseUrl ||
      current.apiKey !== target.apiKey
    )
      throw new Error(
        "The selected job or machine changed. Reopen its details.",
      );
    if (next.job.id !== row.id)
      throw new Error("The machine returned a different queue job.");
    return { target, detail: next, capabilities };
  }
  function retryAuthority(
    row: FleetActiveWork,
    next: QueueJobEntry,
  ): QueueJobAuthority | null {
    const job = next.job;
    return row.instanceId &&
      job.state === "held" &&
      job.retryable === true &&
      job.batch_id?.trim() &&
      job.client_batch_id?.trim()
      ? {
          instanceId: row.instanceId,
          jobId: job.id,
          batchId: job.batch_id.trim(),
          clientBatchId: job.client_batch_id.trim(),
        }
      : null;
  }
  const model = computed(() => {
    const row = selected.value;
    const next = detail.value;
    if (!row || !next) return null;
    const request =
      local.value && !local.value.job.chain
        ? (local.value.job.request as GenerateRequestWire)
        : null;
    const localMetadata =
      request && !("stages" in request)
        ? {
            ...request,
            lora: request.lora?.path ?? null,
            lora_scale: request.lora?.scale ?? null,
            collection:
              request.collection?.name ?? request.collection?.id ?? null,
          }
        : null;
    const value = queueEntryDetailModel({
      entry: next.job,
      hostLabel: row.hostLabel,
      modelLabel: modelDisplayNameForId(
        next.job.model,
        routing.installedModels.value,
      ),
      nowMs: Date.now(),
      metadata: next.job.metadata as QueueDetailMetadata | null,
      mine: local.value !== null,
      localMetadata,
      canCancelRunning:
        row.can_cancel &&
        capabilities.value?.queue?.cooperative_cancellation === true,
      retryAuthority: local.value
        ? local.value.job.retryable &&
          local.value.job.serverId === row.id &&
          local.value.job.durableBatch?.expectedInstanceId === row.instanceId &&
          local.value.job.durableBatch.clientBatchId.trim() &&
          local.value.job.durableBatch.serverBatchId?.trim()
          ? {
              instanceId: local.value.job.durableBatch.expectedInstanceId,
              batchId: local.value.job.durableBatch.serverBatchId,
              clientBatchId: local.value.job.durableBatch.clientBatchId,
              jobId: row.id,
            }
          : null
        : retryAuthority(row, next),
    });
    try {
      targetFor(row);
    } catch {
      value.cancel.available = false;
      value.retry.available = false;
      value.reuse.available = false;
      value.cancel.blockedReason =
        "Reconnect to the original machine before changing this job.";
      value.retry.blockedReason = value.cancel.blockedReason;
      value.reuse.blockedReason = value.cancel.blockedReason;
    }
    if (!row.can_cancel) {
      value.cancel.available = false;
      value.cancel.blockedReason =
        "This machine cannot cancel this job in its current state.";
    }
    if (busy.value || error.value) {
      value.cancel.available = false;
      value.retry.available = false;
      value.reuse.available = false;
    }
    return value;
  });
  const canPause = computed(() => {
    const row = selected.value;
    if (!row || !detail.value || busy.value || error.value) return false;
    try {
      targetFor(row);
    } catch {
      return false;
    }
    return (
      capabilities.value?.queue?.can_pause_job === true &&
      ["queued", "paused"].includes(detail.value.job.state)
    );
  });
  function close() {
    epoch += 1;
    if (timer) clearTimeout(timer);
    timer = null;
    selected.value = null;
    detail.value = null;
    error.value = null;
    busy.value = false;
    pendingAction.value = null;
    capabilities.value = null;
    local.value = null;
    if (opener?.isConnected) opener.focus();
    opener = null;
  }
  function apply(next: Awaited<ReturnType<typeof inspect>>) {
    detail.value = next.detail;
    capabilities.value = next.capabilities;
    error.value = null;
  }
  async function open(
    row: FleetActiveWork,
    provenance?: LocalQueueInspection,
    source?: HTMLElement,
  ) {
    const active = source ?? document.activeElement;
    close();
    opener = active instanceof HTMLElement ? active : null;
    local.value = provenance ?? null;
    selected.value = { ...row };
    const ticket = epoch;
    busy.value = true;
    try {
      apply(await inspect(row, ticket));
    } catch (caught) {
      if (ticket === epoch)
        error.value = caught instanceof Error ? caught.message : String(caught);
    } finally {
      if (ticket === epoch) {
        busy.value = false;
        schedule(row, ticket);
      }
    }
  }
  function schedule(row: FleetActiveWork, ticket: number) {
    if (ticket !== epoch) return;
    timer = setTimeout(async () => {
      if (ticket !== epoch) return;
      if (!busy.value) {
        try {
          const next = await inspect(row, ticket);
          if (ticket === epoch) apply(next);
        } catch (caught) {
          if (ticket === epoch)
            error.value =
              caught instanceof Error ? caught.message : String(caught);
        }
      }
      schedule(row, ticket);
    }, 5000);
  }
  async function snapshot() {
    const row = selected.value;
    if (!row || busy.value) return null;
    const ticket = epoch;
    busy.value = true;
    try {
      const fresh = await inspect(row, ticket);
      apply(fresh);
      return { row, detail: fresh.detail, localJob: local.value?.job ?? null };
    } catch (caught) {
      if (ticket === epoch)
        error.value = caught instanceof Error ? caught.message : String(caught);
      return null;
    } finally {
      if (ticket === epoch) busy.value = false;
    }
  }
  async function act(action: "cancel" | "retry" | "pause" | "resume") {
    const row = selected.value;
    if (!row || busy.value) return;
    const ticket = epoch;
    const provenance = local.value;
    let mutationStarted = false;
    busy.value = true;
    pendingAction.value = action;
    error.value = null;
    try {
      const fresh = await inspect(row, ticket);
      apply(fresh);
      const job = fresh.detail.job;
      if (action === "cancel") {
        if (["completed", "cancelled", "failed", "done"].includes(job.state))
          throw new Error("This job has already settled. Refresh the Queue.");
        if (
          !row.can_cancel ||
          (job.state === "running" &&
            capabilities.value?.queue?.cooperative_cancellation !== true)
        )
          throw new Error(
            "This machine cannot cancel this job in its current state.",
          );
        mutationStarted = true;
        if (provenance) await provenance.cancel();
        else await cancelQueueJob(fresh.target, row.id);
      } else if (action === "retry") {
        if (provenance) {
          const authority = provenance.job.durableBatch;
          if (
            job.state !== "held" ||
            job.retryable !== true ||
            !provenance.job.retryable ||
            !authority?.serverBatchId?.trim() ||
            !authority.clientBatchId.trim() ||
            authority.expectedInstanceId !== row.instanceId ||
            provenance.job.serverId !== row.id
          )
            throw new Error(
              "This local job is not ready to retry. Refresh its machine details.",
            );
          mutationStarted = true;
          await provenance.retry();
        } else {
          const authority = retryAuthority(row, fresh.detail);
          if (!authority)
            throw new Error(
              "This machine has not supplied retry authority for this job.",
            );
          mutationStarted = true;
          const outcome = await retryQueueJobRecoveringAmbiguity(
            fresh.target,
            authority,
          );
          if (outcome.kind === "uncertain") throw new Error(outcome.error);
        }
      } else {
        if (
          capabilities.value?.queue?.can_pause_job !== true ||
          !["queued", "paused"].includes(job.state)
        )
          throw new Error(
            "This job cannot be paused or resumed in its current state.",
          );
        mutationStarted = true;
        await setQueueJobPaused(fresh.target, row.id, action === "pause");
      }
      await refresh();
      if (ticket === epoch) close();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      if (ticket === epoch) error.value = message;
      else if (mutationStarted) toast("error", `${row.hostLabel}: ${message}`);
    } finally {
      if (ticket === epoch) {
        busy.value = false;
        pendingAction.value = null;
      }
    }
  }
  onBeforeUnmount(close);
  return {
    selected,
    detail,
    model,
    error,
    busy,
    pendingAction,
    canPause,
    open,
    close,
    act,
    snapshot,
  };
}

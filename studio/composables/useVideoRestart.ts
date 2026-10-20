import { computed, ref, type ComputedRef, type Ref } from "vue";
import { ApiError } from "../api/client";
import {
  admitChainRestart,
  assertChainRestartDestination,
  chainRestartOptions,
  exportChainRestart,
  type ChainRestartAdmission,
  type ChainRestartOptions,
} from "../api/chainRestart";
import {
  assertChainOwner,
  cancelChainOperation,
  definiteChainRefusal,
  getChainJob,
  sameChainOwner,
  type ChainOwner,
} from "../api/chainJobs";
import type { ChainJobDetail } from "../lib/api/chainTypes";
import type { QueueTransferHost } from "../api/queueTransfer";

export const VIDEO_RESTART_KEY = "mold.video-restarts.admissions.v1";
function owner(host: QueueTransferHost): ChainOwner {
  return {
    hostId: host.id,
    label: host.label,
    instanceId: host.instanceId,
    target: { ...host.target },
  };
}
const message = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause);

/** Read-only opening; every new attempt requires an explicit action. Retained
 * media and credentials stay in memory, never in the recovery journal. */
export function createVideoRestart(
  hosts: ComputedRef<QueueTransferHost[]>,
  fleetBusy: Ref<boolean>,
) {
  const active = ref(false),
    loading = ref(false),
    busy = ref(false),
    cancelling = ref(false),
    error = ref("");
  const source = ref<ChainOwner | null>(null),
    sourceJobId = ref("");
  const options = ref<ChainRestartOptions | null>(null),
    selected = ref("");
  const cancelledBeforeAdmission = ref(false);
  const admission = ref<ChainRestartAdmission | null>(null),
    detail = ref<ChainJobDetail | null>(null);
  let epoch = 0;
  const destinations = computed(() =>
    hosts.value.filter(
      (host) =>
        host.ready &&
        host.generates !== false &&
        (host.instanceId === source.value?.instanceId
          ? options.value?.restart.supported
          : options.value?.send.supported),
    ),
  );
  const filename = computed(
    () => detail.value?.finalizes?.at(-1)?.gallery_filename ?? null,
  );
  const settled = computed(
    () =>
      cancelledBeforeAdmission.value ||
      (!!detail.value &&
        ["completed", "cancelled", "failed"].includes(detail.value.state)),
  );
  const status = computed(() => {
    if (cancelledBeforeAdmission.value) return "Cancelled before queueing";
    if (admission.value?.cancelRequested && !settled.value)
      return "Cancellation requested; checking this machine…";
    if (!detail.value)
      return admission.value ? "Confirming this video on its destination…" : "";
    if (detail.value.state === "completed")
      return filename.value
        ? "Saved to Library"
        : "Completed, but no saved Library video was reported.";
    if (detail.value.execution_phase === "finalizing") return "Joining clips…";
    if (detail.value.state === "running")
      return `Making clip ${Math.min(detail.value.current_stage + 1, detail.value.stage_count)} of ${detail.value.stage_count}`;
    return (
      detail.value.state.charAt(0).toUpperCase() + detail.value.state.slice(1)
    );
  });
  function currentOwner(frozen: ChainOwner): boolean {
    const host = hosts.value.find((entry) => entry.id === frozen.hostId);
    return (
      !!host?.ready &&
      host.instanceId === frozen.instanceId &&
      host.target.baseUrl === frozen.target.baseUrl &&
      host.target.apiKey === frozen.target.apiKey &&
      host.target.connectionMode === frozen.target.connectionMode
    );
  }
  function journal(): ChainRestartAdmission[] {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(VIDEO_RESTART_KEY) ?? "[]",
    );
    if (
      !Array.isArray(parsed) ||
      parsed.some(
        (row) =>
          !row ||
          [
            "operationId",
            "hostId",
            "instanceId",
            "baseUrl",
            "sourceJobId",
            "sourceHostId",
            "sourceInstanceId",
            "sourceRevision",
          ].some((key) => typeof row[key] !== "string"),
      )
    )
      throw new Error(
        "This device’s video recovery record is unreadable. Keep it before repairing storage.",
      );
    return parsed as ChainRestartAdmission[];
  }
  function persist(record: ChainRestartAdmission, show = true) {
    const entries = journal().filter(
      (row) => row.operationId !== record.operationId,
    );
    localStorage.setItem(
      VIDEO_RESTART_KEY,
      JSON.stringify([...entries, record]),
    );
    if (show) admission.value = record;
  }
  function forget(id: string) {
    localStorage.setItem(
      VIDEO_RESTART_KEY,
      JSON.stringify(journal().filter((row) => row.operationId !== id)),
    );
  }
  function destinationFor(record: ChainRestartAdmission): ChainOwner {
    const host = hosts.value.find(
      (candidate) => candidate.id === record.hostId,
    );
    if (!host || !sameChainOwner(record, owner(host)) || !host.ready)
      throw new Error(
        "Reconnect this video’s original destination before recovering it. Its installation must match.",
      );
    return owner(host);
  }
  async function refresh() {
    const known = admission.value,
      attempt = epoch;
    if (!known) return;
    try {
      const destination = destinationFor(known);
      await assertChainOwner(destination);
      if (attempt !== epoch || !currentOwner(destination)) return;
      const result = await getChainJob(destination, known.operationId);
      if (attempt !== epoch || !currentOwner(destination)) return;
      detail.value = result;
      if (!result && known.cancelAcknowledged) {
        cancelledBeforeAdmission.value = true;
        forget(known.operationId);
      }
      if (result && ["completed", "cancelled"].includes(result.state))
        forget(known.operationId);
    } catch (cause) {
      if (attempt === epoch) error.value = message(cause);
    }
  }
  async function open(hostId: string, jobId: string) {
    const attempt = ++epoch;
    active.value = true;
    cancelledBeforeAdmission.value = false;
    loading.value = true;
    error.value = "";
    options.value = null;
    detail.value = null;
    admission.value = null;
    selected.value = "";
    try {
      const host = hosts.value.find((candidate) => candidate.id === hostId);
      if (!host)
        throw new Error("Reconnect this video’s machine before opening it.");
      source.value = owner(host);
      sourceJobId.value = jobId;
      const saved = journal()
        .slice()
        .reverse()
        .find(
          (row) =>
            row.sourceHostId === hostId &&
            row.sourceJobId === jobId &&
            row.sourceInstanceId === host.instanceId,
        );
      if (saved) {
        admission.value = saved;
        selected.value = saved.hostId;
        await refresh();
        return;
      }
      if (!host.ready)
        throw new Error(
          "Reconnect this video’s machine before starting it again.",
        );
      const result = await chainRestartOptions(source.value, jobId);
      if (attempt !== epoch) return;
      if (
        !currentOwner(source.value) ||
        result.source_instance_id !== source.value.instanceId
      )
        throw new Error(
          "The source machine installation changed. Reopen this video.",
        );
      options.value = result;
      selected.value = destinations.value[0]?.id ?? "";
    } catch (cause) {
      if (attempt === epoch)
        error.value =
          cause instanceof ApiError && cause.status === 404
            ? "Update this machine before starting a saved video job again."
            : message(cause);
    } finally {
      if (attempt === epoch) loading.value = false;
    }
  }
  async function start() {
    const frozenSource = source.value,
      frozenOptions = options.value;
    const host = destinations.value.find(
      (entry) => entry.id === selected.value,
    );
    if (
      !frozenSource ||
      !frozenOptions ||
      !host ||
      admission.value ||
      busy.value ||
      fleetBusy.value
    )
      return;
    const destination = owner(host),
      attempt = epoch;
    const isCurrent = () =>
      attempt === epoch &&
      active.value &&
      selected.value === destination.hostId &&
      currentOwner(frozenSource) &&
      currentOwner(destination);
    busy.value = true;
    fleetBusy.value = true;
    error.value = "";
    let pending: ChainRestartAdmission | null = null;
    try {
      await assertChainRestartDestination(destination);
      if (!isCurrent())
        throw new Error(
          "The source or destination machine changed. Reopen this video.",
        );
      const exported = await exportChainRestart(
        frozenSource,
        sourceJobId.value,
        frozenOptions,
        destination.instanceId!,
      );
      if (!isCurrent())
        throw new Error(
          "The source or destination machine changed while reading inputs.",
        );
      await admitChainRestart({
        source: frozenSource,
        sourceJobId: sourceJobId.value,
        options: frozenOptions,
        destination,
        requestJson: exported.request_json,
        operationId: crypto.randomUUID(),
        isCurrent,
        persist: (record) => {
          persist(record);
          pending = record;
        },
      });
    } catch (cause) {
      const refusedDestination =
        cause instanceof ApiError &&
        cause.status === 409 &&
        typeof cause.body === "object" &&
        cause.body !== null &&
        "code" in cause.body &&
        cause.body.code === "CHAIN_DESTINATION_CHANGED";
      if (pending && (definiteChainRefusal(cause) || refusedDestination)) {
        forget((pending as ChainRestartAdmission).operationId);
        if (attempt === epoch) admission.value = null;
      }
      if (attempt === epoch) error.value = message(cause);
    } finally {
      busy.value = false;
      fleetBusy.value = false;
      if (attempt === epoch && admission.value) await refresh();
    }
  }
  async function cancel() {
    const known = admission.value,
      attempt = epoch;
    if (!known || cancelling.value || settled.value) return;
    cancelling.value = true;
    try {
      const destination = destinationFor(known);
      persist({ ...known, cancelRequested: true });
      await cancelChainOperation(
        destination,
        known.operationId,
        () => attempt === epoch && currentOwner(destination),
      );
      persist(
        { ...known, cancelRequested: true, cancelAcknowledged: true },
        attempt === epoch,
      );
      if (attempt === epoch) await refresh();
    } catch (cause) {
      if (attempt === epoch) error.value = message(cause);
    } finally {
      cancelling.value = false;
    }
  }
  function close() {
    epoch++;
    active.value = false;
    loading.value = false;
    options.value = null;
  }
  return {
    active,
    loading,
    busy,
    cancelling,
    error,
    source,
    options,
    selected,
    destinations,
    admission,
    detail,
    settled,
    filename,
    status,
    open,
    start,
    refresh,
    cancel,
    close,
  };
}

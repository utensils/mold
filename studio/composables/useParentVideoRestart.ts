import { computed, ref, shallowRef, type ComputedRef } from "vue";
import { definiteChainRefusal, sameChainOwner } from "../api/chainJobs";
import { type VideoJobOwner, type VideoJobAdmission } from "../api/videoJobs";
import {
  admitVideoRestart,
  exportVideoRestart,
  transferVideoRestartPrefix,
  validateVideoRestart,
  videoRestartOptions,
  type VideoRestartOptions,
} from "../api/videoJobRestart";
import type { QueueTransferHost } from "../api/queueTransfer";
import {
  readVideoJobOperations,
  saveVideoJobOperation,
  forgetVideoJobOperation,
} from "../lib/videoJobRecovery";
import { createVideoJobRun } from "./useVideoJobRun";

/** Explicit new attempt. Original progress/inputs remain on the source machine. */
export function createParentVideoRestart(
  hosts: ComputedRef<QueueTransferHost[]>,
) {
  const active = ref(false),
    loading = ref(false),
    busy = ref(false),
    error = ref(""),
    progress = ref("");
  const source = shallowRef<VideoJobOwner | null>(null),
    sourceJobId = ref(""),
    selected = ref("");
  const options = shallowRef<VideoRestartOptions | null>(null);
  let epoch = 0;
  function current(owner: VideoJobOwner) {
    const host = hosts.value.find(
      (host) => host.id === owner.hostId && host.ready,
    );
    return (
      !!host &&
      host.instanceId === owner.instanceId &&
      host.target.baseUrl === owner.target.baseUrl &&
      host.target.apiKey === owner.target.apiKey &&
      host.target.connectionMode === owner.target.connectionMode
    );
  }
  const run = createVideoJobRun(current);
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
  const destination = () => {
    const host = destinations.value.find((host) => host.id === selected.value);
    return host
      ? {
          hostId: host.id,
          label: host.label,
          instanceId: host.instanceId,
          target: { ...host.target },
        }
      : null;
  };
  async function open(owner: VideoJobOwner, id: string) {
    if (busy.value) return;
    const attempt = ++epoch;
    source.value = { ...owner, target: { ...owner.target } };
    sourceJobId.value = id;
    selected.value = owner.hostId;
    options.value = null;
    error.value = "";
    progress.value = "";
    active.value = true;
    loading.value = true;
    try {
      if (!current(owner))
        throw new Error("Reconnect this video's original machine.");
      const value = await videoRestartOptions(owner, id);
      if (attempt !== epoch || !current(owner)) return;
      options.value = value;
      if (!destinations.value.some((host) => host.id === selected.value))
        selected.value = destinations.value[0]?.id ?? "";
    } catch (cause) {
      if (attempt === epoch)
        error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (attempt === epoch) loading.value = false;
    }
  }
  async function start() {
    if (busy.value || run.uncertain.value) return;
    const original = source.value,
      target = destination(),
      permission = options.value,
      id = sourceJobId.value,
      attempt = epoch;
    if (!original || !target || !permission) return;
    const fresh = () =>
      attempt === epoch &&
      current(original) &&
      current(target) &&
      selected.value === target.hostId;
    let receipt: VideoJobAdmission | null = null;
    busy.value = true;
    error.value = "";
    try {
      if (
        readVideoJobOperations().some((record) =>
          sameChainOwner(record, target),
        )
      )
        throw new Error(
          "Check this machine's saved video attempt in Queue before starting another.",
        );
      progress.value = "Checking saved inputs…";
      const exported = await exportVideoRestart(
        original,
        id,
        permission,
        target,
        fresh,
      );
      if (!fresh())
        throw new Error("The machine changed. Refresh and try again.");
      const upload = await transferVideoRestartPrefix({
        source: original,
        sourceJobId: id,
        destination: target,
        exported,
        uploadId: crypto.randomUUID(),
        isCurrent: fresh,
        progress: (bytes, total) => {
          if (fresh())
            progress.value = `Copying original video: ${bytes} of ${total} bytes`;
        },
      });
      if (!fresh())
        throw new Error("The machine changed. Refresh and try again.");
      const plan = await validateVideoRestart(target, exported, upload);
      if (!fresh())
        throw new Error("The machine changed. Refresh and try again.");
      progress.value = "Queueing new attempt…";
      await admitVideoRestart({
        source: original,
        sourceJobId: id,
        options: permission,
        destination: target,
        exported,
        proof: plan.validation_revision,
        operationId: crypto.randomUUID(),
        ...(upload ? { upload } : {}),
        isCurrent: fresh,
        persist: (value) => {
          // Only exact operation/owner scalars survive; opaque payloads stay private.
          const record: VideoJobAdmission = {
            kind: "video_job",
            operationId: value.operationId,
            hostId: value.hostId,
            instanceId: value.instanceId,
            baseUrl: value.baseUrl,
            createdAt: value.createdAt,
            ...(value.connectionMode
              ? { connectionMode: value.connectionMode }
              : {}),
          };
          saveVideoJobOperation(record);
          receipt = record;
        },
      });
      if (receipt && current(target)) {
        await run.recover(target, receipt);
        progress.value =
          "New attempt queued. The original remains on its machine.";
      }
    } catch (cause) {
      if (receipt) {
        if (definiteChainRefusal(cause)) forgetVideoJobOperation(receipt);
        else if (current(target)) await run.recover(target, receipt);
      }
      if (attempt === epoch)
        error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy.value = false;
    }
  }
  function close() {
    if (busy.value) return;
    active.value = false;
    epoch += 1;
  }
  return {
    active,
    loading,
    busy,
    error,
    progress,
    source,
    options,
    selected,
    destinations,
    run,
    open,
    start,
    close,
    current,
  };
}

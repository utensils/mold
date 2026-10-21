import { computed, ref, shallowRef } from "vue";
import { ApiError } from "../api/client";
import { definiteChainRefusal, sameChainOwner } from "../api/chainJobs";
import {
  admitVideoJob,
  assertVideoJobOwner,
  getVideoJob,
  mutateVideoJob,
  validateVideoJob,
  type PreparedDurationVideo,
  type VideoJobAdmission,
  type VideoJobOwner,
} from "../api/videoJobs";
import type { VideoJob, VideoJobRequest } from "../lib/api/videoJobTypes";
import {
  forgetVideoJobOperation,
  readVideoJobOperations,
  saveVideoJobOperation,
} from "../lib/videoJobRecovery";

const message = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause);
const preCancelled = (cause: unknown) =>
  cause instanceof ApiError &&
  cause.status === 410 &&
  (cause.body as { code?: string } | null)?.code ===
    "VIDEO_OPERATION_CANCELLED";
/** Finite admission and explicit recovery. Acknowledged parents live in the
 * server's durable Queue; only unanswered operations need a local receipt. */
export function createVideoJobRun(
  isOwnerCurrent: (owner: VideoJobOwner) => boolean,
) {
  const detail = shallowRef<VideoJob | null>(null),
    owner = shallowRef<VideoJobOwner | null>(null);
  const admission = shallowRef<VideoJobAdmission | null>(null);
  const busy = ref(false),
    cancelling = ref(false),
    error = ref("");
  const cancelledBeforeAdmission = ref(false),
    removed = ref(false);
  let acknowledgedId: string | null = null;
  const uncertain = computed(
    () =>
      !!admission.value &&
      detail.value == null &&
      !cancelledBeforeAdmission.value,
  );
  let readRevision = 0;
  let epoch = 0,
    disposed = false;
  let stopFlight: Promise<void> | null = null;
  let release: (() => Promise<void>) | undefined;
  const current = (attempt: number, captured: VideoJobOwner) =>
    !disposed && attempt === epoch && isOwnerCurrent(captured);
  function persist(record: VideoJobAdmission) {
    saveVideoJobOperation(record);
    admission.value = record;
  }
  async function settled() {
    if (admission.value) forgetVideoJobOperation(admission.value);
    admission.value = null;
    const cleanup = release;
    release = undefined;
    await cleanup?.().catch(() => undefined);
  }
  async function cancelKnown(attempt: number, captured: VideoJobOwner) {
    const known = admission.value;
    if (!known || known.cancelAcknowledged || !current(attempt, captured))
      return;
    if (stopFlight) return stopFlight;
    persist({ ...known, cancelRequested: true });
    cancelling.value = true;
    const flight = (async () => {
      await mutateVideoJob(captured, known.operationId, "cancel", () =>
        current(attempt, captured),
      );
      if (
        current(attempt, captured) &&
        admission.value?.operationId === known.operationId
      )
        persist({
          ...admission.value,
          cancelRequested: true,
          cancelAcknowledged: true,
        });
    })();
    stopFlight = flight;
    try {
      await flight;
    } finally {
      if (stopFlight === flight) {
        stopFlight = null;
        cancelling.value = false;
      }
    }
  }
  function knownRecord(captured: VideoJobOwner, id: string): VideoJobAdmission {
    return {
      kind: "video_job",
      operationId: id,
      hostId: captured.hostId,
      instanceId: captured.instanceId!,
      baseUrl: captured.target.baseUrl,
      ...(captured.target.connectionMode
        ? { connectionMode: captured.target.connectionMode }
        : {}),
      createdAt: Date.now(),
    };
  }
  async function refresh() {
    const read = ++readRevision;
    const captured = owner.value,
      attempt = epoch;
    const known =
      admission.value ??
      (captured && detail.value
        ? knownRecord(captured, detail.value.id)
        : null);
    if (!known || !captured || !current(attempt, captured)) return;
    error.value = "";
    try {
      await assertVideoJobOwner(captured);
      if (!current(attempt, captured)) return;
      if (known.cancelRequested) await cancelKnown(attempt, captured);
      if (!current(attempt, captured)) return;
      const result = await getVideoJob(captured, known.operationId);
      if (!current(attempt, captured) || read !== readRevision) return;
      if (result?.id && result.id !== known.operationId)
        throw new Error(
          "This machine returned a different video job. Keep the original attempt for recovery.",
        );
      if (!result && acknowledgedId === known.operationId) {
        await retire(known.operationId);
      } else if (result) {
        acknowledgedId = result.id;
        detail.value = result;
        await settled();
      }
    } catch (cause) {
      if (!current(attempt, captured) || read !== readRevision) return;
      if (preCancelled(cause)) {
        cancelledBeforeAdmission.value = true;
        await settled();
      } else error.value = message(cause);
    }
  }
  async function recover(
    captured: VideoJobOwner,
    known: VideoJobAdmission,
    options: { acknowledged?: boolean } = {},
  ) {
    if (
      busy.value ||
      !sameChainOwner(known, captured) ||
      !isOwnerCurrent(captured)
    )
      return;
    epoch += 1;
    acknowledgedId = options.acknowledged ? known.operationId : null;
    removed.value = false;
    owner.value = { ...captured, target: { ...captured.target } };
    admission.value = known;
    detail.value = null;
    error.value = "";
    cancelledBeforeAdmission.value = false;
    await refresh();
  }
  async function submit<Generation extends object>(
    captured: VideoJobOwner,
    request: VideoJobRequest<Generation>,
    options: {
      isDraftCurrent?: () => boolean;
      prepare?: () => Promise<PreparedDurationVideo<Generation>>;
    } = {},
  ) {
    if (busy.value || uncertain.value || !isOwnerCurrent(captured)) return;
    let old: VideoJobAdmission | undefined;
    try {
      old = readVideoJobOperations().find((record) =>
        sameChainOwner(record, captured),
      );
    } catch (cause) {
      error.value = message(cause);
      return;
    }
    if (old) {
      await recover(captured, old);
      return;
    }
    const attempt = ++epoch;
    acknowledgedId = null;
    removed.value = false;
    owner.value = { ...captured, target: { ...captured.target } };
    detail.value = null;
    cancelledBeforeAdmission.value = false;
    error.value = "";
    busy.value = true;
    const fresh = () =>
      current(attempt, captured) && options.isDraftCurrent?.() !== false;
    let attempted = false;
    try {
      const prepared = (await options.prepare?.()) ?? { request };
      release = prepared.release;
      if (!fresh()) return;
      const plan = await validateVideoJob(captured, prepared.request, {
        isCurrent: fresh,
        ...(prepared.media?.retainedMediaSession
          ? { retainedMediaSession: prepared.media.retainedMediaSession }
          : {}),
      });
      if (!fresh()) return;
      const id = crypto.randomUUID();
      const result = await admitVideoJob(
        captured,
        prepared.request,
        plan,
        id,
        (record) => {
          if (!fresh())
            throw new Error("The video or machine changed before submission.");
          persist(record);
          attempted = true;
        },
        fresh,
        prepared.media,
      );
      if (!current(attempt, captured)) return;
      if (result.id !== id)
        throw new Error(
          "This machine returned a different video job. Keep the original attempt for recovery.",
        );
      acknowledgedId = result.id;
      detail.value = result;
      if (admission.value?.cancelRequested)
        await cancelKnown(attempt, captured);
      await settled();
    } catch (cause) {
      if (current(attempt, captured)) {
        error.value = message(cause);
        if (attempted && definiteChainRefusal(cause)) await settled();
      }
    } finally {
      if (!attempted || !admission.value) {
        const cleanup = release;
        release = undefined;
        await cleanup?.().catch(() => undefined);
      }
      if (attempt === epoch) busy.value = false;
    }
  }
  async function cancel() {
    const captured = owner.value,
      attempt = epoch;
    if (!captured || !current(attempt, captured)) return;
    if (!admission.value && detail.value)
      persist(knownRecord(captured, detail.value.id));
    if (!admission.value) return;
    try {
      await cancelKnown(attempt, captured);
      await refresh();
    } catch (cause) {
      if (current(attempt, captured)) error.value = message(cause);
    }
  }
  async function resume() {
    const captured = owner.value,
      attempt = epoch,
      id = detail.value?.id;
    if (
      !captured ||
      !id ||
      !detail.value?.can_resume ||
      busy.value ||
      cancelling.value ||
      !current(attempt, captured)
    )
      return;
    busy.value = true;
    error.value = "";
    try {
      // Explicit Resume supersedes any prior unanswered Stop before dispatch.
      // A lost Resume reply is reconciled with GET, never automatically replayed.
      if (admission.value)
        persist({
          ...admission.value,
          cancelRequested: false,
          cancelAcknowledged: false,
        });
      await mutateVideoJob(captured, id, "resume", () =>
        current(attempt, captured),
      );
      if (current(attempt, captured)) await refresh();
    } catch (cause) {
      if (current(attempt, captured)) error.value = message(cause);
    } finally {
      if (attempt === epoch) busy.value = false;
    }
  }
  async function retire(id: string) {
    if (id !== (detail.value?.id ?? admission.value?.operationId)) return;
    // Persist local retirement before hiding recovery; storage failure stays actionable.
    if (admission.value) forgetVideoJobOperation(admission.value);
    admission.value = null;
    // Invalidate in-flight reads before clearing an acknowledged parent.
    epoch += 1;
    readRevision += 1;
    removed.value = true;
    acknowledgedId = null;
    detail.value = null;
    error.value = "";
    const cleanup = release;
    release = undefined;
    await cleanup?.().catch(() => undefined);
  }
  async function remove() {
    const captured = owner.value,
      id = detail.value?.id;
    if (
      !captured ||
      !id ||
      busy.value ||
      !detail.value?.can_delete ||
      !isOwnerCurrent(captured)
    )
      return;
    const attempt = ++epoch;
    readRevision += 1;
    busy.value = true;
    try {
      await mutateVideoJob(captured, id, "delete", () =>
        current(attempt, captured),
      );
      if (current(attempt, captured)) await retire(id);
    } finally {
      busy.value = false;
    }
  }
  function forgetRemoved(id: string) {
    const captured = owner.value;
    if (captured && isOwnerCurrent(captured) && acknowledgedId === id)
      void retire(id).catch((cause) => {
        error.value = message(cause);
      });
  }
  function dispose() {
    disposed = true;
    epoch += 1;
    busy.value = false;
  }
  return {
    detail,
    owner,
    admission,
    busy,
    cancelling,
    error,
    uncertain,
    cancelledBeforeAdmission,
    removed,
    remove,
    forgetRemoved,
    submit,
    recover,
    refresh,
    cancel,
    resume,
    dispose,
  };
}

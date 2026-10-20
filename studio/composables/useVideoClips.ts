import {
  loadVideoClipsDraft,
  saveVideoClipsDraft,
} from "../lib/chainDraftStorage";
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { apiJsonTo } from "../api/client";
import {
  admitClips,
  assertChainOwner,
  captureChainOwner,
  CHAIN_ADMISSION_KEY,
  fetchChainLimits,
  getChainJob,
  mutateChain,
  cancelChainOperation,
  definiteChainRefusal,
  sameChainOwner,
  sameChainRoute,
  validateClips,
  type ChainAdmission,
  type ChainLimitsWire,
  type ChainOwner,
  type ChainPlan,
} from "../api/chainJobs";
import {
  buildClipsRequest,
  chainInputRefusal,
  editClipOrder,
  type ClipsBaseRequest,
  type VideoClipDraft,
} from "../lib/chainDraft";
import type {
  ChainContinuation,
  ChainJobDetail,
  ChainRequestWire,
} from "../lib/api/chainTypes";
import {
  clampVideoFrames,
  minVideoFrames,
  videoFrameStep,
  videoFrameOffset,
  type VideoFrameContract,
} from "../lib/videoDuration";

export interface VideoClipsOptions {
  request?: ClipsBaseRequest | null | undefined;
  owner?: ChainOwner | null | undefined;
  enabled?: boolean | undefined;
  ownerSelection?: string | undefined;
  frameContract?: VideoFrameContract | null | undefined;
  blockedReason?: string | null | undefined;
  source?: { jobId: string; filename: string } | null | undefined;
}
export interface VideoClipsEvents {
  (event: "active", value: boolean): void;
  (event: "single", value: { prompt: string; frames: number }): void;
  (event: "published", value: { filename: string; owner: ChainOwner }): void;
}
export function useVideoClips(
  props: VideoClipsOptions,
  emit: VideoClipsEvents,
) {
  const active = ref(false),
    loading = ref(false),
    busy = ref(false),
    error = ref("");
  const captured = ref<{
    owner: ChainOwner;
    request: ClipsBaseRequest;
    frameContract?: VideoFrameContract | null;
  } | null>(null);
  const clips = ref<VideoClipDraft[]>([]),
    limits = ref<ChainLimitsWire | null>(null),
    plan = ref<ChainPlan | null>(null);
  const continuationTail = ref<number | null>(null);
  const baseFingerprint = ref("");
  const capturedSelection = ref<string | undefined>();
  const draftKey = ref("");
  const restoredDraft = ref(false);
  const continuation = ref<ChainContinuation>(),
    admission = ref<ChainAdmission | null>(null),
    detail = ref<ChainJobDetail | null>(null);
  let revision = 0,
    timer: ReturnType<typeof setTimeout> | undefined,
    disposed = false;
  const cancelledBeforeAdmission = ref(false);
  const cancelBusy = ref(false);
  const terminal = computed(
    () =>
      cancelledBeforeAdmission.value ||
      (!!detail.value &&
        ["completed", "cancelled"].includes(detail.value.state)),
  );
  const settled = computed(
    () => terminal.value || detail.value?.state === "failed",
  );
  const editing = computed(() => !admission.value && !busy.value);
  const minFrames = computed(() => {
    const grid = limits.value?.authoring;
    const contract = captured.value?.frameContract;
    const step = grid?.frame_step ?? videoFrameStep(contract);
    const offset = grid?.frame_offset ?? videoFrameOffset(contract);
    const tail =
      continuationTail.value ??
      plan.value?.motion_tail_frames ??
      grid?.motion_tail_frames ??
      17;
    const min = Math.max(
      grid?.min_frames ?? minVideoFrames(contract),
      tail + 1,
    );
    return offset + Math.ceil((min - offset) / step) * step;
  });
  const step = computed(
    () =>
      limits.value?.authoring?.frame_step ??
      videoFrameStep(captured.value?.frameContract),
  );
  const maxFrames = computed(
    () => limits.value?.frames_per_clip_cap ?? minFrames.value,
  );
  const lastFilename = computed(
    () => detail.value?.finalizes?.at(-1)?.gallery_filename ?? null,
  );
  const ownerChanged = computed(
    () =>
      captured.value &&
      (props.ownerSelection !== capturedSelection.value ||
        !props.owner ||
        captured.value.owner.hostId !== props.owner.hostId ||
        (props.owner.instanceId &&
          captured.value.owner.instanceId !== props.owner.instanceId) ||
        captured.value.owner.target.baseUrl !== props.owner.target.baseUrl ||
        captured.value.owner.target.apiKey !== props.owner.target.apiKey ||
        captured.value.owner.target.connectionMode !==
          props.owner.target.connectionMode),
  );
  const settingsChanged = computed(
    () =>
      active.value &&
      !props.source &&
      captured.value &&
      JSON.stringify(props.request) !== baseFingerprint.value,
  );
  const refusal = computed(() =>
    settingsChanged.value
      ? "Style or settings changed outside these clips. Go back to one clip and reopen them to review the new settings."
      : captured.value &&
          (
            captured.value.request as ClipsBaseRequest & {
              source_fit?: unknown;
            }
          ).source_fit &&
          limits.value &&
          limits.value.authoring?.source_fit_supported !== true
        ? "Update this machine before using fitted source images with joined clips."
        : (props.blockedReason ??
          (captured.value ? chainInputRefusal(captured.value.request) : null)),
  );
  const inputSummary = computed(() => {
    const fields = captured.value?.request as
      | (ClipsBaseRequest & {
          references?: unknown[];
          keyframes?: Array<{ frame: number }>;
          source_image?: string;
        })
      | undefined;
    const notes: string[] = [];
    if (fields?.references?.length)
      notes.push(
        `${fields.references.length} reference input${fields.references.length === 1 ? "" : "s"} used in every clip.`,
      );
    if (
      fields?.source_image ||
      fields?.keyframes?.some((frame) => frame.frame === 0)
    )
      notes.push("Your first frame starts clip 1.");
    if (fields?.keyframes?.some((frame) => frame.frame > 0))
      notes.push("Your last frame ends the final clip.");
    return notes.join(" ");
  });
  const status = computed(() => {
    const job = detail.value;
    if (cancelledBeforeAdmission.value) return "Cancelled before queueing";
    if (admission.value?.cancelRequested && !settled.value)
      return "Cancellation requested; waiting for this machine…";
    if (!job)
      return admission.value ? "Confirming this video on its machine…" : "";
    if (job.state === "completed")
      return lastFilename.value
        ? "Saved to Library"
        : "Rendering completed, but this machine did not report a saved Library video.";
    if (job.execution_phase === "finalizing") return "Joining clips…";
    if (job.state === "running")
      return `Making clip ${Math.min(job.current_stage + 1, job.stage_count)} of ${job.stage_count}`;
    return job.state.charAt(0).toUpperCase() + job.state.slice(1);
  });
  function newClip(prompt = "", frames?: number): VideoClipDraft {
    const fallback =
      limits.value?.frames_per_clip_recommended ?? minFrames.value;
    return {
      id: crypto.randomUUID(),
      prompt,
      frames: Math.max(
        minFrames.value,
        Math.min(maxFrames.value, frames ?? fallback),
      ),
    };
  }
  function readJournal(): ChainAdmission[] {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(CHAIN_ADMISSION_KEY) ?? "[]",
    );
    if (!Array.isArray(parsed))
      throw new Error("The saved video recovery record is unreadable.");
    if (
      !parsed.every(
        (row) =>
          row &&
          typeof row.operationId === "string" &&
          typeof row.instanceId === "string" &&
          typeof row.hostId === "string" &&
          typeof row.baseUrl === "string",
      )
    )
      throw new Error(
        "The saved video recovery record is unreadable. Keep it until the original video is reconciled.",
      );
    return parsed as ChainAdmission[];
  }
  function persist(record: ChainAdmission) {
    const rows = readJournal().filter(
      (row) => row.operationId !== record.operationId,
    );
    localStorage.setItem(
      CHAIN_ADMISSION_KEY,
      JSON.stringify([...rows, record]),
    );
    admission.value = record;
  }
  function forgetSettled() {
    if (!admission.value || !terminal.value) return;
    const id = admission.value.operationId;
    localStorage.setItem(
      CHAIN_ADMISSION_KEY,
      JSON.stringify(readJournal().filter((row) => row.operationId !== id)),
    );
    admission.value = null;
    detail.value = null;
    cancelledBeforeAdmission.value = false;
  }
  function stopPoll() {
    if (timer) clearTimeout(timer);
    timer = undefined;
  }
  async function refresh() {
    stopPoll();
    const frozen = captured.value,
      pending = admission.value,
      stamp = revision;
    if (
      !frozen ||
      !pending ||
      ownerChanged.value ||
      disposed ||
      !sameChainOwner(pending, frozen.owner)
    )
      return;
    try {
      await assertChainOwner(frozen.owner);
      if (pending.cancelRequested && !pending.cancelAcknowledged) {
        await cancelChainOperation(
          frozen.owner,
          pending.operationId,
          () => stamp === revision && !disposed && !ownerChanged.value,
        );
        if (stamp !== revision || disposed || ownerChanged.value) return;
        persist({ ...pending, cancelAcknowledged: true });
      }
      const result = await getChainJob(frozen.owner, pending.operationId);
      if (stamp !== revision || disposed || ownerChanged.value) return;
      if (result && result.id !== pending.operationId)
        throw new Error(
          "This machine returned a different video job. Reconnect the original machine.",
        );
      detail.value = result;
      cancelledBeforeAdmission.value =
        !result && admission.value?.cancelAcknowledged === true;
      error.value =
        result?.error ??
        (result || cancelledBeforeAdmission.value
          ? ""
          : "The machine has not confirmed this video yet. Keep this recovery record and reconnect; do not submit it again.");
    } catch (cause) {
      if (stamp === revision && !disposed)
        error.value = String(cause instanceof Error ? cause.message : cause);
    }
    if (
      stamp === revision &&
      !disposed &&
      !ownerChanged.value &&
      !settled.value
    )
      timer = setTimeout(() => void refresh(), 2500);
  }
  async function open(recoverOnly = false) {
    if (!props.owner || (!props.request && !props.source)) return;
    const stamp = ++revision;
    const sourceInput = props.source ? { ...props.source } : null;
    baseFingerprint.value = JSON.stringify(props.request);
    capturedSelection.value = props.ownerSelection;
    active.value = true;
    cancelledBeforeAdmission.value = false;
    cancelBusy.value = false;
    busy.value = false;
    clips.value = [];
    limits.value = null;
    plan.value = null;
    admission.value = null;
    detail.value = null;
    continuation.value = undefined;
    continuationTail.value = null;
    loading.value = true;
    error.value = "";
    emit("active", true);
    captured.value = {
      ...(props.frameContract
        ? { frameContract: { ...props.frameContract } }
        : {}),
      owner: { ...props.owner, target: { ...props.owner.target } },
      request: JSON.parse(
        JSON.stringify(props.request ?? {}),
      ) as ClipsBaseRequest,
    };
    try {
      // A local shell may know its route before its status UUID is hydrated.
      // Recovery uses the journal's original installation as the expected
      // authority, then verifies the live server before reading or mutating it.
      if (recoverOnly && !captured.value.owner.instanceId) {
        const pending = readJournal().find((row) =>
          sameChainRoute(row, captured.value!.owner),
        );
        if (pending) captured.value.owner.instanceId = pending.instanceId;
      }
      const before = captured.value.owner;
      const owner = before.instanceId
        ? before
        : await captureChainOwner(before);
      if (stamp !== revision || disposed || ownerChanged.value) return;
      captured.value.owner = owner;
      await assertChainOwner(owner);
      const recovery = readJournal().find((row) => sameChainOwner(row, owner));
      if (recovery) {
        admission.value = recovery;
        await refresh();
        if (stamp !== revision || disposed || ownerChanged.value) return;
        // Existing durable work needs its exact job, not today's model catalog.
        if (!terminal.value || recoverOnly) return;
        forgetSettled();
      }
      let base = captured.value.request;
      if (sourceInput) {
        let source = await getChainJob(owner, sourceInput.jobId);
        const take = source?.finalizes?.at(-1);
        const takeNumber = source?.finalizes?.length;
        if (
          !source ||
          source.state !== "completed" ||
          !source.script ||
          !takeNumber ||
          !take ||
          take.gallery_filename !== sourceInput.filename
        )
          throw new Error(
            "Open the latest completed video part to continue it. Its saved job must still be available on this machine.",
          );
        // Only an explicit source-media read can turn retained references into a new request.
        if (source.script.stage.at(-1)?.generation_references?.length) {
          source = await getChainJob(owner, sourceInput.jobId, true);
          if (
            !source?.script ||
            source.finalizes?.length !== takeNumber ||
            source.finalizes?.at(-1)?.gallery_filename !== take.gallery_filename
          )
            throw new Error(
              "The source video changed while its references were loading. Open its latest take again.",
            );
        }
        const previous = source.script.stage.at(-1);
        const {
          seed: _oldSeed,
          continue_from: _oldLink,
          motion_tail_frames: _oldTail,
          ...settings
        } = source.script.chain;
        base = {
          ...settings,
          prompt: "",
          frames: previous?.frames ?? 97,
          ...(previous?.generation_references?.length
            ? { references: previous.generation_references }
            : {}),
        };
        continuation.value = { job_id: source.id, finalize_take: takeNumber };
        continuationTail.value = source.script.chain.motion_tail_frames;
        captured.value.request = base;
      }
      const result = await fetchChainLimits(owner, base.model, base.fps ?? 24);
      if (stamp !== revision || disposed || ownerChanged.value) return;
      if (!result.supports_sequence)
        throw new Error(
          result.sequence_unsupported_reason ??
            "This style does not support video clips.",
        );
      if (continuation.value && result.linked_parts?.supported !== true)
        throw new Error("Update this machine before continuing video parts.");
      if (!result.authoring && !captured.value.frameContract) {
        const models = await apiJsonTo<Array<VideoFrameContract>>(
          owner.target,
          "/api/models",
        );
        if (stamp !== revision || disposed || ownerChanged.value) return;
        const contract = models.find((model) => model.name === base.model);
        if (!contract)
          throw new Error(
            "This machine cannot describe the video's clip lengths. Update it before continuing.",
          );
        captured.value.frameContract = contract;
      }
      limits.value = result;
      draftKey.value = JSON.stringify([
        owner.hostId,
        owner.instanceId,
        base.model,
        sourceInput?.jobId ?? null,
        continuation.value?.finalize_take ?? null,
      ]);
      const savedDraft = loadVideoClipsDraft(draftKey.value);
      restoredDraft.value = Boolean(savedDraft?.length);
      clips.value = savedDraft?.length
        ? savedDraft
        : sourceInput
          ? [newClip("", base.frames ?? undefined)]
          : [newClip(base.prompt, base.frames ?? undefined), newClip()];
      if (!sourceInput && clips.value.every((clip) => clip.prompt.trim()))
        await validate();
    } catch (cause) {
      if (stamp === revision)
        error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (stamp === revision) loading.value = false;
    }
  }
  function close() {
    const first =
      !props.source && !admission.value && !settingsChanged.value
        ? clips.value[0]
        : null;
    try {
      forgetSettled();
    } catch (cause) {
      error.value = String(cause);
      return;
    }
    if (first) emit("single", { prompt: first.prompt, frames: first.frames });
    revision++;
    stopPoll();
    active.value = false;
    captured.value = null;
    admission.value = null;
    detail.value = null;
    continuation.value = undefined;
    continuationTail.value = null;
    limits.value = null;
    plan.value = null;
    error.value = "";
    emit("active", false);
  }
  function edited() {
    plan.value = null;
    error.value = "";
    revision++;
    if (draftKey.value) {
      try {
        saveVideoClipsDraft(draftKey.value, clips.value);
      } catch {
        error.value =
          "These clips could not be saved on this device. Keep this window open.";
      }
    }
  }
  function add() {
    if (clips.value.length < (limits.value?.max_stages ?? 16)) {
      clips.value.push(newClip());
      edited();
    }
  }
  function move(id: string, delta: -1 | 1) {
    clips.value = editClipOrder(clips.value, id, delta);
    edited();
  }
  function setFrames(clip: VideoClipDraft, raw: string) {
    const authoring = limits.value?.authoring;
    const n = Number(raw);
    clip.frames = authoring
      ? Math.max(
          minFrames.value,
          Math.min(
            maxFrames.value,
            authoring.frame_offset +
              Math.round((n - authoring.frame_offset) / step.value) *
                step.value,
          ),
        )
      : Math.min(
          maxFrames.value,
          clampVideoFrames(
            n,
            captured.value?.request.fps ?? 24,
            captured.value?.frameContract,
          ),
        );
    edited();
  }
  function request(): ChainRequestWire {
    if (!captured.value || !limits.value)
      throw new Error("Reconnect the video's machine first.");
    return buildClipsRequest(
      captured.value.request,
      clips.value,
      continuationTail.value ??
        plan.value?.motion_tail_frames ??
        limits.value.authoring?.motion_tail_frames ??
        17,
      continuation.value,
    );
  }
  async function validate(): Promise<ChainRequestWire | null> {
    if (!captured.value || ownerChanged.value || refusal.value) return null;
    const stamp = revision;
    try {
      await assertChainOwner(captured.value.owner);
      const candidate = request();
      const result = await validateClips(captured.value.owner, candidate);
      if (stamp !== revision || disposed || ownerChanged.value || refusal.value)
        return null;
      plan.value = result;
      error.value = "";
      return { ...candidate, motion_tail_frames: result.motion_tail_frames };
    } catch (cause) {
      if (stamp === revision)
        error.value = cause instanceof Error ? cause.message : String(cause);
      return null;
    }
  }
  async function submit() {
    if (busy.value || admission.value || ownerChanged.value || !captured.value)
      return;
    busy.value = true;
    const attempt = revision;
    const submitted: ChainAdmission[] = [];
    try {
      const candidate = await validate();
      if (
        !candidate ||
        attempt !== revision ||
        ownerChanged.value ||
        refusal.value ||
        disposed
      )
        return;
      await admitClips(
        captured.value.owner,
        candidate,
        (record) => {
          submitted.push(record);
          persist(record);
        },
        undefined,
        () =>
          attempt === revision &&
          !disposed &&
          !ownerChanged.value &&
          !refusal.value,
      );
      if (attempt === revision && !disposed) await refresh();
    } catch (cause) {
      const rejected = definiteChainRefusal(cause) && submitted[0];
      if (rejected) {
        const id = rejected.operationId;
        localStorage.setItem(
          CHAIN_ADMISSION_KEY,
          JSON.stringify(readJournal().filter((row) => row.operationId !== id)),
        );
        if ((admission.value as ChainAdmission | null)?.operationId === id)
          admission.value = null;
      }
      if (attempt !== revision || disposed) return;
      error.value = cause instanceof Error ? cause.message : String(cause);
      if (!rejected && admission.value) void refresh();
    } finally {
      if (attempt === revision && !disposed) busy.value = false;
    }
  }
  async function cancelKnownOperation() {
    const pending = admission.value,
      frozen = captured.value;
    if (
      !pending ||
      !frozen ||
      cancelBusy.value ||
      terminal.value ||
      ownerChanged.value ||
      !sameChainOwner(pending, frozen.owner)
    )
      return;
    cancelBusy.value = true;
    const attempt = revision;
    try {
      persist({ ...pending, cancelRequested: true });
      await refresh();
    } catch (cause) {
      if (attempt === revision && !disposed)
        error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (attempt === revision && !disposed) cancelBusy.value = false;
    }
  }
  async function action(kind: "resume" | "cancel") {
    if (
      !captured.value ||
      !admission.value ||
      busy.value ||
      ownerChanged.value ||
      !sameChainOwner(admission.value, captured.value.owner)
    )
      return;
    busy.value = true;
    const stamp = revision;
    try {
      if (kind === "resume") {
        const {
          cancelRequested: _requested,
          cancelAcknowledged: _acknowledged,
          ...resuming
        } = admission.value;
        persist(resuming);
      }
      await mutateChain(
        captured.value.owner,
        admission.value.operationId,
        kind,
        () => stamp === revision && !disposed && !ownerChanged.value,
      );
      if (stamp === revision && !disposed) await refresh();
    } catch (cause) {
      if (stamp === revision && !disposed)
        error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (stamp === revision && !disposed) busy.value = false;
    }
  }
  watch(
    () => [
      props.owner?.hostId,
      props.owner?.instanceId,
      props.owner?.target.baseUrl,
      props.enabled,
    ],
    () => {
      if (
        active.value ||
        props.source ||
        !props.enabled ||
        !props.owner ||
        !props.request
      )
        return;
      try {
        if (
          readJournal().some(
            (row) =>
              sameChainRoute(row, props.owner!) &&
              (!props.owner!.instanceId || sameChainOwner(row, props.owner!)),
          )
        )
          void open(true);
      } catch {
        /* An explicit open names a corrupt/blocked journal without changing ordinary Create. */
      }
    },
    { immediate: true },
  );
  watch(ownerChanged, (changed) => {
    if (changed) {
      stopPoll();
      error.value =
        "This video's machine selection or connection changed. Reconnect its original machine to recover it.";
    } else if (admission.value) void refresh();
  });
  watch([() => props.source?.jobId, () => props.source?.filename], () => {
    revision++;
    stopPoll();
    active.value = false;
    captured.value = null;
    continuation.value = undefined;
    admission.value = null;
    detail.value = null;
    emit("active", false);
  });
  onBeforeUnmount(() => {
    disposed = true;
    revision++;
    stopPoll();
  });
  return {
    active,
    loading,
    busy,
    error,
    captured,
    clips,
    limits,
    plan,
    admission,
    detail,
    terminal,
    settled,
    cancelBusy,
    editing,
    restoredDraft,
    minFrames,
    step,
    maxFrames,
    lastFilename,
    ownerChanged,
    refusal,
    status,
    inputSummary,
    open,
    close,
    edited,
    add,
    move,
    setFrames,
    validate,
    submit,
    refresh,
    action,
    cancelKnownOperation,
  };
}

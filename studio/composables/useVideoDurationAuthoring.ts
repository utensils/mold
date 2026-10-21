import {
  computed,
  onScopeDispose,
  ref,
  watch,
  type ComputedRef,
  type Ref,
} from "vue";
import { createVideoDurationPlan } from "./useVideoDurationPlan";
import { createVideoJobRun } from "./useVideoJobRun";
import {
  buildDurationVideoRequest,
  initialVideoDuration,
  type VideoDurationAuthoring,
} from "../lib/videoDurationAuthoring";
import { readVideoJobOperations } from "../lib/videoJobRecovery";
import { inspectVideoLibraryExtension } from "../api/videoLibraryExtension";
import {
  videoDurationResetPending,
  videoDurationReusePending,
} from "../lib/videoLibraryExtensionHandoff";
import type { VideoLibraryExtensionSelection } from "../lib/videoLibraryExtensionHandoff";
import { sameChainOwner } from "../api/chainJobs";
import { prepareDurationVideo } from "../api/prepareDurationVideo";
import type { VideoJobOwner } from "../api/videoJobs";
import type {
  ReferenceUploadCapabilities,
  ReferenceUploadLease,
  ReferenceUploadRequest,
} from "../api/referenceUploads";

/** Scalar duration/scenes only. Ordinary source/reference stores still own all
 * media and transport grants never enter this persisted authoring record. */
export function createVideoDurationAuthoring(options: {
  key: string;
  clip: ComputedRef<boolean>;
  legacyActive: ComputedRef<boolean>;
  supported: ComputedRef<boolean>;
  owner: ComputedRef<VideoJobOwner | null>;
  frames: ComputedRef<number | null | undefined>;
  fps: ComputedRef<number | null | undefined>;
  generation: ComputedRef<
    | (ReferenceUploadRequest & {
        source_image?: string | null | undefined;
        keyframes?: { frame: number }[] | null | undefined;
      })
    | null
  >;
}) {
  const storageError = ref("");
  const draft = ref<VideoDurationAuthoring | null>(
    null,
  ) as Ref<VideoDurationAuthoring | null>;
  try {
    if (videoDurationResetPending.value) {
      localStorage.removeItem(options.key);
      draft.value = videoDurationReusePending.value;
      videoDurationReusePending.value = null;
      videoDurationResetPending.value = false;
    }
    const raw = localStorage.getItem(options.key);
    if (raw) {
      const value = JSON.parse(raw) as VideoDurationAuthoring;
      if (
        !value ||
        typeof value !== "object" ||
        !Array.isArray(value.scenes) ||
        value.scenes.length > 4096 ||
        typeof value.scenesEnabled !== "boolean" ||
        (value.reuseNotice != null &&
          (typeof value.reuseNotice !== "string" ||
            value.reuseNotice.length > 512)) ||
        (value.durationMs !== null && typeof value.durationMs !== "string") ||
        value.scenes.some(
          (scene) =>
            typeof scene.id !== "string" ||
            typeof scene.prompt !== "string" ||
            (scene.durationMs !== null && typeof scene.durationMs !== "string"),
        )
      )
        throw new Error(
          "This device’s saved video duration is unreadable. Keep it before repairing storage.",
        );
      if (
        Object.keys(value).some(
          (key) =>
            ![
              "durationMs",
              "scenesEnabled",
              "scenes",
              "extension",
              "reuseNotice",
            ].includes(key),
        )
      )
        throw new Error(
          "The saved video draft has unsupported fields. Keep it before repairing storage.",
        );
      const extension = value.extension;
      if (
        extension &&
        (!extension.owner ||
          !extension.source ||
          typeof extension.source.filename !== "string" ||
          !extension.source.filename ||
          typeof extension.title !== "string" ||
          typeof extension.machine !== "string" ||
          !Array.isArray(extension.references) ||
          extension.references.length > 64 ||
          extension.references.some(
            (ref) =>
              typeof ref.name !== "string" || typeof ref.role !== "string",
          ) ||
          [
            extension.owner.hostId,
            extension.owner.instanceId,
            extension.owner.baseUrl,
          ].some((field) => typeof field !== "string" || !field) ||
          Object.keys(extension.owner).some(
            (key) =>
              !["hostId", "instanceId", "baseUrl", "connectionMode"].includes(
                key,
              ),
          ) ||
          Object.keys(extension.source).some(
            (key) =>
              !["filename", "expected_archive_identity_sha256"].includes(key),
          ))
      )
        throw new Error(
          "The saved Library source is unreadable. Keep it before repairing storage.",
        );
      draft.value = value;
    }
  } catch (cause) {
    storageError.value = cause instanceof Error ? cause.message : String(cause);
  }
  const visible = computed(
    () =>
      options.clip.value &&
      !options.legacyActive.value &&
      (options.supported.value || draft.value !== null),
  );
  const enabled = computed(
    () => visible.value && options.supported.value && !storageError.value,
  );
  watch(
    [enabled, options.frames, options.fps],
    () => {
      if (enabled.value && draft.value === null)
        draft.value = initialVideoDuration(
          options.frames.value,
          options.fps.value,
        );
    },
    { immediate: true },
  );
  watch(
    draft,
    (value) => {
      if (storageError.value) return;
      try {
        if (value) localStorage.setItem(options.key, JSON.stringify(value));
        else localStorage.removeItem(options.key);
      } catch {
        storageError.value =
          "This device could not save the video draft. Make storage available before generating.";
      }
    },
    { deep: true },
  );
  const stamp = computed(() => JSON.stringify(draft.value));
  function ownerCurrent(owner: VideoJobOwner) {
    const selected = options.owner.value;
    return (
      !!selected &&
      selected.hostId === owner.hostId &&
      selected.instanceId === owner.instanceId &&
      selected.target.baseUrl === owner.target.baseUrl &&
      selected.target.apiKey === owner.target.apiKey &&
      selected.target.connectionMode === owner.target.connectionMode
    );
  }
  const run = createVideoJobRun(ownerCurrent);
  const request = computed(() => {
    try {
      return enabled.value && draft.value && options.generation.value
        ? buildDurationVideoRequest(draft.value, {
            ...options.generation.value,
            batch_size: 1,
          })
        : null;
    } catch {
      return null;
    }
  });
  const planner = createVideoDurationPlan(enabled, options.owner, request);
  const sourceChecking = ref(false),
    sourceError = ref("");
  let sourceEpoch = 0,
    verifiedSource = "";
  function sourceIdentity() {
    const owner = options.owner.value,
      source = draft.value?.extension?.source;
    return owner && source
      ? JSON.stringify([
          owner.hostId,
          owner.instanceId,
          owner.target.baseUrl,
          owner.target.apiKey,
          owner.target.connectionMode,
          source,
        ])
      : "";
  }
  watch(
    sourceIdentity,
    async (key) => {
      const extension = draft.value?.extension,
        owner = options.owner.value,
        attempt = ++sourceEpoch;
      sourceError.value = "";
      sourceChecking.value = false;
      if (
        !extension ||
        !owner ||
        !extension.source.expected_archive_identity_sha256 ||
        verifiedSource === key
      ) {
        sourceChecking.value = false;
        return;
      }
      if (
        extension.owner.hostId !== owner.hostId ||
        extension.owner.instanceId !== owner.instanceId ||
        extension.owner.baseUrl !== owner.target.baseUrl
      )
        return;
      sourceChecking.value = true;
      try {
        const selected = await inspectVideoLibraryExtension(
          owner,
          extension.source.filename,
          {},
          () => attempt === sourceEpoch && sourceIdentity() === key,
          extension.title,
          extension.source.expected_archive_identity_sha256,
        );
        if (attempt !== sourceEpoch || sourceIdentity() !== key) return;
        verifiedSource = key;
        extension.references = selected.originalReferences.map(
          (reference, index) => ({
            name:
              reference.member?.display_name ??
              `Original reference ${index + 1}`,
            role: reference.member?.role ?? "reference",
          }),
        );
      } catch (cause) {
        if (attempt === sourceEpoch)
          sourceError.value =
            cause instanceof Error ? cause.message : String(cause);
      } finally {
        if (attempt === sourceEpoch) sourceChecking.value = false;
      }
    },
    { immediate: true },
  );
  const blocker = computed(() => {
    if (!visible.value) return null;
    if (storageError.value) return storageError.value;
    if (sourceChecking.value)
      return "Checking this original video and its retained references…";
    if (sourceError.value) return sourceError.value;
    if (!options.supported.value)
      return "Update this machine before making a video with the saved requested duration.";
    const owner = options.owner.value;
    if (!owner?.instanceId)
      return "Reconnect this machine before making this video.";
    const prefix = draft.value?.extension;
    if (
      prefix &&
      (!owner ||
        prefix.owner.hostId !== owner.hostId ||
        prefix.owner.instanceId !== owner.instanceId ||
        prefix.owner.baseUrl !== owner.target.baseUrl ||
        prefix.owner.connectionMode !== owner.target.connectionMode)
    )
      return `This original video belongs to ${prefix.machine}. Reconnect that machine, or start a new video.`;
    if (
      prefix &&
      planner.plan.value?.join_mode === "continuous" &&
      (options.generation.value?.source_image ||
        options.generation.value?.keyframes?.some((frame) => frame.frame === 0))
    )
      return "The original video's ending supplies the first frame. Remove the selected opening image, or start a new video.";
    try {
      if (
        run.uncertain.value ||
        readVideoJobOperations().some((record) => sameChainOwner(record, owner))
      )
        return "Check the saved video attempt in Queue before making another video.";
      if (!draft.value) return "Choose a duration for this video.";
      buildDurationVideoRequest(
        draft.value,
        options.generation.value ?? { model: "" },
      );
      return null;
    } catch (cause) {
      return cause instanceof Error ? cause.message : String(cause);
    }
  });
  const scenesActive = computed(
    () =>
      visible.value &&
      draft.value?.scenesEnabled === true &&
      draft.value.scenes.length > 0,
  );
  async function submitPrepared<Generation extends ReferenceUploadRequest>(
    owner: VideoJobOwner,
    generation: Generation,
    preparation: {
      referenceUploads?: ReferenceUploadCapabilities | null | undefined;
      preparedReferenceLease?: ReferenceUploadLease<Generation> | undefined;
      retainedSource?:
        { filename: string; memberIds: readonly string[] } | undefined;
      isCurrent?: (() => boolean) | undefined;
      signal?: AbortSignal | undefined;
    } = {},
  ) {
    if (!enabled.value || blocker.value || !draft.value)
      throw new Error(blocker.value ?? "This video draft is unavailable.");
    const captured = stamp.value,
      input = buildDurationVideoRequest(draft.value, generation);
    const fresh = () =>
      enabled.value &&
      stamp.value === captured &&
      ownerCurrent(owner) &&
      preparation.isCurrent?.() !== false;
    await run.submit(owner, input, {
      isDraftCurrent: fresh,
      prepare: () =>
        prepareDurationVideo(owner, input, {
          ...preparation,
          isCurrent: fresh,
        }),
    });
    if (run.error.value) throw new Error(run.error.value);
    if (!run.detail.value)
      throw new Error(
        "The video changed before it could be submitted. Check the draft and try again.",
      );
    return run.detail.value;
  }
  function extend(selection: VideoLibraryExtensionSelection) {
    const owner = selection.owner;
    if (!owner.instanceId || !ownerCurrent(owner))
      throw new Error(
        "Reconnect the original video's machine before extending it.",
      );
    const short = initialVideoDuration(options.frames.value, options.fps.value);
    draft.value = {
      ...short,
      extension: {
        source: {
          filename: selection.source.filename,
          ...(selection.source.expected_archive_identity_sha256
            ? {
                expected_archive_identity_sha256:
                  selection.source.expected_archive_identity_sha256,
              }
            : {}),
        },
        owner: {
          hostId: owner.hostId,
          instanceId: owner.instanceId,
          baseUrl: owner.target.baseUrl,
          ...(owner.target.connectionMode
            ? { connectionMode: owner.target.connectionMode }
            : {}),
        },
        title: selection.title,
        machine: owner.label,
        references: selection.originalReferences
          ? selection.originalReferences.map((reference, index) => ({
              name:
                reference.member?.display_name ??
                String(
                  reference.descriptor.filename ??
                    `Original reference ${index + 1}`,
                ),
              role:
                reference.member?.role ??
                String(reference.descriptor.kind ?? "reference"),
            }))
          : selection.inventory.members
              .filter((member) => member.role.includes("reference"))
              .map((member) => ({
                name: member.display_name,
                role: member.role,
              })),
      },
    };
    verifiedSource = sourceIdentity();
  }
  function reset() {
    draft.value = enabled.value
      ? initialVideoDuration(options.frames.value, options.fps.value)
      : null;
  }
  watch(videoDurationResetPending, (pending) => {
    if (pending) {
      draft.value = videoDurationReusePending.value;
      videoDurationReusePending.value = null;
      if (!draft.value) reset();
      videoDurationResetPending.value = false;
    }
  });
  onScopeDispose(() => {
    sourceEpoch += 1;
    planner.dispose();
    run.dispose();
  });
  return {
    draft,
    visible,
    enabled,
    scenesActive,
    stamp,
    blocker,
    planner,
    run,
    ownerCurrent,
    submitPrepared,
    reset,
    extend,
  };
}

<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { requestChoice, toast, undoableAction } from "../lib/toasts";
import { useRoute, useRouter } from "vue-router";
import ComposerCard from "../components/create/ComposerCard.vue";
import ResultCanvas from "../components/create/ResultCanvas.vue";
import ControlsAside from "../components/create/ControlsAside.vue";
import CreateModelPicker from "../components/create/CreateModelPicker.vue";
import AdvancedDrawer from "../components/create/AdvancedDrawer.vue";
import SourceMediaPanel from "../components/create/SourceMediaPanel.vue";
import ActivityStrip from "../components/create/ActivityStrip.vue";
import EstimateBadge from "../components/create/EstimateBadge.vue";
import { advancedActiveCount } from "../components/create/advancedCount";
import {
  effectiveNegativeDefault,
  restoredNegativePrompt,
} from "@studio/lib/negativePrompt";
import { projectResolution } from "../components/create/resolutionProjection";
import SequenceComposer from "../components/SequenceComposer.vue";
import ExpandModal from "../components/ExpandModal.vue";
import RemixModal from "../components/RemixModal.vue";
import ImagePickerModal from "../components/ImagePickerModal.vue";
import MaskEditorModal from "../components/MaskEditorModal.vue";
import GenerationTemplatesPanel from "../components/GenerationTemplatesPanel.vue";
import ColdStartGuide from "../components/create/ColdStartGuide.vue";
import RecentGrid from "../components/create/RecentGrid.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import { defaultUpscaler } from "../components/create/advanced/upscalers";
import { blobToBase64 } from "../lib/base64";
import Icon from "@ui/components/Icon.vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";
import { ASPECTS } from "@ui/lib/resolution";
import {
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
  floatControlError,
  integerControlError,
  resolutionProfileFinding,
} from "@studio/lib/generationProfile";
import type { DevelopPhase } from "@ui/lib/grain";
import type { ClipRailMedia } from "@ui/components/types";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { createUuid } from "@studio/lib/id";
import { requestNeedsReferenceUpload } from "@studio/api/referenceUploads";
import {
  expansionTaskForRequest,
  type ExpandTask,
} from "@studio/lib/expandTask";
import {
  conditioningFingerprint,
  promptSource,
} from "@studio/lib/promptTransform";
import { isAudioCompletion } from "@studio/lib/ltx2Pipeline";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import {
  canvasMatchesSourceResolution,
  resolveSourceResolution,
  type SourceResolutionResult,
} from "@studio/lib/sourceResolution";
import { copyableError, describeTransportError } from "@studio/lib/errors";
import { normalizeCameraMotionLoraState } from "@studio/lib/cameraMotion";
import {
  guidanceOverrideCount,
  guidanceOverridesError,
  guidanceOverridesFromWire,
} from "@studio/lib/guidanceOverrides";
import {
  wanRecipeCount,
  wanRecipeError,
  wanRecipeFromWire,
} from "@studio/lib/wanRecipe";
import {
  defaultClipFrames,
  friendlySequenceError,
  modelsForOutput,
  modelSupportsSequence,
  sequenceMotionTailFrames,
  type OutputMode,
} from "@studio/lib/sequence";
import {
  buildChainRequest,
  chainScriptToClips,
  type SequenceSharedParams,
} from "@studio/lib/sequenceForm";
import {
  clampClipsToMotionTail,
  isPrintOfChainJob,
  planSequenceReuse,
  sequenceReuseClampNote,
  sequenceReuseNote,
} from "@studio/lib/sequenceReuse";
import {
  pendingSequenceHandoff,
  takeSequenceHandoff,
} from "../composables/useSequenceHandoff";
import {
  pendingGenerationHandoff,
  takeGenerationHandoff,
} from "../composables/useGenerationHandoff";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  sequenceToVM,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import type { AmendRequest } from "@studio/lib/api/chainTypes";
import {
  ApiHttpError,
  chainJobStagePreviewUrl,
  deleteGalleryImage,
  expandPrompt,
  fetchChainLimits,
  fetchPromptHistory,
  getChainJob,
  imageUrl,
  listGallery,
  upscaleStream,
  type StreamTarget,
} from "../api";
import type {
  PromptTransformProvenanceWire,
  RemixDimension,
  RemixResponseWire,
} from "../types";
import { sequenceStageMediaUrl } from "../lib/sequenceMedia";
import { useChainJobs } from "../composables/useChainJobs";
import { sequenceSharedParams } from "../lib/sequenceParams";
import {
  countLeadingCompletedStages,
  normalizeServerChainScript,
} from "@studio/lib/chainScriptWire";
import {
  applyMetadataToForm,
  isQwenImageEditFamily,
  useGenerateForm,
} from "../composables/useGenerateForm";
import { isFlux2DevModel } from "../lib/generateCapabilities";
import { mergeStyleNegative, styleHint } from "../lib/stylePresets";
import {
  activeCanvasJob,
  latestUnresolvedError,
  resolveChainRequest,
  useGenerateStream,
  type Job,
} from "../composables/useGenerateStream";
import {
  completedSeedToLock,
  loadLastSeed,
  storeLastSeed,
} from "../lib/lastSeed";
import { useQueue } from "../composables/useQueue";
import { useLiveActivity } from "../composables/useLiveActivity";
import { useOpenLiveWork } from "../composables/useOpenLiveWork";
import { ORIGIN_HOST_ID, listHosts } from "../lib/hostRegistry";
import {
  localRowHiddenFromStrip,
  sharedRowIsLocallyOwned,
} from "../lib/activityDedup";
import { fetchMergedGallery } from "../lib/multiHostGallery";
import { fetchGalleryBlob } from "../lib/galleryMedia";
import {
  fetchH3BoundaryMedia,
  h3BoundariesNeedingMedia,
} from "@studio/lib/h3BoundaryRestore";
import {
  autoChainFieldList,
  decideGenerateRequestRouting,
} from "../lib/chainRouting";
import { isStandaloneGenerationModel } from "../lib/modelFilters";
import {
  coerceSourceFitForMaskless,
  maskPaddingRectangles,
  parseSourceFitPolicy,
  resolveSourceFitTransform,
} from "@studio/lib/sourceFit";
import {
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
  sha256HexOfBase64,
} from "@studio/lib/generationSourceMedia";
import { useStatusPoll } from "../composables/useStatusPoll";
import {
  useHostRouting,
  type FeasibilityResult,
} from "../composables/useHostRouting";
import { profileConflictMessage } from "@studio/lib/profileFleet";
import { generationCapabilitiesForFamily } from "../lib/generateCapabilities";
import {
  canOfferExtend,
  serverExtendOverlapDefault,
  submitsExtend,
} from "@studio/lib/extend";
import { promptOptional, promptRequired } from "@studio/lib/promptRequirement";
import {
  MINIMAX_H3_PROMPT_PLACEHOLDER,
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Identity,
  minimaxH3AuthoringError,
  minimaxH3TaskForModel,
  setMinimaxH3PickedImageBoundary,
  type MinimaxH3BoundaryEndpoint,
} from "@studio/lib/minimaxH3Authoring";
import {
  firstLastFrameRestoreNotice,
  sourceImageValidationError,
} from "@studio/lib/sourceImageCapability";
import {
  modelDisplayName,
  modelDisplayNameForId,
} from "@studio/lib/modelDisplay";
import { sameHostRoute, type HostRoute } from "../lib/hostRouting";
import type {
  ExpandFormState,
  GalleryImage,
  GenerateRequestWire,
  ModelInfoExtended,
  OutputMetadata,
  SourceFitPolicy,
  SourceImageState,
} from "../types";
import { MAX_LORA_STACK } from "../types";
function loadMuted(): boolean {
  try {
    return localStorage.getItem("mold.gallery.muted") !== "false";
  } catch {
    return true;
  }
}

const form = useGenerateForm();
const pageRoute = useRoute();
const router = useRouter();
const { status } = useStatusPoll();
const queue = useQueue();
const routing = useHostRouting();
const liveActivity = useLiveActivity(routing);
const openLiveWork = useOpenLiveWork(routing);
// The model list follows the routing target: a pinned machine shows its own
// models, Auto / Most capable show the union across ready machines. Single-host
// installs collapse to exactly the origin's `/api/models`, as before.
const models = computed<ModelInfoExtended[]>(() => routing.targetModels.value);
// Gates the cold-start guide (spec §08 G10): only after every listed host's
// models have settled, so a slow remote can't make Create flash "nothing
// installed" while the list is still in flight.
const modelsLoaded = routing.modelsSettled;
const galleryEntries = ref<GalleryImage[]>([]);
const promptHistory = ref<string[]>([]);
const muted = ref(loadMuted());

const showExpand = ref(false);
const showRemix = ref(false);
const remixRoute = ref<HostRoute | null>(null);
const remixTask = ref<ExpandTask>("text-to-image");
const showPicker = ref(false);
// One picker serves both FL2VA boundaries; the target names the slot.
const h3BoundaryPickerTarget = ref<MinimaxH3BoundaryEndpoint | null>(null);
/** Wan's closing still gets its own picker (#779) so attaching one can never
 * overwrite the opening frame the source well holds. */
const showEndFramePicker = ref(false);
const showMask = ref(false);
const showAdvanced = ref(false);
const showTemplates = ref(false);
const templatesHost = ref<HTMLElement | null>(null);
const composerError = ref<string | null>(null);
const preprocessingStatus = ref<string | null>(null);
const submitStatus = computed(
  () =>
    composerError.value ?? preprocessingStatus.value ?? placementStatus.value,
);

function onTemplatesPointerDown(event: PointerEvent) {
  if (
    showTemplates.value &&
    event.target instanceof Node &&
    !templatesHost.value?.contains(event.target)
  ) {
    showTemplates.value = false;
  }
}

function onTemplatesKeydown(event: KeyboardEvent) {
  if (showTemplates.value && event.key === "Escape") {
    event.preventDefault();
    showTemplates.value = false;
  }
}

// ── Expand / variations state (spec §03/§06) ──────────────────────────
// batch = 1 rewrites the prompt in place (undoable); batch > 1 fans out into
// editable variations reviewed in the canvas before queueing.
const prevPrompt = ref<string | null>(null);
const prevOriginalPrompt = ref<string | null>(null);
/**
 * The style state a quick expansion's bake-and-clear replaced: the chip it
 * dropped, and the negative prompt before and after the preset's curated
 * fragments merged in. Undo re-arms it; `baked` lets the negative half bow out
 * when the user has edited the field since.
 */
const prevStyle = ref<{
  preset: string | null;
  negativeBefore: string;
  negativeBaked: string;
} | null>(null);
const expanded = computed(() => prevPrompt.value !== null);
const variations = ref<string[]>([]);
const queueingVariations = ref(false);
const preparingVariations = ref(false);
const expandRoute = ref<HostRoute | null>(null);
interface QuickPreparedExpansion {
  expandedPrompt: string;
  originalPrompt: string;
  model: string;
  family: string;
  task: ExpandTask;
  selectedHostPolicy: string;
  route: HostRoute | null;
  promptTransform?: PromptTransformProvenanceWire;
}
const quickPrepared = ref<QuickPreparedExpansion | null>(null);
interface PreparedWebBatch {
  kind?: "expand" | "remix";
  batchId: string;
  sourcePrompt: string;
  model: string;
  family: string;
  task: ExpandTask;
  requestedCount: number;
  selectedHostPolicy: string;
  baseRequest: GenerateRequestWire;
  decision: Exclude<
    ReturnType<typeof decideGenerateRequestRouting>,
    { kind: "reject" }
  >;
  route: HostRoute;
  rootPrompt?: string;
  sourceKind?: "original" | "current" | "direct";
  conditioningFingerprint?: string;
  remixDimensions?: readonly (readonly RemixDimension[])[];
}
const preparedBatch = ref<PreparedWebBatch | null>(null);
const ordinarySubmitBlocked = computed(
  () => preparingVariations.value || preparedBatch.value !== null,
);

function cloneRoute(route: HostRoute | null): HostRoute | null {
  return route ? { ...route, target: { ...route.target } } : null;
}

function normalizeSubmitRoute(
  route: HostRoute | null,
  request?: GenerateRequestWire,
): HostRoute | null {
  return route?.hostId === "origin" &&
    !(request && requestNeedsReferenceUpload(request))
    ? null
    : route;
}

function sameRoute(
  frozen: HostRoute | null,
  current: HostRoute | null,
): boolean {
  return sameHostRoute(frozen, current);
}

function preparedStaleReasons(batch: PreparedWebBatch): string[] {
  const reasons: string[] = [];
  const currentSource =
    batch.kind === "remix"
      ? promptSource(
          form.state.value.prompt,
          form.state.value.originalPrompt,
          batch.sourceKind === "original" ? "original" : "current",
        ).prompt
      : form.state.value.prompt.trim();
  if (currentSource !== batch.sourcePrompt)
    reasons.push("Source prompt changed after these variations were prepared.");
  if (form.state.value.model !== batch.model)
    reasons.push(
      `Model changed from "${modelDisplayNameForId(batch.model, models.value)}" to "${modelDisplayNameForId(form.state.value.model, models.value)}".`,
    );
  if (currentFamily.value !== batch.family)
    reasons.push(
      `Model family changed from "${batch.family}" to "${currentFamily.value}".`,
    );
  const currentTask = expansionTaskForCurrentOutput(
    form.toRequest(currentModel.value),
  );
  if (currentTask !== batch.task)
    reasons.push(`Conditioning changed from ${batch.task} to ${currentTask}.`);
  if (
    batch.conditioningFingerprint !== undefined &&
    conditioningFingerprint(form.toRequest(currentModel.value)) !==
      batch.conditioningFingerprint
  )
    reasons.push(
      "Conditioning media changed after these remixes were prepared.",
    );
  if (form.state.value.batchSize !== batch.requestedCount)
    reasons.push(
      `Batch changed from ${batch.requestedCount} to ${form.state.value.batchSize}.`,
    );
  if (routing.targetId.value !== batch.selectedHostPolicy)
    reasons.push(
      "The Run on selection changed after these variations were prepared.",
    );
  const currentRoute = routing.resolve(batch.model);
  if (!sameRoute(batch.route, currentRoute))
    reasons.push(
      `${batch.route?.label ?? "This server"} is no longer the prepared generation route.`,
    );
  return reasons;
}

function quickStaleReasons(snapshot: QuickPreparedExpansion): string[] {
  const reasons: string[] = [];
  if (form.state.value.prompt.trim() !== snapshot.expandedPrompt)
    reasons.push("Expanded prompt changed after it was prepared.");
  if (form.state.value.model !== snapshot.model)
    reasons.push(
      `Model changed from "${modelDisplayNameForId(snapshot.model, models.value)}" to "${modelDisplayNameForId(form.state.value.model, models.value)}".`,
    );
  if (currentFamily.value !== snapshot.family)
    reasons.push(
      `Model family changed from "${snapshot.family}" to "${currentFamily.value}".`,
    );
  const currentTask = expansionTaskForCurrentOutput(
    form.toRequest(currentModel.value),
  );
  if (currentTask !== snapshot.task)
    reasons.push(
      `Conditioning changed from ${snapshot.task} to ${currentTask}.`,
    );
  return reasons;
}

function quickRouteIsCurrent(snapshot: QuickPreparedExpansion): boolean {
  return (
    routing.targetId.value === snapshot.selectedHostPolicy &&
    sameRoute(snapshot.route, routing.resolve(snapshot.model))
  );
}
const quickConflictReasons = computed(() =>
  quickPrepared.value ? quickStaleReasons(quickPrepared.value) : [],
);
const quickConflictMessage = computed(() =>
  quickConflictReasons.value.length
    ? `${quickConflictReasons.value.join(" ")} Choose how to continue.`
    : "",
);

async function generateExpandedAnyway(): Promise<void> {
  if (!quickPrepared.value) return;
  composerError.value = null;
  await onSubmit(true);
}

async function reexpandCurrentPrompt(): Promise<void> {
  if (!quickPrepared.value || prevPrompt.value === null) return;
  undoExpand();
  await nextTick();
  await onExpand();
}

async function copyQuickConflict(): Promise<void> {
  await copyErrorMessage(quickConflictMessage.value);
}

async function copyErrorMessage(message: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(message);
  } catch {
    toast("error", "Could not copy the error message.");
  }
}

// Phone surface → the Advanced sheet instead of the inline power column.
let phoneQuery: MediaQueryList | null =
  typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(max-width: 639px)")
    : null;
const isPhone = ref(phoneQuery?.matches ?? false);
function syncPhone() {
  isPhone.value = phoneQuery?.matches ?? false;
}

function mediaUrl(image: SourceImageState): string {
  return `data:${image.mime || "image/png"};base64,${image.base64}`;
}

function loadHtmlImage(image: SourceImageState): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to decode ${image.filename}`));
    img.src = mediaUrl(image);
  });
}

function canvasToSourceImage(
  canvas: HTMLCanvasElement,
  original: SourceImageState,
  suffix: string,
): SourceImageState {
  const dataUrl = canvas.toDataURL("image/png");
  const comma = dataUrl.indexOf(",");
  return {
    ...original,
    filename: original.filename.replace(/(\.[^.]+)?$/, `${suffix}.png`),
    base64: comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl,
    width: canvas.width,
    height: canvas.height,
    mime: "image/png",
  };
}

function drawableFitPolicy(
  policy: SourceFitPolicy | undefined,
): SourceFitPolicy {
  if (!policy) return { mode: "pad-repaint" };
  if (policy.mode === "upscale-then-fit") return policy.fit;
  return policy;
}

// ── Output mode (mockup 1c/3a: a setting, not a place) ────────────────
// The clip list and output mode live in the shared sequence draft store;
// shared generation params stay in the live generate form and are read at
// submit time — the fix for the old stale-inspector sequence bug.
const draft = useSequenceDraftStore();
draft.hydrate();
const sequenceMode = computed(() => draft.output === "sequence");
const promptBridge = {
  getPrompt: () => form.state.value.prompt,
  setPrompt: (v: string) => {
    form.state.value.prompt = v;
  },
};

function setOutput(mode: OutputMode) {
  if (mode === "sequence" && draft.output !== "sequence") {
    // Remember the single-mode model so switching back restores it.
    draft.lastSingleModel = form.state.value.model || null;
  }
  // Leaving Sequence retires the reuse caveat with the rail it described.
  if (mode !== "sequence") sequenceReuseNotice.value = null;
  draft.setOutput(mode, promptBridge, sequenceDefaultFrames.value);
}

// Legacy `?mode=sequence` deep links redirect to `?output=sequence`; that
// query is consumed ONCE (setOutput + strip) so the persisted draft output
// wins on every later visit. Registered from onMounted — the handler leans
// on computeds declared further down.
function queryWithout(
  query: Record<string, unknown>,
  key: string,
  extra?: Record<string, string>,
): Record<string, string> {
  const next: Record<string, unknown> = { ...query, ...(extra ?? {}) };
  delete next[key];
  return next as Record<string, string>;
}

function consumeOutputQuery(query: Record<string, unknown>) {
  if (query.mode === "sequence") {
    void router.replace({
      query: queryWithout(query, "mode", { output: "sequence" }),
    });
    return;
  }
  if (query.output === "sequence") {
    setOutput("sequence");
    void router.replace({ query: queryWithout(query, "output") });
  }
}

const expandClipId = ref<string | null>(null);
const expandStagePrompt = ref("");
const expandTask = ref<ExpandTask>("text-to-image");

function expansionTaskForCurrentOutput(
  request: GenerateRequestWire,
): ExpandTask {
  if (sequenceMode.value) {
    return expansionTaskForRequest(currentFamily.value, {
      source_image: draft.openingImage?.base64,
    });
  }
  return expansionTaskForRequest(currentFamily.value, request);
}
// The composer's style chip steers the main-prompt expansion as natural
// language. Sequence clips are their own text — the style row never touches
// them.
const expandStyleDirective = computed(() =>
  expandClipId.value !== null
    ? null
    : styleHint(form.state.value.stylePreset ?? ""),
);

// Drawer state (mirrors LibraryPage).
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

// Seed of the most recent finished print — powers the seed section's
// "lock last" affordance (desktop InspectorPanel parity). Tracked in a ref
// backed by its own tiny localStorage key: completed cards auto-dismiss from
// the rail ~1.5 s after landing and the persistence watcher then writes the
// shortened list, so the job rail alone forgets the seed across reloads.
// `completedSeedToLock` filters chain completions that fabricate seed_used=0.
const lastSeedUsed = ref<number | null>(loadLastSeed());
const stream = useGenerateStream((job) => {
  const seed = completedSeedToLock(job);
  if (seed === null) return;
  lastSeedUsed.value = seed;
  storeLastSeed(seed);
});
// ── Durable sequence jobs (multi-host, mockup 1c: the chain Jobs list
//    merges with the activity strip) ────────────────────────────────────
const chainJobs = useChainJobs();

function hostLabelFor(hostId: string): string {
  return routing.hosts.value.find((h) => h.id === hostId)?.label ?? hostId;
}

function hostTargetFor(hostId: string): StreamTarget | undefined {
  const host = routing.hosts.value.find((h) => h.id === hostId);
  if (!host) return undefined;
  return { baseUrl: host.url, ...(host.apiKey ? { apiKey: host.apiKey } : {}) };
}

function hostRouteFor(hostId: string): HostRoute | null {
  const host = routing.hosts.value.find((candidate) => candidate.id === hostId);
  if (!host) return null;
  return {
    hostId: host.id,
    label: host.label,
    target: {
      baseUrl: host.url,
      ...(host.apiKey ? { apiKey: host.apiKey } : {}),
    },
    instanceId: host.instanceId ?? null,
  };
}

const watchedProgress = computed(() => {
  const live = chainJobs.state.live;
  return live.activeStage !== null
    ? (live.progress[live.activeStage] ?? null)
    : null;
});

const sequenceVMs = computed<ActivityJobVM[]>(() => {
  const out: ActivityJobVM[] = [];
  const watched = chainJobs.state.watching;
  for (const [hostId, hostState] of Object.entries(chainJobs.state.byHost)) {
    for (const job of hostState.jobs) {
      const progress =
        watched && watched.hostId === hostId && watched.jobId === job.id
          ? watchedProgress.value
          : null;
      out.push(
        sequenceToVM(
          job,
          { hostId, hostLabel: hostLabelFor(hostId) },
          progress,
        ),
      );
    }
  }
  return out;
});

/** Server-owned rows survive this tab and client. Avoid duplicating work that
 * the current Create session is still streaming — but a LOCAL row that has
 * already settled as a failure loses to the live server row, because a host
 * that retained the job across a restart is still rendering it. */
const sharedActivityRows = computed(() =>
  liveActivity.rows.value.filter((row) => {
    if (row.kind === "generation") {
      return !sharedRowIsLocallyOwned(row, stream.jobs.value, ORIGIN_HOST_ID);
    }
    if (row.kind === "sequence") {
      return !sequenceVMs.value.some(
        (job) =>
          job.kind === "sequence" &&
          job.jobId === row.id &&
          job.hostId === row.hostId,
      );
    }
    return true;
  }),
);

/** The other half of that dedup: a settled row the server's view supersedes —
 * a live fleet row for the same job, or a detached settle whose fate the host
 * owns — is dropped here, so a resumed job renders once and never as failed. */
const localActivityJobs = computed(() =>
  stream.jobs.value.filter(
    (job) =>
      !localRowHiddenFromStrip(job, liveActivity.rows.value, ORIGIN_HOST_ID),
  ),
);

// The watched job's progress renders in the canvas region (the old
// ChainJobCard's job): stage/step readout plus the latest stage preview.
const watchedSequenceDetail = computed(() => chainJobs.state.live.detail);
const watchedSequenceActive = computed(() => {
  const d = watchedSequenceDetail.value;
  return !!d && (d.state === "running" || d.state === "queued");
});

/**
 * The watched job once it has settled. The activity strip no longer keeps a
 * settled row, so the canvas is what holds the result: the finished video with
 * Edit sequence / Show in library, or the failure with Resume. Settling must
 * never blank the canvas.
 */
const settledSequence = computed(() => {
  const d = watchedSequenceDetail.value;
  return d && d.state !== "running" && d.state !== "queued" ? d : null;
});

/** The print this job produced, when it landed in this server's gallery. */
const settledSequencePrint = computed(() => {
  const d = settledSequence.value;
  if (!d) return null;
  return (
    galleryEntries.value.find((entry) =>
      isPrintOfChainJob(entry.metadata, d.id),
    ) ?? null
  );
});

const settledSequenceCaption = computed(() => {
  const d = settledSequence.value;
  if (!d) return "";
  const meta = settledSequencePrint.value?.metadata;
  const bits = [
    d.model,
    `${d.stage_count} clip${d.stage_count === 1 ? "" : "s"}`,
  ];
  if (meta) bits.push(`S ${meta.seed}`, `${meta.width}×${meta.height}`);
  const host = chainJobs.state.watching?.hostId;
  if (host) bits.push(hostLabelFor(host));
  return bits.join(" · ");
});

const settledSequenceError = computed(() =>
  settledSequence.value?.state === "failed"
    ? friendlySequenceError(
        settledSequence.value.error ?? "Sequence failed.",
        chainJobs.state.watching?.hostId
          ? hostLabelFor(chainJobs.state.watching.hostId)
          : null,
      )
    : null,
);
const settledSequenceErrorCopy = computed(() => {
  const raw = settledSequence.value?.error;
  return raw && settledSequenceError.value
    ? copyableError(raw, settledSequenceError.value)
    : (settledSequenceError.value ?? "");
});

function onEditSettledSequence() {
  const d = settledSequence.value;
  const host = chainJobs.state.watching?.hostId;
  if (d && host) void editSequence(host, d.id);
}

function onResumeSettledSequence() {
  const d = settledSequence.value;
  const host = chainJobs.state.watching?.hostId;
  if (!d || !host) return;
  void chainJobs
    .resume(host, d.id)
    .catch((error) =>
      toast("error", error instanceof Error ? error.message : String(error)),
    );
}

function onShowSettledSequenceInLibrary() {
  void router.push("/library");
}
const sequenceCanvasPercent = computed(() => {
  const p = watchedProgress.value;
  return p?.total ? Math.round((p.step / p.total) * 100) : 0;
});
const sequenceCanvasStage = computed(() => {
  const d = watchedSequenceDetail.value;
  if (!d) return "";
  const live = chainJobs.state.live;
  if (live.activeStage !== null) {
    const p = watchedProgress.value;
    const step = p ? ` · step ${p.step}/${p.total}` : "";
    return `Clip ${live.activeStage + 1}/${d.stage_count}${step}`;
  }
  return d.state === "queued" ? "Queued" : "Rendering sequence";
});

// Ported from ChainJobCard: authenticated per-stage previews as blob URLs;
// the latest previewed stage feeds the canvas.
const sequencePreviews = ref<Record<number, string>>({});
const sequenceMediaUrls = ref<Record<number, string>>({});
const sequenceStageClipIds = ref<string[]>([]);
const sequenceStageClipIdsByJob = new Map<string, string[]>();
const playingSequenceStage = ref<number | null>(null);
let sequencePreviewKey = "";
let sequencePreviewEpoch = 0;
const pendingSequencePreviews = new Set<string>();
const pendingSequenceMedia = new Set<string>();
function clearSequencePreviews() {
  sequencePreviewEpoch += 1;
  pendingSequencePreviews.clear();
  pendingSequenceMedia.clear();
  for (const url of Object.values(sequencePreviews.value)) {
    URL.revokeObjectURL(url);
  }
  sequencePreviews.value = {};
  sequenceMediaUrls.value = {};
  sequenceStageClipIds.value = [];
  playingSequenceStage.value = null;
  sequencePreviewKey = "";
}
watch(
  () => {
    const d = watchedSequenceDetail.value;
    const watched = chainJobs.state.watching;
    return d && watched
      ? [
          watched.hostId,
          d.id,
          d.stages
            .filter((s) => s.has_preview || s.has_media)
            .map(
              (s) => `${s.idx}:${s.has_preview ? 1 : 0}:${s.has_media ? 1 : 0}`,
            )
            .join(","),
        ].join(":")
      : "";
  },
  () => {
    const d = watchedSequenceDetail.value;
    const watched = chainJobs.state.watching;
    if (!d || !watched) return;
    const nextKey = `${watched.hostId}:${d.id}`;
    if (sequencePreviewKey !== nextKey) {
      clearSequencePreviews();
      sequencePreviewKey = nextKey;
      const script = normalizeServerChainScript(d.script);
      const boundIds = sequenceStageClipIdsByJob.get(nextKey);
      const editing = draft.editing;
      const editingMatches =
        editing?.hostId === watched.hostId &&
        editing.jobId === d.id &&
        draft.clips.length === d.stages.length;
      const matchesDraft =
        script?.stages.length === draft.clips.length &&
        script.stages.every(
          (stage, index) => stage.prompt === draft.clips[index]?.prompt,
        );
      sequenceStageClipIds.value =
        boundIds?.length === d.stages.length
          ? [...boundIds]
          : editingMatches
            ? draft.clips.map((clip) => clip.id)
            : matchesDraft
              ? draft.clips.map((clip) => clip.id)
              : [];
    }
    const stagesByIdx = new Map(d.stages.map((stage) => [stage.idx, stage]));
    for (const [rawIdx, url] of Object.entries(sequencePreviews.value)) {
      const idx = Number(rawIdx);
      if (!stagesByIdx.get(idx)?.has_preview) {
        URL.revokeObjectURL(url);
        const next = { ...sequencePreviews.value };
        delete next[idx];
        sequencePreviews.value = next;
      }
    }
    for (const rawIdx of Object.keys(sequenceMediaUrls.value)) {
      const idx = Number(rawIdx);
      if (!stagesByIdx.get(idx)?.has_media) {
        const next = { ...sequenceMediaUrls.value };
        delete next[idx];
        sequenceMediaUrls.value = next;
        if (playingSequenceStage.value === idx)
          playingSequenceStage.value = null;
      }
    }
    const target = hostTargetFor(watched.hostId);
    const requestEpoch = sequencePreviewEpoch;
    for (const stage of d.stages) {
      const pendingKey = `${requestEpoch}:${nextKey}:${stage.idx}`;
      if (
        stage.has_preview &&
        !sequencePreviews.value[stage.idx] &&
        !pendingSequencePreviews.has(pendingKey)
      ) {
        pendingSequencePreviews.add(pendingKey);
        const headers = target?.apiKey
          ? { "x-api-key": target.apiKey }
          : undefined;
        void fetch(chainJobStagePreviewUrl(d.id, stage.idx, target), {
          ...(headers ? { headers } : {}),
        })
          .then(async (response) => {
            if (!response.ok)
              throw new Error(`Preview failed: ${response.status}`);
            const url = URL.createObjectURL(await response.blob());
            const currentStage = chainJobs.state.live.detail?.stages.find(
              (candidate) => candidate.idx === stage.idx,
            );
            if (
              requestEpoch !== sequencePreviewEpoch ||
              sequencePreviewKey !== nextKey ||
              chainJobs.state.watching?.hostId !== watched.hostId ||
              chainJobs.state.live.detail?.id !== d.id ||
              !currentStage?.has_preview
            ) {
              URL.revokeObjectURL(url);
              return;
            }
            sequencePreviews.value = {
              ...sequencePreviews.value,
              [stage.idx]: url,
            };
          })
          .catch(() => {})
          .finally(() => pendingSequencePreviews.delete(pendingKey));
      }
      if (
        stage.has_media &&
        !sequenceMediaUrls.value[stage.idx] &&
        !pendingSequenceMedia.has(pendingKey)
      ) {
        pendingSequenceMedia.add(pendingKey);
        void sequenceStageMediaUrl(d.id, stage.idx, target)
          .then((url) => {
            const currentStage = chainJobs.state.live.detail?.stages.find(
              (candidate) => candidate.idx === stage.idx,
            );
            if (
              requestEpoch !== sequencePreviewEpoch ||
              sequencePreviewKey !== nextKey ||
              chainJobs.state.watching?.hostId !== watched.hostId ||
              chainJobs.state.live.detail?.id !== d.id ||
              !currentStage?.has_media
            ) {
              return;
            }
            sequenceMediaUrls.value = {
              ...sequenceMediaUrls.value,
              [stage.idx]: url,
            };
          })
          .catch(() => {})
          .finally(() => pendingSequenceMedia.delete(pendingKey));
      }
    }
  },
  { immediate: true },
);
const sequencePreviewSrc = computed(() => {
  const best = Object.entries(sequencePreviews.value)
    .map(([idx, url]) => [Number(idx), url] as const)
    .sort((a, b) => b[0] - a[0])[0];
  return best?.[1];
});
const sequencePlaybackSrc = computed(() => {
  const idx = playingSequenceStage.value;
  return idx === null ? null : (sequenceMediaUrls.value[idx] ?? null);
});
const sequenceFilmstripMediaByClipId = computed<
  Record<string, ClipRailMedia | undefined>
>(() => {
  const detail = watchedSequenceDetail.value;
  if (!detail) return {};
  return Object.fromEntries(
    [...detail.stages]
      .sort((a, b) => a.idx - b.idx)
      .flatMap((stage) => {
        const clipId = sequenceStageClipIds.value[stage.idx];
        if (!clipId) return [];
        const progress = chainJobs.state.live.progress[stage.idx];
        const status: ClipRailMedia["status"] =
          stage.state === "completed"
            ? "ready"
            : stage.state === "failed"
              ? "error"
              : stage.state;
        return [
          [
            clipId,
            {
              stageIdx: stage.idx,
              status,
              posterUrl: sequencePreviews.value[stage.idx] ?? null,
              hasMedia: Boolean(
                stage.has_media && sequenceMediaUrls.value[stage.idx],
              ),
              cacheReady: stage.cache_ready,
              progressPercent:
                progress && progress.total > 0
                  ? (progress.step / progress.total) * 100
                  : null,
              error: stage.error,
            } satisfies ClipRailMedia,
          ],
        ];
      }),
  );
});

function playSequenceStage(stageIdx: number) {
  if (!sequenceMediaUrls.value[stageIdx]) return;
  playingSequenceStage.value =
    playingSequenceStage.value === stageIdx ? null : stageIdx;
}

function playSequenceClip(clipId: string) {
  const stageIdx = sequenceStageClipIds.value.indexOf(clipId);
  if (stageIdx >= 0) playSequenceStage(stageIdx);
}

const playingSequenceClipId = computed(() => {
  const idx = playingSequenceStage.value;
  return idx === null ? null : (sequenceStageClipIds.value[idx] ?? null);
});

function returnToLiveSequence() {
  playingSequenceStage.value = null;
}

// Surface the watched sequence's terminal states without a dedicated card.
watch(
  () => watchedSequenceDetail.value?.state,
  (state, prev) => {
    if (!state || !prev || state === prev) return;
    if (state === "completed") {
      // Verb→noun (§11): the sequence is a thing now, and it has a home.
      toast("info", "Sequence ready — saved to Library");
      void refreshGallery();
    }
  },
);

async function refreshGallery() {
  try {
    galleryEntries.value = await listGallery();
  } catch {
    /* ignore */
  }
}

async function refreshHistory() {
  promptHistory.value = await fetchPromptHistory();
}

const doneJobIds = computed(() =>
  stream.jobs.value
    .filter((j) => j.state === "done")
    .map((j) => j.id)
    .join(","),
);

const seenDoneIds = new Set<string>();
watch(
  doneJobIds,
  () => {
    let added = false;
    for (const j of stream.jobs.value) {
      if (j.state !== "done") continue;
      if (seenDoneIds.has(j.id)) continue;
      seenDoneIds.add(j.id);
      added = true;
    }
    if (added) {
      void refreshGallery();
      void refreshHistory();
    }
  },
  { immediate: true },
);

let galleryTimer: ReturnType<typeof setInterval> | null = null;

function startAutoRefresh() {
  stopAutoRefresh();
  galleryTimer = setInterval(() => {
    if (!document.hidden) void refreshGallery();
  }, 10_000);
  // Models refresh on the host-routing poll — one sweep feeds every machine's
  // model list and queue depth, so Create doesn't duplicate the origin fetch.
}

function stopAutoRefresh() {
  if (galleryTimer) {
    clearInterval(galleryTimer);
    galleryTimer = null;
  }
}

const currentModel = computed(
  () => models.value.find((m) => m.name === form.state.value.model) ?? null,
);

let previousStillSource = "";
let previousStillResolution: SourceResolutionResult | null = null;
let previousOpeningSource = "";
let previousOpeningResolution: SourceResolutionResult | null = null;

function syncSourceCanvas(
  image: {
    base64: string | null;
    width?: number | null;
    height?: number | null;
  } | null,
  previous: { base64: string; resolution: SourceResolutionResult | null },
): { base64: string; resolution: SourceResolutionResult | null } {
  if (!image?.base64) return { base64: "", resolution: null };
  const dimensions =
    image.width && image.height
      ? { width: image.width, height: image.height }
      : imageDimensionsFromBase64(image.base64);
  if (!dimensions) return { base64: image.base64, resolution: null };
  image.width = dimensions.width;
  image.height = dimensions.height;
  const resolution = resolveSourceResolution(
    dimensions,
    currentModel.value ?? form.state.value.modelFamily,
    form.state.value.pipeline,
  );
  const replaced = image.base64 !== previous.base64;
  const wasFollowing =
    previous.resolution !== null &&
    canvasMatchesSourceResolution(
      {
        width: form.state.value.width,
        height: form.state.value.height,
      },
      previous.resolution,
    );
  const isReferenceConditioning = isFlux2DevModel(
    currentModel.value?.name ?? form.state.value.model,
  );
  if (!isReferenceConditioning && (replaced || wasFollowing)) {
    form.state.value.width = resolution.output.width;
    form.state.value.height = resolution.output.height;
  }
  return { base64: image.base64, resolution };
}

watch(
  [
    () => form.state.value.imageAttachments[0]?.base64 ?? null,
    () => currentModel.value?.name ?? form.state.value.model,
    () => form.state.value.pipeline ?? null,
    () => currentModel.value?.generation_profile?.profile_hash ?? null,
    () => currentModel.value?.max_pixels ?? null,
    () => currentModel.value?.max_axis_pixels ?? null,
    () => currentModel.value?.dimension_alignment ?? null,
    () =>
      currentModel.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  () => {
    if (sequenceMode.value) return;
    const next = syncSourceCanvas(
      form.state.value.imageAttachments[0] ?? null,
      {
        base64: previousStillSource,
        resolution: previousStillResolution,
      },
    );
    previousStillSource = next.base64;
    previousStillResolution = next.resolution;
  },
  { immediate: true },
);

watch(
  [
    () => draft.openingImage?.base64 ?? null,
    () => currentModel.value?.name ?? form.state.value.model,
    () => form.state.value.pipeline ?? null,
    () => currentModel.value?.generation_profile?.profile_hash ?? null,
    () => currentModel.value?.max_pixels ?? null,
    () => currentModel.value?.max_axis_pixels ?? null,
    () => currentModel.value?.dimension_alignment ?? null,
    () =>
      currentModel.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  () => {
    if (!sequenceMode.value) return;
    const next = syncSourceCanvas(draft.openingImage, {
      base64: previousOpeningSource,
      resolution: previousOpeningResolution,
    });
    previousOpeningSource = next.base64;
    previousOpeningResolution = next.resolution;
  },
  { immediate: true },
);

const activeSourceDimensions = computed(() => {
  const image = sequenceMode.value
    ? draft.openingImage
    : form.state.value.imageAttachments[0];
  return image?.width && image.height
    ? { width: image.width, height: image.height }
    : null;
});

/** Human title for the selected model — falls back to the raw form value. */
const currentModelLabel = computed(() =>
  currentModel.value
    ? modelDisplayName(currentModel.value)
    : form.state.value.model,
);

const currentFamily = computed(
  () =>
    currentModel.value?.family ??
    form.state.value.modelFamily ??
    (isMinimaxH3Identity(null, form.state.value.model) ? "minimax-h3" : ""),
);

// The fifth argument is the selected row's advertised source-image contract
// (#772). The persisted snapshot backs it up so a form restored before the
// catalog resolves still resolves the same capability the submit gate will.
const capabilities = computed(() =>
  generationCapabilitiesForFamily(
    currentFamily.value,
    form.state.value.model,
    form.state.value.pipeline,
    null,
    currentModel.value?.source_image ?? form.state.value.sourceImageCapability,
    effectiveGenerationRecipe(currentModel.value, form.state.value.pipeline),
  ),
);
const h3FrameError = computed(() =>
  minimaxH3AuthoringError(
    currentFamily.value,
    form.state.value.model,
    form.state.value.h3Authoring,
    capabilities.value.requiresSourceImage,
  ),
);
const h3GenerationInputBlocker = computed<string | null>(() => {
  if (h3FrameError.value) return h3FrameError.value;
  if (
    isMinimaxH3Identity(currentFamily.value, form.state.value.model) &&
    promptRequired({
      family: currentFamily.value,
      model: form.state.value.model,
      sourceImage: form.state.value.h3Authoring?.firstFrame,
    }) &&
    !form.state.value.prompt.trim()
  ) {
    return "Add a prompt before generating.";
  }
  return null;
});

// A conditioned LTX-2 render may go out undescribed — the server admits it,
// so the composer says so instead of implying a prompt is mandatory. Nothing
// here gates submit: `validateSubmit` never required a prompt.
const canSkipPrompt = computed(() =>
  promptOptional({
    family: currentFamily.value,
    imageAttachments: form.state.value.imageAttachments,
    keyframes: form.state.value.keyframes,
    sourceVideo: form.state.value.sourceVideo,
    sourceVideoPath: form.state.value.sourceVideoPath,
    extendVideo: form.state.value.extendVideo,
    extendVideoPath: form.state.value.extendVideoPath,
  }),
);
const requiredPromptPlaceholder = computed(() =>
  isMinimaxH3Identity(currentFamily.value, form.state.value.model)
    ? MINIMAX_H3_PROMPT_PLACEHOLDER
    : "Describe the image you want to create…",
);

// Continuation rides the selected model's own `/api/models` row, which the
// Create surface already holds. A host that predates extend omits the field,
// so the control stays hidden instead of producing a rejected request.
const canExtend = computed(() => canOfferExtend(currentModel.value));
const extendDefaultOverlapFrames = computed(() =>
  serverExtendOverlapDefault(currentModel.value),
);

const supportsChain = computed(() => {
  if (currentModel.value) return modelSupportsSequence(currentModel.value);

  // Keep imported or restored forms usable while the installed-model inventory
  // is still loading. The server remains the final authority at submission.
  const model = form.state.value.model.toLowerCase();
  if (currentFamily.value === "ltx-video") return true;
  if (currentFamily.value !== "ltx2") return false;

  return !model.includes("-dev");
});

const gpuListForPlacement = computed(
  () =>
    status.value?.gpus?.map((g) => ({
      ordinal: g.ordinal,
      name: `GPU ${g.ordinal}`,
    })) ?? [],
);

const chainDecision = computed(() =>
  decideGenerateRequestRouting(
    form.toRequest(currentModel.value),
    currentModel.value?.family ?? null,
  ),
);
const singleShotPreservationNote = computed(() => {
  const decision = chainDecision.value;
  if (decision.kind !== "single" || !decision.preservedAutoChainFields?.length)
    return null;
  return `Will render as one ${form.state.value.frames}-frame clip to preserve ${autoChainFieldList(decision.preservedAutoChainFields)}. This may use more GPU memory than automatic chaining.`;
});

// Keep the preflight small: source/control media do not change the model's
// static peak estimate enough to justify serializing their base64 payloads on
// every form edit.
const estimateRequest = computed<GenerateRequestWire | null>(() => {
  if (!form.state.value.model) return null;
  const request = { ...form.toRequest(currentModel.value) };
  delete request.source_image;
  delete request.mask_image;
  delete request.control_image;
  delete request.edit_images;
  delete request.source_video;
  delete request.audio_file;
  delete request.keyframes;
  return request;
});
const estimateTarget = computed(() =>
  routing.multiHost.value
    ? (routing.resolve(form.state.value.model || null)?.target ?? null)
    : null,
);

// ── Installed generation models for the left rail ─────────────────────
const installedModels = computed(() =>
  models.value.filter((m) => m.downloaded && isStandaloneGenerationModel(m)),
);
const sequenceModels = computed(() =>
  modelsForOutput(installedModels.value, "sequence"),
);
const showSequenceEmpty = computed(
  () =>
    sequenceMode.value &&
    modelsLoaded.value &&
    sequenceModels.value.length === 0,
);
const composerModels = computed(() =>
  modelsForOutput(installedModels.value, draft.output),
);
const sequenceBrowsePath =
  "/models?tab=discover&type=video&kind=checkpoint&intent=sequence";

// Cold start (spec §08 G10): nothing installed to generate with. Only after the
// first load resolves, so the guide replaces the empty canvas rather than
// flashing during boot.
const showColdStart = computed(
  () => modelsLoaded.value && installedModels.value.length === 0,
);

function selectModel(model: ModelInfoExtended) {
  form.applyModelDefaults(model);
}

// The persisted model can name something the routing target doesn't have — it
// was deleted, or the user pinned a machine that never installed it. A <select>
// whose value matches no option renders BLANK, which reads as "this server has
// no models" even when it has several. Re-home the form onto a model that is
// actually there instead.
watch(
  [installedModels, modelsLoaded],
  () => {
    if (!modelsLoaded.value) return;
    const first = installedModels.value[0];
    if (!first) return;
    const current = form.state.value.model;
    const currentRow = current
      ? installedModels.value.find((m) => m.name === current)
      : undefined;
    if (currentRow) {
      // The saved model is still valid, so no defaults reapply — but a
      // pre-#787 snapshot restored without `negativePromptDefault` still
      // needs the advertised default reconciled in (idempotent; typed text
      // and an explicit clear survive).
      form.reconcileNegativeDefault(currentRow);
      return;
    }
    form.applyModelDefaults(first);
  },
  { immediate: true },
);

// Entering Sequence is an intent change: keep only compatible models in the
// picker and move off an image/two-stage model automatically when a runnable
// sequence checkpoint is already installed.
watch(
  [sequenceMode, sequenceModels, modelsLoaded],
  () => {
    if (!sequenceMode.value) return;
    const current = form.state.value.model;
    if (sequenceModels.value.some((model) => model.name === current)) return;
    if (!modelsLoaded.value) {
      if (currentModel.value && !modelSupportsSequence(currentModel.value)) {
        form.state.value.model = "";
        form.state.value.modelFamily = "";
      }
      return;
    }
    const first = sequenceModels.value[0];
    if (first) {
      form.applyModelDefaults(first);
    } else {
      form.state.value.model = "";
      form.state.value.modelFamily = "";
    }
  },
  { immediate: true },
);

// Switching back to One shot restores the model that was selected before the
// Sequence trip (when it is still installed).
watch(sequenceMode, (isSequence, wasSequence) => {
  if (isSequence || !wasSequence) return;
  const name = draft.lastSingleModel;
  if (!name) return;
  const model = installedModels.value.find((m) => m.name === name);
  if (model) form.applyModelDefaults(model);
});

// ── Sequence submit path (live-form projection) ───────────────────────
const sequenceMotionTail = computed(() =>
  sequenceMotionTailFrames({
    name: form.state.value.model,
    family: currentFamily.value,
    // Wan's seam carries context only for an image-conditioned checkpoint
    // (#783); without the advertised contract every wan sequence read as a
    // tail-free join. The persisted snapshot covers a form restored before
    // the inventory lands, exactly as `capabilities` does.
    source_image:
      currentModel.value?.source_image ??
      form.state.value.sourceImageCapability,
  }),
);
const sequenceDefaultFrames = computed(() =>
  defaultClipFrames(currentModel.value, null, sequenceMotionTail.value),
);
const sharedParams = computed<SequenceSharedParams>(() =>
  sequenceSharedParams(form.state.value, currentFamily.value),
);
const sequenceTarget = computed<StreamTarget | undefined>(
  () => routing.resolve(form.state.value.model || null)?.target,
);

// Edit sessions snapshot the shared params at load so shape/detail changes
// mark every clip as re-rendering in the rail's plan badges.
const editBaselineShared = ref<string | null>(null);
/** What a Library reuse could NOT restore, said once and quietly beneath the
 *  rail. It describes one handoff, not a standing property of the draft. */
const sequenceReuseNotice = ref<string | null>(null);
const chainLevelDirty = computed(
  () =>
    editBaselineShared.value !== null &&
    (editBaselineShared.value !== JSON.stringify(sharedParams.value) ||
      JSON.stringify(draft.editing?.baselineOpeningImage ?? null) !==
        JSON.stringify(draft.openingImage)),
);

/** Apply a script's chain-level params onto the LIVE form (edit + import). */
function applySharedToForm(shared: Partial<SequenceSharedParams>) {
  if (shared.model) {
    const model = installedModels.value.find((m) => m.name === shared.model);
    // applyModelDefaults REPLACES state.value — read the form only after.
    if (model) form.applyModelDefaults(model);
    else form.state.value.model = shared.model;
  }
  const s = form.state.value;
  if (shared.width != null) s.width = shared.width;
  if (shared.height != null) s.height = shared.height;
  if (shared.fps != null) s.fps = shared.fps;
  if (shared.steps != null) s.steps = shared.steps;
  if (shared.guidance != null) s.guidance = shared.guidance;
  if (shared.strength != null) s.strength = shared.strength;
  if (shared.seed != null && shared.seed !== "") {
    s.seedMode = "static";
    s.seed = Number(shared.seed);
  }
}

async function onSubmitSequence() {
  composerError.value = null;
  if (
    !sequenceMode.value ||
    !sequenceModels.value.some((model) => model.name === form.state.value.model)
  ) {
    composerError.value =
      "Choose an installed sequence-capable video model on the selected machine.";
    return;
  }
  // Freeze every request-affecting value at the click boundary. Source
  // preprocessing may take minutes; edits during that await belong to the
  // next submission and must not create a hybrid request.
  const editing = draft.editing ? { ...draft.editing } : null;
  const shared = {
    ...sharedParams.value,
    sourceFitPolicy: sharedParams.value.sourceFitPolicy
      ? JSON.parse(JSON.stringify(sharedParams.value.sourceFitPolicy))
      : undefined,
  } satisfies SequenceSharedParams;
  const clips = JSON.parse(JSON.stringify(draft.clips)) as typeof draft.clips;
  const openingSnapshot = draft.openingImage ? { ...draft.openingImage } : null;
  const enableAudio = draft.enableAudio;
  const motionTailFrames = sequenceMotionTail.value;
  const initialRoute = editing
    ? hostRouteFor(editing.hostId)
    : routing.resolve(shared.model || null);
  if (!initialRoute) {
    composerError.value = "The selected sequence host is unavailable.";
    return;
  }
  // Refresh chain limits when stale (30 s cache) — the server still remains
  // the final authority at submission.
  await fetchChainLimits(shared.model, initialRoute.target, shared.fps).catch(
    () => {},
  );
  const preliminaryRequest = buildChainRequest(shared, clips, {
    motionTailFrames,
    enableAudio,
    openingImage: openingSnapshot,
  });
  let route = initialRoute;
  if (!editing) {
    const feasibility = await routing.resolveFeasibleChain(preliminaryRequest);
    if (feasibility.kind !== "route") {
      composerError.value = feasibilityMessage(feasibility, "this sequence");
      return;
    }
    route = feasibility.route;
  }
  let openingImage = openingSnapshot;
  if (openingImage?.base64) {
    const prepared = await prepareStillSourceToRequest(route, {
      source: {
        kind: "upload",
        filename: openingImage.filename,
        base64: openingImage.base64,
        width: openingImage.width,
        height: openingImage.height,
        mime: /\.jpe?g$/i.test(openingImage.filename)
          ? "image/jpeg"
          : "image/png",
      },
      mask: null,
      maskless: true,
      settings: {
        policy: shared.sourceFitPolicy,
        upscalerModel: shared.upscalerModel,
        family: shared.family,
        frames: null,
        width: shared.width,
        height: shared.height,
      },
    });
    if (prepared === false) return;
    openingImage = prepared.source
      ? {
          ...openingImage,
          base64: prepared.source.base64,
          width: prepared.source.width ?? undefined,
          height: prepared.source.height ?? undefined,
        }
      : openingImage;
  }
  const req = buildChainRequest(shared, clips, {
    motionTailFrames,
    enableAudio,
    openingImage,
  });

  if (editing) {
    const amendReq: AmendRequest = {
      stages: req.stages,
      motion_tail_frames: req.motion_tail_frames ?? null,
      fps: req.fps ?? null,
      seed: shared.seed.trim() === "" ? null : shared.seed,
      steps: req.steps,
      guidance: req.guidance,
      strength: req.strength ?? null,
      enable_audio: enableAudio ? true : null,
    };
    try {
      if (!sameRoute(route, hostRouteFor(editing.hostId))) {
        composerError.value =
          "The sequence machine changed during source preparation. Review the machine and Update again.";
        return;
      }
      sequenceStageClipIdsByJob.set(
        `${editing.hostId}:${editing.jobId}`,
        clips.map((clip) => clip.id),
      );
      await chainJobs.amend(
        editing.hostId,
        editing.jobId,
        amendReq,
        route.target,
      );
      draft.stopEditing();
      editBaselineShared.value = null;
      toast("info", "Sequence updated — unchanged clips stay cached.");
      return;
    } catch (error) {
      if (error instanceof ApiHttpError && error.status === 409) {
        toast(
          "error",
          "This sequence changed on its host while you edited. Retry after reloading it, or use Duplicate as new to submit your version.",
        );
        composerError.value =
          "The original sequence changed. Your edits are still here; reload the sequence or choose Duplicate as new.";
        return;
      } else {
        composerError.value =
          error instanceof Error ? error.message : String(error);
        return;
      }
    }
  }

  try {
    const feasibility = await routing.revalidateFeasibleChain(route, req);
    if (feasibility.kind !== "route") {
      throw new Error(
        feasibilityMessage(feasibility, "this finalized sequence"),
      );
    }
    route = feasibility.route;
    const jobId = await chainJobs.create(route.hostId, req);
    sequenceStageClipIdsByJob.set(
      `${route.hostId}:${jobId}`,
      clips.map((clip) => clip.id),
    );
    if (editing) {
      draft.stopEditing();
      editBaselineShared.value = null;
    }
    // The caveat described the handoff, not the submitted job.
    sequenceReuseNotice.value = null;
    toast("info", `Sequence queued on ${route.label}.`);
  } catch (error) {
    composerError.value =
      error instanceof Error ? error.message : String(error);
  }
}

/** Load a durable job's effective script into an edit session. */
async function editSequence(hostId: string, jobId: string) {
  await loadSequence(hostId, jobId, true);
}

async function inspectSequence(hostId: string, jobId: string) {
  await loadSequence(hostId, jobId, false);
}

async function loadSequence(hostId: string, jobId: string, editing: boolean) {
  // An edit session is lossless — any reuse caveat on screen is now stale.
  sequenceReuseNotice.value = null;
  try {
    const detail = await getChainJob(jobId, hostTargetFor(hostId));
    const script = normalizeServerChainScript(detail.script);
    if (!script) throw new Error("This sequence job has no editable script.");
    const loaded = chainScriptToClips(script);
    applySharedToForm(loaded.shared);
    if (loaded.openingImage) {
      form.state.value.sourceFitPolicy = {
        mode: "crop-fill",
        alignX: "center",
        alignY: "center",
      };
    }
    if (editing) {
      draft.loadFromJob(
        {
          jobId,
          hostId,
          baseline: loaded.clips.map((clip) => ({ ...clip })),
          completedStages: countLeadingCompletedStages(detail.stages),
        },
        loaded.clips,
        loaded.enableAudio,
        loaded.openingImage,
      );
      editBaselineShared.value = JSON.stringify(
        sequenceSharedParams(form.state.value, currentFamily.value),
      );
    } else {
      draft.stopEditing();
      editBaselineShared.value = null;
      setOutput("sequence");
      draft.clips.splice(0, draft.clips.length, ...loaded.clips);
      draft.activeClipId = loaded.clips[0]?.id ?? null;
      draft.enableAudio = loaded.enableAudio;
      draft.openingImage = loaded.openingImage;
    }
    if (form.state.value.model) draft.bindSequenceModel(form.state.value.model);
    sequenceStageClipIdsByJob.set(
      `${hostId}:${jobId}`,
      draft.clips.map((clip) => clip.id),
    );
    chainJobs.watch(hostId, jobId);
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

function onDuplicateAsNew() {
  draft.stopEditing();
  editBaselineShared.value = null;
  toast("info", "Detached from the job — Generate now queues a new sequence.");
}

function onDiscardEdit() {
  const editing = draft.editing;
  if (!editing) return;
  draft.clips.splice(
    0,
    draft.clips.length,
    ...editing.baseline.map((clip) => ({ ...clip })),
  );
  draft.openingImage = editing.baselineOpeningImage
    ? { ...editing.baselineOpeningImage }
    : null;
  draft.enableAudio = editing.baselineEnableAudio ?? false;
  draft.stopEditing();
  editBaselineShared.value = null;
  draft.activeClipId = draft.clips[0]?.id ?? null;
}

function onImportShared(shared: Partial<SequenceSharedParams>) {
  applySharedToForm(shared);
  if (draft.openingImage) {
    // Imported opening-image bytes are already the script's prepared source;
    // never inherit a stale client-only upscale policy from the prior draft.
    form.state.value.sourceFitPolicy = { mode: "crop-fill" };
  }
}

function onSequenceAction(action: ActivityAction, vm: ActivityJobVM) {
  if (vm.kind !== "sequence") return;
  const fail = (error: unknown) =>
    toast("error", error instanceof Error ? error.message : String(error));
  switch (action) {
    case "watch":
      void inspectSequence(vm.hostId, vm.jobId);
      break;
    case "cancel":
      void chainJobs.cancel(vm.hostId, vm.jobId).catch(fail);
      break;
    case "resume":
      void chainJobs.resume(vm.hostId, vm.jobId).catch(fail);
      break;
    case "edit":
      void editSequence(vm.hostId, vm.jobId);
      break;
    case "delete":
      void chainJobs.remove(vm.hostId, vm.jobId).catch(fail);
      break;
    case "retake":
      // The strip never offers retake directly; stage-level retakes live in
      // the edit flow.
      break;
  }
}

function cancelPrint(id: string) {
  void stream
    .cancel(id)
    .catch((error) =>
      toast("error", error instanceof Error ? error.message : String(error)),
    );
}

/**
 * Reuse a sequence print's recorded clips as a BRAND-NEW draft: no edit
 * session, nothing cached, Generate queues a fresh job. Shared params ride
 * the same `applyMetadataToForm` path a single print's reuse uses.
 */
function applySequenceReuse(metadata: OutputMetadata) {
  const plan = planSequenceReuse(metadata);
  if (!plan) return;
  const oneShotPrompt = form.state.value.prompt;
  form.state.value = applyMetadataToForm(form.state.value, metadata, {
    models: models.value,
  });
  // Reusing a sequence must not overwrite the parked one-shot prompt.
  form.state.value.prompt = oneShotPrompt;

  // The live tail belongs to the model selected NOW, not the recorded one.
  const { clips, raised } = clampClipsToMotionTail(
    plan.clips,
    sequenceMotionTail.value,
    9,
  );

  draft.stopEditing();
  editBaselineShared.value = null;
  draft.output = "sequence";
  draft.clips.splice(0, draft.clips.length, ...clips);
  draft.activeClipId = clips[0]?.id ?? null;
  draft.enableAudio = metadata.enable_audio === true;
  draft.bindSequenceModel(form.state.value.model);

  const notes = [sequenceReuseNote(clips.length, plan.lossy)];
  if (raised > 0) notes.push(sequenceReuseClampNote(currentModelLabel.value));
  sequenceReuseNotice.value = notes.join(" · ");
}

/** A sequence handed over from the Library: History ▸ Sequences hands over
 *  `inspect`/`edit`; a sequence print hands over `reuse`. One-shot — the slot is
 *  emptied on arrival so a back-nav cannot replay it. */
function applySequenceHandoff() {
  const handoff = takeSequenceHandoff();
  if (!handoff) return;
  if (handoff.kind === "reuse") {
    applySequenceReuse(handoff.metadata);
    return;
  }
  if (handoff.kind === "inspect") {
    void inspectSequence(handoff.hostId, handoff.jobId);
  } else {
    void editSequence(handoff.hostId, handoff.jobId);
  }
}
watch(pendingSequenceHandoff(), applySequenceHandoff, { immediate: true });

/**
 * A first/last-frame print restores every knob except its closing still:
 * saved metadata records each keyframe's name and digest, never the bytes
 * (`applyMetadataToForm` already cleared `endFrame`). Say so, the same way an
 * unrestorable source video does, rather than letting Generate look ready to
 * reproduce the render.
 *
 * Whether the print IS a first/last-frame render is positive knowledge: it
 * needs the origin model's advertised contract. A Library handover lands
 * during setup, before the inventory resolves, so the notice waits for it
 * rather than guessing from the keyframe count alone — LTX-2 keyframes are a
 * different control, and they restore nothing either way.
 */
const pendingEndFrameNotice = ref<OutputMetadata | null>(null);
function noticeFirstLastFrameRestore(metadata: OutputMetadata) {
  if (!modelsLoaded.value) {
    pendingEndFrameNotice.value = metadata;
    return;
  }
  const notice = firstLastFrameRestoreNotice(
    capabilities.value.supportsEndFrame,
    metadata.keyframes,
    // A first/last print carries its opening frame only in keyframes[0], so
    // without a source provenance handle both endpoints need reattaching.
    Boolean(metadata.source_image_sha256 ?? metadata.source_image_name),
  );
  if (notice) toast("error", notice);
}
watch(modelsLoaded, (loaded) => {
  const pending = pendingEndFrameNotice.value;
  if (!loaded || !pending) return;
  pendingEndFrameNotice.value = null;
  noticeFirstLastFrameRestore(pending);
});

/**
 * FL2VA reuse restores its first/last frames as bytes-less reattach
 * descriptors (metadata carries filename + digest only). When the original
 * was a gallery image, the bytes are still on a connected host — fetch them
 * by filename so the well fills itself. Watcher-driven so every reuse entry
 * point (Library reuse, lightbox recreate, generation handoff) is covered;
 * each descriptor is attempted once per session so a genuinely missing file
 * degrades to the existing reattach affordance instead of a fetch loop.
 */
const attemptedH3BoundaryRestores = new Set<string>();
async function restoreReusedH3BoundaryMedia() {
  const s = form.state.value;
  if (minimaxH3TaskForModel(s.model) !== "fl2va") return;
  const wanted = h3BoundariesNeedingMedia(s.h3Authoring).filter((slot) => {
    const key = `${slot.endpoint}|${slot.filename}|${slot.sha256 ?? ""}`;
    if (attemptedH3BoundaryRestores.has(key)) return false;
    attemptedH3BoundaryRestores.add(key);
    return true;
  });
  if (wanted.length === 0) return;
  const modelAtStart = s.model;
  const outcome = await fetchH3BoundaryMedia(
    {
      firstFrame:
        wanted.some((w) => w.endpoint === "firstFrame") && s.h3Authoring
          ? s.h3Authoring.firstFrame
          : null,
      lastFrame:
        wanted.some((w) => w.endpoint === "lastFrame") && s.h3Authoring
          ? s.h3Authoring.lastFrame
          : null,
      references: [],
    },
    async (filename) => {
      const merged = await fetchMergedGallery(listHosts());
      const entry = merged.entries.find((e) => e.filename === filename);
      if (!entry) return null;
      const host = listHosts().find((h) => h.id === entry.hostId);
      if (!host) return null;
      return blobToBase64(await fetchGalleryBlob(host, filename));
    },
  );
  // The fetches can take seconds; never clobber a slot the user has since
  // reattached, and stand down entirely if the model moved on.
  const live = form.state.value;
  if (live.model !== modelAtStart) return;
  let authoring = live.h3Authoring;
  let committed = 0;
  for (const media of outcome.restored) {
    const slot = authoring?.[media.endpoint];
    if (!slot || slot.data || slot.filename !== media.filename) continue;
    const result = setMinimaxH3PickedImageBoundary(authoring, media.endpoint, {
      filename: media.filename,
      base64: media.base64,
    });
    if (result.ok) {
      authoring = result.state;
      committed += 1;
    }
  }
  if (committed > 0) {
    form.state.value = { ...form.state.value, h3Authoring: authoring };
  }
  if (outcome.failed.length > 0) {
    toast(
      "error",
      `Couldn't restore ${outcome.failed.join(", ")} — the file wasn't found on any connected host. Reattach it to generate.`,
    );
  }
}
watch(
  () =>
    h3BoundariesNeedingMedia(form.state.value.h3Authoring)
      .map((slot) => `${slot.endpoint}|${slot.filename}`)
      .join("¦") + `@${form.state.value.model}`,
  () => void restoreReusedH3BoundaryMedia(),
  { immediate: true },
);

function applyGenerationHandoff() {
  const handoff = takeGenerationHandoff();
  if (!handoff) return;
  setOutput("single");
  draft.stopEditing();
  editBaselineShared.value = null;
  const metadata =
    handoff.seedPinned === false
      ? ({ ...handoff.metadata, seed: null } as unknown as OutputMetadata)
      : handoff.metadata;
  form.state.value = applyMetadataToForm(form.state.value, metadata, {
    models: models.value,
  });
  noticeFirstLastFrameRestore(metadata);
}
watch(pendingGenerationHandoff(), applyGenerationHandoff, { immediate: true });

/** The digest chip's one hop. `Clear inactive` and `Clean up disk` moved with
 *  it: they are destructive, host-scoped, and live in the History drawer now. */
function onShowSequenceHistory() {
  void router.push({
    path: "/library",
    query: { panel: "history", tab: "sequences" },
  });
}

const composerCardRef = ref<InstanceType<typeof ComposerCard> | null>(null);

// ⌘K "New print" (spec §06): start a fresh print without leaving Create.
// Reset the advanced knobs to the current model's defaults, clear the prompt,
// source media, and any in-flight expansion/variation review, but KEEP the
// selected model, then focus the prompt. Never nukes the persisted model.
function onNewPrint() {
  const model = currentModel.value;
  if (model) form.applyModelDefaults(model);
  form.state.value.prompt = "";
  form.state.value.stylePreset = null;
  form.state.value.imageAttachments = [];
  form.state.value.endFrame = null;
  form.state.value.maskImage = null;
  form.state.value.controlImage = null;
  variations.value = [];
  preparedBatch.value = null;
  quickPrepared.value = null;
  prevPrompt.value = null;
  prevOriginalPrompt.value = null;
  prevStyle.value = null;
  composerError.value = null;
  preprocessingStatus.value = null;
  void nextTick(() => composerCardRef.value?.focus?.());
}

// Controls rail "Reset" (spec §06): put every generation setting back to the
// current model's defaults. The prompt, style, model and batch size stay —
// this re-tunes the print being composed, it does not start a new one, and
// prepared batch work is never silently resized. Nothing leaves the browser,
// so an undo toast is enough; a blocking confirm would be heavier than the
// action deserves.
function onResetSettings() {
  // resetSettings swaps in a freshly built state object, so the previous one is
  // never mutated and can be handed straight back on undo.
  const previous = form.state.value;
  form.resetSettings(currentModel.value ?? null);
  undoableAction({
    text: "Settings reset to model defaults",
    undo: () => {
      form.state.value = previous;
    },
    commit: () => {},
  });
}

// ── Shape / summary projections ───────────────────────────────────────
const projection = computed(() =>
  projectResolution(form.state.value.width, form.state.value.height),
);
const aspectLabel = computed(
  () =>
    ASPECTS.find((a) => a.id === projection.value.aspectId)?.label ?? "Custom",
);

const advCount = computed(() =>
  sequenceMode.value
    ? Number(Boolean(draft.openingImage)) +
      Number(
        capabilities.value.supportsNegativePrompt &&
          draft.clips.some((clip) => clip.negativePrompt.trim()),
      ) +
      Number(Boolean(draft.clips.some((clip) => clip.cameraControl))) +
      Number(draft.enableAudio)
    : advancedActiveCount({
        negativePrompt: capabilities.value.supportsNegativePrompt
          ? form.state.value.negativePrompt
          : "",
        negativePromptDefault: capabilities.value.supportsNegativePrompt
          ? (form.state.value.negativePromptDefault ?? "")
          : "",
        loraCount: form.state.value.loras.length,
        upscaleOn: form.state.value.upscaleModel.trim() !== "",
        scheduler: capabilities.value.supportsScheduler
          ? form.state.value.scheduler
          : null,
        customSize: projection.value.isCustom,
        videoNonDefault:
          capabilities.value.supportsVideo &&
          form.state.value.frames != null &&
          form.state.value.frames !== 25,
        videoSuite:
          capabilities.value.supportsVideo &&
          (form.state.value.gifPreview ||
            form.state.value.cameraControl != null ||
            form.state.value.enableAudio === false ||
            form.state.value.pipeline != null ||
            form.state.value.icLoraControl != null ||
            form.state.value.audioFile != null ||
            form.state.value.audioFilePath.trim() !== "" ||
            form.state.value.sourceVideo != null ||
            form.state.value.sourceVideoPath.trim() !== "" ||
            form.state.value.keyframes.length > 0 ||
            form.state.value.retakeRange != null ||
            form.state.value.spatialUpscale != null ||
            form.state.value.temporalUpscale != null ||
            guidanceOverrideCount(form.state.value.guidanceOverrides) > 0),
        wanRecipe: capabilities.value.wanRecipe.supported
          ? wanRecipeCount(form.state.value.wanRecipe)
          : 0,
      }),
);

// ── Canvas state ──────────────────────────────────────────────────────
function percentFor(job: Job): number | null {
  const p = job.progress;
  if (p.step !== null && p.totalSteps) {
    return Math.round((p.step / p.totalSteps) * 100);
  }
  if (p.weightBytesLoaded !== null && p.weightBytesTotal) {
    return Math.round((p.weightBytesLoaded / p.weightBytesTotal) * 100);
  }
  return null;
}

// The job the canvas develops: the running job with a live preview (the one
// the server is actually denoising), else the earliest-submitted running job.
// A naive newest-first pick would bind the canvas to a queued batch sibling
// that never previews while an earlier sibling is mid-denoise.
const runningJob = computed(() => {
  const selected = stream.selectedJob.value;
  return selected?.state === "running"
    ? selected
    : activeCanvasJob(stream.jobs.value);
});
const latestDone = computed(() => {
  const selected = stream.selectedJob.value;
  if (selected) return selected.state === "done" ? selected : null;
  let best: Job | null = null;
  for (const j of stream.jobs.value) {
    if (
      j.state === "done" &&
      j.result &&
      (!best || j.startedAt > best.startedAt)
    ) {
      best = j;
    }
  }
  return best;
});

const latestError = computed(() =>
  latestUnresolvedError(
    stream.jobs.value,
    stream.canvasErrorJobId.value,
    stream.selectedJob.value,
  ),
);
const latestErrorMessage = computed(() => {
  const job = latestError.value;
  return job?.error
    ? describeTransportError(job.error, job.hostLabel)
    : "Something went wrong while developing this print.";
});
const latestErrorCopy = computed(() => {
  const job = latestError.value;
  return job?.error
    ? copyableError(job.error, latestErrorMessage.value)
    : latestErrorMessage.value;
});

const canvasMode = computed<
  "empty" | "generating" | "result" | "error" | "variations"
>(() => {
  if (variations.value.length) return "variations";
  if (runningJob.value) return "generating";
  // `latestError` exists only when an exact live/selected failure owns the
  // canvas. Do not second-guess that authority with persisted timestamps:
  // a newer metadata-only success can survive a reload without its image
  // payload and must not mask a recovery error discovered by this boot.
  if (latestError.value) return "error";
  if (latestDone.value) return "result";
  return "empty";
});

const genProgress = computed(() =>
  runningJob.value ? (percentFor(runningJob.value) ?? 0) : 0,
);
const genStage = computed(() => {
  const j = runningJob.value;
  if (!j) return "";
  const p = j.progress;
  if (p.step !== null && p.totalSteps)
    return `Developing ${p.step} / ${p.totalSteps}`;
  return p.stage || "Loading model";
});

// Develop-bed props for the generating canvas (desktop `jobPhase` mapping):
// latent until the first denoise step is reported, developing during, and the
// canvas flips to result/error modes at finish so "fixed" never renders here.
const developPhase = computed<DevelopPhase>(() => {
  const j = runningJob.value;
  if (!j) return "latent";
  return j.progress.step !== null ? "developing" : "latent";
});

const resultSrc = computed(() => {
  const r = latestDone.value?.result;
  if (!r) return "";
  // Audio first: an audio print's `image` is the WAV itself, so the still
  // branch below would build `data:image/wav;…` and render a broken canvas
  // at the end of a render that actually succeeded.
  if (isAudioCompletion(r)) {
    return r.audio_thumbnail
      ? `data:image/png;base64,${r.audio_thumbnail}`
      : "";
  }
  if (r.video_thumbnail) return `data:image/png;base64,${r.video_thumbnail}`;
  return `data:image/${r.format};base64,${r.image}`;
});
/** The playable artifact for an audio-only print; empty for every other kind. */
const resultAudioSrc = computed(() => {
  const r = latestDone.value?.result;
  if (!r || !isAudioCompletion(r) || !r.image) return "";
  return `data:audio/wav;base64,${r.image}`;
});
const resultCaption = computed(() => {
  const job = latestDone.value;
  const r = job?.result;
  if (!r) return "";
  const secs = Math.round(r.generation_time_ms / 1000);
  // Name the machine that actually rendered it — an unrouted job ran here.
  const where = job?.hostLabel ?? "this server";
  return `${modelDisplayNameForId(r.model, models.value)} · seed ${r.seed_used} · ${secs}s · ${where}`;
});

function openLatestResult() {
  if (latestDone.value) openJob(latestDone.value);
}

// ── Source preprocessing / fitting (desktop/mobile parity) ────────────
type PreparedStillSource = {
  source: SourceImageState | null;
  mask: SourceImageState | null;
};

const sourceFitCache = new SourceFitPreprocessCache();

/** Prepare request-only source bytes while preserving the user's editable source. */
async function prepareStillSourceToRequest(
  route: HostRoute | null,
  override?: {
    source: SourceImageState | null;
    mask: SourceImageState | null;
    maskless?: boolean;
    settings?: {
      policy?: SourceFitPolicy;
      upscalerModel?: string;
      family: string;
      frames: number | null;
      width: number;
      height: number;
    };
  },
): Promise<PreparedStillSource | false> {
  let source = override
    ? override.source
    : (form.state.value.imageAttachments[0] ?? null);
  const mask = override ? override.mask : form.state.value.maskImage;
  if (!source) return { source, mask };

  const configuredPolicy =
    override?.settings?.policy ??
    form.state.value.sourceFitPolicy ??
    ({ mode: "pad-repaint" } as const);
  const outerPolicy = override?.maskless
    ? coerceSourceFitForMaskless(configuredPolicy)
    : configuredPolicy;
  if (outerPolicy?.mode === "upscale-then-fit") {
    const model =
      outerPolicy.upscalerModel ||
      override?.settings?.upscalerModel ||
      form.state.value.upscaleModel;
    if (model) {
      try {
        const original = source;
        const base64 = await sourceFitCache.upscale(
          original.base64,
          model,
          async () => {
            preprocessingStatus.value = `Preprocessing source with ${model}`;
            let output: string | null = null;
            await upscaleStream(
              { model, image: original.base64, output_format: "png" },
              {
                onProgress: (evt) => {
                  if (evt.type === "stage_start")
                    preprocessingStatus.value = evt.name;
                  if (evt.type === "stage_done")
                    preprocessingStatus.value = `${evt.name} (done)`;
                  if (evt.type === "info")
                    preprocessingStatus.value = evt.message;
                },
                onComplete: (evt) => (output = evt.image),
                onError: (err) => {
                  composerError.value =
                    err.kind === "http"
                      ? `Source preprocessing failed: ${err.body}`
                      : `Source preprocessing failed: ${err.message}`;
                },
              },
              undefined,
              route?.target,
            );
            if (output === null)
              throw new Error("Source preprocessing did not return an image");
            return output;
          },
        );
        source = {
          ...original,
          filename: original.filename.replace(/(\.[^.]+)?$/, "-prefit.png"),
          base64,
          width: undefined,
          height: undefined,
          mime: "image/png",
        };
      } catch (error) {
        if (!composerError.value) {
          const message =
            error instanceof Error ? error.message : String(error);
          composerError.value = `Source preprocessing failed: ${message}`;
        }
        return false;
      } finally {
        preprocessingStatus.value = null;
      }
    }
  }

  const family =
    override?.settings?.family ??
    currentModel.value?.family ??
    form.state.value.modelFamily;
  if (
    isQwenImageEditFamily(family) ||
    ((override?.settings?.frames ?? form.state.value.frames) &&
      !override?.maskless)
  )
    return { source, mask };
  const target = {
    width: override?.settings?.width ?? form.state.value.width,
    height: override?.settings?.height ?? form.state.value.height,
  };
  if (source.width === target.width && source.height === target.height) {
    return { source, mask };
  }
  const policy = drawableFitPolicy(outerPolicy);
  return sourceFitCache.fit(
    source.base64,
    mask?.base64 ?? null,
    policy,
    target,
    async () => {
      const sourceImg = await loadHtmlImage(source!);
      if (
        sourceImg.naturalWidth === target.width &&
        sourceImg.naturalHeight === target.height
      ) {
        return { source, mask };
      }
      const transform = resolveSourceFitTransform(
        { width: sourceImg.naturalWidth, height: sourceImg.naturalHeight },
        target,
        policy,
      );
      const canvas = document.createElement("canvas");
      canvas.width = transform.outputWidth;
      canvas.height = transform.outputHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return { source, mask };
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(
        sourceImg,
        transform.offsetX,
        transform.offsetY,
        transform.drawWidth,
        transform.drawHeight,
      );
      const fittedSource = canvasToSourceImage(canvas, source!, "-fit");

      if (policy.mode !== "pad-repaint" && !mask)
        return { source: fittedSource, mask };
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = transform.outputWidth;
      maskCanvas.height = transform.outputHeight;
      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) return { source: fittedSource, mask };
      maskCtx.fillStyle = "black";
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      if (mask) {
        const maskImg = await loadHtmlImage(mask);
        maskCtx.drawImage(
          maskImg,
          transform.offsetX,
          transform.offsetY,
          transform.drawWidth,
          transform.drawHeight,
        );
      }
      if (policy.mode === "pad-repaint") {
        maskCtx.fillStyle = "white";
        for (const rect of maskPaddingRectangles(transform)) {
          maskCtx.fillRect(rect.x, rect.y, rect.width, rect.height);
        }
      }
      const fittedMask = canvasToSourceImage(
        maskCanvas,
        mask ?? { kind: source!.kind, filename: "pad-mask.png", base64: "" },
        "-fit-mask",
      );
      return { source: fittedSource, mask: fittedMask };
    },
  );
}

// ── Submit (preserved logic) ──────────────────────────────────────────
function validateSubmit(): boolean {
  composerError.value = null;
  preprocessingStatus.value = null;
  if (!form.state.value.model) {
    showAdvanced.value = false;
    composerError.value = "Pick a model to start.";
    return false;
  }
  const h3Error = h3FrameError.value;
  if (h3Error) {
    composerError.value = h3Error;
    showAdvanced.value = true;
    return false;
  }
  if (h3GenerationInputBlocker.value) {
    composerError.value = h3GenerationInputBlocker.value;
    return false;
  }
  const recipe = effectiveGenerationRecipe(
    currentModel.value,
    form.state.value.pipeline,
  );
  // Fixed recipe controls are not user choices: a stale form value (draft
  // restored before the recipe landed, model swapped under it) snaps to the
  // value the disabled control displays instead of stranding Generate behind
  // an error the user cannot correct. Shared logic with desktop.
  const fixedControls = fixedRecipeControlOverrides(recipe);
  if (fixedControls.steps !== undefined) {
    form.state.value.steps = fixedControls.steps;
  }
  if (fixedControls.guidance !== undefined) {
    form.state.value.guidance = fixedControls.guidance;
  }
  // Resolution constraints are advisory — the server is the authority and
  // its own refusal surfaces as the failed job's error. Only malformed
  // input that cannot form a request blocks the submit (recipe or not).
  if (
    !Number.isInteger(form.state.value.width) ||
    !Number.isInteger(form.state.value.height) ||
    form.state.value.width < 1 ||
    form.state.value.height < 1
  ) {
    composerError.value = "Width and height must be whole numbers.";
    return false;
  }
  const resolutionFinding = resolutionProfileFinding(
    form.state.value.width,
    form.state.value.height,
    recipe?.resolution,
  );
  const profileError =
    (resolutionFinding?.level === "block" ? resolutionFinding.message : null) ??
    integerControlError("Steps", form.state.value.steps, recipe?.steps) ??
    floatControlError("Guidance", form.state.value.guidance, recipe?.guidance);
  if (profileError) {
    composerError.value = profileError;
    return false;
  }
  const qwenImageEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  if (qwenImageEdit && form.state.value.imageAttachments.length === 0) {
    composerError.value = "Qwen image edit needs a target image.";
    return false;
  }
  const referenceEdit =
    qwenImageEdit || isFlux2DevModel(form.state.value.model);
  if (
    !referenceEdit &&
    form.state.value.maskImage &&
    form.state.value.imageAttachments.length === 0
  ) {
    composerError.value = "Mask image needs a source image.";
    return false;
  }
  // The per-model source-image contract (#772) plus wan's first/last-frame
  // pairing (#779), in the same order admission checks them. H3 is excluded:
  // its boundary images have their own authoring validator above, which names
  // the missing one precisely. A continuation carries its own first frames in
  // the tail of the clip it continues, so it satisfies the contract exactly as
  // admission's `request_carries_source_frames` reads it (#783).
  const conditioningError = isMinimaxH3Identity(
    currentFamily.value,
    form.state.value.model,
  )
    ? null
    : sourceImageValidationError({
        capability: capabilities.value.sourceImageCapability,
        hasSourceImage: form.state.value.imageAttachments.length > 0,
        isExtend: submitsExtend({
          family: currentFamily.value,
          extendVideo: form.state.value.extendVideo,
          extendVideoPath: form.state.value.extendVideoPath,
        }),
        hasEndFrame:
          capabilities.value.supportsEndFrame &&
          form.state.value.endFrame != null,
        frames: capabilities.value.supportsVideo
          ? form.state.value.frames
          : null,
        model: form.state.value.model,
      });
  if (conditioningError) {
    composerError.value = conditioningError;
    showAdvanced.value = true;
    return false;
  }
  if (form.state.value.icLoraControl) {
    if (
      !form.state.value.sourceVideo &&
      !form.state.value.sourceVideoPath.trim()
    ) {
      composerError.value = "Reference control requires a guide video.";
      showAdvanced.value = true;
      return false;
    }
    if (form.state.value.loras.length + 1 > 4) {
      composerError.value =
        "Reference control plus custom LoRAs exceeds the four-LoRA limit.";
      showAdvanced.value = true;
      return false;
    }
  }
  // Lip dub always renders in two stages, so stage 1 halves both axes and
  // needs them on the VAE's /32 latent grid afterwards.
  if (
    form.state.value.pipeline === "lip-dub" &&
    (form.state.value.width % 64 !== 0 || form.state.value.height % 64 !== 0)
  ) {
    composerError.value =
      "Lip dub renders in two stages, so width and height must be multiples of 64.";
    showAdvanced.value = true;
    return false;
  }
  // A value the wire cannot carry — an unparsable block list, a fractional
  // skip stride — would otherwise be dropped on the way out, quietly
  // rendering with the pipeline's own constants.
  const guidanceError = guidanceOverridesError(
    form.state.value.guidanceOverrides,
  );
  if (guidanceError) {
    composerError.value = guidanceError;
    showAdvanced.value = true;
    return false;
  }
  // Same contract for the wan recipe: an out-of-band shift or distill strength
  // is dropped by the serializer, so it has to be reported here or the render
  // would silently fall back to the tier's own values.
  const recipeError = wanRecipeError(form.state.value.wanRecipe);
  if (recipeError) {
    composerError.value = recipeError;
    showAdvanced.value = true;
    return false;
  }
  return true;
}

/**
 * The machine this submission dispatches to.
 *
 * `null` means "don't route" — the single-machine case, where requests stay
 * relative to the serving origin exactly as they always have. `false` means the
 * user's pick is currently unreachable: a pinned machine that went offline is
 * an error, never a silent reroute of their print to a different GPU.
 */
function resolveSubmitRoute(): HostRoute | null | false {
  if (!routing.multiHost.value) return null;
  const route = routing.resolve(form.state.value.model || null);
  if (!route) {
    toast(
      "error",
      "The machine you picked isn't reachable. Choose another under Run on.",
    );
    return false;
  }
  return route;
}

async function resolveFeasibleSubmitRoute(
  request: GenerateRequestWire,
  copies = 1,
): Promise<HostRoute | false> {
  const result = await routing.resolveFeasible(request, copies);
  if (result.kind !== "route") {
    toast("error", feasibilityMessage(result, "this print"));
    return false;
  }
  return result.route;
}

function terminalPunctuation(value: string): string {
  return /[.!?]$/.test(value) ? value : `${value}.`;
}

function sentenceCaseHostLabel(label: string): string {
  return label === "this server" ? "This server" : label;
}

function feasibilityMessage(
  result: Exclude<FeasibilityResult, { kind: "route" }>,
  subject: string,
): string {
  if (result.kind === "profile_mismatch") {
    return profileConflictMessage(result.perHost);
  }
  const unreachableMessages = (
    hosts: ReadonlyArray<{ label: string; error: string }>,
  ) =>
    hosts
      .map(
        (host) =>
          `${sentenceCaseHostLabel(host.label)} didn't answer the feasibility check: ${terminalPunctuation(host.error)}`,
      )
      .join(" ");
  const infeasibleMessages = (
    hosts: ReadonlyArray<{
      label: string;
      reason: string;
      missingComponents: ReadonlyArray<{
        name: string;
        present: boolean;
      }>;
    }>,
  ) =>
    hosts
      .map((host) => {
        const missingNames = [
          ...new Set(
            host.missingComponents
              .filter((component) => !component.present)
              .map((component) => component.name),
          ),
        ];
        const missing =
          missingNames.length > 0
            ? ` Missing components: ${missingNames.join(", ")}.`
            : "";
        return `${sentenceCaseHostLabel(host.label)} can't run ${subject}: ${terminalPunctuation(host.reason)}${missing}`;
      })
      .join(" ");
  if (result.kind === "transient") {
    if (result.perHost.length === 0) {
      return `Routing changed while Mold checked ${subject}. Please try again.`;
    }
    const primary = result.perHost
      .map(
        (host) =>
          `${sentenceCaseHostLabel(host.label)} couldn't compute a placement plan right now: ${terminalPunctuation(host.reason)} Try again.`,
      )
      .join(" ");
    return [
      primary,
      result.infeasible ? infeasibleMessages(result.infeasible) : "",
      result.unreachable ? unreachableMessages(result.unreachable) : "",
    ]
      .filter(Boolean)
      .join(" ");
  }
  if (result.kind === "unreachable") {
    if (result.perHost.length === 0) {
      return `No selected machine could be reached to check ${subject}.`;
    }
    return [
      unreachableMessages(result.perHost),
      result.infeasible ? infeasibleMessages(result.infeasible) : "",
    ]
      .filter(Boolean)
      .join(" ");
  }
  if (result.perHost.length === 0) {
    return `No selected machine can run ${subject}.`;
  }
  return [
    infeasibleMessages(result.perHost),
    result.unreachable ? unreachableMessages(result.unreachable) : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function requestCopyCount(request: GenerateRequestWire): number {
  return Math.max(1, Math.floor(request.batch_size ?? 1));
}

function submitRequestCopies(
  request: GenerateRequestWire,
  decision: ReturnType<typeof decideGenerateRequestRouting>,
  route: HostRoute | null,
): void {
  const copies = requestCopyCount(request);
  if (copies === 1) {
    stream.submit(request, decision, normalizeSubmitRoute(route, request));
    return;
  }

  const batchId = createUuid();
  const baseSeed =
    request.seed === null || request.seed === undefined
      ? crypto.getRandomValues(new Uint32Array(1))[0]!
      : request.seed;
  for (let index = 0; index < copies; index += 1) {
    stream.submit(
      {
        ...request,
        batch_size: 1,
        batch_id: batchId,
        batch_index: index + 1,
        batch_count: copies,
        seed: baseSeed + index,
      },
      decision,
      normalizeSubmitRoute(route, request),
    );
  }
}

/** True while a Generate click is being routed/admitted. The feasibility
 * previews are authoritative server roundtrips that can take seconds on a
 * heavy plan; without this the click looked like a silent no-op and a second
 * click could double-queue. */
const submitInFlight = ref(false);
const placementStatus = ref<string | null>(null);
async function onSubmit(allowStaleQuick = false) {
  if (submitInFlight.value) return;
  submitInFlight.value = true;
  placementStatus.value = "Checking fit on the selected machine…";
  try {
    await onSubmitInner(allowStaleQuick);
  } finally {
    submitInFlight.value = false;
    placementStatus.value = null;
  }
}

async function onSubmitInner(allowStaleQuick = false) {
  if (ordinarySubmitBlocked.value) return;
  // The route is settled first, and before source preprocessing, for two
  // reasons: an unreachable pinned machine is the real complaint (its model
  // list is empty, so a model check first would blame the model instead), and
  // an upscale cache miss has to land on the same machine as the render.
  const quick = quickPrepared.value;
  if (quick) {
    const stale = quickStaleReasons(quick);
    if (stale.length && !allowStaleQuick) {
      return;
    }
  }
  let route =
    quick && !allowStaleQuick && quickRouteIsCurrent(quick)
      ? cloneRoute(quick.route)
      : resolveSubmitRoute();
  if (route === false) return;
  if (!validateSubmit()) return;
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  const currentRequest = form.toRequest(currentModel.value);
  const originalSource = form.state.value.imageAttachments[0]
    ? {
        ...form.state.value.imageAttachments[0],
        sourceFit: parseSourceFitPolicy(form.state.value.sourceFitPolicy) ?? {
          mode: "pad-repaint",
        },
      }
    : form.state.value.h3Authoring?.firstFrame?.data
      ? {
          base64: form.state.value.h3Authoring.firstFrame.data,
          filename: form.state.value.h3Authoring.firstFrame.filename,
          width: form.state.value.h3Authoring.firstFrame.width,
          height: form.state.value.h3Authoring.firstFrame.height,
          mime: form.state.value.h3Authoring.firstFrame.mimeType,
          sourceFit: parseSourceFitPolicy(form.state.value.sourceFitPolicy) ?? {
            mode: "pad-repaint",
          },
        }
      : null;
  const copies = requestCopyCount(currentRequest);
  if (quick && !allowStaleQuick) {
    const result = route
      ? decision.kind === "chain"
        ? await routing.revalidateFeasibleChain(
            route,
            resolveChainRequest(currentRequest, decision),
            copies,
          )
        : await routing.revalidateFeasible(route, currentRequest, copies)
      : await resolveFeasibleSubmitRoute(currentRequest, copies);
    if (result === false) return;
    if ("kind" in result && result.kind !== "route") {
      toast("error", feasibilityMessage(result, "this prepared print"));
      return;
    }
    const feasible = "kind" in result ? result.route : result;
    if (!sameRoute(route, feasible)) {
      toast(
        "error",
        "The prepared machine is no longer the feasible route for this print. Re-expand or choose Generate anyway.",
      );
      return;
    }
    route = feasible;
  } else {
    const result =
      decision.kind === "chain"
        ? await routing.resolveFeasibleChain(
            resolveChainRequest(currentRequest, decision),
            copies,
          )
        : await routing.resolveFeasible(currentRequest, copies);
    if (result.kind !== "route") {
      toast("error", feasibilityMessage(result, "this print"));
      return;
    }
    route = result.route;
  }
  const preparedSource = await prepareStillSourceToRequest(route);
  if (preparedSource === false) return;
  const req = form.toRequest(currentModel.value);
  const finalizedCopies = requestCopyCount(req);
  if (quick) req.original_prompt = quick.originalPrompt;
  if (quick?.promptTransform) req.prompt_transform = quick.promptTransform;
  // H3's dedicated FL2VA boundaries own first/last endpoints. The legacy
  // still preprocessor has no corresponding attachment and must never erase
  // that serialized first-frame authority — instead each boundary takes the
  // same client-side fit, coerced maskless.
  if (isMinimaxH3Identity(currentFamily.value, form.state.value.model)) {
    const h3 = form.state.value.h3Authoring;
    const boundaryRoute: HostRoute | null = route || null;
    const fitBoundary = async (
      base64: string,
      boundary: {
        filename: string;
        width: number;
        height: number;
        mimeType: string;
      },
    ): Promise<string | false> => {
      const fitted = await prepareStillSourceToRequest(boundaryRoute, {
        source: {
          kind: "upload",
          filename: boundary.filename,
          base64,
          width: boundary.width,
          height: boundary.height,
          mime: boundary.mimeType,
        },
        mask: null,
        maskless: true,
      });
      if (fitted === false) return false;
      return fitted.source?.base64 ?? base64;
    };
    if (typeof req.source_image === "string" && h3?.firstFrame) {
      const fitted = await fitBoundary(req.source_image, h3.firstFrame);
      if (fitted === false) return;
      req.source_image = fitted;
    }
    const keyframes = (req as { keyframes?: { image: string }[] }).keyframes;
    if (keyframes?.[0] && h3?.lastFrame) {
      const fitted = await fitBoundary(keyframes[0].image, h3.lastFrame);
      if (fitted === false) return;
      keyframes[0].image = fitted;
    }
  } else if ("source_image" in req) {
    req.source_image = preparedSource.source?.base64 ?? null;
    if (preparedSource.mask) req.mask_image = preparedSource.mask.base64;
    else delete req.mask_image;
  }
  if (req.source_image && originalSource) {
    void persistGenerationSourceMedia(req.source_image, originalSource);
  }
  const finalizedResult = route
    ? decision.kind === "chain"
      ? await routing.revalidateFeasibleChain(
          route,
          resolveChainRequest(req, decision),
          finalizedCopies,
        )
      : await routing.revalidateFeasible(route, req, finalizedCopies)
    : decision.kind === "chain"
      ? await routing.resolveFeasibleChain(
          resolveChainRequest(req, decision),
          finalizedCopies,
        )
      : await routing.resolveFeasible(req, finalizedCopies);
  if (finalizedResult.kind !== "route") {
    toast("error", feasibilityMessage(finalizedResult, "this finalized print"));
    return;
  }
  const finalizedRoute = finalizedResult.route;
  if (!sameRoute(route, finalizedRoute)) {
    toast(
      "error",
      "The finalized source request is no longer feasible on the selected machine. Nothing was queued.",
    );
    return;
  }
  route = finalizedRoute;
  submitRequestCopies(req, decision, route);
  quickPrepared.value = null;
  // Push to history immediately so ↑ recalls it before the server round-trips.
  composerCardRef.value?.record(req.prompt);
  if (
    form.state.value.seedMode === "increment" &&
    form.state.value.seed !== null
  ) {
    form.state.value.seed += Math.max(1, form.state.value.batchSize);
  }
}

// ── Expand (spec §03/§06) ─────────────────────────────────────────────
function validateExpandedPrompts(
  prompts: readonly string[],
  expected: number,
): string[] {
  const normalized = prompts.map((prompt) => prompt.trim());
  if (normalized.length !== expected || normalized.some((prompt) => !prompt)) {
    throw new Error(`Expected exactly ${expected} non-empty expanded prompts.`);
  }
  return normalized;
}

async function onExpand() {
  // Desktop parity (`ExpandControl.expand`): expansion rewrites the prompt,
  // so there has to be one. This stays true even where a blank prompt is a
  // legitimate render — there is nothing to enrich.
  if (!form.state.value.prompt.trim()) return;
  if (form.state.value.batchSize > 1) {
    if (preparingVariations.value) return;
    if (!validateSubmit()) return;
    const decision = chainDecision.value;
    if (decision.kind === "reject") {
      toast("error", decision.reason);
      return;
    }
    const sourcePrompt = form.state.value.prompt.trim();
    const count = form.state.value.batchSize;
    const family = currentFamily.value;
    const model = form.state.value.model;
    const selectedHostPolicy = routing.targetId.value;
    const baseRequest = form.toRequest(currentModel.value);
    const task = expansionTaskForCurrentOutput(baseRequest);
    preparingVariations.value = true;
    try {
      const result =
        decision.kind === "chain"
          ? await routing.resolveFeasibleChain(
              resolveChainRequest(baseRequest, decision),
              count,
            )
          : await routing.resolveFeasible(baseRequest, count);
      if (result.kind !== "route") {
        toast("error", feasibilityMessage(result, "every prepared variation"));
        return;
      }
      const route = result.route;
      const submitRoute = normalizeSubmitRoute(route);
      const style = styleHint(form.state.value.stylePreset ?? "");
      composerError.value = null;
      const response = await expandPrompt(
        {
          prompt: sourcePrompt,
          model_family: family,
          variations: count,
          ...(style ? { style } : {}),
          task,
        },
        undefined,
        submitRoute?.target,
      );
      variations.value = validateExpandedPrompts(response.expanded, count);
      preparedBatch.value = {
        batchId: createUuid(),
        sourcePrompt,
        model,
        family,
        task,
        requestedCount: count,
        selectedHostPolicy,
        baseRequest,
        decision,
        route: cloneRoute(route)!,
      };
    } catch (error) {
      composerError.value =
        error instanceof Error ? error.message : String(error);
    } finally {
      preparingVariations.value = false;
    }
    return;
  }
  // batch = 1: server enrichment via the Expand modal, applied in place.
  const route = resolveSubmitRoute();
  if (route === false) return;
  expandRoute.value = cloneRoute(route);
  expandTask.value = expansionTaskForCurrentOutput(
    form.toRequest(currentModel.value),
  );
  showExpand.value = true;
}

async function onRemix() {
  if (!form.state.value.prompt.trim()) return;
  if (!validateSubmit()) return;
  const baseRequest = form.toRequest(currentModel.value);
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  const result =
    decision.kind === "chain"
      ? await routing.resolveFeasibleChain(
          resolveChainRequest(baseRequest, decision),
          3,
        )
      : await routing.resolveFeasible(baseRequest, 3);
  if (result.kind !== "route") {
    toast("error", feasibilityMessage(result, "three reviewed remixes"));
    return;
  }
  remixRoute.value = cloneRoute(result.route);
  remixTask.value = expansionTaskForCurrentOutput(baseRequest);
  showRemix.value = true;
}

function applyRemix(payload: { prompt: string; response: RemixResponseWire }) {
  // The modal may have remixed the durable original while the user continued
  // editing the composer. Undo must return to the live text replaced by Apply,
  // not to the transform backend's (possibly older) source prompt.
  prevPrompt.value = form.state.value.prompt;
  prevOriginalPrompt.value = form.state.value.originalPrompt ?? null;
  form.state.value.originalPrompt =
    payload.response.root_prompt ?? payload.response.source_prompt;
  form.state.value.prompt = payload.prompt;
  quickPrepared.value = {
    expandedPrompt: payload.prompt,
    originalPrompt:
      payload.response.root_prompt ?? payload.response.source_prompt,
    model: form.state.value.model,
    family: currentFamily.value,
    task: remixTask.value,
    selectedHostPolicy: routing.targetId.value,
    route: cloneRoute(remixRoute.value),
    promptTransform: {
      operation: "remix",
      ...(payload.response.root_prompt
        ? { root_prompt: payload.response.root_prompt }
        : {}),
      source_prompt: payload.response.source_prompt,
      source_kind: payload.response.source_kind,
      task: remixTask.value,
      dimensions:
        payload.response.variants.find(
          (variant) => variant.prompt === payload.prompt,
        )?.dimensions ?? [],
    },
  };
  // Remix, like Expand, weaves the active style into the returned prompt.
  // Clear the chip to avoid applying it twice and retain its curated negative
  // in the request; undo restores both through the established snapshot.
  bakeStyleAndClear();
  showRemix.value = false;
}

async function prepareRemixBatch(response: RemixResponseWire) {
  const route = remixRoute.value;
  if (!route) return;
  const baseRequest = form.toRequest(currentModel.value);
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  const variants = response.variants.map((variant) => variant.prompt.trim());
  if (
    variants.length < 2 ||
    variants.length > 3 ||
    variants.some((prompt) => !prompt)
  ) {
    composerError.value = "Select two or three non-empty remix variants.";
    return;
  }
  form.state.value.batchSize = variants.length;
  variations.value = variants;
  preparedBatch.value = {
    kind: "remix",
    batchId: createUuid(),
    sourcePrompt: response.source_prompt,
    ...(response.root_prompt ? { rootPrompt: response.root_prompt } : {}),
    sourceKind: response.source_kind,
    conditioningFingerprint: conditioningFingerprint(baseRequest),
    remixDimensions: response.variants.map((variant) => variant.dimensions),
    model: form.state.value.model,
    family: currentFamily.value,
    task: remixTask.value,
    requestedCount: variants.length,
    selectedHostPolicy: routing.targetId.value,
    baseRequest,
    decision,
    route: cloneRoute(route)!,
  };
  showRemix.value = false;
}

function onExpandClip(clipId: string, prompt: string) {
  const route = resolveSubmitRoute();
  if (route === false) return;
  expandRoute.value = cloneRoute(route);
  expandClipId.value = clipId;
  expandStagePrompt.value = prompt;
  const index = draft.clips.findIndex((clip) => clip.id === clipId);
  const clip = index >= 0 ? draft.clips[index] : null;
  const sourceImage =
    clip?.sourceImage?.base64 ||
    (index === 0 ? draft.openingImage?.base64 : undefined);
  expandTask.value = sourceImage
    ? "image-to-video"
    : index > 0 && clip?.transition === "smooth"
      ? "video-to-video"
      : expansionTaskForRequest(currentFamily.value, {});
  showExpand.value = true;
}

/**
 * Bake-and-clear owes the user the preset's curated negative: the chip is
 * about to be dropped, so submit-time composition will never see it again.
 * The look itself already reached the prompt — through the server's expansion
 * directive, or through the baked variation text — so only the negative half
 * has nowhere else to live. Returns the pre-bake negative for undo.
 */
function bakeStyleAndClear() {
  const preset = form.state.value.stylePreset;
  const negativeBefore = form.state.value.negativePrompt;
  const negativeBaked = mergeStyleNegative(negativeBefore, preset ?? "", {
    supportsNegativePrompt: capabilities.value.supportsNegativePrompt,
  });
  form.state.value.negativePrompt = negativeBaked;
  form.state.value.stylePreset = null;
  prevStyle.value = { preset, negativeBefore, negativeBaked };
}

function applyExpandedPrompt(v: string) {
  if (expandClipId.value !== null) {
    const clip = draft.clips.find((c) => c.id === expandClipId.value);
    if (clip) clip.prompt = v;
    return;
  }
  prevPrompt.value = form.state.value.prompt;
  prevOriginalPrompt.value = form.state.value.originalPrompt ?? null;
  quickPrepared.value = {
    expandedPrompt: v,
    originalPrompt: form.state.value.prompt.trim(),
    model: form.state.value.model,
    family: currentFamily.value,
    task: expandTask.value,
    selectedHostPolicy: routing.targetId.value,
    route: cloneRoute(expandRoute.value),
  };
  form.state.value.originalPrompt = form.state.value.prompt.trim();
  form.state.value.prompt = v;
  // Same bake-and-clear as the desktop app: the rewrite absorbed the look, so
  // leaving the chip lit would apply it twice at submit.
  bakeStyleAndClear();
}

function undoExpand() {
  if (prevPrompt.value === null) return;
  form.state.value.prompt = prevPrompt.value;
  form.state.value.originalPrompt = prevOriginalPrompt.value;
  // Undo re-arms the whole pre-expansion state: prompt, chip, and the negative
  // fragments the bake merged in — unless the user has edited the negative
  // since, which is theirs to keep.
  const style = prevStyle.value;
  if (style) {
    form.state.value.stylePreset = style.preset;
    if (form.state.value.negativePrompt === style.negativeBaked) {
      form.state.value.negativePrompt = style.negativeBefore;
    }
  }
  prevPrompt.value = null;
  prevOriginalPrompt.value = null;
  prevStyle.value = null;
  quickPrepared.value = null;
}

// ── Variations review (batch > 1) ─────────────────────────────────────
function useVariation(index: number) {
  const v = variations.value[index];
  if (v == null) return;
  form.state.value.prompt = v;
  form.state.value.originalPrompt =
    preparedBatch.value?.rootPrompt ??
    preparedBatch.value?.sourcePrompt ??
    null;
  // The variation text already carries the baked look, so the chip clears —
  // and the preset's curated negative comes with it.
  bakeStyleAndClear();
  variations.value = [];
  preparedBatch.value = null;
}

function discardVariations() {
  variations.value = [];
  preparedBatch.value = null;
}

async function queueVariations() {
  if (queueingVariations.value) return;
  const prepared = preparedBatch.value;
  if (!prepared) {
    composerError.value =
      "These variations are no longer prepared. Expand again.";
    return;
  }
  const stale = preparedStaleReasons(prepared);
  if (stale.length) {
    composerError.value = stale.join(" ");
    return;
  }
  const list = variations.value.map((prompt) => prompt.trim());
  if (list.some((prompt) => !prompt)) {
    composerError.value =
      "Every prepared variation needs a prompt before queueing.";
    return;
  }
  queueingVariations.value = true;
  try {
    const revalidated =
      prepared.decision.kind === "chain"
        ? await routing.revalidateFeasibleChain(
            prepared.route,
            resolveChainRequest(prepared.baseRequest, prepared.decision),
            list.length,
          )
        : await routing.revalidateFeasible(
            prepared.route,
            prepared.baseRequest,
            list.length,
          );
    const stillOwned =
      preparedBatch.value?.batchId === prepared.batchId &&
      variations.value.length === list.length &&
      variations.value.every((prompt, index) => prompt.trim() === list[index]);
    if (!stillOwned) {
      composerError.value =
        "The prepared variations changed while the machine was being checked. Nothing was queued; review the current batch and try again.";
      return;
    }
    const refreshedStale = preparedStaleReasons(prepared);
    if (refreshedStale.length) {
      composerError.value = refreshedStale.join(" ");
      return;
    }
    if (
      revalidated.kind !== "route" ||
      !sameRoute(prepared.route, revalidated.route)
    ) {
      composerError.value =
        revalidated.kind === "route"
          ? "The prepared machine can no longer run this complete batch. Nothing was queued; your reviewed variations are preserved."
          : `${feasibilityMessage(revalidated, "this complete batch")} Nothing was queued; your reviewed variations are preserved.`;
      return;
    }
    for (const [index, prompt] of list.entries()) {
      // Each variation already carries the style extras, so it is the final
      // prompt — override the base request's prompt rather than re-appending.
      // Each is one print; the batch size drove the variation count, not the
      // per-job image count.
      const request: GenerateRequestWire = {
        ...prepared.baseRequest,
        prompt,
        batch_size: 1,
        original_prompt: prepared.rootPrompt ?? prepared.sourcePrompt,
        ...(prepared.kind === "remix"
          ? {
              prompt_transform: {
                operation: "remix" as const,
                ...(prepared.rootPrompt
                  ? { root_prompt: prepared.rootPrompt }
                  : {}),
                source_prompt: prepared.sourcePrompt,
                source_kind: prepared.sourceKind ?? "direct",
                task: prepared.task,
                dimensions: [...(prepared.remixDimensions?.[index] ?? [])],
              },
            }
          : {}),
        batch_id: prepared.batchId,
        batch_index: index + 1,
        batch_count: list.length,
      };
      stream.submit(
        request,
        prepared.decision,
        normalizeSubmitRoute(revalidated.route, request),
      );
    }
    variations.value = [];
    preparedBatch.value = null;
  } finally {
    queueingVariations.value = false;
  }
}

// ── Source image handling (preserved) ─────────────────────────────────
async function onClearSource() {
  if (form.state.value.maskImage && !(await resolveMaskSourceConflict()))
    return;
  form.state.value.imageAttachments = [];
}

async function onPickSource(v: SourceImageState[]) {
  if (sequenceMode.value) {
    const first = v[0];
    if (first) {
      draft.openingImage = {
        filename: first.filename,
        base64: first.base64,
        draftId: first.draftId,
        width: first.width ?? undefined,
        height: first.height ?? undefined,
      };
      form.state.value.sourceFitPolicy = coerceSourceFitForMaskless(
        form.state.value.sourceFitPolicy ?? { mode: "crop-fill" },
      );
    }
    composerError.value = null;
    return;
  }
  const qwenEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  const flux2Dev = isFlux2DevModel(form.state.value.model);
  const referenceEdit = qwenEdit || flux2Dev;
  if (
    !referenceEdit &&
    form.state.value.maskImage &&
    form.state.value.imageAttachments.length > 0 &&
    v.length > 0 &&
    !(await resolveMaskSourceConflict())
  ) {
    return;
  }
  form.state.value.imageAttachments = referenceEdit
    ? [...form.state.value.imageAttachments, ...v].slice(
        0,
        flux2Dev ? 4 : undefined,
      )
    : v.slice(0, 1);
  if (!referenceEdit && v.length > 0) {
    // A source-matched canvas differs only by the model's pixel grid or safe
    // downscale. Resize exactly instead of manufacturing narrow repaint bands.
    form.state.value.sourceFitPolicy = { mode: "lanczos-resize" };
  }
  composerError.value = null;
}

function onPickH3Boundary(images: SourceImageState[]): void {
  const endpoint = h3BoundaryPickerTarget.value;
  const image = images[0];
  if (!endpoint || !image) return;
  const result = setMinimaxH3PickedImageBoundary(
    form.state.value.h3Authoring,
    endpoint,
    {
      filename: image.filename,
      base64: image.base64,
      mimeType: image.mime,
      width: image.width,
      height: image.height,
    },
  );
  if (!result.ok) {
    composerError.value = result.error;
    return;
  }
  form.state.value.h3Authoring = result.state;
  composerError.value = null;
}

async function resolveMaskSourceConflict(): Promise<boolean> {
  const choice = await requestChoice({
    title: "Source image has a mask",
    body: "Changing the source affects the inpaint mask.",
    choices: [
      { id: "reset", label: "Clear the mask" },
      { id: "keep", label: "Keep and scale the mask" },
    ],
  });
  if (choice === null) return false;
  if (choice === "keep") return true;
  form.state.value.maskImage = null;
  return true;
}

function onPickEndFrame(picked: SourceImageState[]) {
  const first = picked[0];
  if (first) form.state.value.endFrame = first;
  showEndFramePicker.value = false;
  composerError.value = null;
}

function onClearEndFrame() {
  form.state.value.endFrame = null;
}

function onApplyMask(mask: SourceImageState) {
  form.state.value.maskImage = mask;
  showMask.value = false;
}

// ── Gallery drawer (preserved) ────────────────────────────────────────
function openItem(item: GalleryImage) {
  selectedIndex.value = galleryEntries.value.findIndex(
    (e) => e.filename === item.filename,
  );
  selected.value = item;
}

function recreateFromGallery(item: GalleryImage) {
  // The Create drawer can reuse a normal print without navigating, so retire
  // Sequence explicitly before applying that print's one-shot settings.
  setOutput("single");
  draft.stopEditing();
  editBaselineShared.value = null;
  draft.lastSingleModel = null;
  form.state.value = applyMetadataToForm(form.state.value, item.metadata, {
    format: item.format,
    models: models.value,
  });
  noticeFirstLastFrameRestore(item.metadata);
}

let openJobEpoch = 0;

function openJob(job: Job) {
  const epoch = ++openJobEpoch;
  stream.select(job.id);
  setOutput("single");
  // Activity's print rows call this path; durable sequence rows use the
  // sequence-action handoff below. Keep the broad Job request union at the
  // stream boundary, then restore the complete one-shot wire shape here.
  const request = job.request as GenerateRequestWire;
  const image = (base64: string, filename: string) => ({
    kind: "upload" as const,
    filename,
    base64,
  });
  if ("prompt" in request && typeof request.prompt === "string") {
    form.state.value.prompt = request.prompt;
  }
  form.state.value.originalPrompt = request.original_prompt ?? null;
  form.state.value.stylePreset = null;
  form.state.value.expand = {
    enabled: false,
    variations: 1,
    familyOverride: null,
  };
  form.state.value.sourceFitPolicy = parseSourceFitPolicy(
    request.source_fit,
  ) ?? { mode: "pad-repaint" };
  form.state.value.cameraControl = null;
  form.state.value.model = request.model;
  const requestModel = models.value.find(
    (model) => model.name === request.model,
  );
  form.state.value.modelFamily = requestModel?.family ?? "";
  form.state.value.negativePromptDefault = effectiveNegativeDefault(
    requestModel,
    requestModel?.family ?? "",
  );
  form.state.value.width = request.width;
  form.state.value.height = request.height;
  form.state.value.steps = request.steps;
  form.state.value.guidance = request.guidance ?? form.state.value.guidance;
  form.state.value.seed = request.seed == null ? null : Number(request.seed);
  form.state.value.seedMode = request.seed == null ? "random" : "static";
  // Absence in a queued request means the server materialized the model's
  // default (#787); a recorded "" was the explicit empty-uncond opt-out.
  form.state.value.negativePrompt = restoredNegativePrompt(
    request.negative_prompt,
    form.state.value.negativePromptDefault ?? "",
  );
  form.state.value.scheduler = request.scheduler ?? null;
  form.state.value.cfgPlus = request.cfg_plus ?? false;
  form.state.value.batchSize = request.batch_size ?? 1;
  form.state.value.outputFormat =
    request.output_format ?? form.state.value.outputFormat;
  form.state.value.strength = request.strength ?? 0.75;
  form.state.value.upscaleModel = request.upscale_model ?? "";
  form.state.value.gifPreview = request.gif_preview ?? false;
  form.state.value.placement = request.placement ?? null;
  const source = request.source_image
    ? image(request.source_image, request.source_image_name || "Source image")
    : null;
  form.state.value.imageAttachments = request.edit_images?.length
    ? request.edit_images.map((base64, index) =>
        image(base64, index === 0 ? "Target image" : `Reference ${index}`),
      )
    : source
      ? [source]
      : [];
  if (request.source_image) {
    const effectiveSource = request.source_image;
    void sha256HexOfBase64(effectiveSource)
      .then((sha256) => restoreGenerationSourceMedia(sha256))
      .then(async (restored) => {
        if (
          !restored ||
          epoch !== openJobEpoch ||
          form.state.value.imageAttachments[0]?.base64 !== effectiveSource
        )
          return;
        form.state.value.imageAttachments = [
          {
            kind: restored.kind ?? "upload",
            filename: restored.filename,
            base64: restored.base64,
            width: restored.width ?? undefined,
            height: restored.height ?? undefined,
            mime: restored.mime ?? undefined,
          },
        ];
        await nextTick();
        if (
          epoch !== openJobEpoch ||
          form.state.value.imageAttachments[0]?.base64 !== restored.base64
        )
          return;
        form.state.value.width = request.width;
        form.state.value.height = request.height;
        form.state.value.sourceFitPolicy = parseSourceFitPolicy(
          request.source_fit,
        ) ?? {
          mode: "pad-repaint",
        };
      });
  }
  form.state.value.maskImage = request.mask_image
    ? image(request.mask_image, "Mask")
    : null;
  form.state.value.controlImage = request.control_image
    ? image(request.control_image, "Control image")
    : null;
  form.state.value.controlModel = request.control_model ?? "";
  form.state.value.controlScale = request.control_scale ?? 1;
  form.state.value.loras = (
    request.loras ?? (request.lora ? [request.lora] : [])
  ).map((lora) => ({ ...lora, trainedWords: [] }));
  const camera = normalizeCameraMotionLoraState(
    form.state.value.loras,
    null,
    (path, scale) => ({ path, scale, trainedWords: [] }),
    MAX_LORA_STACK,
  );
  form.state.value.loras = camera.loras;
  form.state.value.cameraControl = camera.cameraControl;
  form.state.value.frames = request.frames ?? null;
  form.state.value.fps = request.fps ?? null;
  form.state.value.enableAudio = request.enable_audio ?? null;
  form.state.value.audioFile = request.audio_file
    ? image(request.audio_file, "Audio input")
    : null;
  form.state.value.audioFilePath = request.audio_file_path ?? "";
  form.state.value.sourceVideo = request.source_video
    ? image(request.source_video, "Video input")
    : null;
  form.state.value.sourceVideoPath = request.source_video_path ?? "";
  form.state.value.keyframes = (request.keyframes ?? []).map((keyframe) => ({
    frame: keyframe.frame,
    image: image(keyframe.image, keyframe.name ?? `Keyframe ${keyframe.frame}`),
  }));
  form.state.value.pipeline = request.pipeline ?? null;
  form.state.value.icLoraControl = request.ic_lora_control ?? null;
  form.state.value.retakeRange = request.retake_range ?? null;
  form.state.value.spatialUpscale = request.spatial_upscale ?? null;
  form.state.value.temporalUpscale = request.temporal_upscale ?? null;
  form.state.value.guidanceOverrides = guidanceOverridesFromWire(
    request.guidance_overrides,
  );
  form.state.value.wanRecipe = wanRecipeFromWire(request);
  if (minimaxH3TaskForModel(request.model)) {
    form.state.value.h3Authoring = emptyMinimaxH3AuthoringState();
    form.state.value.h3Authoring.references = (request.references ?? []).map(
      (reference) => ({
        reference: JSON.parse(JSON.stringify(reference)),
      }),
    );
    if (request.source_image) {
      form.state.value.h3Authoring.firstFrame = {
        filename: request.source_image_name ?? "First frame",
        mimeType: "image/*",
        width: 0,
        height: 0,
        data: request.source_image,
      };
    }
    const finalFrame = (request.frames ?? 1) - 1;
    const last = request.keyframes?.find(
      (keyframe) => keyframe.frame === finalFrame,
    );
    if (last) {
      form.state.value.h3Authoring.lastFrame = {
        filename: last.name ?? "Last frame",
        mimeType: "image/*",
        width: 0,
        height: 0,
        data: last.image,
      };
    }
  }
}

function closeDrawer() {
  selected.value = null;
  selectedIndex.value = -1;
}

function onLightboxReuse(item: GalleryImage) {
  recreateFromGallery(item);
  closeDrawer();
}

async function attachLightboxSource(item: GalleryImage): Promise<boolean> {
  try {
    const res = await fetch(imageUrl(item.filename));
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const base64 = await blobToBase64(await res.blob());
    form.state.value.imageAttachments = [
      { kind: "gallery", filename: item.filename, base64 },
    ];
    return true;
  } catch (err) {
    toast("error", err instanceof Error ? err.message : String(err));
    return false;
  }
}

async function onLightboxUseSource(item: GalleryImage) {
  if (!(await attachLightboxSource(item))) return;
  closeDrawer();
}

async function onLightboxUpscale(item: GalleryImage) {
  if (!(await attachLightboxSource(item))) return;
  form.state.value.upscaleModel ||= defaultUpscaler(models.value);
  closeDrawer();
  openAdvanced();
  toast(
    "info",
    form.state.value.upscaleModel
      ? "Added as source with Upscale after generate enabled."
      : "Added as source — install or choose an upscaler in Advanced.",
  );
}
function stepDrawer(delta: number) {
  if (selectedIndex.value < 0) return;
  const next = selectedIndex.value + delta;
  if (next < 0 || next >= galleryEntries.value.length) return;
  selectedIndex.value = next;
  selected.value = galleryEntries.value[next] ?? null;
}
async function handleDelete(item: GalleryImage) {
  try {
    await deleteGalleryImage(item.filename);
    galleryEntries.value = galleryEntries.value.filter(
      (e) => e.filename !== item.filename,
    );
    if (selected.value && selected.value.filename === item.filename) {
      closeDrawer();
    }
  } catch (e) {
    console.error(e);
  }
}

function openAdvanced() {
  showAdvanced.value = true;
}

onMounted(async () => {
  if (phoneQuery) {
    phoneQuery.addEventListener?.("change", syncPhone);
  }
  // Legacy/entry query handling — registered here (not at setup top level)
  // because the handler reads computeds declared later in the file.
  watch(() => pageRoute.query, consumeOutputQuery, { immediate: true });
  // Durable sequences: refresh every host's list and reattach the watch to
  // any tracked in-flight job.
  void chainJobs.start();
  // Models arrive from the host-routing poll (every machine, not just this
  // one); the watcher above homes the form onto one that's actually installed.
  void routing.refresh();
  try {
    galleryEntries.value = await listGallery();
  } catch (e) {
    console.error(e);
  }
  void refreshHistory();
  window.addEventListener("mold:new-print", onNewPrint);
  document.addEventListener("pointerdown", onTemplatesPointerDown);
  document.addEventListener("keydown", onTemplatesKeydown);
  startAutoRefresh();
});

onBeforeUnmount(() => {
  stopAutoRefresh();
  clearSequencePreviews();
  phoneQuery?.removeEventListener?.("change", syncPhone);
  window.removeEventListener("mold:new-print", onNewPrint);
  document.removeEventListener("pointerdown", onTemplatesPointerDown);
  document.removeEventListener("keydown", onTemplatesKeydown);
  queue.stop();
});
</script>

<template>
  <div
    data-test="generate-shell"
    class="mx-auto w-full max-w-[1600px] px-4 pb-24 pt-5"
  >
    <div
      data-test="generate-workspace"
      class="grid gap-4 md:grid-cols-[minmax(0,1fr)_340px]"
    >
      <!-- Center: activity + composer + canvas + recent -->
      <main class="flex min-w-0 flex-col gap-4">
        <h1
          v-if="isPhone"
          class="font-display text-2xl font-bold tracking-tight text-ink"
          data-test="phone-create-title"
        >
          Create
        </h1>
        <ActivityStrip
          :jobs="localActivityJobs"
          :sequences="sequenceVMs"
          :shared="sharedActivityRows"
          :queue-status="routing.queueStatus.value"
          @cancel="cancelPrint"
          @dismiss="stream.remove"
          @open="openJob"
          @sequence-action="onSequenceAction"
          @show-history="onShowSequenceHistory"
          @shared-open="openLiveWork"
        />

        <div class="flex items-center gap-2">
          <div class="flex-1" />
          <div ref="templatesHost" class="relative">
            <button
              type="button"
              class="flex items-center gap-1.5 rounded-control border border-ce px-3 py-1.5 text-xs text-ink-2 hover:bg-white/5"
              data-test="templates-toggle"
              @click="showTemplates = !showTemplates"
            >
              <Icon name="star" :size="14" />
              Templates
            </button>
            <div
              v-if="showTemplates"
              class="absolute right-0 z-30 mt-2 w-80 rounded-card border border-edge bg-bench p-3 shadow-[var(--shadow-raised)]"
              data-test="templates-popover"
            >
              <GenerationTemplatesPanel
                v-model="form.state.value"
                :models="models"
              />
            </div>
          </div>
        </div>

        <div
          v-if="sequenceMode && (!supportsChain || showSequenceEmpty)"
          class="rounded-card-lg border border-edge bg-bench p-6 shadow-[inset_0_1px_0_var(--card-hi)]"
          data-test="chain-unsupported"
          :data-empty="showSequenceEmpty ? 'true' : 'false'"
        >
          <div class="flex items-start gap-3">
            <span
              class="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-control bg-bath text-ink-3"
            >
              <Icon name="chain" :size="17" />
            </span>
            <div class="min-w-0">
              <p class="font-display text-[15px] font-semibold text-rebate">
                sequences need a video model
              </p>
              <p class="mt-1 font-mono text-[11px] leading-relaxed text-ink-3">
                <template v-if="showSequenceEmpty">
                  No chain-capable video model is installed on the selected
                  machine. Pull an LTX Video or distilled LTX-2 checkpoint to
                  continue.
                </template>
                <template v-else>
                  <span class="text-ink-2">{{ currentModelLabel }}</span>
                  renders single prints only. pick an ltx-video or ltx-2 model
                  to chain clips into a sequence.
                </template>
              </p>
              <div class="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  class="rounded-control border border-ce px-3 py-1.5 font-mono text-[11px] text-ink-2 transition hover:border-safelight hover:text-rebate"
                  data-test="chain-back-to-single"
                  @click="setOutput('single')"
                >
                  back to one shot
                </button>
                <router-link
                  :to="sequenceBrowsePath"
                  data-test="browse-sequence-models"
                  class="rounded-control border border-ce px-3 py-1.5 font-mono text-[11px] text-ink-2 transition hover:border-safelight hover:text-rebate"
                >
                  browse video models →
                </router-link>
              </div>
            </div>
          </div>
        </div>

        <template v-else-if="sequenceMode">
          <template v-if="isPhone">
            <CreateModelPicker
              :models="sequenceModels"
              :model="form.state.value.model"
              :browse-to="sequenceBrowsePath"
              empty-label="No sequence models installed"
              @select="selectModel"
            />
            <!-- Phones lose the right-hand aside, so the Output card (and the
                 shared settings that drive the sequence) render here. -->
            <ControlsAside
              v-model="form.state.value"
              :family="currentFamily"
              :model="currentModel"
              :source-dimensions="activeSourceDimensions"
              :adv-count="advCount"
              :mobile="true"
              :last-seed="lastSeedUsed"
              :output="sequenceMode ? 'sequence' : 'single'"
              :clip-count="sequenceMode ? draft.clips.length : 0"
              data-test="phone-sequence-controls"
              @update:output="setOutput"
              @open-advanced="openAdvanced"
              @reset-settings="onResetSettings"
            />
          </template>
          <SequenceComposer
            :model="form.state.value.model"
            :family="currentFamily"
            :source-image="
              currentModel?.source_image ??
              form.state.value.sourceImageCapability
            "
            :shared="sharedParams"
            :model-default-frames="currentModel?.default_frames ?? null"
            :target="sequenceTarget"
            :chain-level-dirty="chainLevelDirty"
            :stage-media-by-clip-id="sequenceFilmstripMediaByClipId"
            :playing-clip-id="playingSequenceClipId"
            @submit="onSubmitSequence"
            @duplicate-as-new="onDuplicateAsNew"
            @discard-edit="onDiscardEdit"
            @expand-clip="onExpandClip"
            @import-shared="onImportShared"
            @play-clip="playSequenceClip"
          />

          <p
            v-if="sequenceReuseNotice"
            data-test="sequence-reuse-note"
            class="font-mono text-[0.7rem] leading-relaxed text-ink-3"
          >
            {{ sequenceReuseNotice }}
          </p>

          <div
            v-if="composerError"
            class="rounded-control bg-stop/10 px-3 py-2 text-sm leading-relaxed text-stop"
            data-test="sequence-submit-error"
            role="alert"
          >
            <div class="flex items-start gap-2">
              <p class="min-w-0 flex-1">{{ composerError }}</p>
              <button
                type="button"
                class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control border border-stop/40 hover:bg-stop/10"
                aria-label="Copy error message"
                title="Copy error message"
                @click="copyErrorMessage(composerError)"
              >
                <Icon name="copy" :size="16" />
              </button>
            </div>
          </div>

          <div
            v-if="sequencePlaybackSrc"
            class="relative flex min-h-[280px] flex-col overflow-hidden rounded-card border border-edge bg-print"
            data-test="sequence-stage-player"
          >
            <video
              :key="sequencePlaybackSrc"
              :src="sequencePlaybackSrc"
              class="max-h-[60vh] min-h-0 w-full flex-1 object-contain"
              autoplay
              controls
              loop
              playsinline
            />
            <div
              class="absolute left-3 top-3 flex items-center gap-2 rounded-control border border-edge bg-bench/90 px-2 py-1.5 shadow-raised backdrop-blur"
            >
              <span
                class="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-3"
              >
                Clip {{ (playingSequenceStage ?? 0) + 1 }}
              </span>
              <button
                type="button"
                class="rounded-control border border-ce px-2 py-1 text-xs text-ink-2 hover:text-rebate"
                data-test="sequence-return-live"
                @click="returnToLiveSequence"
              >
                Return to live render
              </button>
            </div>
          </div>

          <!-- The canvas region stays in sequence mode: the watched job's
               stage progress and latest clip preview develop here. -->
          <ResultCanvas
            v-else-if="watchedSequenceActive"
            mode="generating"
            :progress="sequenceCanvasPercent"
            :stage="sequenceCanvasStage"
            :preview-src="sequencePreviewSrc"
            :progress-fraction="sequenceCanvasPercent / 100"
            :develop-phase="watchedProgress ? 'developing' : 'latent'"
            :print-width="form.state.value.width"
            :print-height="form.state.value.height"
            data-test="sequence-canvas"
          />

          <!-- Settling must not blank the canvas: the strip no longer keeps a
               settled row, so the result lands here with its actions. -->
          <div
            v-else-if="settledSequence"
            class="flex flex-col gap-2 rounded-card border border-edge bg-bench p-3"
            data-test="sequence-result"
          >
            <video
              v-if="settledSequencePrint"
              :src="imageUrl(settledSequencePrint.filename)"
              class="max-h-[60vh] w-full rounded-media bg-print object-contain"
              controls
              loop
              playsinline
            />
            <ErrorNotice
              v-if="settledSequenceError"
              data-test="sequence-result-error"
              :message="settledSequenceError"
              :copy-message="settledSequenceErrorCopy"
            />
            <p class="data-mono text-caption text-ink-3">
              {{ settledSequenceCaption }}
            </p>
            <div class="flex flex-wrap items-center gap-2">
              <button
                v-if="settledSequenceError"
                type="button"
                class="rounded-control bg-stop px-3 py-1.5 text-sm font-semibold text-on-accent"
                data-test="sequence-result-resume"
                @click="onResumeSettledSequence"
              >
                Resume
              </button>
              <button
                type="button"
                class="rounded-control border border-ce px-3 py-1.5 text-sm text-ink-2 hover:text-rebate"
                data-test="sequence-result-edit"
                @click="onEditSettledSequence"
              >
                Edit sequence
              </button>
              <button
                type="button"
                class="rounded-control border border-ce px-3 py-1.5 text-sm text-ink-2 hover:text-rebate"
                data-test="sequence-result-library"
                @click="onShowSettledSequenceInLibrary"
              >
                Show in library
              </button>
            </div>
          </div>
        </template>

        <template v-else>
          <ComposerCard
            ref="composerCardRef"
            v-model:prompt="form.state.value.prompt"
            v-model:style-preset="form.state.value.stylePreset"
            :aspect-label="aspectLabel"
            :width="form.state.value.width"
            :height="form.state.value.height"
            :steps="form.state.value.steps"
            :batch-size="form.state.value.batchSize"
            :busy="ordinarySubmitBlocked || submitInFlight"
            :disabled-reason="h3GenerationInputBlocker"
            :expanded="expanded"
            :prompt-optional="canSkipPrompt"
            :required-placeholder="requiredPromptPlaceholder"
            :history="promptHistory"
            @submit="onSubmit"
            @expand="onExpand"
            @remix="onRemix"
            @undo-expand="undoExpand"
          >
            <template v-if="isPhone" #mobile-controls>
              <div
                class="mt-3 flex flex-col gap-3"
                data-test="phone-create-controls"
              >
                <CreateModelPicker
                  :models="composerModels"
                  :model="form.state.value.model"
                  browse-to="/models"
                  empty-label="No models installed"
                  @select="selectModel"
                />
                <ControlsAside
                  v-model="form.state.value"
                  :family="currentFamily"
                  :model="currentModel"
                  :source-dimensions="activeSourceDimensions"
                  :adv-count="advCount"
                  :mobile="true"
                  :last-seed="lastSeedUsed"
                  :output="sequenceMode ? 'sequence' : 'single'"
                  :clip-count="sequenceMode ? draft.clips.length : 0"
                  @update:output="setOutput"
                  @open-advanced="openAdvanced"
                  @reset-settings="onResetSettings"
                />
                <SourceMediaPanel
                  v-if="!sequenceMode"
                  v-model="form.state.value"
                  :family="currentFamily"
                  :models="models"
                  @open-picker="showPicker = true"
                  @clear-source="onClearSource"
                  @open-end-frame-picker="showEndFramePicker = true"
                  @clear-end-frame="onClearEndFrame"
                  @open-mask="showMask = true"
                  @open-h3-first-frame-picker="
                    h3BoundaryPickerTarget = 'firstFrame'
                  "
                  @open-h3-last-frame-picker="
                    h3BoundaryPickerTarget = 'lastFrame'
                  "
                />
              </div>
            </template>
          </ComposerCard>
          <EstimateBadge :request="estimateRequest" :target="estimateTarget" />

          <div
            v-if="quickConflictReasons.length"
            class="rounded-control border border-stop/45 bg-stop/10 px-3 py-2.5 text-sm leading-relaxed text-stop"
            role="alert"
            data-test="web-quick-expansion-stale"
          >
            <div class="flex items-start gap-2">
              <p class="min-w-0 flex-1">{{ quickConflictMessage }}</p>
              <button
                type="button"
                class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control border border-stop/40 hover:bg-stop/10"
                aria-label="Copy error message"
                title="Copy error message"
                @click="copyQuickConflict"
              >
                <Icon name="copy" :size="16" />
              </button>
            </div>
            <div class="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                data-test="web-reexpand-current-prompt"
                class="rounded-control bg-stop px-3 py-1.5 font-semibold text-on-accent"
                @click="reexpandCurrentPrompt"
              >
                Re-expand for {{ currentModelLabel }}
              </button>
              <button
                type="button"
                data-test="web-generate-expanded-anyway"
                class="rounded-control border border-stop/50 px-3 py-1.5 font-medium"
                @click="generateExpandedAnyway"
              >
                Generate expanded prompt anyway
              </button>
              <button
                type="button"
                class="rounded-control px-3 py-1.5 text-ink-2 hover:text-ink"
                @click="undoExpand"
              >
                Restore original
              </button>
            </div>
          </div>

          <div
            v-else-if="submitStatus"
            class="rounded-control bg-stop/10 px-3 py-2 text-sm leading-relaxed text-stop"
            data-test="composer-submit-error"
          >
            <div class="flex items-start gap-2">
              <p class="min-w-0 flex-1">{{ submitStatus }}</p>
              <button
                type="button"
                class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control border border-stop/40 hover:bg-stop/10"
                aria-label="Copy error message"
                title="Copy error message"
                @click="copyErrorMessage(submitStatus)"
              >
                <Icon name="copy" :size="16" />
              </button>
            </div>
          </div>

          <div
            v-if="chainDecision.kind === 'chain'"
            class="rounded-control bg-halide/10 px-3 py-1.5 text-xs text-halide"
          >
            Will render as
            <span class="font-semibold">{{ chainDecision.stageCount }}</span>
            chained clips of {{ chainDecision.clipFrames }} frames — expect this
            to take substantially longer than a single clip.
          </div>
          <div
            v-else-if="singleShotPreservationNote"
            class="rounded-control bg-halide/10 px-3 py-1.5 text-xs text-halide"
            data-test="single-shot-preservation-cue"
          >
            {{ singleShotPreservationNote }}
          </div>

          <div
            v-if="canvasMode === 'empty' && showColdStart"
            class="flex min-h-[300px] items-center justify-center rounded-card-lg border border-edge bg-bench p-6 shadow-[inset_0_1px_0_var(--card-hi)]"
          >
            <ColdStartGuide />
          </div>
          <ResultCanvas
            v-else
            :mode="canvasMode"
            :prompt-optional="canSkipPrompt"
            :progress="genProgress"
            :stage="genStage"
            :preview-src="runningJob?.previewUrl ?? undefined"
            :progress-fraction="genProgress / 100"
            :develop-seed="runningJob?.seedVisual"
            :develop-phase="developPhase"
            :print-width="runningJob?.request.width"
            :print-height="runningJob?.request.height"
            :result-src="resultSrc"
            :result-audio-src="resultAudioSrc"
            :result-caption="resultCaption"
            :error="latestErrorMessage"
            :error-copy="latestErrorCopy"
            :variations="variations"
            :variation-batch-id="preparedBatch?.batchId"
            :queueing-variations="queueingVariations"
            @update:variations="variations = $event"
            @use-variation="useVariation"
            @discard="discardVariations"
            @queue="queueVariations"
            @click="canvasMode === 'result' ? openLatestResult() : undefined"
          />
        </template>

        <section>
          <div class="mb-2 flex items-center justify-between">
            <span class="font-display text-[15px] font-semibold text-rebate"
              >Recent</span
            >
            <span class="font-mono text-[11px] text-ink-3"
              >{{ galleryEntries.length }} prints</span
            >
          </div>
          <RecentGrid
            :entries="galleryEntries"
            :limit="isPhone ? 18 : 50"
            @open="openItem"
          />
        </section>
      </main>

      <!-- Right: controls region — model + basics + inline Advanced (spec §06
           v0.12 surface split). On phones the Advanced column collapses into
           the Advanced sheet, opened from the button inside ControlsAside. -->
      <div v-if="!isPhone" class="flex min-w-0 flex-col gap-4">
        <CreateModelPicker
          :models="composerModels"
          :model="form.state.value.model"
          :browse-to="sequenceMode ? sequenceBrowsePath : '/models'"
          :empty-label="
            sequenceMode
              ? 'No sequence models installed'
              : 'No models installed'
          "
          @select="selectModel"
        />
        <ControlsAside
          v-model="form.state.value"
          :family="currentFamily"
          :model="currentModel"
          :source-dimensions="activeSourceDimensions"
          :adv-count="advCount"
          :mobile="false"
          :last-seed="lastSeedUsed"
          :output="sequenceMode ? 'sequence' : 'single'"
          :clip-count="sequenceMode ? draft.clips.length : 0"
          @update:output="setOutput"
          @open-advanced="openAdvanced"
          @reset-settings="onResetSettings"
        />
        <!-- Source media in the primary form: the model dictates whether
             (and how) it renders, exactly like resolutions. -->
        <SourceMediaPanel
          v-if="!sequenceMode"
          v-model="form.state.value"
          :family="currentFamily"
          :models="models"
          @open-picker="showPicker = true"
          @clear-source="onClearSource"
          @open-end-frame-picker="showEndFramePicker = true"
          @clear-end-frame="onClearEndFrame"
          @open-mask="showMask = true"
          @open-h3-first-frame-picker="h3BoundaryPickerTarget = 'firstFrame'"
          @open-h3-last-frame-picker="h3BoundaryPickerTarget = 'lastFrame'"
        />
        <!-- Tablet+ : inline, always-visible Advanced column. -->
        <AdvancedDrawer
          :mobile="false"
          v-model="form.state.value"
          :family="currentFamily"
          :adv-count="advCount"
          :placement-gpus="gpuListForPlacement"
          :models="models"
          :audio-output-supported="currentModel?.supports_audio !== false"
          :can-extend="canExtend"
          :extend-default-overlap-frames="extendDefaultOverlapFrames"
          :output="sequenceMode ? 'sequence' : 'single'"
          :sequence-supports-audio="
            currentFamily === 'ltx2' && currentModel?.supports_audio !== false
          "
          @open-picker="showPicker = true"
          @open-h3-first-frame-picker="h3BoundaryPickerTarget = 'firstFrame'"
          @open-h3-last-frame-picker="h3BoundaryPickerTarget = 'lastFrame'"
          @clear-source="onClearSource"
          @open-end-frame-picker="showEndFramePicker = true"
          @clear-end-frame="onClearEndFrame"
          @open-mask="showMask = true"
          @append-prompt="form.appendPromptPhrase"
        />
      </div>
    </div>

    <!-- Phone: the same Advanced content in a viewport-fixed sheet host, so it
         overlays the scrolling document rather than anchoring in the tall
         controls column (SheetPanel is absolute-in-frame by design). -->
    <div v-if="isPhone && showAdvanced" class="fixed inset-0 z-40">
      <AdvancedDrawer
        :open="true"
        :mobile="true"
        v-model="form.state.value"
        :family="currentFamily"
        :adv-count="advCount"
        :placement-gpus="gpuListForPlacement"
        :models="models"
        :audio-output-supported="currentModel?.supports_audio !== false"
        :can-extend="canExtend"
        :extend-default-overlap-frames="extendDefaultOverlapFrames"
        :output="sequenceMode ? 'sequence' : 'single'"
        :sequence-supports-audio="
          currentFamily === 'ltx2' && currentModel?.supports_audio !== false
        "
        @close="showAdvanced = false"
        @open-picker="showPicker = true"
        @open-h3-first-frame-picker="h3BoundaryPickerTarget = 'firstFrame'"
        @open-h3-last-frame-picker="h3BoundaryPickerTarget = 'lastFrame'"
        @clear-source="onClearSource"
        @open-end-frame-picker="showEndFramePicker = true"
        @clear-end-frame="onClearEndFrame"
        @open-mask="showMask = true"
        @append-prompt="form.appendPromptPhrase"
      />
    </div>

    <ExpandModal
      :open="showExpand"
      :prompt="
        expandClipId !== null ? expandStagePrompt : form.state.value.prompt
      "
      :expand="form.state.value.expand"
      :current-model="currentModel"
      :style-directive="expandStyleDirective"
      :task="expandTask"
      :target="expandRoute?.target"
      @update:expand="(v: ExpandFormState) => (form.state.value.expand = v)"
      @apply-prompt="applyExpandedPrompt"
      @close="
        showExpand = false;
        expandClipId = null;
        expandRoute = null;
      "
    />
    <RemixModal
      :open="showRemix"
      :prompt="form.state.value.prompt"
      :original-prompt="form.state.value.originalPrompt"
      :family="currentFamily"
      :task="remixTask"
      :style="styleHint(form.state.value.stylePreset ?? '')"
      :target="normalizeSubmitRoute(remixRoute)?.target"
      @close="showRemix = false"
      @apply="applyRemix"
      @prepare="prepareRemixBatch"
    />
    <ImagePickerModal
      :open="showPicker"
      :title="
        sequenceMode
          ? 'Opening sequence image'
          : currentFamily === 'qwen-image-edit' ||
              isFlux2DevModel(form.state.value.model)
            ? 'Edit images'
            : 'Source image'
      "
      :multiple="
        !sequenceMode &&
        (currentFamily === 'qwen-image-edit' ||
          isFlux2DevModel(form.state.value.model))
      "
      :gallery-only="
        !sequenceMode &&
        currentFamily !== 'qwen-image-edit' &&
        !isFlux2DevModel(form.state.value.model)
      "
      @pick="onPickSource"
      @close="showPicker = false"
    />
    <ImagePickerModal
      :open="h3BoundaryPickerTarget !== null"
      :title="
        h3BoundaryPickerTarget === 'lastFrame' ? 'Last frame' : 'First frame'
      "
      :multiple="false"
      gallery-only
      @pick="onPickH3Boundary"
      @close="h3BoundaryPickerTarget = null"
    />
    <ImagePickerModal
      :open="showEndFramePicker"
      title="End frame"
      :multiple="false"
      gallery-only
      @pick="onPickEndFrame"
      @close="showEndFramePicker = false"
    />
    <MaskEditorModal
      :open="showMask"
      :source-image="form.state.value.imageAttachments[0] ?? null"
      :initial-mask="form.state.value.maskImage"
      @apply="onApplyMask"
      @close="showMask = false"
    />

    <Lightbox
      :item="selected"
      :models="models"
      :has-prev="selectedIndex > 0"
      :has-next="
        selectedIndex >= 0 && selectedIndex < galleryEntries.length - 1
      "
      :index="selectedIndex"
      :total="galleryEntries.length"
      :muted="muted"
      @close="closeDrawer"
      @prev="stepDrawer(-1)"
      @next="stepDrawer(1)"
      @reuse="onLightboxReuse"
      @use-source="onLightboxUseSource"
      @upscale="onLightboxUpscale"
      @delete="handleDelete"
    />
  </div>
</template>

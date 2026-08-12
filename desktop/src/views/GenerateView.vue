<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { isSupportedDroppedImage, reduceNativeImageDrag } from "../lib/nativeImageDrop";
import { useRoute, useRouter } from "vue-router";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "../lib/generateModels";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import ProgressRing from "@ui/components/ProgressRing.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import type { ClipRailMedia } from "@ui/components/types";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import TemplatesPanel from "../components/generate/TemplatesPanel.vue";
import ExpansionPullStatus from "../components/generate/ExpansionPullStatus.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import GenerateErrorNotice from "../components/generate/GenerateErrorNotice.vue";
import MissingModelDialog from "../components/generate/MissingModelDialog.vue";
import CreateHeader from "../components/create/CreateHeader.vue";
import ActivityStrip from "../components/create/ActivityStrip.vue";
import ComposerCard from "../components/create/ComposerCard.vue";
import InspectorPanel from "../components/create/InspectorPanel.vue";
import SequenceComposer from "../components/create/SequenceComposer.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { filterRestrictedModels } from "@studio/lib/modelAccess";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { useLiveActivityStore } from "../stores/liveActivity";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import {
  canvasMatchesSourceResolution,
  resolveSourceResolution,
  type SourceResolutionResult,
} from "@studio/lib/sourceResolution";
import { useChainJobsStore } from "../stores/chainJobs";
import { buildChainRequest } from "@studio/lib/sequenceForm";
import { chainScriptToClips } from "@studio/lib/sequenceForm";
import {
  defaultClipFrames,
  friendlySequenceError,
  modelSupportsSequence,
  modelsForOutput,
  sequenceMotionTailFrames,
} from "@studio/lib/sequence";
import { OPTIONAL_PROMPT_GUIDANCE, promptRequired } from "@studio/lib/promptRequirement";
import { minimaxH3AuthoringError } from "@studio/lib/minimaxH3Authoring";
import { firstLastFrameRestoreNotice } from "@studio/lib/sourceImageCapability";
import {
  clampClipsToMotionTail,
  isPrintOfChainJob,
  planSequenceReuse,
  sequenceReuseClampNote,
  sequenceReuseNote,
} from "@studio/lib/sequenceReuse";
import type { AmendRequest, ChainLimits } from "@studio/lib/api/chainTypes";
import {
  countLeadingCompletedStages,
  normalizeServerChainScript,
} from "@studio/lib/chainScriptWire";
import { routeForModel } from "../lib/sequenceRoute";
import { sequenceParams } from "../lib/sequenceParams";
import { fetchChainLimits } from "../lib/api/chains";
import { normalizeTargetHost, readyHostSignature } from "../lib/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore, type FeasibleRouteResult } from "../stores/hosts";
import { useConnectionStore } from "../stores/connection";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  needsHostRoute,
  suggestOutputFilename,
  type BatchRequestOptions,
  type Job,
} from "../stores/generation";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { copyBase64ImageToClipboard } from "../lib/clipboard";
import { copyLocalOutputPath } from "../lib/localOutputPath";
import { useUiStore } from "../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import {
  buildAutoChainRequest,
  buildGenerationEstimateRequest,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
  type ChainRoutingDecision,
} from "../lib/chainRouting";
import {
  applyPrefillToForm,
  buildRequest,
  cloneGenerateForm,
  normalizeLegacyNegativeSnapshot,
} from "../lib/generateForm";
import { composeStyle, mergeStyleNegative, styleHint } from "../lib/stylePresets";
import { videoFramesError } from "@studio/lib/videoDuration";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  profileGuidanceValidationError,
  profileStepsValidationError,
  resolutionValidationError,
  sourceConditioningValidationError,
  wanRecipeValidationError,
} from "../lib/generateValidation";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";
import { expansionTaskForRequest } from "@studio/lib/expandTask";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { upscaleImage } from "../lib/api/upscale";
import { expandPrompt } from "../lib/api/expand";
import { remixPrompt } from "../lib/api/remix";
import {
  conditioningFingerprint,
  defaultRemixDimensions,
  promptSource,
  validateRemixVariants,
} from "@studio/lib/promptTransform";
import type { HostRoute } from "../stores/hosts";
import {
  formatTemplateMediaReferences,
  hydrateGenerationTemplate,
  type GenerationTemplate,
} from "../lib/generationTemplates";
import { fetchHistoryAll, type HistoryHostTarget } from "../lib/api/history";
import { randomSeed } from "../stores/generation";
import type { CompleteEvent, GenerateRequest, OutputMetadata } from "../lib/api/types";
import {
  DEFAULT_VIDEO_EXPORT_CAPABILITIES,
  videoExportFilename,
  type VideoExportCapabilities,
  type VideoExportOptions,
} from "@studio/lib/videoExport";
import { saveGalleryMedia, showSavedMediaToast } from "../lib/mediaSave";
import {
  metadataReferencesSource,
  restoreEditImages,
  restoreSourceImage,
  sha256HexOfBase64,
  type SourceRestoreDeps,
} from "../lib/sourceRestore";
import { isMissingModelError } from "../lib/generateErrors";
import { copyableError, describeTransportError } from "../lib/api/errors";
import { startCatalogDownload } from "../lib/api/catalog";
import { computeEtaSeconds, useDownloadsStore, type DownloadsState } from "../stores/downloads";
import { usePullResumeStore } from "../stores/pullResume";
import { modelDisplayNameForId } from "../lib/models";
import { galleryMediaPath, localMediaPath, mediaPath } from "../lib/gallery/media";
import { sequenceStageMediaUrl } from "../lib/sequenceMedia";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import { ApiError, apiFetch, apiFetchTo } from "../lib/api/client";
import { blobToBase64 } from "../lib/image";
import { ipc } from "../lib/ipc";
import { applyDesktopImageDrop } from "../lib/desktopImageDrop";
import { isAudioCompletion } from "@studio/lib/ltx2Pipeline";
import { useGalleryStore } from "../stores/gallery";
import { parseMissingExpandModel } from "../lib/expandErrors";
import {
  expansionPullJobMatchesModel,
  resolveExpansionPullStatus,
  type ExpansionPullPhase,
  type ExpansionPullView,
} from "../lib/expansionPull";
import {
  PreparationRequestGuard,
  createPreparedExpansionBatch,
  preparedExpansionStaleReasons,
  quickExpansionStaleReasons,
  quickExpansionRouteIsCurrent,
  validateExpandedPrompts,
  type PreparedExpansionBatch as PreparedExpansionBatchState,
  type PreparedExpansionInputs,
  type QuickExpansionSnapshot,
} from "../lib/preparedExpansion";

const router = useRouter();
const conn = useConnectionStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const models = useModelStore();
const composer = useComposerStore();
const toasts = useToastStore();
const ui = useUiStore();
const contextMenu = useContextMenuStore();
const videoExportJob = ref<Job | null>(null);
const videoExportBusy = ref(false);
const videoExportError = ref("");
const videoExportCapabilities = ref<VideoExportCapabilities>(DEFAULT_VIDEO_EXPORT_CAPABILITIES);
// Multi-host gallery — source-image restore looks up prints across hosts.
const hostGallery = useGalleryStore();
const downloads = useDownloadsStore();
const pullResume = usePullResumeStore();
const liveActivity = useLiveActivityStore();

function placementFailureMessage(result: Exclude<FeasibleRouteResult, { kind: "route" }>): string {
  if (result.kind === "profile_mismatch") {
    const machines = result.perHost.map((host) => host.label).join(", ");
    return `Settings differ by machine${machines ? ` (${machines})` : ""}. Choose a specific machine before generating. Nothing was queued.`;
  }
  if (result.kind === "infeasible") {
    const details = result.perHost
      .map((failure) => {
        const missing = failure.missingComponents
          .filter((component) => !component.present)
          .map((component) => component.name);
        return `${failure.label}: ${failure.reason}${missing.length ? ` (missing: ${missing.join(", ")})` : ""}`;
      })
      .join("; ");
    return `No selected machine can run this print${details ? ` — ${details}.` : "."} Nothing was queued.`;
  }
  if (result.kind === "mixed") {
    const details = result.perHost
      .map((failure) => {
        if (failure.kind === "infeasible") {
          const missing = failure.missingComponents
            .filter((component) => !component.present)
            .map((component) => component.name);
          return `${failure.label} cannot run it: ${failure.reason}${missing.length ? ` (missing: ${missing.join(", ")})` : ""}`;
        }
        return failure.kind === "unreachable"
          ? `${failure.label} did not answer: ${failure.error}`
          : `${failure.label} could not plan it right now: ${failure.error}`;
      })
      .join("; ");
    return `The selected machines failed for different reasons${details ? ` — ${details}.` : "."} Nothing was queued.`;
  }
  const details = result.perHost.map((failure) => `${failure.label}: ${failure.error}`).join("; ");
  if (result.kind === "unreachable") {
    return `Mold could not check placement${details ? ` — ${details}.` : "."} Nothing was queued.`;
  }
  return `Placement changed or could not be computed right now${details ? ` — ${details}.` : "."} Try again. Nothing was queued.`;
}

/** A generate that 404'd (model not on the routed host) awaiting the user's
 *  pull-and-resume decision. */
const missingModel = ref<{
  model: string;
  route: HostRoute | null;
  request: GenerateRequest;
  batch: number;
  chainRouting: ChainRoutingDecision | null;
  requestOptions: BatchRequestOptions;
} | null>(null);

const missingModelHostLabel = computed(
  () => missingModel.value?.route?.label ?? hosts.primaryHost?.label ?? "this host",
);
/** Known weights size when the model is installed on another host. */
const missingModelSizeGb = computed(() => {
  const name = missingModel.value?.model;
  if (!name) return null;
  return installedModels.value.find((m) => m.name === name)?.size_gb ?? null;
});

/** Start the pull on the routed host and arm the auto-resume. */
async function pullMissingModel() {
  const info = missingModel.value;
  if (!info) return;
  missingModel.value = null;
  const route = info.route;
  const host = route ? (hosts.all.find((h) => h.id === route.hostId) ?? null) : null;
  const label = route?.label ?? hosts.primaryHost?.label ?? "this host";
  // The primary's downloads live in the top-level bucket, not hostStates.
  const bucketId = route && route.hostId !== "local" ? route.hostId : null;
  const armed = {
    model: info.model,
    hostId: bucketId,
    hostLabel: label,
    request: info.request,
    batch: info.batch,
    route,
    chainRouting: info.chainRouting,
    requestOptions: info.requestOptions,
  };
  // The resume watcher is fed by this stream — a dead stream means the
  // promise "generation starts when it's ready" could never be kept, so
  // fail loudly instead of arming a resume that can't fire.
  try {
    await downloads.subscribe(host ?? undefined);
  } catch {
    toasts.push(
      `Couldn't watch downloads on ${label} — pull ${info.model} from the Catalog instead.`,
      "error",
    );
    return;
  }
  try {
    // Watch the EXACT job the server enqueues; a stale completed pull of the
    // same model in history can then never trigger a premature resume.
    const jobId = await startCatalogDownload(info.model, route?.target, route?.kind === "remote");
    pullResume.arm({ ...armed, jobId });
    toasts.push(`Pulling ${info.model} on ${label} — generation starts when it's ready`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      // Already downloading (another client or an earlier click) — watch by
      // model; the running job is live, not terminal, so it can't be stale.
      pullResume.arm({ ...armed, jobId: null });
      toasts.push(`${info.model} is already downloading on ${label} — will generate when ready`);
    } else if (/unknown model/i.test(String(err))) {
      toasts.push(
        `${label} can't pull ${info.model} by name — pull it from the Catalog there, then generate again.`,
        "error",
      );
    } else if (routeRequired.value) {
      toasts.push(String(err), "error");
    }
  }
}

// Store-backed so the model, prompt, and params survive navigating away and
// back — this view unmounts on every route change.
const formStore = useGenerateFormStore();
const form = formStore.form;

const composerRef = ref<InstanceType<typeof ComposerCard> | null>(null);
const workbenchRef = ref<HTMLDivElement | null>(null);
const templatesOpen = ref(false);
const templatesEl = ref<HTMLDivElement | null>(null);
const templatesToggleEl = ref<HTMLButtonElement | null>(null);
/** Recent prompts for the composer's ↑/↓ history cycling. */
const promptHistory = ref<string[]>([]);
const nativeImageDragOver = ref(false);
const DEFAULT_BENCH_HEIGHT = 520;
const MIN_BENCH_HEIGHT = 280;
/**
 * Sequence mode's floor covers the composer's fixed chrome (activity header,
 * edit banner, clip head, prompt, tools, footer) plus the filmstrip's minimum
 * height, so resizing compresses the rail instead of growing a scrollbar.
 */
const MIN_SEQUENCE_BENCH_HEIGHT = 390;
const MIN_CANVAS_HEIGHT = 144;
const BENCH_HEIGHT_KEY = "mold.desktop.create-bench-height.v1";
function readStoredBenchHeight(): number | null {
  try {
    const value = globalThis.localStorage?.getItem(BENCH_HEIGHT_KEY);
    return value == null ? null : Number(value);
  } catch {
    return null;
  }
}
const storedBenchHeight = readStoredBenchHeight();
const benchHeight = ref(
  Number.isFinite(storedBenchHeight)
    ? Math.max(MIN_BENCH_HEIGHT, storedBenchHeight!)
    : DEFAULT_BENCH_HEIGHT,
);
let benchResizeStartY = 0;
let benchResizeStartHeight = 0;

function minBenchHeight(): number {
  return isSequence.value ? MIN_SEQUENCE_BENCH_HEIGHT : MIN_BENCH_HEIGHT;
}

function clampBenchHeight(height: number): number {
  const available = workbenchRef.value?.clientHeight ?? window.innerHeight;
  return Math.round(
    Math.min(
      Math.max(minBenchHeight(), available - MIN_CANVAS_HEIGHT),
      Math.max(minBenchHeight(), height),
    ),
  );
}

function setBenchHeight(height: number) {
  benchHeight.value = clampBenchHeight(height);
  try {
    globalThis.localStorage?.setItem(BENCH_HEIGHT_KEY, String(benchHeight.value));
  } catch {
    // A private or restricted WebView still gets an in-memory resize.
  }
}

function onBenchResizeMove(event: PointerEvent) {
  setBenchHeight(benchResizeStartHeight + benchResizeStartY - event.clientY);
}

function stopBenchResize() {
  document.removeEventListener("pointermove", onBenchResizeMove);
  document.removeEventListener("pointerup", stopBenchResize);
}

function startBenchResize(event: PointerEvent) {
  event.preventDefault();
  benchResizeStartY = event.clientY;
  benchResizeStartHeight = benchHeight.value;
  document.addEventListener("pointermove", onBenchResizeMove);
  document.addEventListener("pointerup", stopBenchResize);
}

function onBenchResizeKeydown(event: KeyboardEvent) {
  if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
  event.preventDefault();
  setBenchHeight(benchHeight.value + (event.key === "ArrowUp" ? 24 : -24));
}

function clampBenchToViewport() {
  benchHeight.value = clampBenchHeight(benchHeight.value);
}
const preparedBatch = ref<PreparedExpansionBatchState | null>(null);
const remixSource = ref<"original" | "current">("original");
const expansionRunning = ref(false);
const expansionError = ref<string | null>(null);
const expansionMissingModel = ref<{ model: string; route: HostRoute } | null>(null);
interface ExpansionPullAttempt {
  id: number;
  model: string;
  route: HostRoute;
  phase: Exclude<ExpansionPullPhase, "missing">;
  jobId: string | null;
  observedJobId: string | null;
  baselineMatchingInFlightId: string | null;
  baselineJobIds: string[];
  allowExistingInFlight: boolean;
  requestError: string | null;
}
const expansionPullAttempt = ref<ExpansionPullAttempt | null>(null);
let expansionPullRequestId = 0;
const expansionAttemptHostLabel = ref<string | null>(null);
const quickExpansionOriginal = ref<string | null>(null);
const quickExpansionSnapshot = ref<QuickExpansionSnapshot | null>(null);
/**
 * The negative prompt before and after a bake-and-clear merged the preset's
 * curated fragments into it. Undo re-arms `before` alongside the prompt and
 * chip; `baked` lets it bow out when the user has since edited the field.
 */
const quickExpansionNegative = ref<{ before: string; baked: string } | null>(null);
const preparedSubmitting = ref(false);
const submissionPlanning = ref(false);
const preparationGuard = new PreparationRequestGuard();
const submissionGuard = new PreparationRequestGuard();
// Completion is detached from authoring once the store accepts a batch. Only
// the newest accepted submission may publish mutable recovery authority:
// older batches can finish later, but must not replace a newer pull/resume
// decision.
let latestAcceptedSubmissionId = 0;

let stopNativeImageDrop: (() => void) | null = null;
let nativeImageDropUnmounted = false;
let nativeImageDragCandidate = false;

async function importDroppedImage(path: string) {
  try {
    const image = await ipc.importSourceImage(path);
    const result = applyDesktopImageDrop(form, image, installedModels.value);
    if (result.metadataApplied && result.attached) {
      toasts.push("Loaded generation settings and attached the image as source.");
    } else if (result.metadataApplied) {
      toasts.push("Loaded generation settings; this model doesn't accept a source image.");
    } else if (result.attached) {
      toasts.push("Source image attached.");
    } else {
      toasts.push("The selected model doesn't accept a source image.", "error");
    }
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

/** Tauri intercepts Finder/file-manager drops before HTML DataTransfer sees
 * them. Bridge its native paths while this route is mounted; browser dev mode
 * keeps SourceImageWell's ordinary DOM drop handler. */
async function listenForNativeImageDrops() {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) return;
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  const unlisten = await getCurrentWebview().onDragDropEvent(({ payload }) => {
    const dragState = reduceNativeImageDrag(
      {
        candidate: nativeImageDragCandidate,
        visible: nativeImageDragOver.value,
      },
      payload,
    );
    nativeImageDragCandidate = dragState.candidate;
    nativeImageDragOver.value = dragState.visible;
    if (payload.type !== "drop") return;
    const path = payload.paths.find(isSupportedDroppedImage);
    if (path) void importDroppedImage(path);
    else toasts.push("Drop a PNG or JPEG image.", "error");
  });
  if (nativeImageDropUnmounted) unlisten();
  else stopNativeImageDrop = unlisten;
}

function onDocumentPointerDown(event: PointerEvent) {
  if (!templatesOpen.value || !templatesEl.value) return;
  if (!event.composedPath().includes(templatesEl.value)) templatesOpen.value = false;
}

function onDocumentKeydown(event: KeyboardEvent) {
  if (!templatesOpen.value || event.defaultPrevented || event.key !== "Escape") return;
  event.preventDefault();
  templatesOpen.value = false;
  void nextTick(() => templatesToggleEl.value?.focus());
}

const job = computed(() => generation.active);
const jobErrorMessage = computed(() =>
  job.value?.error
    ? describeTransportError(job.value.error, job.value.hostLabel)
    : "Something went wrong while developing this print.",
);
const jobErrorCopy = computed(() =>
  job.value?.error ? copyableError(job.value.error, jobErrorMessage.value) : jobErrorMessage.value,
);
const siblings = computed(() => generation.siblings);
// Resolved with the same five inputs the form and its validators use, so the
// view can never disagree with them about the selected checkpoint's advertised
// source-image contract (#772) or its first/last-frame layout (#779).
const caps = computed(() =>
  generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    form.guidanceCapabilities,
    form.sourceImageCapability,
    effectiveGenerationRecipe(selectedEntry.value, form.pipeline),
  ),
);
const formValidationError = computed(
  () =>
    resolutionValidationError(
      form.width,
      form.height,
      selectedEntry.value ?? null,
      form.pipeline,
    ) ??
    profileStepsValidationError(form.steps, selectedEntry.value, form.pipeline) ??
    profileGuidanceValidationError(form.guidance, selectedEntry.value, form.pipeline) ??
    (caps.value.supportsVideo
      ? videoFramesError(form.frames, selectedEntry.value ?? { family: form.family })
      : null) ??
    (caps.value.supportsVideo ? fpsValidationError(form.fps) : null) ??
    cameraControlValidationError(form) ??
    audioOutputValidationError(form) ??
    sourceConditioningValidationError(form) ??
    advancedVideoValidationError(form) ??
    wanRecipeValidationError(form),
);
/** Over-budget video frames on a non-chainable model would fail server-side —
 *  the drawer shows the reason under Frames; this blocks the submit. */
const chainReject = computed(() => {
  if (formValidationError.value) return true;
  if (!caps.value.supportsVideo) return false;
  const request = buildRequest(form);
  const decision = decideGenerateRequestRouting(request, form.family);
  return (
    decision.kind === "reject" ||
    (decision.kind === "chain" && unsupportedAutoChainFields(buildRequest(form)).length > 0)
  );
});
const installedModels = computed(() =>
  mergeInstalledModels(
    filterRestrictedModels(models.installed, hosts.capabilities.local),
    hostModels.unionInstalled,
  ),
);
const modelLabels = computed(
  () =>
    new Map(
      installedModels.value.map((model) => [
        model.name,
        modelDisplayNameForId(model.name, installedModels.value),
      ]),
    ),
);
const showStarterCards = computed(() =>
  shouldShowStarterCards({
    connectionReady: conn.ready,
    primaryLoading: models.loading,
    hostsInitialized: hosts.initialized,
    hostModelsLoading: hostModels.loading,
    allReadyHostsFetched: hostModels.allReadyHostsFetched,
    installed: installedModels.value,
  }),
);

/** The request the estimate badge previews — null until a model is chosen. */
const estimateRequest = computed(() => {
  if (!form.model) return null;
  return buildGenerationEstimateRequest(buildRequest(form), form.family);
});

/** True when submits (and the estimate preflight) must resolve a route:
 *  multiple hosts, or a dead primary while another host can serve. */
const routeRequired = computed(() =>
  needsHostRoute({
    multiHost: hosts.multiHost,
    primaryReady: hosts.primaryHost?.status === "ready",
    anyHostReady: hosts.all.some((h) => h.status === "ready"),
  }),
);

/** Preflight against the host the batch will actually route to. */
const estimateTarget = computed(() =>
  routeRequired.value
    ? (hosts.resolveRoute(appPrefs.settings?.generateTargetHost ?? null, form.model || null)
        ?.target ?? null)
    : null,
);

/** The sticky pick as the Host selector shows it — a ghost host id (removed or
 *  never reconnected) reads as Auto so filtering and expansion never disagree. */
const stickyTarget = computed<string | null>(() =>
  normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
);

const effectiveBatchSize = computed(() =>
  caps.value.forcesBatchSizeOne ||
  (caps.value.sourceImageMode === "references" && form.imageAttachments.length > 0)
    ? 1
    : Math.max(1, Math.floor(form.batchSize)),
);

/** Expansion always resolves a concrete host, even in the one-host case. */
const currentExpansionRoute = computed<HostRoute | null>(() =>
  hosts.resolveRoute(stickyTarget.value, form.model || null),
);
const expansionHostLabel = computed(() => currentExpansionRoute.value?.label ?? null);

const preparedStaleReasons = computed(() => {
  const batch = preparedBatch.value;
  if (!batch) return [];
  const request = buildRequest(form);
  const currentRemixSource =
    batch.kind === "remix"
      ? promptSource(form.prompt, form.originalPrompt, remixSource.value).prompt
      : form.prompt.trim();
  return preparedExpansionStaleReasons(batch, {
    sourcePrompt: currentRemixSource,
    model: form.model,
    family: form.family,
    task: expansionTaskForRequest(form.family, request),
    ...(batch.kind === "remix"
      ? {
          kind: "remix" as const,
          ...(form.originalPrompt ? { rootPrompt: form.originalPrompt } : {}),
          sourceKind: remixSource.value,
          dimensions: defaultRemixDimensions(
            expansionTaskForRequest(form.family, request),
            Boolean(form.stylePreset),
          ),
          conditioningFingerprint: conditioningFingerprint(request),
        }
      : {}),
    requestedCount: effectiveBatchSize.value,
    stylePreset: form.stylePreset || null,
    selectedHostPolicy: stickyTarget.value,
    readyHostIds: new Set(
      hosts.all.filter((host) => host.status === "ready").map((host) => host.id),
    ),
    hostLabels: new Map(hosts.all.map((host) => [host.id, host.label])),
    modelLabels: modelLabels.value,
    hostTargets: new Map(
      hosts.all.flatMap((host) =>
        host.baseUrl
          ? [[host.id, { baseUrl: host.baseUrl, apiKey: host.apiKey, kind: host.kind }] as const]
          : [],
      ),
    ),
  });
});

function currentHostSnapshot() {
  return {
    readyHostIds: new Set(
      hosts.all.filter((host) => host.status === "ready").map((host) => host.id),
    ),
    hostLabels: new Map(hosts.all.map((host) => [host.id, host.label])),
    hostTargets: new Map(
      hosts.all.flatMap((host) =>
        host.baseUrl
          ? [[host.id, { baseUrl: host.baseUrl, apiKey: host.apiKey, kind: host.kind }] as const]
          : [],
      ),
    ),
  };
}

const quickStaleReasons = computed(() => {
  const snapshot = quickExpansionSnapshot.value;
  if (!snapshot) return [];
  return quickExpansionStaleReasons(snapshot, {
    expandedPrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    task: expansionTaskForRequest(form.family, buildRequest(form)),
    selectedHostPolicy: stickyTarget.value,
    modelLabels: modelLabels.value,
    ...currentHostSnapshot(),
  });
});
const quickRouteIsCurrent = computed(() => {
  const snapshot = quickExpansionSnapshot.value;
  if (!snapshot) return false;
  return quickExpansionRouteIsCurrent(snapshot, {
    expandedPrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    task: expansionTaskForRequest(form.family, buildRequest(form)),
    selectedHostPolicy: stickyTarget.value,
    modelLabels: modelLabels.value,
    ...currentHostSnapshot(),
  });
});
const quickStaleMessage = computed(() =>
  quickStaleReasons.value.length
    ? `${quickStaleReasons.value.join(" ")} Choose how to continue.`
    : "",
);
const currentModelLabel = computed(() => modelDisplayNameForId(form.model, installedModels.value));

// ── Output = Sequence (mode is a setting, not a place) ───────────────────────
const activeRoute = useRoute();
const draft = useSequenceDraftStore();
const chains = useChainJobsStore();
const isSequence = computed(() => draft.output === "sequence");
const selectedEntry = computed(() =>
  hostModels.installedEntryForTarget(form.model, stickyTarget.value),
);

let previousStillSource = "";
let previousStillResolution: SourceResolutionResult | null = null;
let previousOpeningSource = "";
let previousOpeningResolution: SourceResolutionResult | null = null;

function applyDecodedSourceResolution(
  base64: string | null,
  previous: {
    base64: string;
    resolution: SourceResolutionResult | null;
  },
  setDimensions: (width: number | null, height: number | null) => void,
): { base64: string; resolution: SourceResolutionResult | null } {
  if (!base64) {
    setDimensions(null, null);
    return { base64: "", resolution: null };
  }
  const dimensions =
    base64 === previous.base64 && previous.resolution
      ? previous.resolution.source
      : imageDimensionsFromBase64(base64);
  if (!dimensions) {
    setDimensions(null, null);
    return { base64, resolution: null };
  }
  setDimensions(dimensions.width, dimensions.height);
  const resolution = resolveSourceResolution(
    dimensions,
    selectedEntry.value ?? form.family,
    form.pipeline,
  );
  const replaced = base64 !== previous.base64;
  const wasFollowing =
    previous.resolution !== null &&
    canvasMatchesSourceResolution({ width: form.width, height: form.height }, previous.resolution);
  if (caps.value.sourceImageMode !== "references" && (replaced || wasFollowing)) {
    form.width = resolution.output.width;
    form.height = resolution.output.height;
  }
  return { base64, resolution };
}

watch(
  [
    () =>
      caps.value.sourceImageMode !== "single"
        ? (form.imageAttachments[0] ?? null)
        : form.sourceImage,
    () => selectedEntry.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => selectedEntry.value?.generation_profile?.profile_hash ?? null,
    () => selectedEntry.value?.max_pixels ?? null,
    () => selectedEntry.value?.max_axis_pixels ?? null,
    () => selectedEntry.value?.dimension_alignment ?? null,
    () =>
      selectedEntry.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  ([base64]) => {
    if (isSequence.value) return;
    // This watcher also runs when Create remounts. Keep it limited to derived
    // dimensions; source attachment boundaries own their one-time fit default
    // so a route change cannot overwrite the user's selected policy.
    const next = applyDecodedSourceResolution(
      base64,
      {
        base64: previousStillSource,
        resolution: previousStillResolution,
      },
      (width, height) => {
        form.sourceImageWidth = width;
        form.sourceImageHeight = height;
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
    () => selectedEntry.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => selectedEntry.value?.generation_profile?.profile_hash ?? null,
    () => selectedEntry.value?.max_pixels ?? null,
    () => selectedEntry.value?.max_axis_pixels ?? null,
    () => selectedEntry.value?.dimension_alignment ?? null,
    () =>
      selectedEntry.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  ([base64]) => {
    if (!isSequence.value) return;
    const next = applyDecodedSourceResolution(
      base64,
      {
        base64: previousOpeningSource,
        resolution: previousOpeningResolution,
      },
      (width, height) => {
        if (!draft.openingImage) return;
        if (width === null) delete draft.openingImage.width;
        else draft.openingImage.width = width;
        if (height === null) delete draft.openingImage.height;
        else draft.openingImage.height = height;
      },
    );
    previousOpeningSource = next.base64;
    previousOpeningResolution = next.resolution;
  },
  { immediate: true },
);
/** Sequence inventory follows the route policy. A pinned host must never
 * inherit a compatible model from another machine's union. Edit sessions are
 * stricter still: their immutable durable route owns the available model. */
const sequenceTargetHostId = computed<string | null>(
  () => draft.editing?.hostId ?? stickyTarget.value,
);
const sequenceTargetModels = computed(() => {
  const target = sequenceTargetHostId.value;
  const fixed = target && target !== "capable";
  const fetched = fixed && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  if (fixed && !fetched) {
    if (target === "local" && !models.loading) return models.installed;
    return [];
  }
  return filterModelsForTarget(
    installedModels.value,
    target,
    fetched ? new Set(hostModels.installedOn(target).map((model) => model.name)) : null,
  );
});
const sequenceCapableModels = computed(() =>
  modelsForOutput(sequenceTargetModels.value, "sequence"),
);
const selectedSequenceEntry = computed(() =>
  findInstalledModel(sequenceCapableModels.value, form.model),
);
const sequenceTarget = computed(() => {
  const entry = selectedSequenceEntry.value;
  if (!entry) return null;
  return draft.editing
    ? (hosts.resolveRoute(draft.editing.hostId, entry.name)?.target ?? null)
    : (routeForModel(entry)?.target ?? null);
});
const sequenceInventorySettled = computed(() => {
  const target = sequenceTargetHostId.value;
  if (target && target !== "capable") {
    if (target === "local" && (hostModels.byHost[target]?.fetchedAt ?? 0) === 0) {
      return !models.loading;
    }
    return !hostModels.loading && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  }
  if (hosts.all.length <= 1) return !models.loading;
  return !models.loading && !hostModels.loading && hostModels.allReadyHostsFetched;
});
const chainLimits = ref<ChainLimits | null>(null);
const sequenceSubmitting = ref(false);
/** Snapshot of the shared params at edit-load time — drives chainLevelDirty. */
const editSharedBaseline = ref<string | null>(null);
/** What a Library reuse could NOT restore, said once and quietly beneath the
 *  rail. Cleared the moment the user submits or leaves Sequence — it describes
 *  one handoff, not a standing property of the draft. */
const sequenceReuseNotice = ref<string | null>(null);

const sequenceMotionTail = computed(() => sequenceMotionTailFrames(selectedSequenceEntry.value));
const sequenceDefaultFrames = computed(() =>
  defaultClipFrames(selectedSequenceEntry.value, chainLimits.value, sequenceMotionTail.value),
);
/** No chain-capable video model on the selected route → guide to Discover. */
const showSequenceEmpty = computed(
  () =>
    isSequence.value &&
    conn.ready &&
    sequenceInventorySettled.value &&
    sequenceCapableModels.value.length === 0,
);

/** Sequence is authoritative over a stale single-image selection. Re-run
 * after host inventory arrives and whenever the target changes. */
watch(
  [isSequence, sequenceCapableModels, sequenceInventorySettled, sequenceTargetHostId],
  () => {
    if (!isSequence.value) return;
    const current = form.model;
    if (sequenceCapableModels.value.some((model) => model.name === current)) return;
    if (current && !draft.lastSingleModel && !draft.editing) {
      draft.lastSingleModel = current;
    }
    const pick = sequenceCapableModels.value[0];
    if (pick) {
      formStore.applyModel(pick);
    } else if (
      sequenceInventorySettled.value ||
      (selectedEntry.value !== null && !modelSupportsSequence(selectedEntry.value))
    ) {
      form.model = "";
      form.family = "";
      chainLimits.value = null;
    }
  },
  { immediate: true },
);

function sharedSnapshot(): string {
  return JSON.stringify({
    ...sequenceParams(form, selectedSequenceEntry.value),
    enableAudio: draft.enableAudio,
    motionTail: sequenceMotionTail.value,
  });
}
const chainLevelDirty = computed(
  () =>
    draft.editing !== null &&
    editSharedBaseline.value !== null &&
    (editSharedBaseline.value !== sharedSnapshot() ||
      JSON.stringify(draft.editing.baselineOpeningImage ?? null) !==
        JSON.stringify(draft.openingImage)),
);

/** `?output=sequence` deep-links (palette, menu, legacy /chains) are consumed
 * ONCE, then stripped; the persisted draft output wins on ordinary visits. */
function consumeOutputQuery() {
  if (activeRoute.query.output !== "sequence") return;
  if (draft.output !== "sequence") {
    // Mirror the inspector's model rule, and swap BEFORE seeding clips: a
    // non-capable selection is remembered and replaced by the first
    // chain-capable model so setOutput's new clips default their frames
    // from the effective selection, not the outgoing still model's.
    const current = selectedEntry.value;
    if (!current || !sequenceCapableModels.value.some((m) => m.name === current.name)) {
      draft.lastSingleModel = form.model || null;
      const pick = sequenceCapableModels.value[0];
      if (pick) formStore.applyModel(pick);
    }
    draft.setOutput(
      "sequence",
      { getPrompt: () => form.prompt, setPrompt: (value) => (form.prompt = value) },
      sequenceDefaultFrames.value,
    );
  }
  void router.replace({ path: "/create" });
}

let chainLimitsFetch = 0;
async function loadChainLimits() {
  const entry = selectedSequenceEntry.value;
  if (!entry) {
    chainLimits.value = null;
    return;
  }
  const version = ++chainLimitsFetch;
  try {
    const target = routeForModel(entry)?.target ?? null;
    const limits = (await fetchChainLimits(entry.name, target)) as ChainLimits;
    if (version !== chainLimitsFetch) return;
    chainLimits.value = limits;
    if (!limits.supports_audio) draft.enableAudio = false;
  } catch {
    if (version === chainLimitsFetch) chainLimits.value = null;
  }
}

// Chain limits are per model AND per host — refetch when either moves.
watch(
  [isSequence, () => form.model, stickyTarget, () => readyHostSignature(hosts.all)],
  () => {
    if (isSequence.value && form.model) void loadChainLimits();
  },
  { immediate: true },
);

// The activity strip lists every connected host's durable jobs.
watch(
  () => readyHostSignature(hosts.all),
  () => void chains.fetchAll(),
  { immediate: true },
);

// A finalized sequence lands in the origin host's gallery — refresh so the
// video appears without a manual reload.
watch(
  () => chains.finalizedTick,
  () => void hostGallery.fetchAll().catch(() => {}),
);

/** The watched durable job, while it is actually rendering. */
const watchedSequence = computed(() => {
  if (!chains.watching) return null;
  const detail = chains.live.detail;
  if (!detail) return null;
  return detail.state === "running" || detail.state === "queued" ? detail : null;
});
const watchedSequencePct = computed(() => {
  const active = chains.live.activeStage;
  if (active === null) return 0;
  const progress = chains.live.progress[active];
  return progress && progress.total > 0 ? (progress.step / progress.total) * 100 : 0;
});

const sequenceStagePosters = ref<Record<number, string>>({});
const sequenceStageMedia = ref<Record<number, string>>({});
const sequenceStageClipIds = ref<string[]>([]);
const sequenceStageClipIdsByJob = new Map<string, string[]>();
const playingSequenceStage = ref<number | null>(null);
let sequenceStageMediaKey = "";
let sequenceStageMediaEpoch = 0;
const pendingSequencePosters = new Set<string>();
const pendingSequenceMedia = new Set<string>();

function clearSequenceStageMedia() {
  sequenceStageMediaEpoch += 1;
  pendingSequencePosters.clear();
  pendingSequenceMedia.clear();
  for (const url of Object.values(sequenceStagePosters.value)) {
    URL.revokeObjectURL(url);
  }
  sequenceStagePosters.value = {};
  sequenceStageMedia.value = {};
  sequenceStageClipIds.value = [];
  playingSequenceStage.value = null;
  sequenceStageMediaKey = "";
}

watch(
  () => {
    const detail = chains.live.detail;
    const watched = chains.watching;
    return detail && watched
      ? [
          watched.hostId,
          detail.id,
          detail.stages
            .filter((stage) => stage.has_preview || stage.has_media)
            .map((stage) => `${stage.idx}:${stage.has_preview ? 1 : 0}:${stage.has_media ? 1 : 0}`)
            .join(","),
        ].join(":")
      : "";
  },
  () => {
    const detail = chains.live.detail;
    const watched = chains.watching;
    if (!detail || !watched) return;
    const key = `${watched.hostId}:${detail.id}`;
    if (key !== sequenceStageMediaKey) {
      clearSequenceStageMedia();
      sequenceStageMediaKey = key;
      const script = normalizeServerChainScript(detail.script);
      const boundIds = sequenceStageClipIdsByJob.get(key);
      const editing = draft.editing;
      const editingMatches =
        editing?.hostId === watched.hostId &&
        editing.jobId === detail.id &&
        draft.clips.length === detail.stages.length;
      if (boundIds?.length === detail.stages.length) {
        sequenceStageClipIds.value = [...boundIds];
      } else if (editingMatches) {
        // The binding cache belongs to this view instance, while the edit
        // session is durable store state. Rebuild positionally on remount
        // even when the user has already changed a prompt from the job script.
        sequenceStageClipIds.value = draft.clips.map((clip) => clip.id);
      } else if (
        script &&
        script.stages.length === draft.clips.length &&
        script.stages.every((stage, idx) => stage.prompt === draft.clips[idx]?.prompt)
      ) {
        sequenceStageClipIds.value = draft.clips.map((clip) => clip.id);
      }
    }
    const stagesByIdx = new Map(detail.stages.map((stage) => [stage.idx, stage]));
    for (const [rawIdx, url] of Object.entries(sequenceStagePosters.value)) {
      const idx = Number(rawIdx);
      if (!stagesByIdx.get(idx)?.has_preview) {
        URL.revokeObjectURL(url);
        const next = { ...sequenceStagePosters.value };
        delete next[idx];
        sequenceStagePosters.value = next;
      }
    }
    for (const rawIdx of Object.keys(sequenceStageMedia.value)) {
      const idx = Number(rawIdx);
      if (!stagesByIdx.get(idx)?.has_media) {
        const next = { ...sequenceStageMedia.value };
        delete next[idx];
        sequenceStageMedia.value = next;
        if (playingSequenceStage.value === idx) playingSequenceStage.value = null;
      }
    }
    const target = chains.targetFor(watched.hostId);
    const requestEpoch = sequenceStageMediaEpoch;
    for (const stage of detail.stages) {
      const pendingKey = `${requestEpoch}:${key}:${stage.idx}`;
      if (
        stage.has_preview &&
        !sequenceStagePosters.value[stage.idx] &&
        !pendingSequencePosters.has(pendingKey)
      ) {
        pendingSequencePosters.add(pendingKey);
        const path = `/api/chain-jobs/${encodeURIComponent(detail.id)}/stages/${stage.idx}/preview`;
        void apiFetchTo(target, path)
          .then((response) => response.blob())
          .then((blob) => {
            const url = URL.createObjectURL(blob);
            const currentStage = chains.live.detail?.stages.find(
              (candidate) => candidate.idx === stage.idx,
            );
            if (
              requestEpoch !== sequenceStageMediaEpoch ||
              sequenceStageMediaKey !== key ||
              chains.watching?.hostId !== watched.hostId ||
              chains.live.detail?.id !== detail.id ||
              !currentStage?.has_preview
            ) {
              URL.revokeObjectURL(url);
              return;
            }
            sequenceStagePosters.value = {
              ...sequenceStagePosters.value,
              [stage.idx]: url,
            };
          })
          .catch(() => {})
          .finally(() => pendingSequencePosters.delete(pendingKey));
      }
      if (
        stage.has_media &&
        !sequenceStageMedia.value[stage.idx] &&
        !pendingSequenceMedia.has(pendingKey)
      ) {
        pendingSequenceMedia.add(pendingKey);
        void sequenceStageMediaUrl(target, detail.id, stage.idx)
          .then((url) => {
            const currentStage = chains.live.detail?.stages.find(
              (candidate) => candidate.idx === stage.idx,
            );
            if (
              requestEpoch !== sequenceStageMediaEpoch ||
              sequenceStageMediaKey !== key ||
              chains.watching?.hostId !== watched.hostId ||
              chains.live.detail?.id !== detail.id ||
              !currentStage?.has_media
            ) {
              return;
            }
            sequenceStageMedia.value = {
              ...sequenceStageMedia.value,
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

const sequencePlaybackSrc = computed(() => {
  const idx = playingSequenceStage.value;
  return idx === null ? null : (sequenceStageMedia.value[idx] ?? null);
});
const sequenceFilmstripMediaByClipId = computed<
  Readonly<Record<string, ClipRailMedia | undefined>>
>(() => {
  const detail = chains.live.detail;
  if (!detail) return {};
  return Object.fromEntries(
    [...detail.stages]
      .sort((a, b) => a.idx - b.idx)
      .flatMap((stage) => {
        const clipId = sequenceStageClipIds.value[stage.idx];
        if (!clipId) return [];
        const progress = chains.live.progress[stage.idx];
        const status: ClipRailMedia["status"] =
          stage.state === "completed" ? "ready" : stage.state === "failed" ? "error" : stage.state;
        return [
          [
            clipId,
            {
              stageIdx: stage.idx,
              status,
              posterUrl: sequenceStagePosters.value[stage.idx] ?? null,
              hasMedia: Boolean(stage.has_media && sequenceStageMedia.value[stage.idx]),
              cacheReady: stage.cache_ready,
              progressPercent:
                progress && progress.total > 0 ? (progress.step / progress.total) * 100 : null,
              error: stage.error,
            } satisfies ClipRailMedia,
          ],
        ];
      }),
  );
});

function playSequenceStage(stageIdx: number) {
  if (!sequenceStageMedia.value[stageIdx]) return;
  playingSequenceStage.value = playingSequenceStage.value === stageIdx ? null : stageIdx;
}

function playSequenceClip(clipId: string) {
  const stageIdx = sequenceStageClipIds.value.indexOf(clipId);
  if (stageIdx >= 0) playSequenceStage(stageIdx);
}

const playingSequenceClipId = computed(() => {
  const stageIdx = playingSequenceStage.value;
  return stageIdx === null ? null : (sequenceStageClipIds.value[stageIdx] ?? null);
});

function returnToLiveSequence() {
  playingSequenceStage.value = null;
}

/**
 * The watched job once it has settled. The Create strip no longer keeps a
 * settled row, so the canvas is what holds the result: the finished video with
 * Edit sequence / Show in library, or the failure with Resume. Settling must
 * never drop the canvas back to the empty state.
 */
const settledSequence = computed(() => {
  if (!chains.watching) return null;
  const detail = chains.live.detail;
  if (!detail) return null;
  return detail.state === "running" || detail.state === "queued" ? null : detail;
});

/** The gallery row this job produced, on the job's OWN host — never a sibling
 *  host's auto-saved copy. Resolves after the finalize refetch; until then the
 *  caption and its actions render over the last stage preview. */
const settledSequencePrint = computed(() => {
  const detail = settledSequence.value;
  const hostId = chains.watching?.hostId;
  if (!detail || !hostId) return null;
  return (
    hostGallery.merged.find(
      (entry) => entry.sourceKey === hostId && isPrintOfChainJob(entry.item.metadata, detail.id),
    ) ?? null
  );
});

/**
 * Largest rect of the settled print's aspect that fits the canvas region —
 * the same pure-CSS containment as `previewFrameStyle`, so the stitched video
 * always shrinks to fit and is never clipped by a width-driven frame.
 */
const settledFrameStyle = computed(() => {
  const meta = settledSequencePrint.value?.item.metadata;
  const w = meta?.width || 16;
  const h = meta?.height || 9;
  return {
    aspectRatio: `${w} / ${h}`,
    width: `min(100cqw, ${(100 * w) / h}cqh)`,
  };
});

const settledSequenceCaption = computed(() => {
  const detail = settledSequence.value;
  if (!detail) return "";
  const print = settledSequencePrint.value;
  const meta = print?.item.metadata;
  const bits = [
    modelDisplayNameForId(detail.model, installedModels.value),
    `${detail.stage_count} clip${detail.stage_count === 1 ? "" : "s"}`,
  ];
  if (meta) bits.push(`S ${meta.seed}`, `${meta.width}×${meta.height}`);
  bits.push(hosts.all.find((h) => h.id === chains.watching?.hostId)?.label ?? "this device");
  return bits.join(" · ");
});

const settledSequenceError = computed(() =>
  settledSequence.value?.state === "failed"
    ? friendlySequenceError(
        settledSequence.value.error ?? "Sequence failed.",
        hosts.all.find((h) => h.id === chains.watching?.hostId)?.label,
      )
    : null,
);
const settledSequenceErrorCopy = computed(() => {
  const raw = settledSequence.value?.error;
  return raw && settledSequenceError.value
    ? copyableError(raw, settledSequenceError.value)
    : (settledSequenceError.value ?? "");
});

function showSettledSequenceInLibrary() {
  const print = settledSequencePrint.value;
  void router.push(
    print ? { path: "/library", query: { print: print.item.filename } } : { path: "/library" },
  );
}

function editSettledSequence() {
  const detail = settledSequence.value;
  const hostId = chains.watching?.hostId;
  if (!detail || !hostId) return;
  void editSequence({ hostId, jobId: detail.id });
}

function resumeSettledSequence() {
  const detail = settledSequence.value;
  const hostId = chains.watching?.hostId;
  if (!detail || !hostId) return;
  void chains.resume(hostId, detail.id).catch((err) => toasts.push(String(err), "error"));
}

async function generateSequence() {
  if (sequenceSubmitting.value) return;
  const entry = selectedSequenceEntry.value;
  if (!entry) {
    toasts.push("Choose an installed sequence-capable video model first.", "error");
    return;
  }
  // Freeze the complete request at the click boundary. Preprocessing can be
  // slow, and edits made while it runs belong to the next submission.
  const editing = draft.editing ? { ...draft.editing } : null;
  const requestForm = cloneGenerateForm(form);
  const clips = JSON.parse(JSON.stringify(draft.clips)) as typeof draft.clips;
  const openingSnapshot = draft.openingImage ? { ...draft.openingImage } : null;
  const enableAudio = draft.enableAudio;
  const motionTailFrames = sequenceMotionTail.value;
  const hostRoute = editing ? hosts.resolveRoute(editing.hostId, entry.name) : routeForModel(entry);
  if (!hostRoute) {
    toasts.push("The selected host isn't reachable. Pick another host.", "error");
    return;
  }
  sequenceSubmitting.value = true;
  try {
    // Refetch stale limits so frames caps/audio gating match the routed host.
    if (!chainLimits.value || chainLimits.value.model !== entry.name) {
      await loadChainLimits();
    }
    requestForm.sourceImage = openingSnapshot?.base64 ?? null;
    requestForm.maskImage = null;
    if (!(await preprocessSourceFit(hostRoute, requestForm))) return;
    const openingImage = openingSnapshot
      ? { ...openingSnapshot, base64: requestForm.sourceImage }
      : null;
    const request = buildChainRequest(sequenceParams(requestForm, entry), clips, {
      motionTailFrames,
      enableAudio,
      openingImage,
    });
    const currentRoute = hosts.resolveRoute(hostRoute.hostId, entry.name);
    if (
      !currentRoute ||
      currentRoute.hostId !== hostRoute.hostId ||
      currentRoute.target.baseUrl !== hostRoute.target.baseUrl ||
      currentRoute.target.apiKey !== hostRoute.target.apiKey ||
      (currentRoute.instanceId ?? null) !== (hostRoute.instanceId ?? null)
    ) {
      toasts.push(
        "The sequence machine changed during source preparation. Review the machine and Generate again.",
        "error",
      );
      return;
    }
    if (editing) {
      const amend: AmendRequest = {
        stages: request.stages,
        motion_tail_frames: request.motion_tail_frames ?? null,
        fps: request.fps ?? null,
        seed: requestForm.seed.trim() === "" ? null : requestForm.seed.trim(),
        steps: request.steps,
        guidance: request.guidance,
        strength: request.strength ?? null,
        // Always explicit: null means "keep current" server-side, which
        // would make turning audio OFF impossible through an edit.
        enable_audio: enableAudio,
      };
      try {
        sequenceStageClipIdsByJob.set(
          `${editing.hostId}:${editing.jobId}`,
          clips.map((clip) => clip.id),
        );
        const outcome = await chains.amend(editing.hostId, editing.jobId, amend, hostRoute.target);
        toasts.push(
          `Sequence updated · ${outcome.preserved_stages} clip${outcome.preserved_stages === 1 ? "" : "s"} kept from cache`,
        );
        draft.stopEditing();
        editSharedBaseline.value = null;
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          toasts.push(
            "This sequence changed on its host while you edited. Use Duplicate as new to submit your version.",
            "error",
          );
          return;
        }
        throw err;
      }
    } else {
      const jobId = await chains.create(hostRoute.hostId, request, hostRoute.target);
      sequenceStageClipIdsByJob.set(
        `${hostRoute.hostId}:${jobId}`,
        clips.map((clip) => clip.id),
      );
      toasts.push("Sequence queued");
    }
    // The caveat described the handoff, not the submitted job.
    sequenceReuseNotice.value = null;
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    sequenceSubmitting.value = false;
  }
}

/** Edit session: submit the current clips as a brand-new job instead. */
async function duplicateSequenceAsNew() {
  draft.stopEditing();
  editSharedBaseline.value = null;
  await generateSequence();
}

/** ActivityStrip Edit: load a durable job's effective script into an edit
 * session — applying its shared params to the form is the explicit action. */
async function editSequence(payload: { hostId: string; jobId: string }) {
  await loadSequence(payload, true);
}

/** Select/watch a durable job without turning a row click into an amend. */
async function inspectSequence(payload: { hostId: string; jobId: string }) {
  await loadSequence(payload, false);
}

async function loadSequence(payload: { hostId: string; jobId: string }, editing: boolean) {
  // An edit session is lossless — any reuse caveat on screen is now stale.
  sequenceReuseNotice.value = null;
  try {
    const detail = await chains.fetchDetail(payload.hostId, payload.jobId);
    const script = normalizeServerChainScript(detail.script);
    if (!script) {
      toasts.push("This job carries no editable script.", "error");
      return;
    }
    const loaded = chainScriptToClips(script);
    const shared = loaded.shared;
    if (shared.model) {
      const entry = findInstalledModel(installedModels.value, shared.model);
      if (entry) formStore.applyModel(entry);
      else form.model = shared.model;
    }
    if (shared.width != null) form.width = shared.width;
    if (shared.height != null) form.height = shared.height;
    if (shared.fps != null) form.fps = shared.fps;
    if (shared.steps != null) form.steps = shared.steps;
    if (shared.guidance != null) form.guidance = shared.guidance;
    if (shared.strength != null) form.strength = shared.strength;
    if (loaded.openingImage) {
      form.sourceFit = { mode: "crop-fill", alignX: "center", alignY: "center" };
    }
    form.seed = shared.seed ?? "";
    if (editing) {
      draft.loadFromJob(
        {
          jobId: payload.jobId,
          hostId: payload.hostId,
          baseline: loaded.clips.map((clip) => ({ ...clip })),
          completedStages: countLeadingCompletedStages(detail.stages),
        },
        loaded.clips,
        loaded.enableAudio,
        loaded.openingImage,
      );
      editSharedBaseline.value = sharedSnapshot();
    } else {
      draft.stopEditing();
      editSharedBaseline.value = null;
      draft.output = "sequence";
      draft.clips.splice(0, draft.clips.length, ...loaded.clips);
      draft.activeClipId = loaded.clips[0]?.id ?? null;
      draft.enableAudio = loaded.enableAudio;
      draft.openingImage = loaded.openingImage;
    }
    sequenceStageClipIdsByJob.set(
      `${payload.hostId}:${payload.jobId}`,
      draft.clips.map((clip) => clip.id),
    );
    await chains.watch(payload.hostId, payload.jobId);
    void loadChainLimits();
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

// Availability data is demand-driven: fetch when the set of ready hosts
// changes. immediate so routing is model-aware on the FIRST Generate click.
// (The inspector's picker force-refreshes on open.)
watch(
  () =>
    hosts.all
      .filter((h) => h.status === "ready")
      .map((h) => h.id)
      .join("\n"),
  () => void hostModels.refresh(),
  { immediate: true },
);

const buttonLabel = computed(() => {
  if (submissionPlanning.value) return "Planning…";
  return generation.pending.length > 0
    ? `Generate (+${generation.pending.length} queued)`
    : "Generate";
});

const previewWidth = computed(() => job.value?.width ?? form.width);
const previewHeight = computed(() => job.value?.height ?? form.height);
/**
 * The frame sizes itself with pure CSS — no measurement, observers, or
 * layout races (a JS-measured frame collapsed to 0×0 whenever the region
 * mounted after the observer, rendering the develop view invisible). The
 * region is a size container; `min(100cqw, ratio·100cqh)` is the largest
 * rect of the print's aspect that fits it. Engines without container-query
 * units drop the invalid width and fall back to the full-width class, where
 * the media object-contains inside the frame instead.
 */
const previewFrameStyle = computed(() => ({
  aspectRatio: `${previewWidth.value} / ${previewHeight.value}`,
  width: `min(100cqw, ${(100 * previewWidth.value) / previewHeight.value}cqh)`,
}));

const liveGenerationStatus = computed(() => {
  const j = job.value;
  if (!j || j.status === "complete" || j.status === "error") return "";
  if (j.status === "queued") return "Queued";
  if (j.status === "loading") return `${j.stage ?? "Preparing"}…`;
  if (j.status === "finishing") return `Fixing — ${j.stage ?? "finishing"}…`;
  return `Developing ${j.step}/${j.total}`;
});

const edgeCode = computed(() => {
  const j = job.value;
  if (!j) return "";
  const name = modelDisplayNameForId(j.model, installedModels.value);
  const s = j.result
    ? `S ${j.result.seed_used}`
    : j.visualSeed.startsWith(`${j.model}·`)
      ? "S random"
      : `S ${j.visualSeed.slice(0, 12)}`;
  const stepPart = `${j.status === "complete" ? j.total : j.step}/${j.total}`;
  const size = j.result ? `${j.result.width}×${j.result.height}` : `${j.width}×${j.height}`;
  const time = j.result ? `${(j.result.generation_time_ms / 1000).toFixed(1)}s` : "";
  return [name, s, stepPart, size, time].filter(Boolean).join("  ");
});

let templateLoadEpoch = 0;
async function loadTemplate(template: GenerationTemplate) {
  const epoch = ++templateLoadEpoch;
  templatesOpen.value = false;
  const hydrated = await hydrateGenerationTemplate(template);
  if (epoch !== templateLoadEpoch) return;
  // buildRequest's pruneRequestForFamily still guards anything the (possibly
  // different) family can't use after the source snapshot is restored. A
  // pre-#787 template lacking `negativePromptDefault` is normalized first so
  // its empty negative reads as "untouched", not the explicit "" opt-out.
  Object.assign(form, normalizeLegacyNegativeSnapshot(hydrated.form, installedModels.value));
  if (form.model && !findInstalledModel(installedModels.value, form.model)) {
    toasts.push(`Model "${form.model}" isn't installed — settings applied anyway.`);
  }
  if (hydrated.missingMediaReferences.length > 0) {
    toasts.push(`Re-add media: ${formatTemplateMediaReferences(hydrated.missingMediaReferences)}.`);
  }
}

function siblingDot(s: Job): string {
  if (s.status === "complete") return "text-ink"; // ◉ developed
  if (s.status === "error") return "text-stop";
  return "text-ink-3"; // ◎ pending
}

/** Whether this job produced a WAV rather than a picture or a clip. */
function isAudioResult(job: { result: CompleteEvent | null } | null): boolean {
  return isAudioCompletion(job?.result);
}

function canvasMenu(): MenuEntry[] {
  const j = job.value;
  if (!j) return [];
  const live = j.status !== "complete" && j.status !== "error";
  return [
    {
      label: "Cancel",
      danger: true,
      disabled: !live,
      action: () => void generation.cancel(j.clientId).then(() => toasts.push("Cancelled")),
    },
    { separator: true },
    {
      label: "Copy prompt",
      action: () => void navigator.clipboard.writeText(j.prompt),
    },
    {
      label: "Copy seed",
      disabled: !j.result,
      action: () => void navigator.clipboard.writeText(String(j.result?.seed_used ?? "")),
    },
    {
      label: "Copy image",
      disabled: !j.result || !!j.result.video_frames || isAudioResult(j),
      action: () => {
        if (!j.result) return;
        const mime = j.result.format === "jpeg" ? "image/jpeg" : `image/${j.result.format}`;
        void copyBase64ImageToClipboard(j.result.image, mime)
          .then(() => toasts.push("Image copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      },
    },
    {
      label: "Save image",
      disabled: !j.result || !!j.result.video_frames || isAudioResult(j) || !j.result.image,
      action: () => {
        if (!j.result?.image) return;
        const filename =
          j.result.filename ??
          suggestOutputFilename(
            j.result.model,
            j.result.seed_used,
            j.result.format,
            j.submittedAtUnixMs,
          );
        void ipc
          .saveMediaBytes(filename, j.result.image)
          .then((saved) => showSavedMediaToast(toasts, saved))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      },
    },
    {
      label: "Export video format…",
      disabled: !j.result?.video_frames || !j.result.filename?.toLowerCase().endsWith(".mp4"),
      action: () => void openGeneratedVideoExport(j),
    },
    {
      label: "Copy file path",
      disabled: !j.result?.filename,
      action: () => {
        if (!j.result?.filename) return;
        void copyLocalOutputPath(j.result.filename)
          .then(() => toasts.push("File path copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      },
    },
    { separator: true },
    {
      label: "Show in Gallery",
      disabled: j.status !== "complete",
      action: () => void router.push("/gallery"),
    },
  ];
}

async function openGeneratedVideoExport(candidate: Job): Promise<void> {
  if (!candidate.result?.filename) return;
  videoExportJob.value = candidate;
  videoExportError.value = "";
  const target = generation.targetForJob(candidate.clientId);
  if (!target) {
    videoExportError.value = "The video’s host is no longer connected.";
    return;
  }
  try {
    const { apiJsonTo } = await import("../lib/api/client");
    videoExportCapabilities.value = await apiJsonTo<VideoExportCapabilities>(
      target,
      "/api/gallery/export-options",
    );
  } catch (error) {
    videoExportCapabilities.value = DEFAULT_VIDEO_EXPORT_CAPABILITIES;
    videoExportError.value = error instanceof Error ? error.message : String(error);
  }
}

async function exportGeneratedVideo(options: VideoExportOptions): Promise<void> {
  const candidate = videoExportJob.value;
  const filename = candidate?.result?.filename;
  if (!candidate || !filename || videoExportBusy.value) return;
  const target = generation.targetForJob(candidate.clientId);
  if (!target) {
    videoExportError.value = "The video’s host is no longer connected.";
    return;
  }
  videoExportBusy.value = true;
  videoExportError.value = "";
  try {
    const saved = await saveGalleryMedia(
      target,
      filename,
      videoExportFilename(filename, options.format),
      options,
    );
    videoExportJob.value = null;
    showSavedMediaToast(toasts, saved);
  } catch (error) {
    videoExportError.value = error instanceof Error ? error.message : String(error);
  } finally {
    videoExportBusy.value = false;
  }
}

function expansionInputs(count: number): PreparedExpansionInputs {
  const request = buildRequest(form);
  return {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    task: expansionTaskForRequest(form.family, request),
    requestedCount: count,
    stylePreset: form.stylePreset || null,
    selectedHostPolicy: stickyTarget.value,
  };
}

async function remixForCurrentPrompt(replacePrepared = false) {
  if (!form.prompt.trim() || !form.model || expansionRunning.value) return;
  submissionGuard.invalidate();
  const route = currentExpansionRoute.value;
  if (!route) {
    expansionError.value = unavailableExpansionHostMessage();
    return;
  }
  const request = buildRequest(form);
  const task = expansionTaskForRequest(form.family, request);
  const source = promptSource(form.prompt, form.originalPrompt, remixSource.value);
  const stylePreset = form.stylePreset || null;
  const dimensions = defaultRemixDimensions(task, Boolean(stylePreset));
  // Match the Batch value the composer actually presents. Capability/source
  // constraints can force the effective value to one while preserving the
  // user's saved raw preference for a later compatible model.
  const requestedCount = effectiveBatchSize.value;
  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = null;
  expansionAttemptHostLabel.value = route.label;
  try {
    const style = styleHint(stylePreset ?? "");
    const response = await remixPrompt(
      {
        source_prompt: source.prompt,
        ...(source.rootPrompt ? { root_prompt: source.rootPrompt } : {}),
        source_kind: source.kind,
        model_family: form.family,
        variations: requestedCount,
        task,
        ...(style ? { style } : {}),
        dimensions,
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const variants = validateRemixVariants(response.variants, requestedCount);
    const currentRequest = buildRequest(form);
    if (
      form.model !== request.model ||
      expansionTaskForRequest(form.family, currentRequest) !== task ||
      conditioningFingerprint(currentRequest) !== conditioningFingerprint(request)
    ) {
      expansionError.value =
        "The model or conditioning changed while Remix was running. Remix again.";
      return;
    }
    if (requestedCount === 1) {
      const selected = variants[0]!;
      form.prompt = selected.prompt;
      form.originalPrompt = response.root_prompt ?? response.source_prompt;
      quickExpansionOriginal.value = response.source_prompt;
      bakeStyleNegative(stylePreset ?? "", form.family);
      form.stylePreset = "";
      quickExpansionSnapshot.value = {
        requestToken: token,
        originalPrompt: response.root_prompt ?? response.source_prompt,
        expandedPrompt: selected.prompt,
        model: form.model,
        family: form.family,
        task,
        stylePreset,
        selectedHostPolicy: stickyTarget.value,
        route: { ...route, target: { ...route.target } },
        promptTransform: {
          operation: "remix",
          ...(response.root_prompt ? { root_prompt: response.root_prompt } : {}),
          source_prompt: response.source_prompt,
          source_kind: response.source_kind,
          task,
          dimensions: [...selected.dimensions],
        },
      };
      preparedBatch.value = null;
      void nextTick(() => composerRef.value?.focus?.());
      return;
    }
    form.batchSize = requestedCount;
    const batch = createPreparedExpansionBatch(
      {
        kind: "remix",
        sourcePrompt: response.source_prompt,
        ...(response.root_prompt ? { rootPrompt: response.root_prompt } : {}),
        sourceKind: response.source_kind,
        dimensions,
        conditioningFingerprint: conditioningFingerprint(request),
        model: form.model,
        family: form.family,
        task,
        requestedCount,
        stylePreset,
        selectedHostPolicy: stickyTarget.value,
      },
      route,
      variants.map((variant) => variant.prompt),
      token,
    );
    batch.prompts.forEach((prompt, index) => {
      prompt.dimensions = variants[index]?.dimensions ?? [];
    });
    preparedBatch.value = batch;
    if (replacePrepared) void nextTick(() => composerRef.value?.focus?.());
  } catch (error) {
    if (preparationGuard.isCurrent(token))
      expansionError.value = describeExpansionError(error, route);
  } finally {
    if (preparationGuard.isCurrent(token)) expansionRunning.value = false;
  }
}

function unavailableExpansionHostMessage(): string {
  const selection = stickyTarget.value;
  const selected = selection ? hosts.all.find((host) => host.id === selection) : null;
  if (selected) {
    return `${selected.label} isn't reachable. Expansion will not fall back to another host.`;
  }
  return "No generation host is reachable. Connect the selected host before expanding.";
}

function describeExpansionError(error: unknown, route: HostRoute): string {
  const message = error instanceof Error ? error.message : String(error);
  const missingModel = parseMissingExpandModel(message);
  expansionMissingModel.value = missingModel ? { model: missingModel, route } : null;
  expansionPullAttempt.value = null;
  return missingModel
    ? `The expansion model ${missingModel} isn't installed on ${route.label}.`
    : `Expansion failed on ${route.label}: ${message}`;
}

/**
 * Resolve once, expand on that target, then retain the same route in the
 * prepared batch. A refresh can resolve the current policy again, but an
 * already prepared batch never does so implicitly.
 */
async function expandForCurrentBatch(
  replacePrepared = false,
  routeOverride: HostRoute | null = null,
) {
  const count = effectiveBatchSize.value;
  const inputs = expansionInputs(count);
  if (
    !inputs.sourcePrompt ||
    !inputs.model ||
    expansionRunning.value ||
    (preparedBatch.value && count === 1 && !replacePrepared)
  )
    return;

  const preparedSection = document.querySelector<HTMLElement>(
    '[data-test="prepared-expansion-batch"]',
  );
  const replacementOwnedFocus =
    replacePrepared &&
    !!preparedSection &&
    !!document.activeElement &&
    preparedSection.contains(document.activeElement);
  // Starting another expansion supersedes any Generate still preprocessing
  // an older quick snapshot. Its late completion must not queue or clear the
  // replacement snapshot created below.
  submissionGuard.invalidate();

  const route = routeOverride ?? currentExpansionRoute.value;
  if (!route) {
    expansionAttemptHostLabel.value = null;
    expansionError.value = unavailableExpansionHostMessage();
    expansionMissingModel.value = null;
    return;
  }
  expansionAttemptHostLabel.value = route.label;
  const capability = hosts.capabilities[route.hostId]?.expand;
  if (capability?.configured === false) {
    expansionError.value = `Prompt expansion isn't configured on ${route.label}. Configure that host before retrying.`;
    expansionMissingModel.value = null;
    return;
  }

  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = null;
  expansionMissingModel.value = null;
  try {
    // The active chip travels as a natural-language directive the server
    // weaves into the expander's system message — never the literal suffix.
    const styleDirective = styleHint(inputs.stylePreset ?? "");
    const response = await expandPrompt(
      inputs.sourcePrompt,
      {
        variations: count,
        ...(inputs.family ? { modelFamily: inputs.family } : {}),
        task: inputs.task,
        ...(styleDirective ? { style: styleDirective } : {}),
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const prompts = validateExpandedPrompts(response.expanded, count);
    if (count === 1) {
      // Quick expansion has no review workspace. Never overwrite edits or a
      // target change that happened while its request was in flight.
      const current = expansionInputs(1);
      const hostStillReady = hosts.all.some(
        (host) => host.id === route.hostId && host.status === "ready",
      );
      if (
        current.sourcePrompt !== inputs.sourcePrompt ||
        current.model !== inputs.model ||
        current.family !== inputs.family ||
        current.task !== inputs.task ||
        current.stylePreset !== inputs.stylePreset ||
        current.selectedHostPolicy !== inputs.selectedHostPolicy ||
        !hostStillReady
      ) {
        expansionError.value =
          "The prompt, style, or generation host changed while expansion was running. Expand again to use the current inputs.";
        return;
      }
      quickExpansionOriginal.value = inputs.sourcePrompt;
      form.prompt = prompts[0]!;
      form.originalPrompt = inputs.sourcePrompt;
      quickExpansionSnapshot.value = {
        requestToken: token,
        originalPrompt: inputs.sourcePrompt,
        expandedPrompt: prompts[0]!,
        model: inputs.model,
        family: inputs.family,
        task: inputs.task,
        stylePreset: inputs.stylePreset,
        selectedHostPolicy: inputs.selectedHostPolicy,
        route: { ...route, target: { ...route.target } },
      };
      // Bake-and-clear: the rewrite absorbed the style (the server received
      // it as a directive), so the chip clears here — leaving it lit would
      // apply the look twice at submit. Prepared batches below KEEP the chip:
      // it is the frozen-style indicator for the reviewed set (a style change
      // is a named staleness axis) and their submit path never re-composes it
      // into the reviewed prompt text.
      bakeStyleNegative(inputs.stylePreset ?? "", inputs.family);
      form.stylePreset = "";
      if (replacePrepared) {
        const active = document.activeElement;
        const shouldRestoreFocus =
          replacementOwnedFocus &&
          (active === document.body || (!!active && preparedSection?.contains(active)));
        preparedBatch.value = null;
        if (shouldRestoreFocus) void nextTick(() => composerRef.value?.focus?.());
      }
      return;
    }
    preparedBatch.value = createPreparedExpansionBatch(inputs, route, prompts, token);
    quickExpansionSnapshot.value = null;
  } catch (error) {
    if (!preparationGuard.isCurrent(token)) return;
    expansionError.value = describeExpansionError(error, route);
  } finally {
    if (preparationGuard.isCurrent(token)) expansionRunning.value = false;
  }
}

/**
 * Bake-and-clear owes the user the preset's curated negative: the chip is
 * about to be dropped, so submit-time composition will never see it again.
 * The look itself already reached the rewritten prompt through the expansion
 * directive — only the negative half has nowhere else to live.
 */
function bakeStyleNegative(presetId: string, family: string) {
  quickExpansionNegative.value = null;
  const merged = mergeStyleNegative(form.negativePrompt, presetId, {
    supportsNegativePrompt: generationCapabilitiesForFamily(family).supportsNegativePrompt,
  });
  if (merged === form.negativePrompt) return;
  quickExpansionNegative.value = { before: form.negativePrompt, baked: merged };
  form.negativePrompt = merged;
}

function restoreQuickExpansion() {
  const original = quickExpansionOriginal.value;
  if (original === null) return;
  submissionGuard.invalidate();
  // Undo re-arms the whole pre-expansion state, including the chip the
  // bake-and-clear apply removed and the negative fragments it merged in —
  // unless the user has edited the negative since, which is theirs to keep.
  const snapshot = quickExpansionSnapshot.value;
  if (snapshot) form.stylePreset = snapshot.stylePreset ?? "";
  const negative = quickExpansionNegative.value;
  if (negative && form.negativePrompt === negative.baked) {
    form.negativePrompt = negative.before;
  }
  quickExpansionNegative.value = null;
  form.prompt = original;
  form.originalPrompt = null;
  quickExpansionOriginal.value = null;
  quickExpansionSnapshot.value = null;
  expansionError.value = null;
}

async function generateExpandedAnyway() {
  if (!quickExpansionSnapshot.value) return;
  submissionGuard.invalidate();
  quickExpansionSnapshot.value = null;
  expansionError.value = null;
  await generate();
}

async function reexpandAndGenerate() {
  if (!quickExpansionSnapshot.value || quickExpansionOriginal.value === null) return;
  restoreQuickExpansion();
  await nextTick();
  await expandForCurrentBatch();
  if (quickExpansionSnapshot.value && quickStaleReasons.value.length === 0) {
    await generate();
  }
}

function editPreparedPrompt(payload: { id: string; text: string }) {
  if (preparedSubmitting.value) return;
  const prompt = preparedBatch.value?.prompts.find((candidate) => candidate.id === payload.id);
  if (prompt) prompt.text = payload.text;
}

function removePreparedPrompt(id: string) {
  if (preparedSubmitting.value) return;
  const batch = preparedBatch.value;
  if (!batch || batch.prompts.length <= 2) return;
  batch.prompts = batch.prompts.filter((prompt) => prompt.id !== id);
  batch.requestedCount = batch.prompts.length;
  form.batchSize = batch.prompts.length;
}

function collapsePreparedBatch(removedId: string) {
  if (preparedSubmitting.value) return;
  const batch = preparedBatch.value;
  if (!batch || batch.prompts.length !== 2) return;
  const remaining = batch.prompts.find((prompt) => prompt.id !== removedId);
  if (!remaining) return;
  preparationGuard.invalidate();
  preparedBatch.value = null;
  expansionRunning.value = false;
  expansionError.value = null;
  expansionMissingModel.value = null;
  expansionAttemptHostLabel.value = null;
  form.batchSize = 1;
  form.prompt = remaining.text;
  form.originalPrompt = batch.sourcePrompt;
  // Same bake-and-clear rule as a quick apply: the surviving reviewed text
  // absorbed the frozen style, so keeping the chip would re-apply the look —
  // and the frozen style's negative moves into the form with it.
  bakeStyleNegative(batch.stylePreset ?? "", batch.family);
  form.stylePreset = "";
  quickExpansionOriginal.value = batch.sourcePrompt;
  quickExpansionSnapshot.value = null;
  void nextTick(() => composerRef.value?.focus?.());
}

function applyPreparedRemix(id: string) {
  const batch = preparedBatch.value;
  const selected = batch?.prompts.find((prompt) => prompt.id === id);
  if (!batch || batch.kind !== "remix" || !selected?.text.trim()) return;
  preparationGuard.invalidate();
  form.prompt = selected.text.trim();
  form.originalPrompt = batch.rootPrompt ?? batch.sourcePrompt;
  quickExpansionOriginal.value = batch.sourcePrompt;
  bakeStyleNegative(batch.stylePreset ?? "", batch.family);
  form.stylePreset = "";
  quickExpansionSnapshot.value = {
    requestToken: preparationGuard.begin(),
    originalPrompt: batch.rootPrompt ?? batch.sourcePrompt,
    expandedPrompt: selected.text.trim(),
    model: batch.model,
    family: batch.family,
    task: batch.task,
    stylePreset: batch.stylePreset,
    selectedHostPolicy: batch.selectedHostPolicy,
    route: { ...batch.route, target: { ...batch.route.target } },
    promptTransform: {
      operation: "remix",
      ...(batch.rootPrompt ? { root_prompt: batch.rootPrompt } : {}),
      source_prompt: batch.sourcePrompt,
      source_kind: batch.sourceKind ?? "direct",
      task: batch.task,
      dimensions: [...(selected.dimensions ?? batch.dimensions ?? [])],
    },
  };
  form.batchSize = 1;
  preparedBatch.value = null;
  expansionError.value = null;
  void nextTick(() => composerRef.value?.focus?.());
}

function discardPreparedBatch() {
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  preparedBatch.value = null;
  expansionRunning.value = false;
  expansionError.value = null;
  expansionMissingModel.value = null;
  expansionAttemptHostLabel.value = null;
  void nextTick(() => composerRef.value?.focus?.());
}

async function pullExpansionModel() {
  const missing = expansionMissingModel.value;
  if (!missing) return;
  const route = missing.route;
  const bucket = downloadBucketForRoute(route);
  const baselineInFlight = [...bucket.activeJobs, ...bucket.queued];
  const attempt: ExpansionPullAttempt = {
    id: ++expansionPullRequestId,
    model: missing.model,
    route: { ...route, target: { ...route.target } },
    phase: "connecting",
    jobId: null,
    observedJobId: null,
    baselineMatchingInFlightId:
      baselineInFlight.find((job) => expansionPullJobMatchesModel(job, missing.model))?.id ?? null,
    baselineJobIds: [...bucket.activeJobs, ...bucket.queued, ...bucket.history].map(
      (job) => job.id,
    ),
    allowExistingInFlight: false,
    requestError: null,
  };
  expansionPullAttempt.value = attempt;
  const host = hosts.all.find((candidate) => candidate.id === route.hostId) ?? null;
  const streamHost = host
    ? {
        ...host,
        label: route.label,
        baseUrl: route.target.baseUrl,
        apiKey: route.target.apiKey,
      }
    : null;
  try {
    if (route.kind === "remote" && !host) {
      throw new Error(`${route.label} is no longer connected.`);
    }
    await downloads.subscribe(streamHost ?? undefined);
    if (expansionPullAttempt.value?.id !== attempt.id) return;
    expansionPullAttempt.value.phase = "starting";
    const jobId = await startCatalogDownload(missing.model, route.target, route.kind === "remote");
    if (expansionPullAttempt.value?.id !== attempt.id) return;
    expansionPullAttempt.value.jobId = jobId;
  } catch (error) {
    if (expansionPullAttempt.value?.id !== attempt.id) return;
    if (error instanceof ApiError && error.status === 409) {
      expansionPullAttempt.value.phase = "starting";
      expansionPullAttempt.value.allowExistingInFlight = true;
      const conflictId =
        typeof error.body === "object" &&
        error.body !== null &&
        "id" in error.body &&
        typeof error.body.id === "string" &&
        error.body.id.trim()
          ? error.body.id
          : null;
      expansionPullAttempt.value.observedJobId ??=
        conflictId ?? expansionPullAttempt.value.baselineMatchingInFlightId;
      return;
    }
    expansionPullAttempt.value.requestError =
      error instanceof Error ? error.message : `Couldn't pull ${missing.model} on ${route.label}.`;
  }
}

function downloadBucketForRoute(route: HostRoute): DownloadsState {
  if (route.kind === "local" || route.hostId === downloads.primaryHostId) {
    return downloads;
  }
  return downloads.hostStates[route.hostId] ?? { activeJobs: [], queued: [], history: [] };
}

const expansionPullBucket = computed<DownloadsState>(() => {
  const route = expansionPullAttempt.value?.route ?? expansionMissingModel.value?.route;
  return route ? downloadBucketForRoute(route) : { activeJobs: [], queued: [], history: [] };
});

watch(expansionMissingModel, (missing) => {
  if (!missing) expansionPullAttempt.value = null;
});

watch(
  [expansionPullAttempt, expansionPullBucket],
  ([attempt, bucket]) => {
    if (!attempt || attempt.jobId || attempt.observedJobId) return;
    const baseline = new Set(attempt.baselineJobIds);
    const candidate = [...bucket.activeJobs, ...bucket.queued].find(
      (job) =>
        expansionPullJobMatchesModel(job, attempt.model) &&
        (attempt.allowExistingInFlight || !baseline.has(job.id)),
    );
    if (candidate) attempt.observedJobId = candidate.id;
  },
  { deep: true, flush: "sync" },
);

const expansionPullStatus = computed<ExpansionPullView | null>(() => {
  const missing = expansionMissingModel.value;
  if (!missing) return null;
  const attempt = expansionPullAttempt.value;
  if (
    !attempt ||
    attempt.model !== missing.model ||
    attempt.route.hostId !== missing.route.hostId
  ) {
    return { kind: "missing", job: null };
  }
  return resolveExpansionPullStatus({
    model: attempt.model,
    phase: attempt.phase,
    jobId: attempt.jobId,
    observedJobId: attempt.observedJobId,
    baselineJobIds: attempt.baselineJobIds,
    allowExistingInFlight: attempt.allowExistingInFlight,
    activeJobs: expansionPullBucket.value.activeJobs,
    queued: expansionPullBucket.value.queued,
    history: expansionPullBucket.value.history,
    requestError: attempt.requestError,
  });
});

const expansionPullEtaSeconds = computed(() => {
  const attempt = expansionPullAttempt.value;
  const job = expansionPullStatus.value?.job;
  if (!attempt || !job || job.status !== "active") return null;
  const scope = attempt.route.kind === "local" ? "primary" : attempt.route.hostId;
  return computeEtaSeconds(downloads.rateSamples[`${scope}:${job.id}`] ?? [], job.bytes_total);
});

function retryExpansionAfterPull() {
  const route = expansionPullAttempt.value?.route ?? expansionMissingModel.value?.route;
  if (!route) return;
  void expandForCurrentBatch(!!preparedBatch.value, route);
}
function appendPromptWord(word: string) {
  const trimmed = word.trim();
  if (!trimmed) return;
  form.prompt = form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed;
}

/** Status line while the source is upscaled/refit ahead of the submit. */
const preprocessingStatus = ref<string | null>(null);
/** One last-result cache belongs to this Generate composer instance. */
const sourceFitCache = new SourceFitPreprocessCache();

/**
 * Apply the source-fit policy to the attached source (and mask) before the
 * request is built: canvas-fit a mismatched source, generate the pad mask
 * for pad-repaint, and for upscale-then-fit run the source through
 * `POST /api/upscale/stream` first. `route` is the ALREADY-RESOLVED
 * generation host so a cache miss auto-downloads/runs the upscaler on the
 * same machine the job will run on. Returns false when the submit must abort.
 */
async function preprocessSourceFit(
  route: HostRoute | null,
  draft: ReturnType<typeof cloneGenerateForm>,
): Promise<boolean> {
  const draftCaps = generationCapabilitiesForFamily(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
  );
  if (!draftCaps.supportsImg2img || draftCaps.sourceImageMode !== "single") return true;
  if (!draft.sourceImage) return true;
  try {
    const result = await applySourceFitPreprocess(
      {
        source: draft.sourceImage,
        // Maskless families (LTX-2 img2video) can't ship the repaint mask —
        // coerce defensively even if a stale pad-repaint policy survived.
        mask: draftCaps.supportsMask ? draft.maskImage : null,
        policy: draftCaps.supportsMask
          ? draft.sourceFit
          : coerceSourceFitForMaskless(draft.sourceFit),
        target: { width: draft.width, height: draft.height },
      },
      {
        ops: domCanvasOps,
        cache: sourceFitCache,
        upscale: (image, model) =>
          upscaleImage({
            model,
            image,
            ...(route ? { target: route.target } : {}),
            onProgress: (message) => (preprocessingStatus.value = message),
          }),
        onStatus: (message) => (preprocessingStatus.value = message),
      },
    );
    // Only the frozen request draft receives processed bytes. Keeping the
    // composer on the user's original source makes repeat submits hit the
    // content-keyed cache and avoids replacing the editable source/mask.
    draft.sourceImage = result.source;
    draft.maskImage = result.mask;
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    toasts.push(`Source preprocessing failed: ${message}`, "error");
    return false;
  } finally {
    preprocessingStatus.value = null;
  }
}

function sourcePreprocessingNeedsRoute(draft: ReturnType<typeof cloneGenerateForm>): boolean {
  const draftCaps = generationCapabilitiesForFamily(draft.family, draft.model);
  return (
    draftCaps.supportsImg2img &&
    draftCaps.sourceImageMode === "single" &&
    Boolean(draft.sourceImage) &&
    draft.sourceFit.mode === "upscale-then-fit" &&
    Boolean(draft.sourceFit.upscalerModel)
  );
}

// Moves in lockstep with ComposerCard's `promptMissing`: one is the disabled
// Generate button, the other this silent early return, and an enabled control
// that quietly does nothing is exactly the dead end the prepared-expansion
// invariant forbids.
const promptMissing = computed(() => promptRequired(form) && !form.prompt.trim());
const h3AuthoringError = computed(() =>
  minimaxH3AuthoringError(form.family, form.model, form.h3Authoring),
);

const emptyCanvasGuidance = computed(() =>
  promptRequired(form)
    ? "Describe an image below, pick a look, and press Generate. Everything runs on your own machine."
    : OPTIONAL_PROMPT_GUIDANCE,
);

async function generate() {
  if (
    promptMissing.value ||
    h3AuthoringError.value ||
    !form.model ||
    chainReject.value ||
    expansionRunning.value ||
    preparedSubmitting.value ||
    submissionPlanning.value
  )
    return;
  const prepared = preparedBatch.value;
  if (
    prepared &&
    (preparedStaleReasons.value.length > 0 ||
      prepared.prompts.some((prompt) => !prompt.text.trim()))
  ) {
    return;
  }
  if (quickExpansionSnapshot.value && quickStaleReasons.value.length > 0) {
    return;
  }

  const preparedSubmission = prepared
    ? {
        batchId: prepared.batchId,
        batch: prepared.prompts.length,
        promptIds: prepared.prompts.map((prompt) => prompt.id),
        prompts: prepared.prompts.map((prompt) => prompt.text.trim()),
        originalPrompt: prepared.rootPrompt ?? prepared.sourcePrompt,
        promptTransform:
          prepared.kind === "remix"
            ? {
                operation: "remix" as const,
                ...(prepared.rootPrompt ? { root_prompt: prepared.rootPrompt } : {}),
                source_prompt: prepared.sourcePrompt,
                source_kind: prepared.sourceKind ?? "direct",
                task: prepared.task,
                dimensions: [...(prepared.dimensions ?? [])],
              }
            : undefined,
        promptTransforms:
          prepared.kind === "remix"
            ? prepared.prompts.map((prompt) => ({
                operation: "remix" as const,
                ...(prepared.rootPrompt ? { root_prompt: prepared.rootPrompt } : {}),
                source_prompt: prepared.sourcePrompt,
                source_kind: prepared.sourceKind ?? "direct",
                task: prepared.task,
                dimensions: [...(prompt.dimensions ?? [])],
              }))
            : undefined,
        route: { ...prepared.route, target: { ...prepared.route.target } },
      }
    : null;
  const quickSubmission = !preparedSubmission
    ? quickExpansionSnapshot.value
      ? {
          requestToken: quickExpansionSnapshot.value.requestToken,
          route: quickRouteIsCurrent.value
            ? {
                ...quickExpansionSnapshot.value.route,
                target: { ...quickExpansionSnapshot.value.route.target },
              }
            : null,
        }
      : null
    : null;
  const submitToken = submissionGuard.begin();
  preparedSubmitting.value = preparedSubmission !== null;
  submissionPlanning.value = true;
  try {
    const draft = cloneGenerateForm(form);
    const draftCaps = generationCapabilitiesForFamily(draft.family, draft.model);
    // The composer style preset is baked into the OUTGOING request at submit —
    // the textarea and negative field are never mutated. Reviewed prepared
    // prompts ship verbatim (the style already reached them through the
    // expansion directive; staleness pins the chip to the frozen style), so
    // the prompt half only applies to the ordinary path. The preset negative
    // is separate from the reviewed prompt text and merges for BOTH paths,
    // gated on the family's negative-prompt support.
    const styled = composeStyle(draft.prompt, draft.stylePreset, {
      supportsNegativePrompt: draftCaps.supportsNegativePrompt,
      negative: draft.negativePrompt,
    });
    if (!preparedSubmission) draft.prompt = styled.prompt;
    draft.negativePrompt = styled.negative ?? "";
    const batch = preparedSubmission
      ? preparedSubmission.batch
      : draftCaps.forcesBatchSizeOne
        ? 1
        : draft.batchSize;
    // With multiple live hosts — or a dead primary while another host can
    // serve — route the batch (sticky pick, Auto = least busy, or Most
    // capable) — model-aware, so hosts that already have the weights win.
    // A pinned host that went away is an error, not a reroute. Only
    // upscale-then-fit needs a route before preprocessing; local fit policies
    // can finalize the source first and avoid a duplicate placement preview.
    let route: HostRoute | null = preparedSubmission?.route ?? quickSubmission?.route ?? null;
    const routeRequiredForPreprocessing = sourcePreprocessingNeedsRoute(draft);
    let routeResolvedAgainstFinalRequest = false;
    let sourcePreprocessed = false;
    if (!route && !routeRequiredForPreprocessing) {
      if (!(await preprocessSourceFit(null, draft))) return;
      if (!submissionGuard.isCurrent(submitToken)) return;
      sourcePreprocessed = true;
    }
    const preliminaryRequest = buildRequest(draft);
    const preliminaryRouting = decideGenerateRequestRouting(preliminaryRequest, draft.family);
    if (preliminaryRouting.kind === "reject") {
      toasts.push(preliminaryRouting.reason, "error");
      return;
    }
    if (
      preliminaryRouting.kind === "chain" &&
      unsupportedAutoChainFields(preliminaryRequest).length > 0
    ) {
      toasts.push(
        "Long-video chaining can’t preserve the selected advanced options. Remove them or reduce Frames to 97 or fewer.",
        "error",
      );
      return;
    }
    const planningRequest =
      preliminaryRouting.kind === "chain"
        ? buildAutoChainRequest(preliminaryRequest, preliminaryRouting)
        : preliminaryRequest;
    if (route) {
      // Prepared work already froze the concrete host. Preserve that
      // authority through source preprocessing and perform exactly one
      // placement preview against the finalized request below.
      const feasible = hosts.resolveRoute(route.hostId);
      if (
        !feasible ||
        feasible.hostId !== route.hostId ||
        feasible.target.baseUrl !== route.target.baseUrl ||
        feasible.target.apiKey !== route.target.apiKey ||
        (feasible.instanceId ?? null) !== (route.instanceId ?? null)
      ) {
        toasts.push(
          "The prepared machine no longer has an authoritative route for this print. Nothing was queued.",
          "error",
        );
        return;
      }
      route = feasible;
    } else {
      const feasibility = await hosts.resolveFeasible(
        appPrefs.settings?.generateTargetHost ?? null,
        planningRequest,
        batch,
      );
      if (feasibility.kind !== "route") {
        toasts.push(placementFailureMessage(feasibility), "error");
        return;
      }
      route = feasibility.route;
      routeResolvedAgainstFinalRequest = sourcePreprocessed;
    }
    if (!sourcePreprocessed) {
      if (!(await preprocessSourceFit(route, draft))) return;
      if (!submissionGuard.isCurrent(submitToken)) return;
    }
    if (preparedSubmission) {
      const current = preparedBatch.value;
      const unchanged =
        current?.batchId === preparedSubmission.batchId &&
        current.prompts.length === preparedSubmission.prompts.length &&
        current.prompts.every(
          (prompt, index) =>
            prompt.id === preparedSubmission.promptIds[index] &&
            prompt.text.trim() === preparedSubmission.prompts[index],
        );
      if (!unchanged || preparedStaleReasons.value.length > 0) {
        expansionError.value =
          "Prepared inputs changed while source preprocessing was running. Nothing was queued; refresh or discard the preserved batch.";
        return;
      }
    } else if (quickSubmission) {
      const current = quickExpansionSnapshot.value;
      if (
        !current ||
        current.requestToken !== quickSubmission.requestToken ||
        quickStaleReasons.value.length > 0
      ) {
        // Restore/re-expand/snapshot replacement intentionally superseded
        // this queued intent. Preserve whichever newer quick state exists.
        return;
      }
    }
    const request = buildRequest(draft);
    if (quickExpansionSnapshot.value?.promptTransform) {
      request.prompt_transform = quickExpansionSnapshot.value.promptTransform;
    }
    const chainRouting = decideGenerateRequestRouting(request, draft.family);
    if (chainRouting.kind === "reject") {
      toasts.push(chainRouting.reason, "error");
      return;
    }
    if (chainRouting.kind === "chain" && unsupportedAutoChainFields(request).length > 0) {
      toasts.push(
        "Long-video chaining can’t preserve the selected advanced options. Remove them or reduce Frames to 97 or fewer.",
        "error",
      );
      return;
    }
    const finalizedPlanningRequest =
      chainRouting.kind === "chain" ? buildAutoChainRequest(request, chainRouting) : request;
    if (route && !routeResolvedAgainstFinalRequest) {
      const finalized = await hosts.resolveFeasible(route.hostId, finalizedPlanningRequest, batch);
      const finalizedRoute = finalized.kind === "route" ? finalized.route : null;
      if (
        !finalizedRoute ||
        finalizedRoute.hostId !== route.hostId ||
        finalizedRoute.target.baseUrl !== route.target.baseUrl ||
        finalizedRoute.target.apiKey !== route.target.apiKey ||
        (finalizedRoute.instanceId ?? null) !== (route.instanceId ?? null)
      ) {
        toasts.push(
          finalized.kind === "route"
            ? "The selected machine changed while the finalized source request was being checked. Nothing was queued."
            : placementFailureMessage(finalized),
          "error",
        );
        return;
      }
      route = finalizedRoute;
    }
    // Stash exact img2img/Qwen-edit bytes by the hashes the server records so
    // Reuse settings can restore local files and fitted sources later.
    // Fire-and-forget — never blocks the submit.
    for (const sourceB64 of [
      ...(request.source_image ? [request.source_image] : []),
      ...(request.edit_images ?? []),
    ]) {
      void sha256HexOfBase64(sourceB64)
        .then((sha) => ipc.sourceStashPut(sha, sourceB64))
        .catch(() => {});
    }
    // Submitting while another print develops queues server-side; each job
    // snapshots its own model + params, so tweaking the form afterwards is safe.
    const requestOptions = preparedSubmission
      ? {
          prompts: preparedSubmission.prompts,
          originalPrompt: preparedSubmission.originalPrompt,
          batchId: preparedSubmission.batchId,
          ...(preparedSubmission.promptTransform
            ? { promptTransform: preparedSubmission.promptTransform }
            : {}),
          ...(preparedSubmission.promptTransforms
            ? { promptTransforms: preparedSubmission.promptTransforms }
            : {}),
        }
      : {};
    const { settled } = generation.submitBatch(request, batch, route, chainRouting, requestOptions);
    const acceptedSubmissionId = ++latestAcceptedSubmissionId;
    missingModel.value = null;
    if (preparedSubmission) {
      preparationGuard.invalidate();
      preparedBatch.value = null;
      expansionError.value = null;
      expansionMissingModel.value = null;
      // The store has synchronously snapshotted every sibling and its exact
      // route. The composer no longer owns this work: release it immediately
      // so another batch can be prepared while these jobs remain queued or
      // generating. Completion feedback is handled by the detached promise
      // below.
      preparedSubmitting.value = false;
      void nextTick(() => composerRef.value?.focus?.());
    }
    if (
      !quickSubmission ||
      quickExpansionSnapshot.value?.requestToken === quickSubmission.requestToken
    ) {
      quickExpansionSnapshot.value = null;
    }
    composerRef.value?.record?.(preparedSubmission?.originalPrompt ?? request.prompt);
    void settled.then((done) => {
      void loadPromptHistory();
      const ok = done.filter((s) => s.status === "complete").length;
      const failedCount = done.filter((s) => s.status === "error").length;
      const failed = done.find((s) => s.status === "error");
      if (ok > 0) {
        if (failedCount > 0) {
          toasts.push(
            `Generated ${ok} of ${done.length} variations. ${failedCount} failed; successful prints were saved to Gallery.`,
            "error",
          );
        } else {
          toasts.push(
            ok === 1 ? "Generated, saved to Gallery" : `Generated ${ok} prints, saved to Gallery`,
          );
        }
        // Gallery refresh is handled by the generation store's complete hook
        // (per-origin bucket) plus the SSE / fallback-poll paths.
      } else if (failed?.error && failed.error !== "Cancelled") {
        // A 404 also fires on proxy/base-URL mismatches — only offer the pull
        // when the availability snapshot doesn't CONTRADICT "model missing"
        // (unknown availability still offers; the pull endpoint will say no).
        const routedId = route?.hostId ?? "local";
        const hostSaysInstalled =
          (hostModels.byHost[routedId]?.fetchedAt ?? 0) > 0 &&
          hostModels.installedOn(routedId).some((m) => m.name === request.model);
        if (
          acceptedSubmissionId === latestAcceptedSubmissionId &&
          isMissingModelError(failed.error) &&
          !hostSaysInstalled
        ) {
          // The routed host doesn't have the model — offer pull-and-resume
          // instead of the raw HTTP error.
          missingModel.value = {
            model: request.model,
            route,
            request,
            batch,
            chainRouting: chainRouting.kind === "chain" ? chainRouting : null,
            requestOptions,
          };
        }
      }
    });
  } finally {
    preparedSubmitting.value = false;
    submissionPlanning.value = false;
  }
}

async function loadPromptHistory() {
  try {
    const targets: HistoryHostTarget[] = hosts.all.flatMap((host) =>
      host.status === "ready" && host.baseUrl
        ? [
            {
              hostId: host.id,
              label: host.label,
              target: { baseUrl: host.baseUrl, apiKey: host.apiKey },
            },
          ]
        : [],
    );
    const history = await fetchHistoryAll(targets);
    promptHistory.value = history.entries.map((entry) => entry.prompt);
  } catch {
    // No history API (older engine / DB off) — arrows just move the caret.
  }
}

// Auto-select a default model only when none is set. With the form persisted
// in a store, the `!form.model` guard now means "the first ever visit" — a
// remount after the user chose a model leaves their choice untouched.
watch(
  () => installedModels.value,
  (installed) => {
    if (!form.model && installed.length > 0) {
      const preferred = preferredInstalledModel(installed);
      if (preferred) formStore.applyModel(preferred);
    }
  },
  { immediate: true },
);

// Refetch on every visit — a pull may have finished since the last look.
watch(
  () => conn.ready,
  (ready) => {
    if (ready) {
      void models.fetch();
      void loadPromptHistory();
    }
  },
  { immediate: true },
);

// The boot fetch usually races the remote hosts' reconnect and captures only
// this device's prompts; refetch the merged history whenever the set of
// ready hosts changes so ↑/↓ recall spans every connected machine.
watch(
  () => readyHostSignature(hosts.all),
  () => {
    if (conn.ready) void loadPromptHistory();
  },
);

/** Monotonic token: only the latest prefill's async source restore may touch
 *  the form — a superseded restore (newer prefill, ⌘N, user edits) is
 *  dropped silently. Bumped by every prefill and by ⌘N. */
let restoreEpoch = 0;

function applyPrefill() {
  const prefill = composer.take();
  if (!prefill) return;
  restoreEpoch += 1;
  // A gallery/history metadata prefill represents one rendered print. Sequence
  // prints use the separate sequence handoff above; every other print must
  // restore into One shot even when the persisted Create draft is currently a
  // sequence. Switch first so the sequence-only model guard cannot replace the
  // print's model after its settings land.
  if ("metadata" in prefill && prefill.metadata && isSequence.value) {
    draft.setOutput(
      "single",
      { getPrompt: () => form.prompt, setPrompt: (value) => (form.prompt = value) },
      sequenceDefaultFrames.value,
    );
    draft.stopEditing();
    draft.lastSingleModel = null;
  }
  // Gallery reuse ships full metadata (full-fidelity restore); palette /
  // history / jobs keep the legacy scalar copy.
  applyPrefillToForm(form, prefill, installedModels.value);
  if ("metadata" in prefill && prefill.metadata) {
    // A first/last-frame print restores every knob except its closing still:
    // saved metadata records each keyframe's name and digest, never the bytes
    // (`applyMetadataToForm` already cleared `form.endFrame`). Say so, the same
    // way an unrestorable source video does, rather than letting Generate look
    // ready to reproduce the render.
    const endFrameNotice = firstLastFrameRestoreNotice(
      caps.value.supportsEndFrame,
      prefill.metadata.keyframes,
      // A first/last print carries its opening frame only in keyframes[0];
      // the stash/gallery restore below keys on source provenance, so
      // without it both endpoints need reattaching and the notice says so.
      Boolean(prefill.metadata.source_image_sha256 ?? prefill.metadata.source_image_name),
    );
    if (endFrameNotice) toasts.push(endFrameNotice, "error");
    void restorePrefillSource(prefill.metadata, restoreEpoch);
  }
  void nextTick(() => composerRef.value?.focus?.());
}

/**
 * Best-effort input-image restore for Reuse settings: local source stash by
 * sha first (covers uploads and canvas-fitted sources), then a cross-host
 * gallery filename match fetched from the print's own origin. Old prints
 * without provenance keys are silently skipped; a keyed print that can't be
 * found gets a toast.
 */
async function restorePrefillSource(metadata: OutputMetadata, epoch: number) {
  if (!metadataReferencesSource(metadata)) return;
  if (!caps.value.supportsImg2img) return;
  const attachmentMode = caps.value.sourceImageMode !== "single";
  if (attachmentMode ? form.imageAttachments.length > 0 : Boolean(form.sourceImage)) return;
  const modelAtStart = form.model;
  const deps: SourceRestoreDeps = {
    stashGet: (sha) => ipc.sourceStashGet(sha),
    galleryLookup: async (filename) => {
      await hostGallery.fetchAll().catch(() => {});
      const entry = hostGallery.merged.find((e) => e.item.filename === filename);
      if (!entry) return null;
      if (hostGallery.mediaSourceOf(entry.sourceKey) === "local") {
        const res = await fetch(localMediaPath(filename));
        if (!res.ok) return null;
        return blobToBase64(await res.blob());
      }
      const target = hostGallery.targetOf(entry.sourceKey);
      const path = mediaPath(filename);
      const res = await (target ? apiFetchTo(target, path) : apiFetch(path));
      return blobToBase64(await res.blob());
    },
  };
  const editRestore = attachmentMode ? await restoreEditImages(metadata, deps) : null;
  const restored = attachmentMode ? null : await restoreSourceImage(metadata, deps);
  // The lookups can take seconds (cold gallery, cross-host fetch). Bail if
  // this restore was superseded: a newer prefill or ⌘N bumped the epoch, the
  // user attached their own source, the model changed under us, or the new
  // family can't take an image at all.
  if (epoch !== restoreEpoch || form.model !== modelAtStart) return;
  if (!caps.value.supportsImg2img) return;
  if (attachmentMode ? form.imageAttachments.length > 0 : Boolean(form.sourceImage)) return;
  if (attachmentMode && caps.value.sourceImageMode !== "single" && editRestore?.images.length) {
    const restoredImages =
      caps.value.sourceImageMode === "references"
        ? editRestore.images.slice(0, 4)
        : editRestore.images;
    const omitted = editRestore.images.length - restoredImages.length;
    form.imageAttachments = restoredImages;
    if (editRestore.missing > 0 || omitted > 0) {
      toasts.push(
        `Restored ${restoredImages.length} source ${
          restoredImages.length === 1 ? "image" : "images"
        }; ${editRestore.missing + omitted} ${
          editRestore.missing + omitted === 1 ? "was" : "were"
        } unavailable or beyond this model's limit.`,
        "error",
      );
    }
  } else if (!attachmentMode && caps.value.sourceImageMode === "single" && restored) {
    form.sourceImage = restored.base64;
    form.sourceImageName = restored.filename;
  } else {
    toasts.push(
      attachmentMode
        ? "Couldn't restore the edit images — the original local files are no longer available."
        : "Couldn't restore the source image — the original file wasn't found on any connected host.",
      "error",
    );
  }
}

// Apply a prefill whenever one arrives (Reuse settings, history, "Generate
// with <model>"), including one already queued before this view mounted.
watch(() => composer.prefill, applyPrefill, { immediate: true });

/**
 * Reuse a sequence print's recorded clips as a BRAND-NEW draft: no edit
 * session, nothing cached, `Generate sequence` queues a fresh job. Shared
 * params ride the same `applyPrefillToForm` path a single print's reuse uses,
 * so model defaults, legacy normalization and capability gating stay in one
 * place.
 */
function applySequenceReuse(metadata: OutputMetadata) {
  const plan = planSequenceReuse(metadata);
  if (!plan) return;
  const oneShotPrompt = form.prompt;
  applyPrefillToForm(form, { metadata }, installedModels.value);
  // Reusing a sequence must not overwrite the parked one-shot prompt.
  form.prompt = oneShotPrompt;

  // The live tail belongs to the model that is selected NOW, not the one the
  // print recorded — raise anything that no longer clears it, and say so.
  const tail = sequenceMotionTailFrames(selectedEntry.value);
  const { clips, raised } = clampClipsToMotionTail(plan.clips, tail, 9);

  draft.stopEditing();
  editSharedBaseline.value = null;
  draft.output = "sequence";
  draft.clips.splice(0, draft.clips.length, ...clips);
  draft.activeClipId = clips[0]?.id ?? null;
  draft.enableAudio = metadata.enable_audio === true;

  const notes = [sequenceReuseNote(clips.length, plan.lossy)];
  if (raised > 0) {
    notes.push(sequenceReuseClampNote(modelDisplayNameForId(form.model, installedModels.value)));
  }
  sequenceReuseNotice.value = notes.join(" · ");
  void loadChainLimits();
}

/** A sequence handed over from elsewhere: Library ▸ History ▸ Sequences and
 *  the canvas hand over `edit`; a Library sequence print hands over `reuse`.
 *  One-shot — the slot is emptied on arrival so a back-nav cannot replay it. */
function applySequenceHandoff() {
  const handoff = composer.takeSequence();
  if (!handoff) return;
  if (handoff.kind === "reuse") {
    applySequenceReuse(handoff.metadata);
    return;
  }
  if (handoff.kind === "inspect") {
    void inspectSequence({ hostId: handoff.hostId, jobId: handoff.jobId });
  } else {
    void editSequence({ hostId: handoff.hostId, jobId: handoff.jobId });
  }
}
watch(() => composer.pendingSequence, applySequenceHandoff, { immediate: true });
// Leaving Sequence retires the caveat with the rail it described.
watch(isSequence, (on) => {
  if (!on) sequenceReuseNotice.value = null;
  // The bench floor is mode-aware; a persisted one-shot height may sit below
  // the sequence floor and would otherwise scroll the composer.
  clampBenchToViewport();
});

// ⌘N — clear the composer for a fresh generation, keeping the model.
watch(
  () => ui.newGenerationTick,
  () => {
    restoreEpoch += 1; // an in-flight source restore must not repopulate ⌘N
    preparationGuard.invalidate();
    preparedBatch.value = null;
    expansionRunning.value = false;
    expansionError.value = null;
    expansionMissingModel.value = null;
    expansionAttemptHostLabel.value = null;
    quickExpansionOriginal.value = null;
    quickExpansionSnapshot.value = null;
    quickExpansionNegative.value = null;
    submissionGuard.invalidate();
    formStore.clearComposer();
    void nextTick(() => composerRef.value?.focus?.());
  },
);

// ⌘R — randomize the seed.
watch(
  () => ui.randomizeSeedTick,
  () => {
    form.seed = String(randomSeed());
  },
);

// Menu ▸ Generate / Expand Prompt reuse the composer actions. In sequence
// output the same intent submits the sequence.
watch(
  () => ui.generateTick,
  () => (isSequence.value ? void generateSequence() : void generate()),
);
watch(
  () => ui.expandTick,
  () => composerRef.value?.expand?.(),
);

onMounted(() => {
  if (!import.meta.env.TEST) liveActivity.start();
  document.addEventListener("pointerdown", onDocumentPointerDown);
  document.addEventListener("keydown", onDocumentKeydown);
  window.addEventListener("resize", clampBenchToViewport);
  clampBenchToViewport();
  void listenForNativeImageDrops();
  // The persisted sequence draft wins on ordinary visits; a ?output=sequence
  // deep-link is consumed once and stripped.
  draft.hydrate();
  consumeOutputQuery();
});

onBeforeUnmount(() => {
  if (!import.meta.env.TEST) liveActivity.stop();
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  clearSequenceStageMedia();
  nativeImageDropUnmounted = true;
  stopNativeImageDrop?.();
  stopBenchResize();
  window.removeEventListener("resize", clampBenchToViewport);
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  document.removeEventListener("keydown", onDocumentKeydown);
});
</script>

<template>
  <StarterCards v-if="showStarterCards" @browse="router.push('/models')" />

  <div v-else data-test="generate-layout" class="relative flex h-full min-h-0 overflow-hidden">
    <div
      v-if="nativeImageDragOver"
      data-test="native-image-drop-overlay"
      class="pointer-events-none absolute inset-3 z-40 flex items-center justify-center rounded-chrome border-2 border-dashed border-safelight bg-bath/90 text-body-lg text-safelight shadow-raised"
    >
      Drop image to load settings and use as source
    </div>

    <!-- Main column: header / canvas / activity / composer -->
    <div class="flex min-w-0 flex-1 flex-col">
      <CreateHeader :form="form" />

      <div
        ref="workbenchRef"
        data-test="generate-workbench"
        class="relative flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        <!-- Templates popover (relocated from the inspector) -->
        <div ref="templatesEl" class="absolute right-3 top-3 z-20">
          <button
            ref="templatesToggleEl"
            type="button"
            data-test="templates-toggle"
            class="border-edge rounded-control border bg-bench/80 px-2.5 py-1 text-caption text-ink-2 backdrop-blur transition-colors hover:text-ink"
            :aria-expanded="templatesOpen"
            @click="templatesOpen = !templatesOpen"
          >
            Templates
          </button>
          <div
            v-if="templatesOpen"
            class="border-edge absolute right-0 mt-1 w-72 rounded-chrome border bg-bench p-3 shadow-raised"
          >
            <TemplatesPanel :form="form" :models="hostModels.unionInstalled" @load="loadTemplate" />
          </div>
        </div>

        <!-- Canvas -->
        <div
          class="flex min-h-[144px] flex-1 items-center justify-center overflow-hidden bg-desk p-7"
        >
          <!-- Prepared variations review (prototype: this replaces the canvas) -->
          <PreparedExpansionBatch
            v-if="preparedBatch"
            :batch="preparedBatch"
            :stale-reasons="preparedStaleReasons"
            :preparing="expansionRunning"
            :error="expansionError"
            :pull-status="expansionPullStatus"
            :pull-model="expansionMissingModel?.model ?? null"
            :pull-host-label="expansionMissingModel?.route.label ?? null"
            :pull-eta-seconds="expansionPullEtaSeconds"
            :active-host-label="expansionAttemptHostLabel"
            :submitting="preparedSubmitting"
            :models="hostModels.unionInstalled"
            class="ms-fade-up w-full max-w-2xl"
            @edit="editPreparedPrompt"
            @remove="removePreparedPrompt"
            @collapse="collapsePreparedBatch"
            @regenerate="
              preparedBatch.kind === 'remix'
                ? remixForCurrentPrompt(true)
                : expandForCurrentBatch(true)
            "
            @refresh="
              preparedBatch.kind === 'remix'
                ? remixForCurrentPrompt(true)
                : expandForCurrentBatch(true)
            "
            @discard="discardPreparedBatch"
            @pull="pullExpansionModel"
            @retry-expansion="retryExpansionAfterPull"
            @generate="generate"
            @apply="applyPreparedRemix"
          />

          <!-- Developing / result -->
          <!-- Must stretch to the canvas: the preview region's flex-1 height
               is what the frame is measured against — a content-sized column
               would collapse the frame to 0×0 (an invisible develop view). -->
          <div v-else-if="job" class="flex h-full w-full min-h-0 flex-col items-center">
            <div
              data-test="preview-region"
              class="grid min-h-0 w-full flex-1 place-items-center self-stretch overflow-hidden [container-type:size]"
            >
              <div
                class="relative max-h-full w-full max-w-full overflow-hidden rounded-media border border-control-edge bg-print-surface"
                data-test="preview-frame"
                :style="previewFrameStyle"
                @contextmenu="contextMenu.open($event, canvasMenu())"
              >
                <!-- Audio is checked first: an audio print has no frames, so
                     the video probe falls through and the <img> below renders
                     a WAV — a broken canvas at the end of a render that
                     actually succeeded. The waveform is the visual; the
                     transport sits over it. -->
                <div
                  v-if="job?.resultUrl && isAudioResult(job)"
                  class="absolute inset-0 flex flex-col items-center justify-center gap-3 p-3"
                  data-test="preview-audio"
                >
                  <img
                    v-if="job.result?.audio_thumbnail"
                    :src="`data:image/png;base64,${job.result.audio_thumbnail}`"
                    alt="Waveform of the generated audio"
                    class="min-h-0 w-full flex-1 object-contain"
                  />
                  <audio class="w-full shrink-0" controls :src="job.resultUrl" />
                </div>
                <video
                  v-else-if="job?.resultUrl && job.result?.video_frames"
                  :src="job.resultUrl"
                  class="absolute inset-0 h-full w-full object-contain"
                  autoplay
                  loop
                  controls
                  disablepictureinpicture
                />
                <img
                  v-else-if="job?.resultUrl"
                  :src="job.resultUrl"
                  alt=""
                  class="absolute inset-0 h-full w-full object-contain transition-opacity duration-500"
                />
                <!-- Live latent preview: a tiny PNG upscaled by CSS; the blur
                     tightens as denoising progresses and the grain resolves
                     over it, so the print literally develops on the canvas. -->
                <img
                  v-if="job && job.status !== 'complete' && job.previewUrl"
                  :src="job.previewUrl"
                  alt=""
                  class="absolute inset-0 h-full w-full object-contain"
                  :style="{ filter: `blur(${Math.max(2, 14 - 12 * jobProgress(job))}px)` }"
                />
                <!-- The grain canvas paints edge-to-edge (temperature wash), so
                     once previews exist it thins out with progress to reveal
                     the forming print underneath. -->
                <DevelopCanvas
                  v-if="job && job.status !== 'complete'"
                  :seed="job.visualSeed"
                  :progress="jobProgress(job)"
                  :phase="jobPhase(job)"
                  class="absolute inset-0"
                  :style="{
                    opacity: job.previewUrl
                      ? String(Math.max(0.18, 1 - jobProgress(job) * 0.9))
                      : '1',
                  }"
                />
                <!-- Grain is the signature; the ring overlays it until the
                     first latent preview arrives, after which the forming
                     print itself takes over. The status stays below the frame
                     where the grain cannot obscure it. -->
                <div
                  v-if="job && job.status !== 'complete' && !job.previewUrl"
                  data-test="develop-progress"
                  class="pointer-events-none absolute inset-0 flex items-center justify-center"
                >
                  <ProgressRing :value="jobProgress(job) * 100" :size="96" show-label />
                </div>
              </div>
            </div>

            <div
              v-if="liveGenerationStatus"
              data-test="generation-live-status"
              class="edge-code mt-2 max-w-full text-center text-safelight"
            >
              {{ liveGenerationStatus }}
            </div>

            <div
              data-test="generation-edge-code"
              class="edge-code mt-2 max-w-full truncate"
              :title="edgeCode"
            >
              {{ edgeCode }}
            </div>

            <!-- Batch dots -->
            <div v-if="siblings.length > 1" class="mt-2 flex items-center gap-1.5">
              <span
                v-for="(s, i) in siblings"
                :key="i"
                class="data-mono text-body"
                :class="siblingDot(s)"
                :title="`Variation ${i + 1} of ${siblings.length}: ${s.status}${s.error ? `. ${s.error}` : ''}`"
                :aria-label="`Variation ${i + 1} of ${siblings.length}: ${s.status}${s.error ? `. ${s.error}` : ''}`"
              >
                {{ s.status === "complete" ? "◉" : s.status === "error" ? "◉" : "◎" }}
              </span>
              <span class="edge-code ml-1">
                {{ siblings.filter((s) => s.status === "complete").length }} of
                {{ siblings.length }}
              </span>
            </div>

            <GenerateErrorNotice
              v-if="job?.status === 'error'"
              class="mt-3 w-full"
              :message="jobErrorMessage"
              :copy-message="jobErrorCopy"
            />
          </div>

          <div
            v-else-if="isSequence && sequencePlaybackSrc"
            data-test="sequence-stage-player"
            class="relative flex h-full w-full min-h-0 flex-col items-center"
          >
            <video
              :key="sequencePlaybackSrc"
              :src="sequencePlaybackSrc"
              class="min-h-0 w-full flex-1 rounded-media border border-control-edge bg-print-surface object-contain"
              autoplay
              controls
              loop
              playsinline
            />
            <div
              class="absolute left-3 top-3 flex items-center gap-2 rounded-control border border-edge bg-bench/90 px-2 py-1.5 shadow-raised backdrop-blur"
            >
              <span class="edge-code">Clip {{ (playingSequenceStage ?? 0) + 1 }}</span>
              <button
                type="button"
                data-test="sequence-return-live"
                class="rounded-control border border-edge px-2 py-1 text-caption text-ink-2 hover:text-ink"
                @click="returnToLiveSequence"
              >
                Return to live render
              </button>
            </div>
          </div>

          <!-- Watched sequence: denoise progress in the develop chrome -->
          <div
            v-else-if="isSequence && watchedSequence"
            data-test="sequence-develop"
            class="pointer-events-none flex flex-col items-center justify-center gap-3"
          >
            <ProgressRing :value="watchedSequencePct" :size="96" show-label />
            <span class="edge-code text-safelight">
              clip {{ (chains.live.activeStage ?? watchedSequence.current_stage) + 1 }}/{{
                watchedSequence.stage_count
              }}
              · developing…
            </span>
          </div>

          <!-- Settled sequence: the canvas holds the result, because the
               activity strip no longer does. -->
          <div
            v-else-if="isSequence && settledSequence"
            data-test="sequence-result"
            class="flex h-full w-full min-h-0 flex-col items-center"
          >
            <div
              v-if="settledSequencePrint"
              data-test="sequence-result-stage"
              class="grid min-h-0 w-full flex-1 place-items-center self-stretch overflow-hidden [container-type:size]"
            >
              <div
                data-test="sequence-result-frame"
                class="relative max-h-full w-full max-w-full overflow-hidden rounded-media border border-control-edge bg-print-surface"
                :style="settledFrameStyle"
              >
                <AuthedMedia
                  class="absolute inset-0"
                  video
                  controls
                  :path="
                    galleryMediaPath(
                      settledSequencePrint.item.filename,
                      hostGallery.mediaSourceOf(settledSequencePrint.sourceKey),
                    )
                  "
                  :target="hostGallery.targetOf(settledSequencePrint.sourceKey)"
                  :cache-key="settledSequencePrint.sourceKey"
                  :alt="settledSequenceCaption"
                />
              </div>
            </div>
            <GenerateErrorNotice
              v-else-if="settledSequenceError"
              data-test="sequence-failed"
              class="w-full"
              :message="settledSequenceError"
              :copy-message="settledSequenceErrorCopy"
            />
            <div v-else class="grid min-h-0 w-full flex-1 place-items-center">
              <span class="edge-code text-ink-3">saved to Library</span>
            </div>

            <div class="edge-code mt-2 max-w-full truncate" :title="settledSequenceCaption">
              {{ settledSequenceCaption }}
            </div>
            <div class="mt-2 flex items-center gap-2">
              <button
                v-if="settledSequenceError"
                type="button"
                data-test="sequence-resume"
                class="rounded-control bg-stop px-3 py-1 text-body font-semibold text-on-accent transition-colors hover:brightness-105 active:translate-y-px"
                @click="resumeSettledSequence"
              >
                Resume
              </button>
              <button
                type="button"
                data-test="sequence-edit"
                class="border-ce rounded-control border px-3 py-1 text-body text-ink-2 transition-colors hover:text-rebate"
                @click="editSettledSequence"
              >
                Edit sequence
              </button>
              <button
                type="button"
                data-test="sequence-show-in-library"
                class="border-ce rounded-control border px-3 py-1 text-body text-ink-2 transition-colors hover:text-rebate"
                @click="showSettledSequenceInLibrary"
              >
                Show in library
              </button>
            </div>
          </div>

          <!-- Empty -->
          <EmptyStateBlock
            v-else
            data-test="empty-canvas"
            brand
            icon="image"
            headline="Your print develops here"
            :guidance="emptyCanvasGuidance"
          />
        </div>

        <!-- Batch-1 expansion status (prepared batches carry their own inline
             surfaces; these are the composer-level ones). -->
        <GenerateErrorNotice
          v-if="quickStaleReasons.length && !preparedBatch"
          data-test="quick-expansion-stale"
          class="mx-5 mb-2"
          :dismissible="false"
          :message="quickStaleMessage"
        >
          <template #actions>
            <button
              type="button"
              data-test="reexpand-and-generate"
              class="rounded-control bg-stop px-3 py-1.5 text-body font-semibold text-on-accent transition-colors hover:brightness-105 active:translate-y-px disabled:opacity-50"
              :disabled="expansionRunning || preparedSubmitting"
              @click="reexpandAndGenerate"
            >
              Re-expand for {{ currentModelLabel }} and generate
            </button>
            <button
              type="button"
              data-test="generate-expanded-anyway"
              class="border-stop/50 rounded-control border px-3 py-1.5 text-body font-medium text-stop transition-colors hover:bg-stop/10 active:translate-y-px disabled:opacity-50"
              :disabled="expansionRunning || preparedSubmitting"
              @click="generateExpandedAnyway"
            >
              Generate expanded prompt anyway
            </button>
            <button
              type="button"
              data-test="restore-expanded-original"
              class="rounded-control px-3 py-1.5 text-body text-ink-2 transition-colors hover:text-ink active:translate-y-px"
              @click="restoreQuickExpansion"
            >
              Restore original
            </button>
          </template>
        </GenerateErrorNotice>
        <GenerateErrorNotice
          v-else-if="expansionError && !preparedBatch && !expansionMissingModel"
          class="mx-5 mb-2"
          :message="expansionError"
        />
        <ExpansionPullStatus
          v-if="expansionError && expansionMissingModel && expansionPullStatus && !preparedBatch"
          :model="expansionMissingModel.model"
          :host-label="expansionMissingModel.route.label"
          :error="expansionError"
          :status="expansionPullStatus"
          :eta-seconds="expansionPullEtaSeconds"
          :models="hostModels.unionInstalled"
          class="mx-5 mb-2"
          @pull="pullExpansionModel"
          @retry-expansion="retryExpansionAfterPull"
        />

        <div
          data-test="create-bench-resizer"
          class="group relative z-10 flex h-3 shrink-0 cursor-row-resize touch-none items-center justify-center border-y border-edge bg-bath/80"
          role="separator"
          aria-label="Resize Activity and sequence editor"
          aria-orientation="horizontal"
          :aria-valuenow="benchHeight"
          :aria-valuemin="minBenchHeight()"
          :aria-valuemax="
            Math.max(minBenchHeight(), (workbenchRef?.clientHeight ?? 0) - MIN_CANVAS_HEIGHT)
          "
          tabindex="0"
          title="Drag to resize · double-click to reset"
          @pointerdown="startBenchResize"
          @keydown="onBenchResizeKeydown"
          @dblclick="setBenchHeight(DEFAULT_BENCH_HEIGHT)"
        >
          <span
            class="h-1 w-12 rounded-full bg-ink-3/50 transition-colors group-hover:bg-safelight group-focus-visible:bg-safelight"
            aria-hidden="true"
          />
        </div>

        <div
          data-test="create-bottom-panel"
          class="flex min-h-0 shrink-0 flex-col overflow-hidden bg-desk"
          :style="{
            height: `${benchHeight}px`,
            containerType: 'size',
            containerName: 'create-bench',
          }"
        >
          <ActivityStrip @edit-sequence="editSequence" />
          <p
            v-if="isSequence && sequenceReuseNotice"
            data-test="sequence-reuse-note"
            class="edge-code shrink-0 px-1 pt-1.5 text-ink-3"
          >
            {{ sequenceReuseNotice }}
          </p>

          <!-- Sequence bench replaces the single-print composer in-place -->
          <EmptyStateBlock
            v-if="showSequenceEmpty"
            data-test="sequence-empty"
            class="flex-1 py-6"
            icon="image"
            headline="Sequences need a video model"
            guidance="Pull a chain-capable LTX Video or distilled LTX-2 checkpoint, then tell the story one clip at a time."
          >
            <template #action>
              <button
                type="button"
                data-test="sequence-browse-models"
                class="rounded-control bg-safelight px-3 py-1.5 text-body font-semibold text-on-accent"
                @click="
                  router.push('/models?tab=discover&type=video&kind=checkpoint&intent=sequence')
                "
              >
                Browse video models
              </button>
            </template>
          </EmptyStateBlock>
          <!-- Protect the sequence's hard chrome floor from Activity. The
               composer itself stays min-h-0 so its filmstrip can squash, but
               this parent flex item makes Activity yield before the footer or
               Generate button can clip at the supported 390px bench floor. -->
          <div
            v-else-if="isSequence"
            data-test="generate-sequence-shell"
            class="flex min-h-[300px] flex-[1_0_300px] overflow-hidden"
          >
            <SequenceComposer
              data-test="generate-sequence-composer"
              class="min-h-0 flex-1"
              :form="form"
              :selected-model="selectedSequenceEntry"
              :chain-limits="chainLimits"
              :installed-models="installedModels"
              :submitting="sequenceSubmitting"
              :chain-level-dirty="chainLevelDirty"
              :stage-media-by-clip-id="sequenceFilmstripMediaByClipId"
              :playing-clip-id="playingSequenceClipId"
              :target="sequenceTarget"
              @submit="generateSequence"
              @duplicate="duplicateSequenceAsNew"
              @play-clip="playSequenceClip"
            />
          </div>
          <ComposerCard
            v-else
            ref="composerRef"
            data-test="generate-composer"
            class="flex-1"
            :form="form"
            :effective-batch-size="effectiveBatchSize"
            :expansion-running="expansionRunning"
            :expansion-host-label="expansionHostLabel"
            :can-undo="quickExpansionOriginal !== null"
            :prepared-blocked="!!preparedBatch && effectiveBatchSize === 1"
            :has-prepared="!!preparedBatch"
            :chain-reject="chainReject"
            :submitting="submissionPlanning"
            :button-label="buttonLabel"
            :estimate-request="estimateRequest"
            :estimate-target="estimateTarget"
            :preprocessing-status="preprocessingStatus"
            :history="promptHistory"
            :remix-source="remixSource"
            @generate="generate"
            @expand="expandForCurrentBatch()"
            @remix="remixForCurrentPrompt()"
            @update:remix-source="remixSource = $event"
            @restore="restoreQuickExpansion"
          />
        </div>
      </div>
    </div>

    <!-- Inspector (persisted, left-edge resizable width) -->
    <InspectorPanel
      :form="form"
      :last-seed="generation.lastSeedUsed"
      :chain-limits="chainLimits"
      @append-word="appendPromptWord"
    />

    <MissingModelDialog
      v-if="missingModel"
      :model="missingModel.model"
      :host-label="missingModelHostLabel"
      :size-gb="missingModelSizeGb"
      :models="hostModels.unionInstalled"
      @confirm="pullMissingModel"
      @close="missingModel = null"
    />
    <VideoExportDialog
      v-if="videoExportJob?.result?.filename"
      :open="true"
      :filename="videoExportJob.result.filename"
      :formats="videoExportCapabilities.formats"
      :busy="videoExportBusy"
      :error="videoExportError"
      @close="videoExportJob = null"
      @export="exportGeneratedVideo"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import { expandPrompt } from "../lib/api/expand";
import { summarizeStatusGpuMemory } from "../lib/api/gpuStatus";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { createUuid } from "@studio/lib/id";
import { modelSupportsSequence } from "@studio/lib/sequence";
import { upscaleImage } from "../lib/api/upscale";
import { generationCapabilitiesForFamily, outputFormatsForFamily } from "../lib/capabilities";
import { modelDisplayName, modelDisplayNameForId } from "../lib/models";
import type {
  CompleteEvent,
  ChainJobDetail,
  ChainRequest,
  CreateChainJobResponse,
  DownloadJob,
  ExpandCapabilities,
  GalleryImage,
  GenerateRequest,
  ModelEntry,
  ServerStatus,
} from "../lib/api/types";
import {
  buildGenerationEstimateRequest,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
} from "../lib/chainRouting";
import {
  applyModelDefaults,
  buildRequest,
  cloneGenerateForm,
  newGenerateForm,
  reconcileModelCapabilities,
  resetFormToModelDefaults,
  type GenerateForm,
} from "../lib/generateForm";
import { formatTemplateMediaReferences, type GenerationTemplate } from "../lib/generationTemplates";
import { galleryMediaPath, isVideoItem } from "../lib/gallery/media";
import { isUpscaledImage } from "../lib/gallery/upscaled";
import { percent } from "../lib/format";
import { composeStyle, mergeStyleNegative, styleHint } from "../lib/stylePresets";
import {
  guidanceValidationError,
  inlineGenerationMediaBytes,
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  MOBILE_MEDIA_BUDGET_ERROR,
  mobileMediaBudgetValidationError,
  stepsValidationError,
} from "../lib/generateValidation";
import { blobToBase64, isStillImageFile } from "../lib/image";
import { parseMissingExpandModel } from "../lib/expandErrors";
import {
  expansionPullJobMatchesModel,
  resolveExpansionPullStatus,
  type ExpansionPullPhase,
  type ExpansionPullView,
} from "../lib/expansionPull";
import type { DownloadsState } from "../lib/downloads";
import {
  PreparationRequestGuard,
  createPreparedExpansionBatch,
  preparedExpansionStaleReasons,
  quickExpansionStaleReasons,
  validateExpandedPrompts,
  type PreparedExpansionBatch as PreparedExpansionBatchState,
  type PreparedExpansionInputs,
  type QuickExpansionSnapshot,
} from "../lib/preparedExpansion";
import { isGenerationModel } from "../stores/models";
import type { HostRoute } from "../stores/hosts";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";
import {
  isCancelledError,
  jobPhase,
  jobProgress,
  jobStatusCode,
  railOrder,
  useGenerationStore,
  type Job,
} from "../stores/generation";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import { mobileHostTarget, normalizeRemoteAddress, remoteHostId, type MobileHost } from "./hosts";
import { applyMobileGalleryMetadata } from "./reuse";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileCatalogView from "./MobileCatalogView.vue";
import MobileExpansionPullStatus from "./MobileExpansionPullStatus.vue";
import MobileGalleryViewer from "./MobileGalleryViewer.vue";
import MobileGenerateParameters from "./MobileGenerateParameters.vue";
import MobileHostDetail from "./MobileHostDetail.vue";
import MobileLoraControls from "./MobileLoraControls.vue";
import MobilePromptTools from "./MobilePromptTools.vue";
import MobilePreparedExpansionBatch from "./MobilePreparedExpansionBatch.vue";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";
import MobileSeedPicker from "./MobileSeedPicker.vue";
import MobileSequenceComposer from "./MobileSequenceComposer.vue";
import MobileSettingsView from "./MobileSettingsView.vue";
import MobileSourceControls from "./MobileSourceControls.vue";
import MobileStyleChips from "./MobileStyleChips.vue";
import MobileTemplates from "./MobileTemplates.vue";
import {
  loadMobileSettings,
  updateMobileSettings as persistMobileSettings,
  type MobileSettings,
} from "./settings";
import { useMobileDownloadsStore } from "./mobileDownloads";
import {
  createMobileExpansionRecovery,
  mobileExpansionRecoveryStaleReason,
  type MobileExpansionRecoveryRecord,
} from "./mobileExpansionRecovery";
import { reconcileInterruptedGenerationJobs } from "./mobileGenerationRecovery";

type Tab = "generate" | "gallery" | "catalog" | "hosts";
type CreateMode = "single" | "sequence";

interface DiscoveredHost {
  name: string;
  host: string;
  port: number;
}

interface GalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  target: ApiTarget;
  thumbnailUrl: string;
}

interface PendingGalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  target: ApiTarget;
}

interface MobileExpansionPullAttempt {
  id: number;
  recoveryId: number;
  phase: Exclude<ExpansionPullPhase, "missing">;
  jobId: string | null;
  observedJobId: string | null;
  baselineJobIds: string[];
  allowExistingInFlight: boolean;
  requestError: string | null;
  terminalJob: DownloadJob | null;
}

const STORAGE_KEY = "mold.mobile.hosts.v1";
const SELECTED_KEY = "mold.mobile.selected-host.v1";
const LIBRARY_SEEN_AT_KEY = "mold.mobile.library-seen-at.v1";
const LEGACY_LIBRARY_SEEN_KEY = "mold.mobile.library-seen.v1";
const LIBRARY_VISITED_KEY = "mold.mobile.library-visited.v1";
const CREATE_MODE_KEY = "mold.mobile.create-mode.v1";
const SEQUENCE_RECOVERY_KEY = "mold.mobile.sequence-job.v1";
const HOST_PROBE_TIMEOUT_MS = 9_000;
const tab = ref<Tab>("generate");
const savedCreateMode = localStorage.getItem(CREATE_MODE_KEY);
const createMode = ref<CreateMode>(savedCreateMode === "sequence" ? "sequence" : "single");
const mobileContent = ref<HTMLElement | null>(null);
const settingsOpen = ref(false);
const settingsButton = ref<HTMLButtonElement | null>(null);
const settingsBackButton = ref<HTMLButtonElement | null>(null);
const mobileSettings = reactive<MobileSettings>(loadMobileSettings());
const appVersion = ref(import.meta.env.DEV ? "Development build" : "Current build");
const hosts = ref<MobileHost[]>(loadHosts());
const selectedHostId = ref(localStorage.getItem(SELECTED_KEY) ?? hosts.value[0]?.id ?? "");
const catalogHostId = ref(selectedHostId.value || hosts.value[0]?.id || "");
const hostDetailId = ref("");
const hostInput = reactive({ name: "", address: "", apiKey: "" });
const discovered = ref<DiscoveredHost[]>([]);
const discovering = ref(false);
const hostError = ref("");
const models = ref<ModelEntry[]>([]);
const modelsHostId = ref("");
const loadingModels = ref(false);
const modelLoadError = ref("");
const sequenceJob = ref<ChainJobDetail | null>(null);
const sequenceStarting = ref(false);
const sequenceError = ref("");
let sequencePollTimer: ReturnType<typeof setInterval> | null = null;
let sequenceRoute: { hostId: string; target: ApiTarget } | null = null;
const expandCapabilities = reactive<Record<string, ExpandCapabilities | null | undefined>>({});
const form = reactive<GenerateForm>(newGenerateForm());
const seedValid = ref(true);
const parameterValid = ref(true);
const sourceValid = ref(true);
const resolutionValid = ref(true);
const advancedSheetOpen = ref(false);

/** Count of advanced settings that differ from their defaults — drives the
 * "Advanced" trigger badge and the sheet header badge. */
const advancedActiveCount = computed(() => {
  let count = 0;
  if (form.negativePrompt.trim()) count += 1;
  if (form.sourceImage || form.controlImage || form.imageAttachments.length) count += 1;
  if (form.loras.length) count += 1;
  if (form.upscaleModel) count += 1;
  if (form.scheduler !== "default") count += 1;
  if (form.cfgPlus) count += 1;
  return count;
});

function openAdvancedSheet(): void {
  advancedSheetOpen.value = true;
}

function closeAdvancedSheet(): void {
  advancedSheetOpen.value = false;
}

/** Restore every generation knob to the selected model's defaults, keeping the
 * prompt, the model, and any prepared batch size. Same contract as the desktop
 * inspector's Reset — the sheet's scoped reset below is deliberately narrower. */
function resetCreateSettings(): void {
  resetFormToModelDefaults(form, selectedGenerationModel.value);
}

/** Restore only the advanced-tier fields to their defaults; prompt, model,
 * dimensions, steps, guidance, seed, and batch are left untouched. */
function resetAdvancedSettings(): void {
  const defaults = newGenerateForm();
  form.negativePrompt = defaults.negativePrompt;
  form.scheduler = defaults.scheduler;
  form.cfgPlus = defaults.cfgPlus;
  form.upscaleModel = defaults.upscaleModel;
  form.loras = [];
  form.strength = defaults.strength;
  form.sourceImage = defaults.sourceImage;
  form.sourceImageName = defaults.sourceImageName;
  form.imageAttachments = [];
  form.sourceFit = { mode: "pad-repaint" };
  form.maskImage = defaults.maskImage;
  form.controlImage = defaults.controlImage;
  form.controlModel = defaults.controlModel;
  form.controlScale = defaults.controlScale;
}
const preparingGeneration = ref(false);
const preparedBatch = ref<PreparedExpansionBatchState | null>(null);
const expansionRunning = ref(false);
const expansionError = ref("");
const expansionRecovery = ref<MobileExpansionRecoveryRecord | null>(null);
const expansionPullAttempt = ref<MobileExpansionPullAttempt | null>(null);
const quickExpansionOriginal = ref<string | null>(null);
const quickExpansionSnapshot = ref<QuickExpansionSnapshot | null>(null);
/**
 * The negative prompt before and after a bake-and-clear merged the preset's
 * curated fragments into it. Undo re-arms `before` alongside the prompt and
 * chip; `baked` lets it bow out when the user has since edited the field.
 */
const quickExpansionNegative = ref<{ before: string; baked: string } | null>(null);
const preparedSubmitting = ref(false);
const preparationGuard = new PreparationRequestGuard();
const submissionGuard = new PreparationRequestGuard();
let expansionPullRequestId = 0;
let expansionRecoveryId = 0;
let submissionUiId = 0;
let recoveryRetryId = 0;
let unmounted = false;
const downloadConsumerId = `mobile-generate-${createUuid()}`;
const progress = ref("Ready");
/** Whether the status line under Develop currently shows a failure. Set with
 * `setGenerationStatus` so error styling never depends on string sniffing. */
const progressIsError = ref(false);
const generationAnnouncement = ref("");
const gallery = ref<GalleryPrint[]>([]);
const galleryLoading = ref(false);
const galleryLoadingMore = ref(false);
const galleryError = ref("");
const galleryRemaining = ref(0);
const selectedPrint = ref<GalleryPrint | null>(null);
const generatedViewerOpen = ref(false);
const reusingPrint = ref(false);
const usingPrintAsSource = ref(false);
const reusePrintError = ref("");
const latestResultClientId = ref<number | null>(null);
const resultMediaLoadKey = ref(0);
const objectUrls = new Set<string>();
const handledGenerationClientIds = new Set<number>();
let pendingGallery: PendingGalleryPrint[] = [];
let modelLoadEpoch = 0;
let galleryRefreshRequested = false;
let galleryRefreshDeferred = false;
let galleryRefreshTask: Promise<void> | null = null;
let galleryOperationTail: Promise<void> = Promise.resolve();
let resultMediaRecoveryClientId: number | null = null;
let resultMediaRecoveryAttempts = 0;
let hostProbeTimer: ReturnType<typeof setInterval> | null = null;
let hostProbeEpoch = 0;
const hostProbes = new Map<
  string,
  { epoch: number; controller: AbortController; timeout: ReturnType<typeof setTimeout> }
>();
const generation = useGenerationStore();
const mobileDownloads = useMobileDownloadsStore();
function loadLibrarySeenAt(): Record<string, number> {
  try {
    const parsed = JSON.parse(localStorage.getItem(LIBRARY_SEEN_AT_KEY) ?? "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, number] =>
          typeof entry[1] === "number" && Number.isFinite(entry[1]) && entry[1] >= 0,
      ),
    );
  } catch {
    return {};
  }
}
let librarySeenAtBaseline = loadLibrarySeenAt();
let libraryPreviouslyVisited = localStorage.getItem(LIBRARY_VISITED_KEY) === "true";

// Ephemeral per-host telemetry from the /api/status probe (VRAM + queue), kept
// out of the persisted host identity so the Machines cards can mirror the host
// detail view's live figures. Cleared when a host goes offline.
interface HostTelemetry {
  vramUsedMb: number | null;
  vramTotalMb: number | null;
  queueDepth: number | null;
}
const hostTelemetry = reactive<Record<string, HostTelemetry>>({});

function hostMemLabel(id: string): string {
  const telemetry = hostTelemetry[id];
  if (!telemetry || telemetry.vramUsedMb == null || telemetry.vramTotalMb == null) return "—";
  return `${(telemetry.vramUsedMb / 1000).toFixed(1)} / ${(telemetry.vramTotalMb / 1000).toFixed(1)} GB`;
}

function hostVramPercent(id: string): number {
  const telemetry = hostTelemetry[id];
  if (!telemetry || telemetry.vramUsedMb == null || !telemetry.vramTotalMb) return 0;
  return percent(telemetry.vramUsedMb, telemetry.vramTotalMb);
}

function hostQueueLabel(id: string): string {
  return String(hostTelemetry[id]?.queueDepth ?? 0);
}

function captureHostTelemetry(hostId: string, status: ServerStatus): void {
  const memory = summarizeStatusGpuMemory(status);
  hostTelemetry[hostId] = {
    vramUsedMb: memory?.usedMb ?? null,
    vramTotalMb: memory?.totalMb ?? null,
    queueDepth: status.queue_depth ?? null,
  };
}

const selectedHost = computed(() => hosts.value.find((host) => host.id === selectedHostId.value));
const hostDetail = computed(() => hosts.value.find((host) => host.id === hostDetailId.value));
const selectedPrintIndex = computed(() => {
  const selected = selectedPrint.value;
  if (!selected) return -1;
  return gallery.value.findIndex(
    (print) => print.hostId === selected.hostId && print.filename === selected.filename,
  );
});
const canUseSelectedPrintAsSource = computed(
  () =>
    !!selectedPrint.value &&
    isStillImageFile(selectedPrint.value.filename) &&
    caps.value.supportsImg2img,
);
const selectedTarget = computed<ApiTarget | null>(() => {
  const host = selectedHost.value;
  return host ? mobileHostTarget(host) : null;
});
const caps = computed(() => generationCapabilitiesForFamily(form.family));
const effectiveBatchSize = computed(() =>
  caps.value.forcesBatchSizeOne ? 1 : Math.max(1, Math.floor(form.batchSize)),
);
const selectedRoute = computed<HostRoute | null>(() => {
  const host = selectedHost.value;
  return host ? routeForMobileHost(host) : null;
});
const expansionMissingModel = computed(() => {
  const recovery = expansionRecovery.value;
  return recovery ? { model: recovery.model, route: recovery.route, host: recovery.host } : null;
});
const preparedStaleReasons = computed(() => {
  const batch = preparedBatch.value;
  if (!batch) return [];
  return preparedExpansionStaleReasons(batch, {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    requestedCount: effectiveBatchSize.value,
    stylePreset: form.stylePreset || null,
    selectedHostPolicy: selectedHostId.value || null,
    readyHostIds: new Set(hosts.value.filter((host) => host.online).map((host) => host.id)),
    hostLabels: new Map(hosts.value.map((host) => [host.id, host.name])),
    modelLabels: new Map(models.value.map((model) => [model.name, modelLabel(model.name)])),
    hostTargets: new Map(
      hosts.value.map((host) => [
        host.id,
        {
          ...mobileHostTarget(host),
          kind: "remote" as const,
          instanceId: host.instanceId ?? null,
        },
      ]),
    ),
  });
});
const quickStaleReasons = computed(() => {
  const snapshot = quickExpansionSnapshot.value;
  if (!snapshot) return [];
  return quickExpansionStaleReasons(snapshot, {
    expandedPrompt: form.prompt,
    model: form.model,
    family: form.family,
    selectedHostPolicy: selectedHostId.value || null,
    readyHostIds: new Set(hosts.value.filter((host) => host.online).map((host) => host.id)),
    hostLabels: new Map(hosts.value.map((host) => [host.id, host.name])),
    modelLabels: new Map(models.value.map((model) => [model.name, modelLabel(model.name)])),
    hostTargets: new Map(
      hosts.value.map((host) => [
        host.id,
        {
          ...mobileHostTarget(host),
          kind: "remote" as const,
          instanceId: host.instanceId ?? null,
        },
      ]),
    ),
  });
});
const generationModels = computed(() =>
  models.value.filter((model) => model.downloaded && isGenerationModel(model)),
);
const sequenceModels = computed(() =>
  generationModels.value.filter((model) => modelSupportsSequence(model)),
);
const modelLabel = (name: string) => modelDisplayNameForId(name, generationModels.value);
const upscalers = computed(() =>
  models.value.filter((model) => model.family === "upscaler" || model.family === "real-esrgan"),
);
const controlModels = computed(() => models.value.filter((model) => model.family === "controlnet"));
const sourceSectionTitle = computed(() =>
  caps.value.sourceImageMode === "qwen-edit" ? "Pictures" : "Source image",
);
const sourceSectionSummary = computed(() => {
  if (caps.value.sourceImageMode === "qwen-edit") {
    const count = form.imageAttachments.length;
    return count === 0 ? "Target required" : `${count} photo${count === 1 ? "" : "s"}`;
  }
  if (form.sourceImage) return form.sourceImageName || "Selected";
  return form.controlImage ? "Control photo selected" : "Optional";
});
const outputFormats = computed(() => outputFormatsForFamily(form.family));
const selectedModelAvailable = computed(
  () =>
    modelsHostId.value === selectedHostId.value &&
    generationModels.value.some((model) => model.name === form.model),
);
const selectedGenerationModel = computed(
  () => generationModels.value.find((model) => model.name === form.model) ?? null,
);
const sourceControlsValid = computed(() => !caps.value.supportsImg2img || sourceValid.value);
const stepsError = computed(() => stepsValidationError(form.steps));
const guidanceError = computed(() => guidanceValidationError(form.guidance));
const basicParametersValid = computed(() => !stepsError.value && !guidanceError.value);
const mobileMediaBudgetError = computed(() => mobileMediaBudgetValidationError(form));
const estimateRequest = computed(() => {
  if (!form.model) return null;
  return buildGenerationEstimateRequest(buildRequest(form), form.family);
});
const queuedJobs = computed(() => railOrder(generation.pending));
const activeGeneration = computed(() => {
  const active = generation.active;
  return active && active.status !== "complete" && active.status !== "error" ? active : null;
});
const latestResultJob = computed(() => {
  const latest = generation.jobs.find((job) => job.clientId === latestResultClientId.value);
  // Once a completion is promoted, never put an older print underneath its
  // new seed/status while a saved-file URL is loading or has failed.
  if (latestResultClientId.value !== null) {
    return latest?.status === "complete" ? latest : null;
  }
  for (let index = generation.jobs.length - 1; index >= 0; index -= 1) {
    const job = generation.jobs[index];
    if (job?.status === "complete") return job;
  }
  return null;
});
const resultUrl = computed(() => latestResultJob.value?.resultUrl ?? "");
const resultIsVideo = computed(() => latestResultJob.value?.result?.format === "mp4");
const generatedPreviewItem = computed<GalleryImage | null>(() => {
  const job = latestResultJob.value;
  const result = job?.result;
  if (!job || !result || !resultUrl.value) return null;
  return {
    filename: result.filename ?? `generated-${result.seed_used}.${result.format}`,
    timestamp: Math.floor(job.submittedAtUnixMs / 1000),
    format: result.format,
    metadata: result.metadata ?? {
      prompt: job.prompt,
      model: result.model,
      seed: result.seed_used,
      steps: job.total,
      guidance: job.guidance,
      width: result.width,
      height: result.height,
    },
  };
});
const generatedPreviewTarget = computed<ApiTarget>(() => {
  const job = latestResultJob.value;
  const host = hosts.value.find((candidate) => candidate.id === job?.hostId) ?? selectedHost.value;
  return host ? mobileHostTarget(host) : { baseUrl: "", apiKey: null };
});
const resultPreviewError = computed(() => {
  const job = latestResultJob.value;
  return job?.resultError ? describeTransportError(job.resultError, job.hostLabel) : "";
});
const developButtonLabel = computed(() =>
  preparingGeneration.value
    ? "Preparing source…"
    : `${form.batchSize > 1 ? `Develop ${form.batchSize} prints` : "Develop print"}${
        queuedJobs.value.length > 0 ? ` (+${queuedJobs.value.length} queued)` : ""
      }`,
);
const expansionPullBucket = computed<DownloadsState>(() => {
  const route = expansionRecovery.value?.route;
  return route
    ? (mobileDownloads.downloadsByHost[route.hostId] ?? {
        activeJobs: [],
        queued: [],
        history: [],
      })
    : { activeJobs: [], queued: [], history: [] };
});
const expansionPullStatus = computed<ExpansionPullView | null>(() => {
  const missing = expansionMissingModel.value;
  if (!missing) return null;
  const attempt = expansionPullAttempt.value;
  const recovery = expansionRecovery.value;
  if (!attempt || !recovery || attempt.recoveryId !== recovery.id) {
    return { kind: "missing", job: null };
  }
  const pending = mobileDownloads.pendingPulls.get(`${recovery.route.hostId}:${recovery.model}`);
  return resolveExpansionPullStatus({
    model: recovery.model,
    phase: pending?.phase ?? attempt.phase,
    jobId: attempt.jobId,
    observedJobId: attempt.observedJobId,
    baselineJobIds: attempt.baselineJobIds,
    allowExistingInFlight: attempt.allowExistingInFlight,
    activeJobs: expansionPullBucket.value.activeJobs,
    queued: expansionPullBucket.value.queued,
    history: attempt.terminalJob
      ? [
          attempt.terminalJob,
          ...expansionPullBucket.value.history.filter((job) => job.id !== attempt.terminalJob?.id),
        ]
      : expansionPullBucket.value.history,
    requestError: attempt.requestError,
  });
});
const expansionPullEtaSeconds = computed(() => {
  const attempt = expansionPullAttempt.value;
  const recovery = expansionRecovery.value;
  const job = expansionPullStatus.value?.job;
  return attempt && recovery && job ? mobileDownloads.etaFor(recovery.route.hostId, job.id) : null;
});
const queueAnnouncement = computed(() => {
  const count = queuedJobs.value.length;
  return count === 0
    ? "No active generations."
    : `${count} active generation${count === 1 ? "" : "s"}.`;
});
const generationStatus = computed(() => {
  const active = activeGeneration.value;
  if (!active) return progress.value;
  switch (active.status) {
    case "queued":
      return active.queuePosition && active.queuePosition > 0
        ? `Queued #${active.queuePosition}`
        : "Queued";
    case "loading":
      return active.stage ?? "Loading model";
    case "denoising":
      return `Developing ${active.step} / ${active.total}`;
    case "finishing":
      return active.stage ?? "Finalizing";
    default:
      return jobStatusCode(active);
  }
});
const generationStatusIsError = computed(() => !activeGeneration.value && progressIsError.value);

/**
 * The status line under Develop shows generation lifecycle and explicit
 * user-action outcomes only — background poll failures must use their own
 * surfaces (e.g. the model-load banner), never this slot.
 */
function setGenerationStatus(message: string, isError = false): void {
  progress.value = message;
  progressIsError.value = isError;
}

/** Seed/time line for a completed result; the time is omitted when a resumed
 * reconciliation lost the true duration with the stream. */
function completionSummary(result: CompleteEvent): string {
  const timing =
    result.generation_time_ms > 0 ? `${(result.generation_time_ms / 1000).toFixed(1)}s · ` : "";
  return `${timing}seed ${result.seed_used}`;
}

async function saveCompletedStillToPhotos(result: CompleteEvent, target: ApiTarget): Promise<void> {
  if (!mobileSettings.autoSavePhotos) return;
  const filenames = [result.original_filename, result.filename].filter(
    (filename, index, all): filename is string =>
      !!filename && isStillImageFile(filename) && all.indexOf(filename) === index,
  );
  const saves = filenames.map(async (filename) => {
    const response = await apiFetchTo(target, galleryMediaPath(filename, "host"));
    await invoke("save_image_to_photos", {
      dataB64: await blobToBase64(await response.blob()),
    });
  });
  const settled = await Promise.allSettled(saves);
  for (const save of settled) {
    if (save.status === "rejected") {
      console.warn("Unable to auto-save generated image to Photos", save.reason);
    }
  }
}

function routeForMobileHost(host: MobileHost): HostRoute {
  return {
    hostId: host.id,
    label: host.name,
    kind: "remote",
    target: { ...mobileHostTarget(host) },
    instanceId: host.instanceId ?? null,
  };
}

function loadHosts(): MobileHost[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as MobileHost[];
    return raw.map((host) => ({ ...host, apiKey: "", online: false }));
  } catch {
    return [];
  }
}

function persistHosts(): void {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(hosts.value.map(({ apiKey: _apiKey, ...host }) => host)),
  );
}

async function hydrateApiKeys(): Promise<void> {
  await Promise.all(
    hosts.value.map(async (host) => {
      host.apiKey =
        (await invoke<string | null>("keychain_get_api_key", { hostId: host.id })) ?? "";
    }),
  );
}

async function connectHost(address?: string, discoveredName?: string): Promise<void> {
  hostError.value = "";
  try {
    const baseUrl = normalizeRemoteAddress(address ?? hostInput.address);
    const target = { baseUrl, apiKey: hostInput.apiKey.trim() || null };
    const status = await apiJsonTo<ServerStatus>(target, "/api/status");
    const instanceId = status.instance_id ?? undefined;
    const existing = hosts.value.find(
      (host) =>
        host.baseUrl === baseUrl ||
        (instanceId &&
          (host.instanceId === instanceId || host.id === instanceId) &&
          (!host.hostname || !status.hostname || host.hostname === status.hostname)),
    );
    // URL identity keeps two machines that copied the same MOLD_HOME distinct;
    // a compatible saved alias keeps its existing keychain id.
    const id = existing?.id ?? remoteHostId(baseUrl);
    const saved: MobileHost = {
      id,
      name: hostInput.name.trim() || discoveredName || status.hostname || new URL(baseUrl).hostname,
      baseUrl,
      apiKey: hostInput.apiKey.trim(),
      hostname: status.hostname ?? undefined,
      version: status.version,
      instanceId,
      online: true,
    };
    if (existing) Object.assign(existing, saved);
    else hosts.value.push(saved);
    if (saved.apiKey) {
      await invoke("keychain_set_api_key", { hostId: saved.id, apiKey: saved.apiKey });
    } else {
      await invoke("keychain_delete_api_key", { hostId: saved.id });
    }
    persistHosts();
    selectedHostId.value = saved.id;
    catalogHostId.value = saved.id;
    tab.value = "generate";
    hostInput.name = "";
    hostInput.address = "";
    hostInput.apiKey = "";
    await refreshModels();
  } catch (error) {
    const label = hostInput.name.trim() || discoveredName || (address ?? hostInput.address).trim();
    hostError.value = describeTransportError(error, label);
  }
}

async function discoverHosts(): Promise<void> {
  discovering.value = true;
  hostError.value = "";
  try {
    discovered.value = await invoke<DiscoveredHost[]>("discover_mold_hosts", { timeoutMs: 2500 });
  } catch (error) {
    hostError.value = describeTransportError(error);
  } finally {
    discovering.value = false;
  }
}

async function selectHost(id: string): Promise<void> {
  selectedHostId.value = id;
  await refreshModels();
}

function showHostDetail(id: string): void {
  hostDetailId.value = id;
}

function renameHost(payload: { id: string; name: string }): void {
  const host = hosts.value.find((candidate) => candidate.id === payload.id);
  if (!host) return;
  host.name = payload.name;
  persistHosts();
}

function updateHostStatus(payload: { id: string; status: ServerStatus | null }): void {
  const host = hosts.value.find((candidate) => candidate.id === payload.id);
  if (!host) return;
  host.online = payload.status !== null;
  if (payload.status) {
    host.version = payload.status.version;
    host.hostname = payload.status.hostname ?? undefined;
    host.instanceId = payload.status.instance_id ?? host.instanceId;
    captureHostTelemetry(host.id, payload.status);
  } else {
    delete hostTelemetry[host.id];
  }
}

function cancelHostProbe(id: string): void {
  const probe = hostProbes.get(id);
  if (!probe) return;
  probe.controller.abort();
  clearTimeout(probe.timeout);
  hostProbes.delete(id);
}

async function probeHost(host: MobileHost): Promise<void> {
  cancelHostProbe(host.id);
  const controller = new AbortController();
  const epoch = ++hostProbeEpoch;
  const timeout = setTimeout(() => controller.abort(), HOST_PROBE_TIMEOUT_MS);
  const probe = { epoch, controller, timeout };
  hostProbes.set(host.id, probe);
  try {
    const status = await apiJsonTo<ServerStatus>(mobileHostTarget(host), "/api/status", {
      signal: controller.signal,
    });
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    updateHostStatus({ id: host.id, status });
  } catch {
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    updateHostStatus({ id: host.id, status: null });
  } finally {
    if (hostProbes.get(host.id)?.epoch === epoch) hostProbes.delete(host.id);
    clearTimeout(timeout);
  }
}

function probeHosts(): void {
  for (const host of hosts.value) void probeHost(host);
}

function removeHost(id: string): void {
  cancelHostProbe(id);
  const removedSelectedHost = selectedHostId.value === id;
  const removedCatalogHost = catalogHostId.value === id;
  if (hostDetailId.value === id) hostDetailId.value = "";
  hosts.value = hosts.value.filter((host) => host.id !== id);
  if (removedSelectedHost) {
    selectedHostId.value = hosts.value[0]?.id ?? "";
    models.value = [];
    modelsHostId.value = "";
    void refreshModels();
  }
  if (removedCatalogHost) catalogHostId.value = hosts.value[0]?.id ?? "";
  persistHosts();
  void invoke("keychain_delete_api_key", { hostId: id });
}

function selectCatalogHost(id: string): void {
  if (hosts.value.some((host) => host.id === id)) catalogHostId.value = id;
}

function openCatalog(id?: string): void {
  if (id && hosts.value.some((host) => host.id === id)) catalogHostId.value = id;
  else if (!hosts.value.some((host) => host.id === catalogHostId.value)) {
    catalogHostId.value = selectedHostId.value || hosts.value[0]?.id || "";
  }
  hostDetailId.value = "";
  tab.value = "catalog";
}

function catalogModelsChanged(hostId: string): void {
  if (hostId === selectedHostId.value) void refreshModels();
}

async function refreshModels(): Promise<boolean> {
  const epoch = ++modelLoadEpoch;
  const host = selectedHost.value;
  if (!host) {
    models.value = [];
    modelsHostId.value = "";
    loadingModels.value = false;
    modelLoadError.value = "";
    return false;
  }
  const hostId = host.id;
  const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
  loadingModels.value = true;
  modelLoadError.value = "";
  models.value = [];
  modelsHostId.value = "";
  try {
    const [status, entries, capabilities] = await Promise.all([
      apiJsonTo<ServerStatus>(target, "/api/status"),
      apiJsonTo<ModelEntry[]>(target, "/api/models"),
      apiJsonTo<{ expand?: ExpandCapabilities | null }>(target, "/api/capabilities").catch(
        () => null,
      ),
    ]);
    if (unmounted || epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    host.online = true;
    host.version = status.version;
    host.hostname = status.hostname ?? undefined;
    host.instanceId = status.instance_id ?? host.instanceId;
    captureHostTelemetry(hostId, status);
    expandCapabilities[hostId] = capabilities?.expand;
    // Keep auxiliary entries for the Upscale and ControlNet pickers, while
    // the main Model select uses `generationModels` so those tools can never
    // become the active generation model.
    models.value = entries;
    modelsHostId.value = hostId;
    const selectedEntry = generationModels.value.find((model) => model.name === form.model);
    if (selectedEntry) {
      reconcileModelCapabilities(form, selectedEntry);
    } else if (generationModels.value[0]) {
      applyModelDefaults(form, generationModels.value[0]);
    }
    return true;
  } catch (error) {
    if (unmounted || epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    host.online = false;
    // The banner above the model select owns this failure; the generation
    // status line keeps showing generation state, not background loads.
    const detail = describeTransportError(error, host.name);
    modelLoadError.value = `Couldn’t load generation models from ${host.name}. ${detail}`;
    return false;
  } finally {
    if (epoch === modelLoadEpoch && selectedHostId.value === hostId) loadingModels.value = false;
  }
}

function clearSequencePoll(): void {
  if (sequencePollTimer) clearInterval(sequencePollTimer);
  sequencePollTimer = null;
}

function clearSequenceRecovery(): void {
  try {
    localStorage.removeItem(SEQUENCE_RECOVERY_KEY);
  } catch {
    // Recovery persistence is best effort; the live job remains authoritative.
  }
}

function persistSequenceRecovery(host: MobileHost, jobId: string): void {
  try {
    localStorage.setItem(
      SEQUENCE_RECOVERY_KEY,
      JSON.stringify({
        hostId: host.id,
        baseUrl: host.baseUrl,
        instanceId: host.instanceId ?? null,
        jobId,
      }),
    );
  } catch {
    // A storage failure must never turn an accepted durable job into an error.
  }
}

async function pollSequenceJob(): Promise<void> {
  const route = sequenceRoute;
  const jobId = sequenceJob.value?.id;
  if (!route || !jobId) return;
  try {
    const job = await apiJsonTo<ChainJobDetail>(
      route.target,
      `/api/chain-jobs/${encodeURIComponent(jobId)}`,
    );
    if (sequenceRoute !== route) return;
    sequenceJob.value = job;
    sequenceError.value = "";
    if (job.state === "completed" || job.state === "failed" || job.state === "cancelled") {
      clearSequencePoll();
      clearSequenceRecovery();
      generationAnnouncement.value =
        job.state === "completed"
          ? `Sequence completed on ${hosts.value.find((host) => host.id === route.hostId)?.name ?? "the selected host"}.`
          : `Sequence ${job.state}. ${job.error ?? ""}`.trim();
      if (job.state === "completed" && tab.value === "gallery") {
        void refreshGallery();
      }
    }
  } catch (error) {
    if (sequenceRoute !== route) return;
    sequenceError.value = describeTransportError(
      error,
      hosts.value.find((host) => host.id === route.hostId)?.name ?? "sequence host",
    );
  }
}

function watchSequenceJob(hostId: string, target: ApiTarget, jobId: string): void {
  clearSequencePoll();
  sequenceRoute = { hostId, target: { ...target } };
  sequenceJob.value = {
    id: jobId,
    state: "queued",
    model: "",
    stage_count: 0,
    current_stage: 0,
    created_at_unix_ms: Date.now(),
    updated_at_unix_ms: Date.now(),
    error: null,
    ephemeral: false,
    stages: [],
    finalizes: [],
    retakes: [],
    script: {
      schema: "mold.chain.v1",
      chain: {
        model: "",
        width: 0,
        height: 0,
        fps: 24,
        steps: 0,
        guidance: 0,
        strength: 1,
        motion_tail_frames: 0,
        output_format: "mp4",
      },
      stage: [],
    },
  };
  void pollSequenceJob();
  sequencePollTimer = setInterval(() => void pollSequenceJob(), 2_000);
}

async function submitMobileSequence(request: ChainRequest): Promise<void> {
  const host = selectedHost.value;
  if (!host || sequenceStarting.value) return;
  const target = { ...mobileHostTarget(host) };
  sequenceStarting.value = true;
  sequenceError.value = "";
  try {
    const response = await apiJsonTo<CreateChainJobResponse>(target, "/api/chain-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    persistSequenceRecovery(host, response.job_id);
    watchSequenceJob(host.id, target, response.job_id);
  } catch (error) {
    sequenceError.value = describeTransportError(error, host.name);
  } finally {
    sequenceStarting.value = false;
  }
}

async function cancelMobileSequence(): Promise<void> {
  const route = sequenceRoute;
  const job = sequenceJob.value;
  if (!route || !job) return;
  sequenceError.value = "";
  try {
    await apiFetchTo(route.target, `/api/chain-jobs/${encodeURIComponent(job.id)}/cancel`, {
      method: "POST",
    });
    await pollSequenceJob();
  } catch (error) {
    sequenceError.value = describeTransportError(
      error,
      hosts.value.find((host) => host.id === route.hostId)?.name ?? "sequence host",
    );
  }
}

function dismissMobileSequence(): void {
  clearSequencePoll();
  clearSequenceRecovery();
  sequenceRoute = null;
  sequenceJob.value = null;
  sequenceError.value = "";
}

function recoverMobileSequence(): void {
  let saved: {
    hostId?: string;
    baseUrl?: string;
    instanceId?: string | null;
    jobId?: string;
  } | null = null;
  try {
    saved = JSON.parse(localStorage.getItem(SEQUENCE_RECOVERY_KEY) ?? "null") as {
      hostId?: string;
      baseUrl?: string;
      instanceId?: string | null;
      jobId?: string;
    } | null;
  } catch {
    clearSequenceRecovery();
    return;
  }
  if (!saved?.hostId || !saved.jobId) return;
  const host = hosts.value.find((candidate) => candidate.id === saved!.hostId);
  if (
    !host ||
    host.baseUrl !== saved.baseUrl ||
    (saved.instanceId != null && saved.instanceId !== host.instanceId)
  ) {
    clearSequenceRecovery();
    sequenceError.value = "The exact machine for this saved sequence is no longer available.";
    return;
  }
  watchSequenceJob(host.id, mobileHostTarget(host), saved.jobId);
}

function changeModel(): void {
  const model = generationModels.value.find((entry) => entry.name === form.model);
  if (model) applyModelDefaults(form, model);
}

function clearHostScopedGenerationSelections(): void {
  form.loras = [];
  form.upscaleModel = "";
  form.controlModel = "";
  form.cameraControl = null;
  if (form.sourceFit.mode === "upscale-then-fit") {
    form.sourceFit = { ...form.sourceFit, upscalerModel: "" };
  }
}

function appendPromptWord(word: string): void {
  const trimmed = word.trim();
  if (!trimmed) return;
  form.prompt = form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed;
}

function loadTemplate(template: GenerationTemplate): void {
  Object.assign(form, template.form);
  const sameHost = !!template.scopeId && template.scopeId === selectedHostId.value;
  if (!sameHost) clearHostScopedGenerationSelections();
  const selectedEntry = generationModels.value.find((model) => model.name === form.model);
  if (selectedEntry) reconcileModelCapabilities(form, selectedEntry);
  const available = generationModels.value.some((model) => model.name === form.model);
  setGenerationStatus(
    available
      ? `Loaded ${template.name}`
      : `Loaded ${template.name}. Install ${form.model} from Catalog or choose another model.`,
  );
  const mediaMessage = template.mediaReferences.length
    ? ` Re-add ${formatTemplateMediaReferences(template.mediaReferences)}.`
    : "";
  generationAnnouncement.value = sameHost
    ? `Template loaded.${mediaMessage}`
    : `Template loaded. Re-add host-specific LoRAs and auxiliary models.${mediaMessage}`;
}

function expansionInputs(count: number): PreparedExpansionInputs {
  return {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    requestedCount: count,
    stylePreset: form.stylePreset || null,
    selectedHostPolicy: selectedHostId.value || null,
  };
}

function sameFrozenHost(route: HostRoute, host: MobileHost | undefined): boolean {
  if (!host || !host.online || host.id !== route.hostId) return false;
  const target = mobileHostTarget(host);
  return (
    target.baseUrl === route.target.baseUrl &&
    target.apiKey === route.target.apiKey &&
    (route.instanceId === undefined || (host.instanceId ?? null) === route.instanceId)
  );
}

interface ReplacementFocusOwnership {
  preparedRoot: HTMLElement | null;
  pullRoot: HTMLElement | null;
  owned: boolean;
}

function captureReplacementFocus(replacePrepared: boolean): ReplacementFocusOwnership {
  const preparedRoot = replacePrepared
    ? document.querySelector<HTMLElement>("[data-test='mobile-prepared-expansion']")
    : null;
  const pullRoot = document.querySelector<HTMLElement>(".mobile-expansion-pull");
  const active = document.activeElement;
  return {
    preparedRoot,
    pullRoot,
    owned: !!active && (!!preparedRoot?.contains(active) || !!pullRoot?.contains(active)),
  };
}

function restoreReplacementFocus(
  ownership: ReplacementFocusOwnership,
  target: "prompt" | "prepared",
): void {
  const active = document.activeElement;
  const restore =
    ownership.owned &&
    (active === document.body ||
      (!!active &&
        (!!ownership.preparedRoot?.contains(active) || !!ownership.pullRoot?.contains(active))));
  if (!restore) return;
  void nextTick(() => {
    if (unmounted) return;
    if (target === "prompt") {
      document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus();
    } else {
      document
        .querySelector<HTMLTextAreaElement>("[data-test='mobile-prepared-expansion'] textarea")
        ?.focus();
    }
  });
}

function clearExpansionRecovery(invalidateRetry = true): void {
  const recovery = expansionRecovery.value;
  if (invalidateRetry) recoveryRetryId += 1;
  expansionPullRequestId += 1;
  expansionPullAttempt.value = null;
  expansionRecovery.value = null;
  if (recovery) releaseExpansionPullLease(recovery);
  else mobileDownloads.unregisterConsumer(downloadConsumerId);
}

function releaseExpansionPullLease(recovery: MobileExpansionRecoveryRecord): void {
  mobileDownloads.releaseFrozenPull(recovery.leaseId);
  mobileDownloads.unregisterConsumer(downloadConsumerId);
}

function markExpansionRecoveryStale(recovery: MobileExpansionRecoveryRecord, reason: string): void {
  expansionError.value = reason;
  const attempt = expansionPullAttempt.value;
  if (attempt?.recoveryId === recovery.id) attempt.requestError = reason;
  releaseExpansionPullLease(recovery);
}

function setExpansionFailure(
  error: unknown,
  inputs: PreparedExpansionInputs,
  route: HostRoute,
  requestToken: number,
  replacePrepared: boolean,
): string {
  const message = error instanceof Error ? error.message : String(error);
  const missingModel = parseMissingExpandModel(message);
  clearExpansionRecovery();
  if (missingModel) {
    const id = ++expansionRecoveryId;
    expansionRecovery.value = createMobileExpansionRecovery({
      id,
      leaseId: `${downloadConsumerId}:recovery-${id}`,
      model: missingModel,
      inputs,
      route,
      requestToken,
      replacePrepared,
    });
  }
  return missingModel
    ? `The expansion model ${missingModel} isn't installed on ${route.label}.`
    : `Expansion failed on ${route.label}: ${describeTransportError(error, route.label)}`;
}

function recoveryStaleReason(recovery: MobileExpansionRecoveryRecord): string | null {
  return mobileExpansionRecoveryStaleReason(recovery, {
    inputs: expansionInputs(recovery.inputs.requestedCount),
    currentHost: hosts.value.find((host) => host.id === recovery.route.hostId),
    tokenCurrent:
      !unmounted &&
      expansionRecovery.value?.id === recovery.id &&
      preparationGuard.isCurrent(recovery.requestToken),
  });
}

function syncExpansionDownloadConsumer(recovery: MobileExpansionRecoveryRecord): void {
  const host = { ...recovery.host };
  mobileDownloads.registerConsumer(downloadConsumerId, [host], {
    onEvent: ({ event }) => {
      if (
        event.type !== "job_done" &&
        event.type !== "job_failed" &&
        event.type !== "job_cancelled"
      ) {
        return;
      }
      const attempt = expansionPullAttempt.value;
      if (
        expansionRecovery.value?.id !== recovery.id ||
        attempt?.recoveryId !== recovery.id ||
        (attempt.jobId !== event.id && attempt.observedJobId !== event.id)
      ) {
        return;
      }
      const bucket = mobileDownloads.downloadsByHost[recovery.route.hostId];
      attempt.terminalJob = bucket?.history.find((job) => job.id === event.id) ?? null;
      releaseExpansionPullLease(recovery);
    },
    onStreamError: (failedHost, cause) => {
      const attempt = expansionPullAttempt.value;
      if (
        expansionRecovery.value?.id === recovery.id &&
        attempt?.recoveryId === recovery.id &&
        recovery.route.hostId === failedHost.id
      ) {
        attempt.requestError = describeTransportError(cause, recovery.route.label);
        releaseExpansionPullLease(recovery);
      }
    },
  });
}

function commitExpandedPrompts(
  inputs: PreparedExpansionInputs,
  route: HostRoute,
  prompts: string[],
  requestToken: number,
  replacePrepared: boolean,
  focus: ReplacementFocusOwnership,
): void {
  if (inputs.requestedCount === 1) {
    quickExpansionOriginal.value = inputs.sourcePrompt;
    form.prompt = prompts[0]!;
    form.originalPrompt = inputs.sourcePrompt;
    quickExpansionSnapshot.value = {
      requestToken,
      originalPrompt: inputs.sourcePrompt,
      expandedPrompt: prompts[0]!,
      model: inputs.model,
      family: inputs.family,
      stylePreset: inputs.stylePreset,
      selectedHostPolicy: inputs.selectedHostPolicy,
      route: { ...route, target: { ...route.target } },
    };
    // Bake-and-clear: the rewrite absorbed the style (the server received it
    // as a directive), so the chip clears here — leaving it lit would apply
    // the look twice at submit. Prepared batches below KEEP the chip: it is
    // the frozen-style indicator for the reviewed set (a style change is a
    // named staleness axis) and their submit path never re-composes it into
    // the reviewed prompt text.
    bakeStyleNegative(inputs.stylePreset ?? "", inputs.family);
    form.stylePreset = "";
    if (replacePrepared) preparedBatch.value = null;
    clearExpansionRecovery(false);
    if (replacePrepared) restoreReplacementFocus(focus, "prompt");
    return;
  }
  preparedBatch.value = createPreparedExpansionBatch(inputs, route, prompts, requestToken);
  quickExpansionSnapshot.value = null;
  clearExpansionRecovery(false);
  if (replacePrepared) restoreReplacementFocus(focus, "prepared");
}

async function expandForCurrentBatch(
  replacePrepared = false,
  routeOverride: HostRoute | null = null,
): Promise<void> {
  const count = effectiveBatchSize.value;
  const inputs = expansionInputs(count);
  const host = routeOverride
    ? hosts.value.find((candidate) => candidate.id === routeOverride.hostId)
    : selectedHost.value;
  const route = routeOverride ?? selectedRoute.value;
  const replacementFocus = captureReplacementFocus(replacePrepared);
  if (
    !inputs.sourcePrompt ||
    !inputs.model ||
    !host ||
    !route ||
    expansionRunning.value ||
    (preparedBatch.value && count === 1 && !replacePrepared)
  ) {
    return;
  }
  if (!sameFrozenHost(route, host)) {
    expansionError.value = `${route.label} isn't reachable with the frozen connection. Expansion will not fall back.`;
    return;
  }
  if (expandCapabilities[route.hostId]?.configured === false) {
    expansionError.value = `Prompt expansion isn't configured on ${route.label}. Configure that host before retrying.`;
    clearExpansionRecovery();
    return;
  }

  clearExpansionRecovery();
  submissionGuard.invalidate();
  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = "";
  try {
    // The active chip travels as a natural-language directive the server
    // weaves into the expander's system message — never the literal suffix.
    const styleDirective = styleHint(inputs.stylePreset ?? "");
    const response = await expandPrompt(
      inputs.sourcePrompt,
      {
        variations: count,
        ...(inputs.family ? { modelFamily: inputs.family } : {}),
        ...(styleDirective ? { style: styleDirective } : {}),
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const prompts = validateExpandedPrompts(response.expanded, count);
    const currentHost = hosts.value.find((candidate) => candidate.id === route.hostId);
    const current = expansionInputs(count);
    if (
      current.sourcePrompt !== inputs.sourcePrompt ||
      current.model !== inputs.model ||
      current.family !== inputs.family ||
      current.requestedCount !== inputs.requestedCount ||
      current.stylePreset !== inputs.stylePreset ||
      current.selectedHostPolicy !== inputs.selectedHostPolicy ||
      !sameFrozenHost(route, currentHost)
    ) {
      expansionError.value =
        "The prompt, model, style, Batch, or host changed while expansion was running. Expand again with the current inputs.";
      return;
    }
    commitExpandedPrompts(inputs, route, prompts, token, replacePrepared, replacementFocus);
  } catch (error) {
    if (!preparationGuard.isCurrent(token)) return;
    expansionError.value = setExpansionFailure(error, inputs, route, token, replacePrepared);
  } finally {
    if (!unmounted && preparationGuard.isCurrent(token)) expansionRunning.value = false;
  }
}

/**
 * Bake-and-clear owes the user the preset's curated negative: the chip is
 * about to be dropped, so submit-time composition will never see it again.
 * The look itself already reached the rewritten prompt through the expansion
 * directive — only the negative half has nowhere else to live (mirrors
 * desktop).
 */
function bakeStyleNegative(presetId: string, family: string): void {
  quickExpansionNegative.value = null;
  const merged = mergeStyleNegative(form.negativePrompt, presetId, {
    supportsNegativePrompt: generationCapabilitiesForFamily(family).supportsNegativePrompt,
  });
  if (merged === form.negativePrompt) return;
  quickExpansionNegative.value = { before: form.negativePrompt, baked: merged };
  form.negativePrompt = merged;
}

function restoreQuickExpansion(): void {
  if (quickExpansionOriginal.value === null) return;
  submissionGuard.invalidate();
  preparationGuard.invalidate();
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
  form.prompt = quickExpansionOriginal.value;
  form.originalPrompt = null;
  quickExpansionOriginal.value = null;
  quickExpansionSnapshot.value = null;
  expansionError.value = "";
  clearExpansionRecovery();
  expansionRunning.value = false;
  preparedSubmitting.value = false;
  preparingGeneration.value = false;
  submissionUiId += 1;
}

async function developExpandedAnyway(): Promise<void> {
  if (!quickExpansionSnapshot.value) return;
  submissionGuard.invalidate();
  quickExpansionSnapshot.value = null;
  expansionError.value = "";
  await generate();
}

async function reexpandAndDevelop(): Promise<void> {
  if (!quickExpansionSnapshot.value || quickExpansionOriginal.value === null) return;
  restoreQuickExpansion();
  await nextTick();
  await expandForCurrentBatch();
  if (quickExpansionSnapshot.value && quickStaleReasons.value.length === 0) {
    await generate();
  }
}

async function copyQuickExpansionError(): Promise<void> {
  await copyMobileError(`${quickStaleReasons.value.join(" ")} Choose how to continue.`);
}

async function copyMobileError(message: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(message);
  } catch {
    setGenerationStatus("Could not copy the error message.", true);
  }
}

function editPreparedPrompt(payload: { id: string; text: string }): void {
  if (preparedSubmitting.value || expansionRunning.value) return;
  supersedePreparedReplacement();
  submissionGuard.invalidate();
  const prompt = preparedBatch.value?.prompts.find((candidate) => candidate.id === payload.id);
  if (prompt) prompt.text = payload.text;
}

function removePreparedPrompt(id: string): void {
  const batch = preparedBatch.value;
  if (!batch || batch.prompts.length <= 2 || preparedSubmitting.value || expansionRunning.value)
    return;
  supersedePreparedReplacement();
  submissionGuard.invalidate();
  batch.prompts = batch.prompts.filter((prompt) => prompt.id !== id);
  batch.requestedCount = batch.prompts.length;
  form.batchSize = batch.prompts.length;
}

function supersedePreparedReplacement(): void {
  if (!expansionRecovery.value?.replacePrepared) return;
  preparationGuard.invalidate();
  clearExpansionRecovery();
  expansionRunning.value = false;
  expansionError.value = "Pending replacement cancelled; your reviewed variations were kept.";
}

function collapsePreparedBatch(removedId: string): void {
  const batch = preparedBatch.value;
  if (!batch || batch.prompts.length !== 2 || preparedSubmitting.value || expansionRunning.value)
    return;
  const remaining = batch.prompts.find((prompt) => prompt.id !== removedId);
  if (!remaining) return;
  const preparedRoot = document.querySelector<HTMLElement>(
    "[data-test='mobile-prepared-expansion']",
  );
  const restoreFocus = !!preparedRoot?.contains(document.activeElement);
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  submissionUiId += 1;
  preparedBatch.value = null;
  expansionRunning.value = false;
  preparedSubmitting.value = false;
  preparingGeneration.value = false;
  expansionError.value = "";
  clearExpansionRecovery();
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
  if (restoreFocus) {
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  }
}

function discardPreparedBatch(): void {
  const preparedRoot = document.querySelector<HTMLElement>(
    "[data-test='mobile-prepared-expansion']",
  );
  const restoreFocus = !!preparedRoot?.contains(document.activeElement);
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  submissionUiId += 1;
  preparedBatch.value = null;
  expansionRunning.value = false;
  preparedSubmitting.value = false;
  preparingGeneration.value = false;
  expansionError.value = "";
  clearExpansionRecovery();
  if (restoreFocus) {
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  }
}

async function pullExpansionModel(): Promise<void> {
  const recovery = expansionRecovery.value;
  if (!recovery || expansionRunning.value) return;
  releaseExpansionPullLease(recovery);
  const stale = recoveryStaleReason(recovery);
  if (stale) {
    markExpansionRecoveryStale(recovery, stale);
    return;
  }
  const bucket = mobileDownloads.downloadsByHost[recovery.route.hostId] ?? {
    activeJobs: [],
    queued: [],
    history: [],
  };
  const attempt: MobileExpansionPullAttempt = {
    id: ++expansionPullRequestId,
    recoveryId: recovery.id,
    phase: "connecting",
    jobId: null,
    observedJobId: null,
    baselineJobIds: [...bucket.activeJobs, ...bucket.queued, ...bucket.history].map(
      (job) => job.id,
    ),
    allowExistingInFlight: false,
    requestError: null,
    terminalJob: null,
  };
  expansionPullAttempt.value = attempt;
  syncExpansionDownloadConsumer(recovery);
  try {
    const result = await mobileDownloads.startPullFrozen(
      { id: recovery.model, name: recovery.model },
      { ...recovery.host },
      recovery.leaseId,
    );
    if (
      unmounted ||
      expansionPullAttempt.value?.id !== attempt.id ||
      expansionRecovery.value?.id !== recovery.id
    ) {
      return;
    }
    const changed = recoveryStaleReason(recovery);
    if (changed) {
      markExpansionRecoveryStale(recovery, changed);
      return;
    }
    if (result.kind === "started" || result.kind === "conflict") {
      expansionPullAttempt.value.jobId = result.jobId;
    } else {
      expansionPullAttempt.value.allowExistingInFlight = true;
      const state = mobileDownloads.downloadsByHost[recovery.route.hostId];
      expansionPullAttempt.value.observedJobId = state
        ? ([...state.activeJobs, ...state.queued].find((job) =>
            expansionPullJobMatchesModel(job, recovery.model),
          )?.id ?? null)
        : null;
    }
    expansionPullAttempt.value.phase = "starting";
    const terminalId =
      expansionPullAttempt.value.jobId ?? expansionPullAttempt.value.observedJobId ?? null;
    const terminalJob = terminalId
      ? mobileDownloads.terminalJobFor(
          { id: recovery.model, name: recovery.model },
          recovery.route.hostId,
          terminalId,
        )
      : null;
    if (terminalJob) {
      expansionPullAttempt.value.terminalJob = terminalJob;
      releaseExpansionPullLease(recovery);
    }
  } catch (error) {
    if (
      unmounted ||
      expansionPullAttempt.value?.id !== attempt.id ||
      expansionRecovery.value?.id !== recovery.id
    ) {
      return;
    }
    expansionPullAttempt.value.requestError = describeTransportError(error, recovery.route.label);
    releaseExpansionPullLease(recovery);
  }
}

async function retryExpansionAfterPull(): Promise<void> {
  const recovery = expansionRecovery.value;
  const attempt = expansionPullAttempt.value;
  if (!recovery || !attempt || attempt.recoveryId !== recovery.id || expansionRunning.value) return;
  const stale = recoveryStaleReason(recovery);
  if (stale) {
    markExpansionRecoveryStale(recovery, stale);
    return;
  }
  if (expansionPullStatus.value?.kind !== "ready") {
    expansionError.value = `${recovery.model} is not ready on ${recovery.route.label}.`;
    return;
  }

  const retryId = ++recoveryRetryId;
  const focus = captureReplacementFocus(recovery.replacePrepared);
  expansionRunning.value = true;
  expansionError.value = "";
  try {
    // The immutable recovery record owns the style: a resumed pull re-requests
    // with exactly the directive the user saw frozen, not the live chip.
    const styleDirective = styleHint(recovery.inputs.stylePreset ?? "");
    const response = await expandPrompt(
      recovery.inputs.sourcePrompt,
      {
        variations: recovery.inputs.requestedCount,
        ...(recovery.inputs.family ? { modelFamily: recovery.inputs.family } : {}),
        ...(styleDirective ? { style: styleDirective } : {}),
      },
      recovery.route.target,
    );
    if (unmounted || retryId !== recoveryRetryId || expansionRecovery.value?.id !== recovery.id) {
      return;
    }
    const changed = recoveryStaleReason(recovery);
    if (changed) {
      markExpansionRecoveryStale(recovery, changed);
      return;
    }
    const prompts = validateExpandedPrompts(response.expanded, recovery.inputs.requestedCount);
    commitExpandedPrompts(
      { ...recovery.inputs },
      { ...recovery.route, target: { ...recovery.route.target } },
      prompts,
      recovery.requestToken,
      recovery.replacePrepared,
      focus,
    );
  } catch (error) {
    if (unmounted || retryId !== recoveryRetryId || expansionRecovery.value?.id !== recovery.id) {
      return;
    }
    expansionError.value = `Expansion failed on ${recovery.route.label}: ${describeTransportError(
      error,
      recovery.route.label,
    )}`;
  } finally {
    if (!unmounted && retryId === recoveryRetryId) expansionRunning.value = false;
  }
}

function revokeObjectUrl(url: string): void {
  URL.revokeObjectURL(url);
  objectUrls.delete(url);
}

const sourceFitCache = new SourceFitPreprocessCache();

async function prepareGenerationRequest(
  target: ApiTarget,
  draft: GenerateForm,
  isCurrent: () => boolean = () => true,
) {
  const draftCaps = generationCapabilitiesForFamily(draft.family);
  if (draftCaps.supportsImg2img && draftCaps.sourceImageMode === "single" && draft.sourceImage) {
    const result = await applySourceFitPreprocess(
      {
        source: draft.sourceImage,
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
            image,
            model,
            target,
            onProgress: (message) => {
              if (isCurrent()) setGenerationStatus(message);
            },
          }),
        onStatus: (message) => {
          if (isCurrent()) setGenerationStatus(message);
        },
      },
    );
    draft.sourceImage = result.source;
    draft.maskImage = result.mask;
  }
  const mediaBudgetError = mobileMediaBudgetValidationError(draft);
  if (mediaBudgetError) throw new Error(mediaBudgetError);
  return buildRequest(draft);
}

async function generate(): Promise<void> {
  const prepared = preparedBatch.value;
  if (effectiveBatchSize.value > 1 && !prepared) {
    await expandForCurrentBatch();
    return;
  }
  if (
    prepared &&
    (preparedStaleReasons.value.length > 0 ||
      prepared.prompts.some((prompt) => !prompt.text.trim()))
  ) {
    return;
  }
  if (quickExpansionSnapshot.value && quickStaleReasons.value.length > 0) {
    expansionError.value = `${quickStaleReasons.value.join(" ")} Undo or expand again before developing.`;
    return;
  }

  const preparedSubmission = prepared
    ? {
        batchId: prepared.batchId,
        promptIds: prepared.prompts.map((prompt) => prompt.id),
        prompts: prepared.prompts.map((prompt) => prompt.text.trim()),
        originalPrompt: prepared.sourcePrompt,
        route: { ...prepared.route, target: { ...prepared.route.target } },
      }
    : null;
  const preparedSection = preparedSubmission
    ? document.querySelector<HTMLElement>("[data-test='mobile-prepared-expansion']")
    : null;
  const preparedSubmissionOwnedFocus = !!preparedSection?.contains(document.activeElement);
  const quickSubmission = !preparedSubmission
    ? quickExpansionSnapshot.value
      ? {
          requestToken: quickExpansionSnapshot.value.requestToken,
          route: {
            ...quickExpansionSnapshot.value.route,
            target: { ...quickExpansionSnapshot.value.route.target },
          },
        }
      : null
    : null;
  const host = preparedSubmission
    ? hosts.value.find((candidate) => candidate.id === preparedSubmission.route.hostId)
    : quickSubmission
      ? hosts.value.find((candidate) => candidate.id === quickSubmission.route.hostId)
      : selectedHost.value;
  const route = preparedSubmission?.route ?? quickSubmission?.route ?? selectedRoute.value;
  const target = route?.target ?? null;
  if (
    !host ||
    !route ||
    !target ||
    !form.prompt.trim() ||
    !selectedModelAvailable.value ||
    !seedValid.value ||
    !parameterValid.value ||
    !sourceControlsValid.value ||
    !resolutionValid.value ||
    !basicParametersValid.value ||
    !!mobileMediaBudgetError.value ||
    preparingGeneration.value
  )
    return;

  const draft = cloneGenerateForm(form);
  const draftCaps = generationCapabilitiesForFamily(draft.family);
  // The composer style preset is baked into the OUTGOING request at submit —
  // the textarea and negative field are never mutated. Reviewed prepared
  // prompts ship verbatim (the style already reached them through the
  // expansion directive; staleness pins the chip to the frozen style), so the
  // prompt half only applies to the ordinary path. The preset negative is
  // separate from the reviewed prompt text and merges for BOTH paths, gated
  // on the family's negative-prompt support (mirrors desktop).
  const styled = composeStyle(draft.prompt, draft.stylePreset, {
    supportsNegativePrompt: draftCaps.supportsNegativePrompt,
    negative: draft.negativePrompt,
  });
  if (!preparedSubmission) draft.prompt = styled.prompt;
  draft.negativePrompt = styled.negative ?? "";
  const batchSize = preparedSubmission
    ? preparedSubmission.prompts.length
    : draftCaps.forcesBatchSizeOne
      ? 1
      : draft.batchSize;
  const guardedSubmission = !!preparedSubmission || !!quickSubmission;
  const liveFormIdentity = guardedSubmission ? JSON.stringify(cloneGenerateForm(form)) : "";
  const token = submissionGuard.begin();
  const uiId = ++submissionUiId;
  const ownsPreparedSubmission = () =>
    !unmounted &&
    uiId === submissionUiId &&
    submissionGuard.isCurrent(token) &&
    (!preparedSubmission || preparedBatch.value?.batchId === preparedSubmission.batchId);
  const releasePreparedSubmission = () => {
    if (ownsPreparedSubmission()) preparedSubmitting.value = false;
  };
  let request: GenerateRequest;
  preparingGeneration.value = true;
  preparedSubmitting.value = !!preparedSubmission;
  try {
    request = await prepareGenerationRequest(target, draft, () => submissionGuard.isCurrent(token));
  } catch (error) {
    if (!ownsPreparedSubmission()) return;
    setGenerationStatus(describeTransportError(error, route.label), true);
    generationAnnouncement.value = `Couldn’t prepare the source image. ${progress.value}`;
    releasePreparedSubmission();
    return;
  } finally {
    if (!unmounted && uiId === submissionUiId) preparingGeneration.value = false;
  }

  if (!submissionGuard.isCurrent(token)) {
    return;
  }
  if (guardedSubmission && JSON.stringify(cloneGenerateForm(form)) !== liveFormIdentity) {
    setGenerationStatus("The generation inputs changed while the source was being prepared.");
    generationAnnouncement.value = `${progress.value} Review the current inputs before developing.`;
    releasePreparedSubmission();
    return;
  }
  if (
    !sameFrozenHost(
      route,
      hosts.value.find((candidate) => candidate.id === route.hostId),
    )
  ) {
    expansionError.value = `${route.label}'s connection details changed. Refresh or undo before developing.`;
    releasePreparedSubmission();
    return;
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
      releasePreparedSubmission();
      return;
    }
  } else if (quickSubmission) {
    if (
      quickExpansionSnapshot.value?.requestToken !== quickSubmission.requestToken ||
      quickStaleReasons.value.length > 0
    ) {
      releasePreparedSubmission();
      return;
    }
  }

  const chainRouting = decideGenerateRequestRouting(request, draft.family);
  if (chainRouting.kind === "reject") {
    setGenerationStatus(chainRouting.reason);
    generationAnnouncement.value = chainRouting.reason;
    releasePreparedSubmission();
    return;
  }
  if (chainRouting.kind === "chain" && unsupportedAutoChainFields(request).length > 0) {
    setGenerationStatus("These options can’t be preserved during long-video chaining.");
    generationAnnouncement.value = `${progress.value} Remove the highlighted options or reduce Frames to 97 or fewer.`;
    releasePreparedSubmission();
    return;
  }

  const requestOptions = preparedSubmission
    ? {
        prompts: preparedSubmission.prompts,
        originalPrompt: preparedSubmission.originalPrompt,
        batchId: preparedSubmission.batchId,
      }
    : {};
  const { settled } = generation.submitBatch(
    request,
    batchSize,
    {
      hostId: route.hostId,
      label: route.label,
      kind: "remote",
      target: { ...route.target },
      mirrorRemoteOutput: false,
      retainEncodedResult: false,
      metadataOnlyCompletion: true,
    },
    chainRouting,
    requestOptions,
  );
  releasePreparedSubmission();
  if (preparedSubmission) {
    const active = document.activeElement;
    const restoreFocus =
      preparedSubmissionOwnedFocus &&
      (active === document.body || (!!active && !!preparedSection?.contains(active)));
    preparedBatch.value = null;
    if (restoreFocus) {
      void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
    }
  }
  if (
    quickSubmission &&
    quickExpansionSnapshot.value?.requestToken === quickSubmission.requestToken
  ) {
    quickExpansionSnapshot.value = null;
  }
  setGenerationStatus("Queued");
  generationAnnouncement.value = "";
  void settled.then(async (jobs) => {
    if (unmounted || jobs.length === 0) return;
    // iOS suspension kills every held SSE socket: jobs that settled with a
    // dead-transport error are re-queried against their frozen submission
    // route (finished prints render, still-running jobs re-attach, zombies
    // are cleared) BEFORE any summary copy is composed, so raw transport
    // text never reaches the status line or the announcement channel.
    await reconcileInterruptedGenerationJobs(jobs, {
      target: { ...route.target },
      hostLabel: route.label,
      chain: chainRouting.kind === "chain",
      refreshResultUrl: (clientId) =>
        void generation.refreshRemoteResultUrl(clientId).catch(() => {
          // The reactive job carries the directed, user-visible error.
        }),
      isActive: () => !unmounted,
    });
    if (unmounted) return;
    for (const candidate of jobs) handledGenerationClientIds.add(candidate.clientId);
    const completed = jobs.filter(
      (candidate) => candidate.status === "complete" && candidate.result,
    );
    for (const candidate of completed) {
      void saveCompletedStillToPhotos(candidate.result!, route.target);
    }
    const latestCompleted = completed.at(-1);
    const unconfirmedCancellation = jobs.find((candidate) =>
      candidate.error?.includes("remote cancellation was not confirmed"),
    );
    const failed = jobs.find((candidate) => candidate.error && !isCancelledError(candidate.error));
    const failedError = failed?.error ? describeTransportError(failed.error, route.label) : null;
    const failedVariations = preparedSubmission
      ? jobs.flatMap((candidate, index) => {
          if (!candidate.error || isCancelledError(candidate.error)) return [];
          const prompt =
            candidate.prompt.length > 120 ? `${candidate.prompt.slice(0, 117)}…` : candidate.prompt;
          return [
            `Variation ${index + 1}, “${prompt}”, failed: ${describeTransportError(
              candidate.error,
              route.label,
            )}`,
          ];
        })
      : [];
    const preparedFailureSummary = failedVariations.join(" ");
    const failedCount = jobs.filter(
      (candidate) => candidate.error && !isCancelledError(candidate.error),
    ).length;
    const cancelled = jobs.find((candidate) => isCancelledError(candidate.error));

    if (latestCompleted?.result) {
      latestResultClientId.value = latestCompleted.clientId;
      if (latestCompleted.resultError) {
        const previewDetail = describeTransportError(latestCompleted.resultError, route.label);
        setGenerationStatus(previewDetail, true);
        generationAnnouncement.value = `${completed.length} of ${jobs.length} generations completed, but the latest preview is unavailable. ${previewDetail}`;
      } else {
        setGenerationStatus(
          `${completed.length > 1 ? `${completed.length} prints · ` : ""}${completionSummary(
            latestCompleted.result,
          )}`,
        );
        generationAnnouncement.value =
          completed.length === 1 && jobs.length === 1
            ? "Generation completed."
            : `${completed.length} of ${jobs.length} generations completed.`;
      }
      if (unconfirmedCancellation?.error || failedError) {
        setGenerationStatus(
          [
            `${completed.length} of ${jobs.length} completed`,
            failedError,
            unconfirmedCancellation?.error,
          ]
            .filter(Boolean)
            .join(" · "),
          true,
        );
        generationAnnouncement.value = [
          `${completed.length} generations completed.`,
          failedError
            ? preparedSubmission
              ? `${failedCount} failed. ${preparedFailureSummary}`
              : `${failedCount} failed. ${failedError}`
            : "",
          unconfirmedCancellation?.error
            ? `Cancellation failed. ${unconfirmedCancellation.error}`
            : "",
        ]
          .filter(Boolean)
          .join(" ");
      }
      if (tab.value === "gallery") void refreshGallery();
    } else if (unconfirmedCancellation?.error || failedError) {
      setGenerationStatus(
        [failedError, unconfirmedCancellation?.error].filter(Boolean).join(" · "),
        true,
      );
      generationAnnouncement.value = [
        failedError
          ? preparedSubmission
            ? `Generation failed. ${preparedFailureSummary}`
            : `Generation failed. ${failedError}`
          : "",
        unconfirmedCancellation?.error
          ? `Cancellation failed. ${unconfirmedCancellation.error}`
          : "",
      ]
        .filter(Boolean)
        .join(" ");
    } else if (cancelled) {
      setGenerationStatus("Cancelled");
      generationAnnouncement.value = `${jobs.length} generation${jobs.length === 1 ? "" : "s"} cancelled.`;
    }
    // Only terminal jobs whose callbacks have run are eligible: multiple
    // completion microtasks cannot prune one another before they promote the
    // correct latest result. The UI renders one result, so retain one Blob.
    generation.prune(1, latestResultClientId.value, handledGenerationClientIds);
    for (const clientId of handledGenerationClientIds) {
      if (!generation.jobs.some((candidate) => candidate.clientId === clientId)) {
        handledGenerationClientIds.delete(clientId);
      }
    }
  });
}

async function cancelGeneration(job: Job): Promise<void> {
  try {
    await generation.cancel(job.clientId);
    if (job.status === "complete" && job.result) {
      latestResultClientId.value = job.clientId;
      setGenerationStatus(completionSummary(job.result));
      generationAnnouncement.value = "Generation completed.";
      if (tab.value === "gallery") void refreshGallery();
    } else if (job.error && !isCancelledError(job.error)) {
      const detail = describeTransportError(job.error, job.hostLabel);
      setGenerationStatus(detail, true);
      generationAnnouncement.value = `Generation failed. ${detail}`;
    } else {
      setGenerationStatus("Cancelled");
      generationAnnouncement.value = "Generation cancelled.";
    }
  } catch (error) {
    setGenerationStatus(describeTransportError(error, job.hostLabel), true);
    generationAnnouncement.value = `Cancellation failed. ${progress.value}`;
  }
}

function renewGeneratedResult(force: boolean): void {
  const job = latestResultJob.value;
  if (!job?.metadataOnlyCompletion || !job.result || job.resultUrlLoading) return;
  const previousUrl = job.resultUrl;
  void generation
    .refreshRemoteResultUrl(job.clientId, force)
    .then(() => {
      if (latestResultClientId.value !== job.clientId || job.resultError || !job.resultUrl) return;
      if (force) resultMediaLoadKey.value += 1;
      setGenerationStatus(completionSummary(job.result!));
      if (force || job.resultUrl !== previousUrl) {
        generationAnnouncement.value = "Result preview refreshed.";
      }
    })
    .catch(() => {
      // The store exposes the directed failure through resultError.
    });
}

function generatedMediaReady(): void {
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
}

function recoverGeneratedMedia(): void {
  const job = latestResultJob.value;
  if (!job || job.resultUrlLoading) return;
  if (resultMediaRecoveryClientId !== job.clientId) {
    resultMediaRecoveryClientId = job.clientId;
    resultMediaRecoveryAttempts = 0;
  }
  if (resultMediaRecoveryAttempts === 0) {
    resultMediaRecoveryAttempts = 1;
    renewGeneratedResult(true);
    return;
  }

  if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
  job.resultUrl = null;
  job.resultUrlIsObjectUrl = false;
  job.resultUrlExpiresAt = null;
  job.resultError = "Couldn’t load this generated print from the host.";
}

function retryGeneratedPreview(): void {
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
  renewGeneratedResult(true);
}

async function thumbnailUrl(target: ApiTarget, filename: string): Promise<string> {
  const response = await apiFetchTo(target, galleryMediaPath(filename, "host", true));
  const url = URL.createObjectURL(await response.blob());
  objectUrls.add(url);
  return url;
}

function refreshGallery(): Promise<void> {
  if (selectedPrint.value) {
    // The viewer uses the grid thumbnail as its placeholder/poster and
    // returns focus to that tile. Keep both alive until the viewer closes.
    galleryRefreshDeferred = true;
    return Promise.resolve();
  }
  galleryRefreshRequested = true;
  if (!galleryRefreshTask) {
    const operation = enqueueGalleryOperation(async () => {
      while (galleryRefreshRequested) {
        if (selectedPrint.value) {
          galleryRefreshRequested = false;
          galleryRefreshDeferred = true;
          break;
        }
        galleryRefreshRequested = false;
        await performGalleryRefresh();
      }
    });
    galleryRefreshTask = operation.then(
      async () => {
        galleryRefreshTask = null;
        // A request can arrive after the loop's final condition but before
        // this continuation. Adopt the re-armed task so every caller waits
        // for the refresh it requested.
        if (galleryRefreshRequested) await refreshGallery();
      },
      (error: unknown) => {
        galleryRefreshTask = null;
        throw error;
      },
    );
  }
  return galleryRefreshTask;
}

function enqueueGalleryOperation(operation: () => Promise<void>): Promise<void> {
  const task = galleryOperationTail.then(operation, operation);
  galleryOperationTail = task.catch(() => {});
  return task;
}

async function performGalleryRefresh(): Promise<void> {
  galleryLoading.value = true;
  galleryError.value = "";
  const prior = gallery.value;
  gallery.value = [];
  for (const item of prior) revokeObjectUrl(item.thumbnailUrl);
  const results = await Promise.allSettled(
    hosts.value.map(async (host) => {
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
      const prints = await apiJsonTo<GalleryImage[]>(target, "/api/gallery");
      return prints.map((print) => ({
        ...print,
        hostId: host.id,
        hostName: host.name,
        target,
      }));
    }),
  );
  pendingGallery = results
    .flatMap((result) => (result.status === "fulfilled" ? result.value : []))
    .sort((a, b) => b.timestamp - a.timestamp);
  const failed = results.filter((result) => result.status === "rejected").length;
  if (failed) galleryError.value = `${failed} host${failed === 1 ? "" : "s"} unavailable`;
  await loadMoreGalleryPage();
  galleryLoading.value = false;
}

function loadMoreGallery(): Promise<void> {
  return enqueueGalleryOperation(loadMoreGalleryPage);
}

async function loadMoreGalleryPage(): Promise<void> {
  galleryLoadingMore.value = true;
  const page = pendingGallery.splice(0, 40);
  for (let offset = 0; offset < page.length; offset += 4) {
    const batch = await Promise.allSettled(
      page.slice(offset, offset + 4).map(async ({ target, ...print }) => ({
        ...print,
        target,
        thumbnailUrl: await thumbnailUrl(target, print.filename),
      })),
    );
    gallery.value.push(
      ...batch.flatMap((result) => (result.status === "fulfilled" ? [result.value] : [])),
    );
  }
  markMobileLibrarySeen(gallery.value);
  galleryRemaining.value = pendingGallery.length;
  galleryLoadingMore.value = false;
}

async function reusePrint(print: GalleryPrint): Promise<void> {
  if (reusingPrint.value || print.metadata_synthetic || !print.metadata.prompt?.trim()) return;
  reusingPrint.value = true;
  reusePrintError.value = "";
  try {
    if (selectedHostId.value !== print.hostId) {
      selectedHostId.value = print.hostId;
    }
    if (modelsHostId.value !== print.hostId) {
      if (!(await refreshModels())) {
        reusePrintError.value = `Couldn’t load models from ${print.hostName}. Check the host and try again.`;
        return;
      }
    }
    if (generationModels.value.length === 0) {
      reusePrintError.value = `${print.hostName} has no downloaded models available.`;
      return;
    }
    const reuse = applyMobileGalleryMetadata(form, print.metadata, generationModels.value);
    setGenerationStatus(
      reuse.substitutedModel
        ? `The original model isn’t installed on ${print.hostName}; using ${reuse.modelName}.`
        : "Prompt settings restored",
    );
    selectedPrint.value = null;
    // The next Gallery visit performs its normal refresh; do not refetch the
    // grid while navigating directly to the restored prompt.
    galleryRefreshDeferred = false;
    tab.value = "generate";
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  } finally {
    reusingPrint.value = false;
  }
}

async function useSelectedPrintAsSource(): Promise<void> {
  const print = selectedPrint.value;
  if (!print || !canUseSelectedPrintAsSource.value || usingPrintAsSource.value) return;
  usingPrintAsSource.value = true;
  reusePrintError.value = "";
  try {
    const response = await apiFetchTo(print.target, galleryMediaPath(print.filename, "host"));
    const qwenEdit = caps.value.sourceImageMode === "qwen-edit";
    const existingBytes = inlineGenerationMediaBytes(form, qwenEdit ? null : "sourceImage");
    const exceedsBudget = (incomingBytes: number) =>
      existingBytes + incomingBytes > MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES;
    // Reject from Content-Length before materialising the response Blob when
    // the host provides it; then verify the actual Blob size for older hosts
    // and chunked responses before the ~4/3 base64 expansion.
    const declaredBytes = Number(response.headers?.get("content-length") ?? Number.NaN);
    if (Number.isFinite(declaredBytes) && declaredBytes >= 0 && exceedsBudget(declaredBytes)) {
      throw new Error(MOBILE_MEDIA_BUDGET_ERROR);
    }
    const blob = await response.blob();
    if (blob.size === 0) throw new Error("That gallery image is empty.");
    if (exceedsBudget(blob.size)) throw new Error(MOBILE_MEDIA_BUDGET_ERROR);
    const base64 = await blobToBase64(blob);
    if (qwenEdit) {
      form.imageAttachments = [base64, ...form.imageAttachments];
      setGenerationStatus("Added gallery print as the edit target");
    } else {
      form.sourceImage = base64;
      form.sourceImageName = print.filename;
      setGenerationStatus("Gallery print selected as source");
    }
    selectedPrint.value = null;
    galleryRefreshDeferred = false;
    tab.value = "generate";
  } catch (error) {
    reusePrintError.value = describeTransportError(error, print.hostName);
  } finally {
    usingPrintAsSource.value = false;
  }
}

function openPrint(print: GalleryPrint): void {
  reusePrintError.value = "";
  selectedPrint.value = print;
}

function isFreshMobilePrint(print: GalleryPrint): boolean {
  const seenAt = librarySeenAtBaseline[print.hostId];
  return libraryPreviouslyVisited && seenAt != null && print.timestamp > seenAt;
}

function markMobileLibrarySeen(prints: GalleryPrint[]): void {
  const seenAt = { ...librarySeenAtBaseline };
  for (const print of prints) {
    seenAt[print.hostId] = Math.max(seenAt[print.hostId] ?? 0, print.timestamp);
  }
  localStorage.setItem(LIBRARY_SEEN_AT_KEY, JSON.stringify(seenAt));
  localStorage.removeItem(LEGACY_LIBRARY_SEEN_KEY);
  localStorage.setItem(LIBRARY_VISITED_KEY, "true");
}

function navigateSelectedPrint(delta: -1 | 1): void {
  const next = gallery.value[selectedPrintIndex.value + delta];
  if (!next || reusingPrint.value || usingPrintAsSource.value) return;
  reusePrintError.value = "";
  selectedPrint.value = next;
}

function closePrint(): void {
  if (reusingPrint.value || usingPrintAsSource.value) return;
  reusePrintError.value = "";
  selectedPrint.value = null;
  if (galleryRefreshDeferred || galleryRefreshRequested) {
    galleryRefreshDeferred = false;
    // The viewer normally restores focus to its tile. A deferred refresh — or
    // one still queued behind Load older — will replace that tile, so move
    // focus to the stable Gallery tab first.
    void nextTick(() => {
      document.querySelector<HTMLButtonElement>("[data-test='mobile-tab-gallery']")?.focus();
      void refreshGallery();
    });
  }
}

function reuseSelectedPrint(): void {
  const print = selectedPrint.value;
  if (print) void reusePrint(print);
}

function openSettings(): void {
  settingsOpen.value = true;
  void nextTick(() => settingsBackButton.value?.focus());
}

function closeSettings(): void {
  settingsOpen.value = false;
  void nextTick(() => settingsButton.value?.focus());
}

function manageHostsFromSettings(): void {
  settingsOpen.value = false;
  tab.value = "hosts";
  void nextTick(() => {
    document.querySelector<HTMLButtonElement>("[data-test='mobile-tab-hosts']")?.focus();
  });
}

function updateSettings(patch: Partial<MobileSettings>): void {
  Object.assign(mobileSettings, persistMobileSettings(mobileSettings, patch));
}

watch(selectedHostId, (id, previousId) => {
  if (id !== previousId) clearHostScopedGenerationSelections();
  if (id) localStorage.setItem(SELECTED_KEY, id);
  else localStorage.removeItem(SELECTED_KEY);
});

watch(createMode, (mode) => {
  try {
    localStorage.setItem(CREATE_MODE_KEY, mode);
  } catch {
    // The selected mode can fall back to Single on the next launch.
  }
});

watch(
  () => {
    const attempt = expansionPullAttempt.value;
    const recovery = expansionRecovery.value;
    if (
      !attempt ||
      !recovery ||
      attempt.recoveryId !== recovery.id ||
      attempt.jobId ||
      attempt.observedJobId
    ) {
      return "";
    }
    const bucket = mobileDownloads.downloadsByHost[recovery.route.hostId];
    return bucket
      ? [...bucket.activeJobs, ...bucket.queued]
          .map((job) => `${job.id}:${job.model}:${job.catalog_id ?? ""}`)
          .join("|")
      : "";
  },
  () => {
    const attempt = expansionPullAttempt.value;
    const recovery = expansionRecovery.value;
    if (
      !attempt ||
      !recovery ||
      attempt.recoveryId !== recovery.id ||
      attempt.jobId ||
      attempt.observedJobId
    ) {
      return;
    }
    const bucket = mobileDownloads.downloadsByHost[recovery.route.hostId];
    const match = bucket
      ? [...bucket.activeJobs, ...bucket.queued].find(
          (job) =>
            expansionPullJobMatchesModel(job, recovery.model) &&
            (attempt.allowExistingInFlight || !attempt.baselineJobIds.includes(job.id)),
        )
      : undefined;
    if (match) attempt.observedJobId = match.id;
  },
  { flush: "sync" },
);

watch(
  () => {
    const recovery = expansionRecovery.value;
    const attempt = expansionPullAttempt.value;
    if (!recovery || !attempt || attempt.recoveryId !== recovery.id) return "";
    const host = hosts.value.find((candidate) => candidate.id === recovery.route.hostId);
    return [
      form.prompt,
      form.model,
      form.batchSize,
      selectedHostId.value,
      host?.online ?? false,
      host?.baseUrl ?? "",
      host?.apiKey ?? "",
      host?.instanceId ?? "",
    ].join("\u0000");
  },
  () => {
    const recovery = expansionRecovery.value;
    const attempt = expansionPullAttempt.value;
    if (!recovery || !attempt || attempt.recoveryId !== recovery.id) return;
    const stale = recoveryStaleReason(recovery);
    if (stale) markExpansionRecoveryStale(recovery, stale);
  },
  { flush: "sync" },
);

watch(tab, (next) => {
  // The primary destinations share this one WebView scroller. Reset it after
  // Vue swaps the destination so a long Library cannot open Models, Machines,
  // or Create at the same inherited offset.
  void nextTick(() => {
    if (!mobileContent.value) return;
    mobileContent.value.scrollTop = 0;
    mobileContent.value.scrollLeft = 0;
  });
  if (next === "gallery") {
    librarySeenAtBaseline = loadLibrarySeenAt();
    libraryPreviouslyVisited = localStorage.getItem(LIBRARY_VISITED_KEY) === "true";
    void refreshGallery();
  }
  if (next !== "hosts") hostDetailId.value = "";
});

// One failed model load must not become a manual-Retry dead end: the 10s
// probe already self-heals `host.online`, so a false→true transition on the
// selected host re-runs the (epoch-guarded) model load automatically.
watch(
  () => selectedHost.value?.online ?? false,
  (online, wasOnline) => {
    if (online && !wasOnline && modelLoadError.value && !loadingModels.value) {
      void refreshModels();
    }
  },
);

/**
 * iOS froze this WebView and tore down every socket while the app was
 * backgrounded. On return: probe hosts immediately (instead of waiting out
 * the 10s cadence), recover a failed model list, renew the promoted result's
 * media ticket if it aged out, and refresh the Library if it is on screen.
 * Interrupted generation streams settle through their own reconciliation in
 * the submit path's settled handler.
 */
function handleForegroundResume(): void {
  if (unmounted || document.visibilityState === "hidden") return;
  probeHosts();
  if (modelLoadError.value && !loadingModels.value) void refreshModels();
  renewGeneratedResult(false);
  if (tab.value === "gallery") void refreshGallery();
}

watch(resultPreviewError, (error) => {
  if (!error) return;
  setGenerationStatus(error, true);
  generationAnnouncement.value = `Generation completed, but its preview is unavailable. ${error}`;
});

onMounted(async () => {
  if ("__TAURI_INTERNALS__" in window) {
    void import("@tauri-apps/api/app")
      .then(({ getVersion }) => getVersion())
      .then((version) => {
        appVersion.value = version;
      })
      .catch(() => {});
  }
  await hydrateApiKeys();
  if (unmounted) return;
  recoverMobileSequence();
  // Start the cadence before awaiting individual tailnet hosts. One slow host
  // must not prevent every other saved host from being probed on schedule.
  hostProbeTimer = setInterval(probeHosts, 10_000);
  document.addEventListener("visibilitychange", handleForegroundResume);
  window.addEventListener("pageshow", handleForegroundResume);
  if (selectedHost.value) {
    await Promise.all([
      refreshModels(),
      ...hosts.value
        .filter((host) => host.id !== selectedHostId.value)
        .map((host) => probeHost(host)),
    ]);
  } else {
    tab.value = "hosts";
  }
});

onBeforeUnmount(() => {
  unmounted = true;
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  submissionUiId += 1;
  recoveryRetryId += 1;
  expansionPullRequestId += 1;
  clearExpansionRecovery();
  document.removeEventListener("visibilitychange", handleForegroundResume);
  window.removeEventListener("pageshow", handleForegroundResume);
  if (hostProbeTimer) clearInterval(hostProbeTimer);
  hostProbeTimer = null;
  clearSequencePoll();
  for (const id of [...hostProbes.keys()]) cancelHostProbe(id);
  generation.resetJobs();
  for (const url of objectUrls) URL.revokeObjectURL(url);
});
</script>

<template>
  <main class="mobile-shell" :class="{ 'is-settings-open': settingsOpen }">
    <header v-if="settingsOpen" class="mobile-header mobile-settings-nav">
      <button
        ref="settingsBackButton"
        class="mobile-settings-back"
        type="button"
        data-test="mobile-settings-back"
        @click="closeSettings"
      >
        <span aria-hidden="true">‹</span>
        Back
      </button>
      <strong>Settings</strong>
      <span class="mobile-settings-nav-spacer" aria-hidden="true" />
    </header>
    <header v-else class="mobile-header">
      <div class="mobile-wordmark">Mold</div>
      <div class="mobile-header-actions">
        <div class="host-chip">{{ selectedHost?.name ?? "Remote only" }}</div>
        <button
          ref="settingsButton"
          class="mobile-settings-button"
          type="button"
          aria-label="Open settings"
          data-test="mobile-open-settings"
          @click="openSettings"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M7 14v6" />
          </svg>
        </button>
      </div>
    </header>

    <p class="sr-only" aria-live="polite" aria-atomic="true">
      {{ queueAnnouncement }}
    </p>
    <p class="sr-only" aria-live="polite" aria-atomic="true">
      {{ generationAnnouncement }}
    </p>

    <section ref="mobileContent" class="mobile-content">
      <MobileSettingsView
        v-if="settingsOpen"
        :settings="mobileSettings"
        :host-count="hosts.length"
        :app-version="appVersion"
        @update="updateSettings"
        @manage-hosts="manageHostsFromSettings"
      />
      <template v-else-if="tab === 'generate'">
        <div v-if="!selectedHost" class="empty-state">
          <div>
            <h1 class="section-title">Connect a host</h1>
            <p>Generation runs on a remote Mold engine.</p>
            <button class="primary-button" type="button" @click="tab = 'hosts'">Add host</button>
          </div>
        </div>
        <template v-else>
          <h1 class="section-title">Create</h1>
          <p class="section-note">Develop on {{ selectedHost.name }}</p>
          <label v-if="hosts.length > 1" class="field">
            <span>Host</span>
            <select
              class="control"
              :value="selectedHostId"
              data-test="mobile-generate-host"
              @change="selectHost(($event.target as HTMLSelectElement).value)"
            >
              <option v-for="host in hosts" :key="host.id" :value="host.id">
                {{ host.name }}{{ host.online ? "" : " · offline" }}
              </option>
            </select>
          </label>
          <div
            class="mobile-create-mode"
            role="radiogroup"
            aria-label="Create mode"
            data-test="mobile-create-mode"
          >
            <button
              type="button"
              role="radio"
              :aria-checked="createMode === 'single'"
              :class="{ active: createMode === 'single' }"
              @click="createMode = 'single'"
            >
              Single
            </button>
            <button
              type="button"
              role="radio"
              :aria-checked="createMode === 'sequence'"
              :class="{ active: createMode === 'sequence' }"
              @click="createMode = 'sequence'"
            >
              Sequence
            </button>
          </div>
          <MobileSequenceComposer
            v-if="createMode === 'sequence' && selectedTarget"
            :models="sequenceModels"
            :target="selectedTarget"
            :host-name="selectedHost.name"
            :job="sequenceJob"
            :starting="sequenceStarting"
            :error="sequenceError"
            @submit="submitMobileSequence"
            @cancel="cancelMobileSequence"
            @dismiss="dismissMobileSequence"
            @browse="openCatalog(selectedHost.id)"
          />
          <template v-else>
            <label class="field">
              <span>Model</span>
              <select
                v-model="form.model"
                class="control"
                :disabled="loadingModels || generationModels.length === 0"
                @change="changeModel"
              >
                <option v-if="!form.model" value="" disabled>
                  {{ loadingModels ? "Loading models…" : "No generation models available" }}
                </option>
                <option v-if="form.model && !selectedModelAvailable" :value="form.model" disabled>
                  {{ modelLabel(form.model) }} · not installed
                </option>
                <option v-for="model in generationModels" :key="model.name" :value="model.name">
                  {{ modelDisplayName(model) }}
                </option>
              </select>
            </label>
            <div
              v-if="modelLoadError"
              class="mobile-model-state is-error"
              role="alert"
              data-test="mobile-model-error"
            >
              <p>{{ modelLoadError }}</p>
              <button
                class="secondary-button"
                type="button"
                data-test="mobile-model-retry"
                @click="refreshModels"
              >
                Retry
              </button>
            </div>
            <div
              v-else-if="!loadingModels && generationModels.length === 0"
              class="mobile-model-state"
              data-test="mobile-model-empty"
            >
              <p>No downloaded generation model is available on {{ selectedHost.name }}.</p>
              <button class="secondary-button" type="button" @click="openCatalog(selectedHost.id)">
                Open Catalog
              </button>
            </div>
            <label class="field">
              <span>Prompt</span>
              <textarea
                id="mobile-prompt"
                v-model="form.prompt"
                class="control"
                placeholder="Describe the print…"
              />
            </label>
            <MobileStyleChips v-model="form.stylePreset" />
            <MobilePromptTools
              v-if="selectedTarget"
              :form="form"
              :target="selectedTarget"
              :running="expansionRunning"
              :can-undo="quickExpansionOriginal !== null"
              :blocked="!!preparedBatch"
              :models="generationModels"
              @expand="expandForCurrentBatch()"
              @undo="restoreQuickExpansion"
            />
            <div
              v-if="quickStaleReasons.length"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-quick-expansion-stale"
            >
              <div class="mobile-generate-validation-copy">
                <p>{{ quickStaleReasons.join(" ") }} Choose how to continue.</p>
                <button
                  class="mobile-error-copy"
                  type="button"
                  aria-label="Copy error message"
                  title="Copy error message"
                  @click="copyQuickExpansionError"
                >
                  <svg
                    width="17"
                    height="17"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.7"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    aria-hidden="true"
                  >
                    <rect x="8" y="8" width="11" height="11" rx="2" />
                    <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
                  </svg>
                </button>
              </div>
              <div class="mobile-generate-validation-actions">
                <button
                  class="primary-button mobile-touch-action"
                  type="button"
                  data-test="mobile-reexpand-and-develop"
                  :disabled="expansionRunning || preparedSubmitting"
                  @click="reexpandAndDevelop"
                >
                  Re-expand and Develop
                </button>
                <button
                  class="secondary-button mobile-touch-action"
                  type="button"
                  data-test="mobile-develop-expanded-anyway"
                  :disabled="expansionRunning || preparedSubmitting"
                  @click="developExpandedAnyway"
                >
                  Develop anyway
                </button>
                <button
                  class="secondary-button mobile-touch-action"
                  type="button"
                  @click="restoreQuickExpansion"
                >
                  Restore original
                </button>
              </div>
            </div>
            <div
              v-if="expansionError && !expansionMissingModel && !preparedBatch"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-expansion-error"
            >
              <div class="mobile-generate-validation-copy">
                <p>{{ expansionError }}</p>
                <button
                  class="mobile-error-copy"
                  type="button"
                  aria-label="Copy error message"
                  title="Copy error message"
                  @click="copyMobileError(expansionError)"
                >
                  <svg
                    width="17"
                    height="17"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.7"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    aria-hidden="true"
                  >
                    <rect x="8" y="8" width="11" height="11" rx="2" />
                    <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
                  </svg>
                </button>
              </div>
            </div>
            <MobileExpansionPullStatus
              v-if="expansionMissingModel && expansionPullStatus"
              :model="expansionMissingModel.model"
              :host-label="expansionMissingModel.route.label"
              :error="expansionError"
              :status="expansionPullStatus"
              :eta-seconds="expansionPullEtaSeconds"
              :models="generationModels"
              @pull="pullExpansionModel"
              @retry-expansion="retryExpansionAfterPull"
            />
            <MobilePreparedExpansionBatch
              v-if="preparedBatch"
              :batch="preparedBatch"
              :stale-reasons="preparedStaleReasons"
              :preparing="expansionRunning"
              :error="expansionMissingModel ? '' : expansionError"
              :submitting="preparedSubmitting"
              @edit="editPreparedPrompt"
              @remove="removePreparedPrompt"
              @collapse="collapsePreparedBatch"
              @regenerate="expandForCurrentBatch(true, preparedBatch.route)"
              @refresh="expandForCurrentBatch(true)"
              @discard="discardPreparedBatch"
              @generate="generate"
            />
            <p
              v-if="mobileMediaBudgetError"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-media-budget-error"
            >
              {{ mobileMediaBudgetError }}
            </p>

            <MobileResolutionPicker
              v-model:width="form.width"
              v-model:height="form.height"
              :family="form.family"
              :disabled="loadingModels"
              @validity-change="resolutionValid = $event"
            />
            <div class="field-grid">
              <label class="field">
                <span>Steps</span>
                <input
                  v-model.number="form.steps"
                  class="control"
                  type="number"
                  inputmode="numeric"
                  min="1"
                  max="100"
                  :aria-invalid="stepsError ? 'true' : undefined"
                />
              </label>
              <label class="field"
                ><span>Guidance</span
                ><input
                  v-model.number="form.guidance"
                  class="control"
                  type="number"
                  inputmode="decimal"
                  step="0.1"
                  min="0"
                  max="100"
                  :aria-invalid="guidanceError ? 'true' : undefined"
              /></label>
            </div>
            <p
              v-if="stepsError || guidanceError"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-basic-parameter-error"
            >
              {{ stepsError || guidanceError }}
            </p>
            <MobileSeedPicker
              v-model="form.seed"
              :last-seed="generation.lastSeedUsed"
              @validity-change="seedValid = $event"
            />

            <div class="mobile-advanced-row">
              <button
                class="mobile-advanced-trigger"
                type="button"
                data-test="mobile-open-advanced"
                @click="openAdvancedSheet"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                <span>Advanced (sampler, LoRA, source)</span>
                <span
                  v-if="advancedActiveCount > 0"
                  class="mobile-advanced-trigger-badge"
                  data-test="mobile-advanced-trigger-count"
                  >{{ advancedActiveCount }}</span
                >
              </button>
              <button
                class="mobile-settings-reset"
                type="button"
                data-test="mobile-settings-reset"
                aria-label="Reset settings to model defaults"
                @click="resetCreateSettings"
              >
                ↺ Reset
              </button>
            </div>

            <MobileAdvancedSheet
              :open="advancedSheetOpen"
              :count="advancedActiveCount"
              @close="closeAdvancedSheet"
              @reset="resetAdvancedSettings"
            >
              <MobileGenerateParameters
                :form="form"
                :upscalers="upscalers"
                :audio-output-supported="selectedGenerationModel?.supports_audio !== false"
                @validity-change="parameterValid = $event"
              />
              <label v-if="form.model && caps.supportsNegativePrompt" class="field">
                <span>Negative prompt</span>
                <input v-model="form.negativePrompt" class="control" placeholder="Optional" />
              </label>
              <details
                v-if="form.model && caps.supportsImg2img"
                class="mobile-native-disclosure"
                :open="
                  !!(form.sourceImage || form.controlImage || form.imageAttachments.length) ||
                  caps.sourceImageMode === 'qwen-edit'
                "
                data-test="mobile-source-disclosure"
              >
                <summary>
                  <span>{{ sourceSectionTitle }}</span>
                  <small>{{ sourceSectionSummary }}</small>
                </summary>
                <MobileSourceControls
                  :form="form"
                  :target="selectedTarget"
                  :control-models="controlModels"
                  :upscalers="upscalers"
                  @validity-change="sourceValid = $event"
                />
              </details>
              <MobileLoraControls
                v-if="selectedTarget"
                :form="form"
                :target="selectedTarget"
                @append-word="appendPromptWord"
              />
              <label class="field"
                ><span>Format</span
                ><select v-model="form.outputFormat" class="control">
                  <option v-for="format in outputFormats" :key="format" :value="format">
                    {{ format.toUpperCase() }}
                  </option>
                </select></label
              >
            </MobileAdvancedSheet>

            <MobileTemplates
              :form="form"
              :host-id="selectedHost.id"
              :models="generationModels"
              @load="loadTemplate"
            />

            <div class="mobile-estimate">
              <EstimateBadge :request="estimateRequest" :target="selectedTarget" />
            </div>
            <button
              v-if="!preparedBatch"
              class="primary-button"
              type="button"
              :disabled="
                !form.prompt.trim() ||
                !selectedModelAvailable ||
                !seedValid ||
                !parameterValid ||
                !sourceControlsValid ||
                !resolutionValid ||
                !basicParametersValid ||
                !!mobileMediaBudgetError ||
                preparingGeneration
              "
              data-test="mobile-develop-button"
              @click="generate"
            >
              {{ developButtonLabel }}
            </button>
            <section
              v-if="queuedJobs.length"
              class="mobile-generation-queue"
              aria-label="Generation queue"
              data-test="mobile-generation-queue"
            >
              <div class="mobile-generation-queue-head">
                <h2>Queue</h2>
                <span>{{ queuedJobs.length }} active</span>
              </div>
              <ol>
                <li
                  v-for="job in queuedJobs"
                  :key="job.clientId"
                  class="mobile-generation-job"
                  data-test="mobile-generation-job"
                >
                  <div class="mobile-generation-job-copy">
                    <p>{{ job.prompt }}</p>
                    <span>{{ modelLabel(job.model) }} · {{ job.hostLabel }}</span>
                  </div>
                  <div class="mobile-generation-job-action">
                    <span data-test="mobile-generation-status">{{ jobStatusCode(job) }}</span>
                    <button
                      class="mobile-generation-cancel"
                      type="button"
                      :aria-label="`Cancel ${job.prompt}`"
                      data-test="mobile-generation-cancel"
                      @click="cancelGeneration(job)"
                    >
                      Cancel
                    </button>
                  </div>
                </li>
              </ol>
            </section>
            <!-- Live develop bed: once the host streams latent previews the
               active print literally forms here — the preview's blur tightens
               with denoise progress while the Develop grain thins over it.
               Without previews the status line below stands alone, and the
               grain's rAF loop stays parked (nothing mounts), which keeps the
               WebKit compositor free during model load. -->
            <div
              v-if="activeGeneration && activeGeneration.previewUrl"
              class="mobile-develop-bed"
              data-test="mobile-develop-bed"
              aria-hidden="true"
              :style="{
                aspectRatio: `${activeGeneration.width} / ${activeGeneration.height}`,
                // The 55vh height cap rides the width axis (see mobile.css) so a
                // portrait bed shrinks instead of distorting its layered media.
                '--bed-ar': `${activeGeneration.width / Math.max(1, activeGeneration.height)}`,
              }"
            >
              <img
                class="mobile-develop-preview"
                data-test="mobile-develop-preview"
                :src="activeGeneration.previewUrl"
                alt=""
                :style="{
                  filter: `blur(${Math.max(2, 14 - 12 * jobProgress(activeGeneration))}px)`,
                }"
              />
              <DevelopCanvas
                :seed="activeGeneration.visualSeed"
                :progress="jobProgress(activeGeneration)"
                :phase="jobPhase(activeGeneration)"
                class="mobile-develop-grain"
                :style="{
                  opacity: String(Math.max(0.18, 1 - jobProgress(activeGeneration) * 0.9)),
                }"
              />
            </div>
            <div
              class="status-line"
              :class="{ 'error-text': generationStatusIsError }"
              data-test="mobile-generation-summary"
            >
              {{ generationStatus }}
            </div>
            <div v-if="resultPreviewError" class="result-preview-error" role="alert">
              <p class="status-line error-text">{{ resultPreviewError }}</p>
              <button class="secondary-button" type="button" @click="retryGeneratedPreview">
                Try preview again
              </button>
            </div>
            <video
              v-if="resultUrl && resultIsVideo"
              :key="`${latestResultJob?.clientId}:${resultMediaLoadKey}`"
              class="result-media"
              :src="resultUrl"
              controls
              playsinline
              @play="renewGeneratedResult(false)"
              @loadedmetadata="generatedMediaReady"
              @error="recoverGeneratedMedia"
            />
            <button
              v-else-if="resultUrl"
              class="result-media-button"
              type="button"
              data-test="mobile-generated-result"
              aria-label="Expand generated print"
              @click="generatedViewerOpen = true"
            >
              <img
                :key="`${latestResultJob?.clientId}:${resultMediaLoadKey}`"
                class="result-media"
                :src="resultUrl"
                alt="Generated print"
                draggable="false"
                @load="generatedMediaReady"
                @error="recoverGeneratedMedia"
                @contextmenu.prevent
              />
            </button>
          </template>
        </template>
      </template>

      <template v-else-if="tab === 'gallery'">
        <h1 class="section-title">Library</h1>
        <p class="section-note">Prints from every saved host</p>
        <p v-if="galleryError" class="status-line error-text">{{ galleryError }}</p>
        <div v-if="galleryLoading" class="empty-state">Loading prints…</div>
        <div v-else-if="gallery.length" class="gallery-grid">
          <button
            v-for="print in gallery"
            :key="`${print.hostId}:${print.filename}`"
            class="gallery-item"
            type="button"
            :aria-label="`Open ${print.filename} from ${print.hostName}`"
            data-test="gallery-item"
            @click="openPrint(print)"
          >
            <img
              :src="print.thumbnailUrl"
              :alt="print.metadata.prompt || print.filename"
              loading="lazy"
            />
            <span v-if="isVideoItem(print)" class="gallery-video-badge" aria-hidden="true">▶</span>
            <span v-if="isFreshMobilePrint(print)" class="gallery-new-badge" data-test="new-badge">
              New
            </span>
            <span
              v-if="isUpscaledImage(print)"
              class="gallery-upscaled-badge"
              data-test="upscaled-badge"
            >
              Upscaled
            </span>
          </button>
        </div>
        <div v-else class="empty-state">No prints found.</div>
        <button
          v-if="!galleryLoading && galleryRemaining"
          class="secondary-button gallery-more"
          type="button"
          :disabled="galleryLoading || galleryLoadingMore"
          @click="loadMoreGallery"
        >
          {{ galleryLoadingMore ? "Loading…" : `Load older prints (${galleryRemaining})` }}
        </button>
      </template>

      <template v-else-if="tab === 'hosts'">
        <MobileHostDetail
          v-if="hostDetail"
          :host="hostDetail"
          :active="hostDetail.id === selectedHostId"
          @back="hostDetailId = ''"
          @select="selectHost"
          @rename="renameHost"
          @forget="removeHost"
          @catalog="openCatalog"
          @status="updateHostStatus"
        />
        <template v-else>
          <h1 class="section-title">Machines</h1>
          <p class="section-note">LAN discovery, Tailscale MagicDNS, or an address</p>
          <button
            class="secondary-button"
            type="button"
            :disabled="discovering"
            @click="discoverHosts"
          >
            {{ discovering ? "Scanning…" : "Discover nearby" }}
          </button>
          <div v-for="host in discovered" :key="`${host.host}:${host.port}`" class="host-row">
            <div class="host-row-head">
              <div>
                <div class="host-name">{{ host.name }}</div>
                <div class="host-url">{{ host.host }}:{{ host.port }}</div>
              </div>
              <button
                class="secondary-button"
                type="button"
                @click="connectHost(`${host.host}:${host.port}`, host.name)"
              >
                Connect
              </button>
            </div>
          </div>
          <form style="margin-top: 20px" @submit.prevent="connectHost()">
            <label class="field"
              ><span>Name</span
              ><input
                v-model="hostInput.name"
                class="control"
                placeholder="Studio Mac (optional)"
                autocomplete="off"
            /></label>
            <label class="field"
              ><span>Address or MagicDNS name</span
              ><input
                v-model="hostInput.address"
                class="control"
                placeholder="studio.tailnet.ts.net or 192.168.1.20"
                autocapitalize="none"
                autocomplete="url"
                required
            /></label>
            <label class="field"
              ><span>API key</span
              ><input
                v-model="hostInput.apiKey"
                class="control"
                type="password"
                placeholder="If required"
                autocomplete="off"
            /></label>
            <button class="primary-button" type="submit">Test and save</button>
          </form>
          <p v-if="hostError" class="status-line error-text">{{ hostError }}</p>
          <div v-for="host in hosts" :key="host.id" class="host-row">
            <button
              class="host-row-button"
              type="button"
              :aria-label="`View ${host.name}`"
              data-test="mobile-host-row"
              @click="showHostDetail(host.id)"
            >
              <span class="host-row-head">
                <span>
                  <span class="host-name">{{ host.name }}</span>
                  <span class="host-url">{{ host.baseUrl }}</span>
                </span>
                <span class="host-row-state">
                  <span class="status-dot" :class="host.online ? 'is-ready' : 'is-error'" />
                  <span class="host-chip">{{
                    host.online ? `v${host.version ?? ""}` : "offline"
                  }}</span>
                  <span aria-hidden="true">›</span>
                </span>
              </span>
            </button>
            <div v-if="host.online" class="host-telemetry" data-test="mobile-host-telemetry">
              <div class="host-telemetry-row">
                <span class="host-telemetry-mem">{{ hostMemLabel(host.id) }}</span>
                <span class="host-telemetry-queue">queue {{ hostQueueLabel(host.id) }}</span>
              </div>
              <div
                class="meter host-telemetry-meter"
                role="meter"
                :aria-label="`VRAM usage on ${host.name}`"
                :aria-valuenow="Math.round(hostVramPercent(host.id))"
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <span :style="{ width: `${hostVramPercent(host.id)}%` }" />
              </div>
            </div>
            <div class="row-actions">
              <button
                class="secondary-button"
                type="button"
                :disabled="host.id === selectedHostId"
                @click="selectHost(host.id)"
              >
                {{ host.id === selectedHostId ? "Active" : "Use host" }}
              </button>
            </div>
          </div>
        </template>
      </template>

      <KeepAlive>
        <MobileCatalogView
          v-if="!settingsOpen && tab === 'catalog'"
          :hosts="hosts"
          :selected-host-id="catalogHostId"
          @select-host="selectCatalogHost"
          @models-changed="catalogModelsChanged"
        />
      </KeepAlive>
    </section>

    <MobileGalleryViewer
      v-if="selectedPrint"
      :item="selectedPrint"
      :target="selectedPrint.target"
      :cache-key="selectedPrint.hostId"
      :host-name="selectedPrint.hostName"
      :thumbnail-url="selectedPrint.thumbnailUrl"
      :reusing="reusingPrint"
      :can-use-as-source="canUseSelectedPrintAsSource"
      :using-source="usingPrintAsSource"
      :reuse-error="reusePrintError"
      :generation-announcement="generationAnnouncement"
      :position="selectedPrintIndex + 1"
      :total="gallery.length"
      :has-previous="selectedPrintIndex > 0"
      :has-next="selectedPrintIndex >= 0 && selectedPrintIndex < gallery.length - 1"
      @close="closePrint"
      @reuse="reuseSelectedPrint"
      @use-source="useSelectedPrintAsSource"
      @previous="navigateSelectedPrint(-1)"
      @next="navigateSelectedPrint(1)"
    />

    <MobileGalleryViewer
      v-if="generatedViewerOpen && generatedPreviewItem && resultUrl"
      :item="generatedPreviewItem"
      :target="generatedPreviewTarget"
      :cache-key="latestResultJob?.hostId ?? 'generated'"
      :host-name="latestResultJob?.hostLabel ?? selectedHost?.name ?? 'Mold host'"
      :thumbnail-url="resultUrl"
      :media-url-override="resultUrl"
      :generation-announcement="generationAnnouncement"
      @close="generatedViewerOpen = false"
      @reuse="generatedViewerOpen = false"
    />

    <nav v-if="!settingsOpen" class="mobile-tabs" aria-label="Primary">
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'generate' ? 'page' : undefined"
        data-test="mobile-tab-generate"
        @click="tab = 'generate'"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z" />
        </svg>
        <span>Create</span>
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'gallery' ? 'page' : undefined"
        data-test="mobile-tab-gallery"
        @click="tab = 'gallery'"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
          <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
          <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
          <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
        </svg>
        <span>Library</span>
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'catalog' ? 'page' : undefined"
        data-test="mobile-tab-catalog"
        @click="openCatalog()"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3l8.5 4.3-8.5 4.3L3.5 7.3z" />
          <path d="M3.5 12L12 16.3 20.5 12" />
        </svg>
        <span>Models</span>
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'hosts' ? 'page' : undefined"
        data-test="mobile-tab-hosts"
        @click="tab = 'hosts'"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3" y="4" width="18" height="7" rx="2" />
          <rect x="3" y="13" width="18" height="7" rx="2" />
        </svg>
        <span>Machines</span>
      </button>
    </nav>
  </main>
</template>

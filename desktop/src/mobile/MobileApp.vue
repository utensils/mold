<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  shallowRef,
  watch,
} from "vue";
import { invoke } from "@tauri-apps/api/core";
import {
  cancel as cancelBarcodeScanner,
  checkPermissions as checkBarcodeScannerPermissions,
  Format,
  requestPermissions as requestBarcodeScannerPermissions,
  scan,
} from "@tauri-apps/plugin-barcode-scanner";
import { getCurrent, onOpenUrl } from "@tauri-apps/plugin-deep-link";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import { expandPrompt } from "../lib/api/expand";
import { remixPrompt } from "../lib/api/remix";
import { summarizeStatusGpuMemory } from "../lib/api/gpuStatus";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { createUuid } from "@studio/lib/id";
import { confirmCancellation } from "@studio/lib/cancellationRetry";
import { filterRestrictedModels, modelAccessRestrictionFor } from "@studio/lib/modelAccess";
import { expansionTaskForRequest } from "@studio/lib/expandTask";
import {
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
} from "@studio/lib/generationProfile";
import {
  conditioningFingerprint,
  defaultRemixDimensions,
  promptSource,
  remixDimensionsForTask,
  validateRemixVariants,
  DEFAULT_REMIX_VARIATIONS,
} from "@studio/lib/promptTransform";
import { claimPairingSession, parseMobilePairingPayload } from "@studio/api/pairing";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import {
  resolveDefaultSourceResolution,
  resolveSourceConditioningTarget,
  resolveSourceCanvasTransition,
  resolveSourceResolution,
  type SourceDimensions,
  type SourceResolutionResult,
} from "@studio/lib/sourceResolution";
import type { CanvasIntent } from "@studio/lib/outputShape";
import { groupLogicalGalleryPrints } from "@studio/lib/galleryPrintIdentity";
import {
  collectionSlug,
  collectionSlugResolver,
  displayTitle,
  planOrganizationFanout,
  tagKey,
  trashRetentionSummary,
  type OrganizationMutation,
  type OrganizationUnion,
} from "@studio/lib/libraryOrganization";
import type { Collection, TagCount } from "@studio/lib/api/galleryOrganization";
import {
  createCollection as createHostCollection,
  deleteCollection as deleteHostCollection,
  emptyTrash as emptyHostTrash,
  listCollections as listHostCollections,
  listTags as listHostTags,
  listTrash as listHostTrash,
  updateCollection as updateHostCollection,
} from "@studio/api/galleryOrganization";
import {
  defaultClipFrames,
  modelsForOutput,
  sequenceMotionTailFrames,
  type OutputMode,
} from "@studio/lib/sequence";
import { promptPlaceholder, promptRequired } from "@studio/lib/promptRequirement";
import { applyAuthoredPrompt } from "@studio/lib/promptProvenance";
import {
  appendMinimaxH3GalleryImageReference,
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Identity,
  minimaxH3AuthoringError,
  minimaxH3TaskForModel,
  setMinimaxH3GalleryImageFirstFrame,
  setMinimaxH3PickedImageBoundary,
} from "@studio/lib/minimaxH3Authoring";
import { h3BoundariesNeedingMedia } from "@studio/lib/h3BoundaryRestore";
import {
  classifyPlacementPreview,
  comparePlacementPreviews,
  previewChainPlacement,
  previewGenerationPlacement,
  previewRequestForSiblingFanout,
  requiresAuthoritativePlacement,
  type GenerationPlacementPreview,
} from "@studio/api/generationPlacement";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  chooseRoutedHost,
  hostIdsForModel,
  isAutomaticTarget,
  pickAutoHost,
  pickMostCapableHost,
  unionModelsByName,
} from "@studio/lib/hostRouting";
import { profileConflictMessage, profileHashConflict } from "@studio/lib/profileFleet";
import {
  MOBILE_AUTO_ROUTING_HINT,
  MOBILE_CAPABLE_ROUTING_HINT,
  loadMobileGenerateTarget,
  mobileGenerateTargetLabel,
  mobileAutoRoutingAvailable,
  mobileModelAvailabilityTag,
  mobileRoutingHosts,
  resolveMobileGenerateTarget,
  saveMobileGenerateTarget,
} from "./generateTarget";
import {
  activityAnnouncement,
  activityCountLabel,
  mergeActivity,
  sequenceToVM,
  withLiveQueueStatus,
  type ActivityJobVM,
  type PrintActivityVM,
} from "@studio/lib/activity";
import {
  buildQueueStatusIndex,
  queueStatusFor,
  queueWaitCode,
  queueWaitLabel,
  resolveQueueWait,
} from "@studio/lib/queuePosition";
import {
  mergeFleetActivity,
  listActiveWork,
  reconcileActivityHost,
  type ActivityHostSnapshot,
  type FleetActiveWork,
} from "@studio/api/activity";
import { listQueue, type QueueListing } from "@studio/api/queuePlan";
import { buildChainRequest } from "@studio/lib/sequenceForm";
import { chainScriptToClips } from "@studio/lib/sequenceForm";
import { normalizeServerChainScript } from "@studio/lib/chainScriptWire";
import { sequenceReuseClampNote, sequenceReuseNote } from "@studio/lib/sequenceReuse";
import {
  firstLastFrameRestoreNotice,
  parseSourceImageCapability,
} from "@studio/lib/sourceImageCapability";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ChainJobDetail, ChainLimits } from "@studio/lib/api/chainTypes";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import { upscaleImage } from "../lib/api/upscale";
import {
  generationCapabilitiesForFamily,
  isFlux2DevModel,
  outputFormatsForFamily,
} from "../lib/capabilities";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { modelDisplayName, modelDisplayNameForId } from "../lib/models";
import type {
  CompleteEvent,
  CreateChainJobResponse,
  DownloadJob,
  ExpandCapabilities,
  GalleryImage,
  GenerateRequest,
  Ltx2ControlAdapterInfo,
  Ltx2CameraControlInfo,
  ModelEntry,
  OutputMetadata,
  PromptTransformProvenance,
  RemixDimension,
  RemixSourceKind,
  ServerCapabilities,
  ServerStatus,
} from "../lib/api/types";
import {
  isCameraMotionPreset,
  parseCameraControlAvailability,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import { guidanceOverridesAreEmpty } from "@studio/lib/guidanceOverrides";
import { negativePromptWireValue } from "@studio/lib/negativePrompt";
import { wanRecipeCount } from "@studio/lib/wanRecipe";
import {
  buildAutoChainRequest,
  buildGenerationEstimateRequest,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
} from "../lib/chainRouting";
import {
  applyModelDefaults,
  applyMetadataToForm,
  applyRequestToForm,
  buildRequest,
  cloneGenerateForm,
  newGenerateForm,
  normalizeLegacyNegativeSnapshot,
  reconcileModelCapabilities,
  resetAdvancedToModelDefaults,
  resetFormToModelDefaults,
  type GenerateForm,
} from "../lib/generateForm";
import {
  formatTemplateMediaReferences,
  hydrateGenerationTemplate,
  type GenerationTemplate,
} from "../lib/generationTemplates";
import { galleryMediaPath, isAudioItem, isVideoItem } from "../lib/gallery/media";
import { isUpscaledImage } from "../lib/gallery/upscaled";
import { percent } from "../lib/format";
import { composeStyle, mergeStyleNegative, styleHint } from "../lib/stylePresets";
import {
  profileGuidanceValidationError,
  profileStepsValidationError,
  inlineGenerationMediaBytes,
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  MOBILE_MEDIA_BUDGET_ERROR,
  mobileMediaBudgetValidationError,
  sourceConditioningValidationError,
} from "../lib/generateValidation";
import { blobToBase64, isStillImageFile } from "../lib/image";
import { parseMissingExpandModel } from "../lib/expandErrors";
import { resolveExpansionRoute } from "@studio/lib/expansionRouting";
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
import { sequenceParams } from "../lib/sequenceParams";
import { isGenerationModel } from "../stores/models";
import type { HostRoute } from "../stores/hosts";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { applyH3BoundaryFit, applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { coerceSourceFitForMaskless, parseSourceFitPolicy } from "@studio/lib/sourceFit";
import {
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
  sha256HexOfBase64,
} from "@studio/lib/generationSourceMedia";
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
import {
  mobileHostMatchesRoute,
  mobileHostTarget,
  normalizeRemoteAddress,
  remoteHostId,
  type MobileHost,
} from "./hosts";
import { applyMobileGalleryMetadata } from "./reuse";
import {
  EMPTY_LIBRARY_FILTERS,
  MOBILE_LIBRARY_SCOPES,
  MOBILE_LIBRARY_SCOPE_LABELS,
  buildOrganizationIndex,
  collectionCards,
  collectionOnHost,
  deleteActionCopy,
  fanoutFailureMessage,
  filterLibraryPrints,
  libraryOrganizationSupport,
  logicalCopiesOf,
  mergeHostTags,
  mergeTrashSnapshot,
  mergedCollectionsFor,
  purgeChipLabel,
  requestTitle,
  runOrganizationFanout,
  selectionDeleteKind,
  tagChipPlan,
  trashRetentionHosts,
  validateCollectionName,
  type MobileCollectionCard,
  type MobileGalleryImage,
  type MobileLibraryScope,
} from "./libraryOrganization";
import {
  captureCachedHostFence,
  clearCachedGalleryHosts,
  loadCachedGallery,
  loadCachedGalleryMedia,
  loadCachedHostPresentation,
  patchCachedGalleryPrints,
  pruneCachedGalleryMedia,
  removeCachedGalleryPrints,
  storeCachedGallery,
  storeCachedGalleryMedia,
  storeCachedHostPresentation,
  type CachedHostPresentation,
} from "./galleryCache";
import {
  MOBILE_GALLERY_COLUMNS_MAX,
  MOBILE_GALLERY_COLUMNS_MIN,
  createPinchZoom,
  isPinching,
  loadMobileGalleryColumns,
  pinchPointerDown,
  pinchPointerMove,
  pinchPointerUp,
  resetPinch,
  saveMobileGalleryColumns,
  tracksPointer,
} from "./galleryZoom";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileCatalogView from "./MobileCatalogView.vue";
import MobileExpansionPullStatus from "./MobileExpansionPullStatus.vue";
import MobileGalleryViewer from "./MobileGalleryViewer.vue";
import MobileGenerateParameters from "./MobileGenerateParameters.vue";
import MobileHostDetail from "./MobileHostDetail.vue";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";
import MobileLoraControls from "./MobileLoraControls.vue";
import MobilePromptTools from "./MobilePromptTools.vue";
import MobileRemixReview, { type MobileRemixReviewVariant } from "./MobileRemixReview.vue";
import MobilePreparedExpansionBatch from "./MobilePreparedExpansionBatch.vue";
import MobileSequenceComposer from "./MobileSequenceComposer.vue";
import MobileSettingsView from "./MobileSettingsView.vue";
import MobileSharedParams from "./MobileSharedParams.vue";
import MobileSourceControls from "./MobileSourceControls.vue";
import MobileStyleChips from "./MobileStyleChips.vue";
import MobileTemplates from "./MobileTemplates.vue";
import {
  loadMobileSettings,
  updateMobileSettings as persistMobileSettings,
  type MobileSettings,
} from "./settings";
import { useMobileDownloadsStore } from "./mobileDownloads";
import { isNativeIOSRuntime } from "./platform";
import {
  createMobileExpansionRecovery,
  mobileExpansionRecoveryStaleReason,
  type MobileRemixRecoveryPayload,
  type MobileExpansionRecoveryRecord,
} from "./mobileExpansionRecovery";
import { reconcileInterruptedGenerationJobs } from "../lib/generationRecovery";
import { watchChainJob, type SequenceWatchHandle } from "./sequenceWatch";

type Tab = "generate" | "gallery" | "catalog" | "hosts";

/** Deep-link payload for the Discover shelf; `token` re-fires the intent even
 *  when the (KeepAlive-cached) catalog is asked for the same filters twice. */
interface CatalogFilterIntent {
  mediaType: "all" | "image" | "video";
  kind: "checkpoint" | "";
  token: number;
}

/** One queue row, already resolved to the print or sequence it renders. */
type ActivityRow =
  | {
      key: string;
      print: Job;
      /** Live dispatch order from `/api/queue`; absent on older hosts. */
      queuePosition: number | null;
      blockedReason: string | null;
      sequence: null;
    }
  | { key: string; print: null; sequence: Extract<ActivityJobVM, { kind: "sequence" }> };

interface DiscoveredHost {
  name: string;
  host: string;
  port: number;
}

interface GalleryPrint extends MobileGalleryImage {
  hostId: string;
  cacheKey: string;
  hostName: string;
  target: ApiTarget;
  thumbnailUrl: string;
}

interface PendingGalleryPrint extends MobileGalleryImage {
  hostId: string;
  cacheKey: string;
  hostName: string;
  target: ApiTarget;
}

interface ModelSnapshotIdentity {
  cacheKey: string;
  updatedAt: number;
  serverVersion: string | null;
}

/** The Library's bottom-sheet editors (one open at a time). */
type LibrarySheet =
  | { kind: "collections" }
  | { kind: "tags" }
  | { kind: "new-collection" }
  | { kind: "rename-collection"; slug: string; name: string }
  | { kind: "more-tags" };

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

interface MobileRemixReviewState {
  sourcePrompt: string;
  rootPrompt?: string;
  sourceKind: RemixSourceKind;
  visiblePrompt: string;
  model: string;
  family: string;
  task: ReturnType<typeof expansionTaskForRequest>;
  stylePreset: string | null;
  dimensions: RemixDimension[];
  conditioningFingerprint: string;
  selectedHostPolicy: string | null;
  route: HostRoute;
  variants: MobileRemixReviewVariant[];
  requestToken: number;
}

interface MobileRemixUndoSnapshot {
  prompt: string;
  originalPrompt: string | null;
  stylePreset: string;
}

interface MobileAppliedRemix {
  prompt: string;
  rootPrompt?: string;
  sourcePrompt: string;
  sourceKind: RemixSourceKind;
  task: ReturnType<typeof expansionTaskForRequest>;
  dimensions: RemixDimension[];
}

function sentence(text: string): string {
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

function mobilePlacementFailure(
  preview: GenerationPlacementPreview | null,
  hostLabel: string,
  subject: "print" | "sequence",
): string {
  const classification = classifyPlacementPreview(preview);
  if (classification === "infeasible" && preview) {
    const missing = (preview.missing_components ?? [])
      .filter((component) => !component.present)
      .map((component) => component.name);
    const reason =
      typeof preview.reason === "string" && preview.reason.trim()
        ? sentence(preview.reason.trim())
        : sentence(`the server reported that this ${subject} is infeasible`);
    return `${hostLabel} cannot run this ${subject}: ${reason}${missing.length ? ` Missing components: ${missing.join(", ")}.` : ""} Nothing was queued.`;
  }
  if (classification === "temporarily_unavailable") {
    const reason =
      typeof preview?.reason === "string" && preview.reason.trim()
        ? ` Reason: ${sentence(preview.reason.trim())}`
        : "";
    return `${hostLabel} could not compute a placement plan right now.${reason} Try again. Nothing was queued.`;
  }
  return `${hostLabel} returned an invalid placement response. Nothing was queued.`;
}

const STORAGE_KEY = "mold.mobile.hosts.v1";
const SELECTED_KEY = "mold.mobile.selected-host.v1";
const LIBRARY_SEEN_AT_KEY = "mold.mobile.library-seen-at.v1";
const LEGACY_LIBRARY_SEEN_KEY = "mold.mobile.library-seen.v1";
const LIBRARY_VISITED_KEY = "mold.mobile.library-visited.v1";
const SEQUENCE_RECOVERY_KEY = "mold.mobile.sequence-job.v1";
const LIVE_ACTIVITY_KEY = "mold.mobile.live-activity.v1";
const HOST_PROBE_TIMEOUT_MS = 9_000;
/** How long automatic routing keeps waiting for slower machines once one has
 *  answered `planned`. Nothing is abandoned before that first plan exists —
 *  a deadline that fires with no route would manufacture a dead end. */
const MOBILE_PLACEMENT_SETTLE_MS = 1_500;
const GALLERY_HOST_TIMEOUT_MS = 9_000;
const REUSE_PRESENTATION_TIMEOUT_MS = 9_000;
const OUTPUT_OPTIONS = [
  { value: "single" as const, label: "One shot" },
  { value: "sequence" as const, label: "Sequence" },
];
const tab = ref<Tab>("generate");
// Output is a setting of Create, not a place. The store hydrates here (before
// first paint) so the legacy `mold.mobile.create-mode.v1` key migrates into
// the shared draft and existing installs land back in Sequence.
const draft = useSequenceDraftStore();
draft.hydrate();
const mobileContent = ref<HTMLElement | null>(null);
const settingsOpen = ref(false);
const settingsButton = ref<HTMLButtonElement | null>(null);
const settingsBackButton = ref<HTMLButtonElement | null>(null);
const mobileSettings = reactive<MobileSettings>(loadMobileSettings());
const appVersion = ref(import.meta.env.DEV ? "Development build" : "Current build");
const hosts = ref<MobileHost[]>(loadHosts());
function loadMobileActivity(): Record<string, ActivityHostSnapshot> {
  try {
    const saved = JSON.parse(localStorage.getItem(LIVE_ACTIVITY_KEY) ?? "{}") as Record<
      string,
      ActivityHostSnapshot
    >;
    for (const snapshot of Object.values(saved)) {
      snapshot.routeUrl ??= snapshot.target?.baseUrl ?? "";
      snapshot.instanceId ??= null;
      snapshot.unavailableKinds ??= [];
      snapshot.stale = true;
      snapshot.error ??= "Waiting to reconnect";
    }
    return saved;
  } catch {
    return {};
  }
}
const liveActivityHosts = ref(loadMobileActivity());
/** Live `/api/queue` listings per host, in memory only. Unlike the activity
 *  snapshot this carries full prompts, so it is deliberately never persisted. */
const liveQueues = ref<Record<string, QueueListing>>({});
const liveActivityEpochs: Record<string, number> = {};
let liveActivityTimer: ReturnType<typeof setInterval> | null = null;
let liveActivityRefreshing = false;
const connectedHosts = computed(() => hosts.value.filter((host) => host.connected !== false));
const selectedHostId = ref(localStorage.getItem(SELECTED_KEY) ?? connectedHosts.value[0]?.id ?? "");
const catalogHostId = ref(selectedHostId.value || connectedHosts.value[0]?.id || "");
const catalogFilterIntent = ref<CatalogFilterIntent | null>(null);
let catalogIntentToken = 0;
const hostDetailId = ref("");
const hostInput = reactive({ name: "", address: "", apiKey: "" });
const discovered = ref<DiscoveredHost[]>([]);
const discovering = ref(false);
const pairing = ref(false);
const pairingScannerOpen = ref(false);
let pairingScannerCancelled = false;
let stopPairingDeepLinks: (() => void) | null = null;
const hostError = ref("");
const models = ref<ModelEntry[]>([]);
const modelsHostId = ref("");
/** Per-machine `/api/models` snapshots, the input to model-aware routing and
 *  to the union picker the automatic policies need. The browsed machine's copy
 *  is written by `refreshModels`; peers are filled in by `refreshRoutingModels`. */
const modelsByHost = ref<Record<string, ModelEntry[]>>({});
/** Identity and freshness authority for each in-memory model snapshot. */
const modelSnapshotIdentities = ref<Record<string, ModelSnapshotIdentity>>({});
/** Persisted generation target: a machine id, `auto`, or `capable`. */
const generateTargetPolicy = ref(loadMobileGenerateTarget());
const loadingModels = ref(false);
const modelLoadError = ref("");
const sequenceJob = ref<ChainJobDetail | null>(null);
const sequenceStarting = ref(false);
let sequenceCancellationRequest: (() => Promise<unknown>) | null = null;
const sequenceError = ref("");
const sequenceProgress = ref<{ step: number; total: number } | null>(null);
const chainLimits = ref<ChainLimits | null>(null);
/**
 * The frozen route the durable job was created on. Identity IS the staleness
 * guard, so this must be a `shallowRef`: a plain `ref` hands back a reactive
 * proxy and every `sequenceRoute.value !== route` check would fire against a
 * live watch, silently dropping its own events.
 */
const sequenceRoute = shallowRef<{ hostId: string; target: ApiTarget } | null>(null);
let sequenceWatch: SequenceWatchHandle | null = null;
const expandCapabilities = reactive<Record<string, ExpandCapabilities | null | undefined>>({});
const serverCapabilities = reactive<Record<string, ServerCapabilities | null | undefined>>({});
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
  // Differs-from-default (#787): an untouched wan default is not "active",
  // while an explicit clear (the empty-uncond opt-out) is.
  if (
    generationCapabilitiesForFamily(
      form.family,
      form.model,
      form.pipeline,
      form.guidanceCapabilities,
    ).supportsNegativePrompt &&
    negativePromptWireValue(form.negativePrompt, form.negativePromptDefault) !== undefined
  )
    count += 1;
  // Source media lives in the primary form now, not the Advanced sheet.
  if (form.loras.length) count += 1;
  if (form.upscaleModel) count += 1;
  if (form.scheduler !== "default") count += 1;
  if (form.cfgPlus) count += 1;
  if (form.cameraControl) count += 1;
  if (
    generationCapabilitiesForFamily(form.family).supportsAdvancedVideo &&
    !guidanceOverridesAreEmpty(form.guidanceOverrides)
  ) {
    count += 1;
  }
  if (generationCapabilitiesForFamily(form.family, form.model).wanRecipe.supported) {
    count += wanRecipeCount(form.wanRecipe);
  }
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
  // The canvas is part of what Reset restores, so its authority resets with
  // it — otherwise the next model change would re-snap the reset canvas back
  // onto the attached source (#1166).
  canvasIntent.value = "model-default";
  // Sequence output keeps its source media in the primary stack too, so the
  // primary Reset owns the opening image exactly as it owns the one-shot
  // wells `resetFormToModelDefaults` just discarded. `clearOpeningImage` is
  // the narrow store write: clips stay, and the persisted blob is reclaimed.
  if (isSequence.value) {
    draft.enableAudio = false;
    draft.clearOpeningImage();
  }
}

/** Match the desktop Advanced reset: restore model-owned generation controls
 * while preserving the prompt, selected model, batch, and staged source media. */
function resetAdvancedSettings(): void {
  resetAdvancedToModelDefaults(form, selectedGenerationModel.value);
  // The canvas comes back to the model's default, so its authority does too.
  canvasIntent.value = "model-default";
}
const preparingGeneration = ref(false);
const generationSubmissionPhase = ref<"preparing" | "placement" | null>(null);
const preparedBatch = ref<PreparedExpansionBatchState | null>(null);
const expansionRunning = ref(false);
const expansionError = ref("");
const expansionRecovery = ref<MobileExpansionRecoveryRecord | null>(null);
const expansionPullAttempt = ref<MobileExpansionPullAttempt | null>(null);
const quickExpansionOriginal = ref<string | null>(null);
const quickExpansionSnapshot = ref<QuickExpansionSnapshot | null>(null);
const remixSource = ref<RemixSourceKind>("original");
const remixDimensions = ref<RemixDimension[]>([]);
const remixReview = ref<MobileRemixReviewState | null>(null);
const remixUndo = ref<MobileRemixUndoSnapshot | null>(null);
const appliedRemix = ref<MobileAppliedRemix | null>(null);
/**
 * The negative prompt before and after a bake-and-clear merged the preset's
 * curated fragments into it. Undo re-arms `before` alongside the prompt and
 * chip; `baked` lets it bow out when the user has since edited the field.
 */
const quickExpansionNegative = ref<{ before: string; baked: string } | null>(null);
const preparedSubmitting = ref(false);
const preparationGuard = new PreparationRequestGuard();
const submissionGuard = new PreparationRequestGuard();
const sequenceSubmissionGuard = new PreparationRequestGuard();
let expansionPullRequestId = 0;
let expansionRecoveryId = 0;
let submissionUiId = 0;
let recoveryRetryId = 0;
let unmounted = false;
let keyboardViewportRestoreTimer: ReturnType<typeof setTimeout> | null = null;
const KEYBOARD_VIEWPORT_SETTLE_MS = 400;
const NON_KEYBOARD_INPUT_TYPES = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "hidden",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);
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
const gallerySentinel = ref<HTMLElement | null>(null);
const gallerySentinelVisible = ref(false);
const gallerySelectMode = ref(false);
const gallerySelection = ref<Set<string>>(new Set());
const galleryDeleteConfirming = ref(false);
/** Library thumbnail size, driven by the pinch gesture below. */
const galleryColumns = ref(loadMobileGalleryColumns());
const galleryZoom = createPinchZoom(galleryColumns.value);
const galleryZoomAnnouncement = ref("");
const galleryPinchSurface = ref<HTMLElement | null>(null);
const galleryDeleting = ref(false);
const selectedPrint = ref<GalleryPrint | null>(null);
// ── Library organization (V3 "Shelf") ───────────────────────────────────────
/** Prints | Collections | Trash. Collections and Trash appear only when a
 * connected host advertises `capabilities.gallery.organize` / `.trash`. */
const libraryScope = ref<MobileLibraryScope>("prints");
const libraryFilters = reactive({ ...EMPTY_LIBRARY_FILTERS });
/** Per-host `/api/gallery/collections` and `/api/gallery/tags` listings. */
const hostCollections = reactive<Record<string, Collection[]>>({});
const hostTags = reactive<Record<string, TagCount[]>>({});
/** Merged title / ♥ / tags / collections per physical print key. */
const galleryOrganization = ref<Map<string, OrganizationUnion>>(new Map());
/** Per-host print counts behind the host chips (physical copies). */
const libraryHostCounts = ref<Record<string, number>>({});
/** Logical live prints (the Prints segment count). */
const libraryPrintCount = ref(0);
/** Inline (never a toast) organization failure banner. */
const organizationError = ref("");
const organizationBusy = ref(false);
/** Trash listing, fetched lazily the first time the Trash scope opens. */
let trashCopies: PendingGalleryPrint[] = [];
const trashCount = ref(0);
const trashLoaded = ref(false);
const trashLoading = ref(false);
const trashError = ref("");
const emptyTrashConfirming = ref(false);
const emptyingTrash = ref(false);
const galleryRestoring = ref(false);
const collectionMenuSlug = ref<string | null>(null);
const collectionDeleteConfirmSlug = ref<string | null>(null);
const collectionCovers = reactive<Record<string, string>>({});
const librarySheet = ref<LibrarySheet | null>(null);
const librarySheetInput = ref("");
const librarySheetError = ref("");
const librarySheetBusy = ref(false);
/** Create ▸ Title — rides every mobile-built `GenerateRequest` as `title`. */
const printTitle = ref("");
const generatedViewerOpen = ref(false);
const reusingPrint = ref(false);
const usingPrintAsSource = ref(false);
const reusePrintError = ref("");
const latestResultClientId = ref<number | null>(null);
const resultMediaLoadKey = ref(0);
const GENERATED_VIDEO_RECOVERY_DELAYS_MS = [250, 750, 1_500] as const;
const objectUrls = new Set<string>();
const handledGenerationClientIds = new Set<number>();
let pendingGallery: PendingGalleryPrint[] = [];
/** Every concrete device copy behind the deduplicated Library tiles. */
let galleryCopies: PendingGalleryPrint[] = [];
let modelLoadEpoch = 0;
let reusePrintEpoch = 0;
let reusePrintController: AbortController | null = null;
let sourceUseEpoch = 0;
let sourceUseController: AbortController | null = null;
let resultMediaRecoveryTimer: ReturnType<typeof setTimeout> | null = null;
let galleryRefreshRequested = false;
let galleryRefreshDeferred = false;
let galleryRefreshTask: Promise<void> | null = null;
let galleryOperationTail: Promise<void> = Promise.resolve();
let galleryLoadMoreQueued = false;
let gallerySentinelObserver: IntersectionObserver | null = null;
let galleryChainedFetches = 0;
const MAX_GALLERY_CHAINED_FETCHES = 3;
let galleryDragPointerId: number | null = null;
let galleryDragActive = false;
let galleryDragSelect = true;
let galleryDragStartX = 0;
let galleryDragStartY = 0;
let galleryDragClientX = 0;
let galleryDragClientY = 0;
let galleryDragFrame: number | null = null;
let galleryDragPendingClicks = 0;
let galleryPinchPendingClicks = 0;
let galleryDragSelectionBaseline: Set<string> | null = null;
const galleryDragVisited = new Set<string>();
const GALLERY_DRAG_INTENT_THRESHOLD = 8;
const GALLERY_DRAG_SCROLL_EDGE = 72;
const GALLERY_DRAG_SCROLL_MAX = 18;
let resultMediaRecoveryClientId: number | null = null;
let resultMediaRecoveryAttempts = 0;
let hostProbeTimer: ReturnType<typeof setInterval> | null = null;
let hostProbeEpoch = 0;
const hostProbes = new Map<
  string,
  { epoch: number; controller: AbortController; timeout: ReturnType<typeof setTimeout> }
>();
const knownHostReachability = new Set<string>();
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
  /** `gpu_info.backend` as this machine reported it; null on servers ≤ 0.16,
   *  where the routing ladder falls back to inferring it from the GPU name. */
  gpuBackend: string | null;
  gpuName: string | null;
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
    gpuBackend: status.gpu_info?.backend ?? null,
    gpuName: status.gpu_info?.name ?? null,
  };
}

const selectedHost = computed(() =>
  connectedHosts.value.find((host) => host.id === selectedHostId.value),
);

// --- Generation routing -----------------------------------------------------
// The phone is remote-only: Auto and Most capable choose between CONNECTED
// machines and never fall back to a local engine. Both are offered only while
// at least two machines are reachable; with one machine the picker keeps
// today's single-host behaviour exactly.
const routingHosts = computed(() => mobileRoutingHosts(connectedHosts.value));
const autoRoutingAvailable = computed(() => mobileAutoRoutingAvailable(connectedHosts.value));
const generateTarget = computed(() =>
  resolveMobileGenerateTarget(
    generateTargetPolicy.value,
    connectedHosts.value,
    selectedHostId.value,
  ),
);
const automaticRouting = computed(
  () => autoRoutingAvailable.value && isAutomaticTarget(generateTarget.value),
);
const routingHint = computed(() =>
  generateTarget.value === CAPABLE_TARGET_ID
    ? MOBILE_CAPABLE_ROUTING_HINT
    : MOBILE_AUTO_ROUTING_HINT,
);
/** The header chip names where work lands: a machine, or the active policy. */
const headerTargetLabel = computed(() =>
  automaticRouting.value
    ? mobileGenerateTargetLabel(generateTarget.value, connectedHosts.value)
    : (selectedHost.value?.name ?? "Remote only"),
);
const developOnNote = computed(() => {
  if (!automaticRouting.value) return `Develop on ${selectedHost.value?.name ?? "this machine"}`;
  return generateTarget.value === CAPABLE_TARGET_ID
    ? "Develop on the most capable machine"
    : "Develop on the least busy machine";
});
/** Where the Create empty states say a model is (or is not) installed. */
const modelScopeLabel = computed(() =>
  automaticRouting.value ? "your connected machines" : (selectedHost.value?.name ?? "this machine"),
);

/** The routing view of one machine: the phone has no home host, so the
 *  pickers see status/queue/GPU only and break dead heats on the id. */
function routingHostView(host: MobileHost) {
  const telemetry = hostTelemetry[host.id];
  return {
    id: host.id,
    status: "ready" as const,
    queueDepth: telemetry?.queueDepth ?? null,
    gpu: {
      backend: telemetry?.gpuBackend ?? null,
      name: telemetry?.gpuName ?? null,
      vramTotalMb: telemetry?.vramTotalMb ?? null,
    },
    host,
  };
}
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
/**
 * Where model-scoped read-only lookups go — chain limits, LoRAs, control
 * adapters, source preprocessing, and the estimate. The browsed machine
 * answers whenever it has the model (the stable, pinned-path answer); under an
 * automatic policy a model that lives only on a peer is asked of the machine
 * that policy would route it to, so the union picker cannot show a model whose
 * own capabilities nothing can read.
 */
const generationTargetHost = computed<MobileHost | null>(() => {
  const browsed = selectedHost.value ?? null;
  if (!automaticRouting.value || !form.model) return browsed;
  const owners = modelHostIds(form.model);
  if (!browsed || !owners.includes(browsed.id)) {
    // Deliberately NOT the Auto pick: these answers are identical on every
    // owner, and depending on live queue depth here would hand children a new
    // target object on every telemetry tick.
    const owner = routingHosts.value
      .filter((host) => owners.includes(host.id))
      .sort((left, right) => left.id.localeCompare(right.id))[0];
    if (owner) return owner;
  }
  return browsed;
});
const generationTarget = computed<ApiTarget | null>(() =>
  generationTargetHost.value ? mobileHostTarget(generationTargetHost.value) : null,
);
const controlAdapters = ref<Ltx2ControlAdapterInfo[]>([]);
const cameraControls = ref<Ltx2CameraControlInfo[]>([]);
const cameraControlsLoaded = ref(false);
const cameraUnsupportedReason = ref<string | null>(null);
let controlAdaptersEpoch = 0;
watch(
  [selectedHostId, () => form.model, () => selectedHost.value?.online],
  async () => {
    const epoch = ++controlAdaptersEpoch;
    // Drop the previous model's reason immediately; keeping it while the
    // new request is in flight shows a stale explanation for the wrong model.
    cameraUnsupportedReason.value = null;
    controlAdapters.value = [];
    cameraControls.value = [];
    cameraControlsLoaded.value = false;
    const target = generationTarget.value;
    if (!target || !generationTargetHost.value?.online || form.family !== "ltx2" || !form.model) {
      form.icLoraControl = null;
      return;
    }
    const controlsRequest = apiJsonTo<Ltx2ControlAdapterInfo[]>(
      target,
      `/api/capabilities/ltx2-control-adapters?model=${encodeURIComponent(form.model)}`,
    )
      .then((options) => {
        if (epoch !== controlAdaptersEpoch) return;
        controlAdapters.value = options;
        if (form.icLoraControl && !options.some((adapter) => adapter.id === form.icLoraControl)) {
          form.icLoraControl = null;
        }
      })
      .catch(() => {
        if (epoch !== controlAdaptersEpoch) return;
        controlAdapters.value = [];
        form.icLoraControl = null;
      });
    const cameraRequest = apiJsonTo<unknown>(
      target,
      `/api/capabilities/ltx2-camera-controls?model=${encodeURIComponent(form.model)}&detail=1`,
    )
      .then((body) => {
        if (epoch !== controlAdaptersEpoch) return;
        const availability = parseCameraControlAvailability(body);
        const cameras = availability.controls;
        cameraControls.value = cameras;
        cameraUnsupportedReason.value = availability.unsupportedReason;
        cameraControlsLoaded.value = true;
        const compatible = (value: string | null) =>
          !value || !isCameraMotionPreset(value) || cameras.some((camera) => camera.id === value);
        if (!compatible(form.cameraControl)) {
          form.loras = syncCameraMotionLora(
            form.loras,
            form.cameraControl,
            null,
            (path, scale) => ({ path, name: path, scale, trainedWords: [] }),
          );
          form.cameraControl = null;
        }
        for (const clip of draft.clips) {
          if (!compatible(clip.cameraControl)) clip.cameraControl = null;
        }
      })
      .catch(() => {
        if (epoch !== controlAdaptersEpoch) return;
        cameraControls.value = [];
        cameraUnsupportedReason.value = null;
        cameraControlsLoaded.value = false;
      });
    await Promise.allSettled([controlsRequest, cameraRequest]);
  },
  { immediate: true },
);
const caps = computed(() =>
  generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    selectedGenerationModel.value?.guidance_capabilities,
    // Per-model source-image contract (#772): the picked row when the host's
    // listing has it, otherwise the form's snapshot of it. Without this the
    // source well would render for a text-to-video wan checkpoint that
    // rejects one, and the End frame well (#779) would never appear.
    selectedGenerationModel.value?.source_image ?? form.sourceImageCapability,
    effectiveGenerationRecipe(selectedGenerationModel.value, form.pipeline),
  ),
);
/** The model's image-attachment shape — one shared policy (`sourceMediaPlan`).
 * `none` hides the source section outright; `h3-references` keeps the H3
 * ordered-reference editor in the Advanced sheet. */
const sourcePlan = computed(() => sourceMediaPlan(caps.value));
const showSourceMedia = computed(
  () => sourcePlan.value.kind !== "none" && sourcePlan.value.kind !== "h3-references",
);
const h3AuthoringError = computed(() =>
  minimaxH3AuthoringError(
    form.family,
    form.model,
    form.h3Authoring,
    caps.value.requiresSourceImage,
  ),
);
const effectiveBatchSize = computed(() =>
  caps.value.forcesBatchSizeOne ||
  (caps.value.sourceImageMode === "references" && form.imageAttachments.length > 0)
    ? 1
    : Math.max(1, Math.floor(form.batchSize)),
);
const currentExpansionTask = computed(() =>
  expansionTaskForRequest(form.family, buildRequest(form)),
);
watch(
  [currentExpansionTask, () => form.stylePreset] as const,
  ([task], previous) => {
    const styleLocked = Boolean(form.stylePreset?.trim());
    const available = remixDimensionsForTask(task, styleLocked);
    const retained = remixDimensions.value.filter((dimension) => available.includes(dimension));
    remixDimensions.value =
      previous === undefined || retained.length === 0
        ? defaultRemixDimensions(task, styleLocked)
        : retained;
  },
  { immediate: true },
);
const selectedRoute = computed<HostRoute | null>(() => {
  const host = selectedHost.value;
  return host ? routeForMobileHost(host) : null;
});

/** A finished single-result transform is prompt content, not work owned by the
 * machine that rewrote it. Carry its semantic snapshot to an explicitly chosen
 * ready machine while dropping the previous route/download authority. Reviewed
 * multi-variation batches deliberately keep their original frozen route. */
function carryQuickTransformToHost(hostId: string): void {
  const snapshot = quickExpansionSnapshot.value;
  const host = hosts.value.find((candidate) => candidate.id === hostId);
  if (
    !snapshot ||
    preparedBatch.value ||
    !host ||
    host.connected === false ||
    !host.online ||
    form.prompt !== snapshot.expandedPrompt ||
    form.model !== snapshot.model ||
    form.family !== snapshot.family ||
    currentExpansionTask.value !== snapshot.task
  ) {
    return;
  }
  quickExpansionSnapshot.value = {
    ...snapshot,
    selectedHostPolicy: hostId,
    route: routeForMobileHost(host),
  };
  if (expansionRecovery.value?.route.hostId !== hostId) clearExpansionRecovery();
  expansionError.value = "";
}

watch(selectedHostId, (hostId, previous) => {
  if (hostId && hostId !== previous) carryQuickTransformToHost(hostId);
});
// Keep the peer model snapshots current while an automatic policy is in force:
// the reachable set changes as machines wake, sleep, and are added.
watch(
  [automaticRouting, () => routingHosts.value.map((host) => host.id).join(",")],
  ([automatic]) => {
    if (automatic) void refreshRoutingModels();
  },
);
const expansionMissingModel = computed(() => {
  const recovery = expansionRecovery.value;
  return recovery ? { model: recovery.model, route: recovery.route, host: recovery.host } : null;
});
const preparedStaleReasons = computed(() => {
  const batch = preparedBatch.value;
  if (!batch) return [];
  return preparedExpansionStaleReasons(batch, {
    sourcePrompt:
      batch.kind === "remix"
        ? promptSource(form.prompt, form.originalPrompt, batch.sourceKind).prompt
        : form.prompt.trim(),
    model: form.model,
    family: form.family,
    task: currentExpansionTask.value,
    requestedCount: batch.kind === "remix" ? batch.prompts.length : effectiveBatchSize.value,
    ...(batch.kind === "remix"
      ? {
          dimensions: [...remixDimensions.value],
          conditioningFingerprint: conditioningFingerprint(buildRequest(form)),
        }
      : {}),
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
    task: expansionTaskForRequest(form.family, buildRequest(form)),
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
const remixStaleReasons = computed(() => {
  const review = remixReview.value;
  if (!review) return [];
  const reasons: string[] = [];
  const source = promptSource(form.prompt, form.originalPrompt, review.sourceKind);
  if (source.prompt !== review.sourcePrompt || source.kind !== review.sourceKind) {
    reasons.push("Remix source changed after these variants were prepared.");
  }
  if (form.model !== review.model) reasons.push("Model changed after this remix was prepared.");
  if (form.family !== review.family)
    reasons.push("Model family changed after this remix was prepared.");
  if (currentExpansionTask.value !== review.task)
    reasons.push(`Conditioning changed from ${review.task} to ${currentExpansionTask.value}.`);
  if (conditioningFingerprint(buildRequest(form)) !== review.conditioningFingerprint)
    reasons.push("Conditioning media changed after this remix was prepared.");
  if ((form.stylePreset || null) !== review.stylePreset)
    reasons.push("Style changed after this remix was prepared.");
  if (selectedHostId.value !== review.selectedHostPolicy)
    reasons.push("Selected machine changed after this remix was prepared.");
  if (JSON.stringify(remixDimensions.value) !== JSON.stringify(review.dimensions))
    reasons.push("Remix dimensions changed after these variants were prepared.");
  const host = hosts.value.find((candidate) => candidate.id === review.route.hostId);
  if (!sameFrozenHost(review.route, host))
    reasons.push(`${review.route.label}'s connection details changed.`);
  return reasons;
});
/** Under an automatic policy the picker is the union across every reachable
 *  machine — a model installed on any of them is routable — while a pinned
 *  machine keeps showing exactly what it has. */
const generationModels = computed(() => {
  const entries = automaticRouting.value
    ? unionModelsByName(
        modelsByHost.value,
        routingHosts.value.map((host) => host.id),
      )
    : models.value;
  return entries.filter((model) => model.downloaded && isGenerationModel(model));
});
/** Which reachable machines hold a model, for the picker's availability tag. */
function modelHostIds(name: string): string[] {
  return hostIdsForModel(
    modelsByHost.value,
    name,
    routingHosts.value.map((host) => host.id),
  );
}
function modelAvailabilityTag(name: string): string | null {
  if (!automaticRouting.value) return null;
  return mobileModelAvailabilityTag(modelHostIds(name), connectedHosts.value);
}
const isSequence = computed(() => draft.output === "sequence");
const sequenceModels = computed(() => modelsForOutput(generationModels.value, "sequence"));
/** Sequence output narrows the picker to chain-capable video models. */
const pickerModels = computed(() => modelsForOutput(generationModels.value, draft.output));
const sequenceMotionTail = computed(() => sequenceMotionTailFrames(selectedGenerationModel.value));
const sequenceDefaultFrames = computed(() =>
  defaultClipFrames(selectedGenerationModel.value, chainLimits.value, sequenceMotionTail.value),
);
const sequenceSettingsSummary = computed(
  () => `${form.width}×${form.height} · ${form.fps}fps · ${form.steps} steps`,
);
const modelLabel = (name: string) => modelDisplayNameForId(name, generationModels.value);
const upscalers = computed(() =>
  models.value.filter((model) => model.family === "upscaler" || model.family === "real-esrgan"),
);
const controlModels = computed(() => models.value.filter((model) => model.family === "controlnet"));
const sourceSectionTitle = computed(() => {
  if (sourcePlan.value.kind === "h3-boundaries") return "Frame endpoints";
  return caps.value.sourceImageMode !== "single"
    ? isFlux2DevModel(form.model)
      ? "References"
      : "Pictures"
    : "Source image";
});
const sourceSectionSummary = computed(() => {
  if (sourcePlan.value.kind === "h3-boundaries") {
    const first = form.h3Authoring?.firstFrame?.filename;
    if (!first) return caps.value.requiresSourceImage ? "First frame required" : "Optional";
    const last = form.h3Authoring?.lastFrame?.filename;
    return last ? `${first} · ${last}` : first;
  }
  if (caps.value.sourceImageMode !== "single") {
    const count = form.imageAttachments.length;
    if (isFlux2DevModel(form.model)) {
      return count === 0 ? "Optional · up to 4" : `${count} reference${count === 1 ? "" : "s"}`;
    }
    return count === 0 ? "Target required" : `${count} photo${count === 1 ? "" : "s"}`;
  }
  if (form.sourceImage) {
    const opening = form.sourceImageName || "Selected";
    // A first/last-frame pair is the whole point of the section for wan, so
    // the collapsed summary has to say the closing still is there too.
    return form.endFrame && caps.value.supportsEndFrame ? `${opening} · end frame` : opening;
  }
  if (caps.value.requiresSourceImage) return "Required";
  return form.controlImage ? "Control photo selected" : "Optional";
});
const outputFormats = computed(
  () => caps.value.outputFormats as ReturnType<typeof outputFormatsForFamily>,
);
const selectedModelAvailable = computed(
  () =>
    (automaticRouting.value || modelsHostId.value === selectedHostId.value) &&
    generationModels.value.some((model) => model.name === form.model),
);
const selectedGenerationModel = computed(
  () => generationModels.value.find((model) => model.name === form.model) ?? null,
);

function generationProfileHashForHost(hostId: string, model: string): string | null {
  const entries = modelsByHost.value[hostId];
  if (!entries) return null;
  return (
    entries.find((candidate) => candidate.name === model)?.generation_profile?.profile_hash ?? null
  );
}

let previousStillSource = "";
let previousStillResolution: SourceResolutionResult | null = null;
let previousStillAutomaticResolution: SourceDimensions | null = null;
let previousOpeningSource = "";
let previousOpeningResolution: SourceResolutionResult | null = null;
let previousOpeningAutomaticResolution: SourceDimensions | null = null;
const canvasIntent = ref<CanvasIntent>("model-default");
let preservedSourceReplacement = "";
function setCanvasIntent(intent: CanvasIntent) {
  canvasIntent.value = intent;
}
function preserveRestoredSourceCanvas(base64: string) {
  preservedSourceReplacement = base64;
  canvasIntent.value = "manual";
}

function applyMobileSourceResolution(
  base64: string | null,
  previous: {
    base64: string;
    resolution: SourceResolutionResult | null;
    automaticResolution: SourceDimensions | null;
  },
  setDimensions: (width: number | null, height: number | null) => void,
): {
  base64: string;
  resolution: SourceResolutionResult | null;
  automaticResolution: SourceDimensions | null;
} {
  if (!base64) {
    setDimensions(null, null);
    // Clearing the source hands the canvas back to the model, but never
    // overrides a size the user chose deliberately.
    if (canvasIntent.value !== "manual") canvasIntent.value = "model-default";
    return { base64: "", resolution: null, automaticResolution: null };
  }
  const dimensions =
    base64 === previous.base64 && previous.resolution
      ? previous.resolution.source
      : imageDimensionsFromBase64(base64);
  if (!dimensions) {
    setDimensions(null, null);
    return { base64, resolution: null, automaticResolution: null };
  }
  setDimensions(dimensions.width, dimensions.height);
  const resolution = resolveSourceResolution(
    dimensions,
    selectedGenerationModel.value ?? form.family,
    form.pipeline,
  );
  const automaticResolution = resolveDefaultSourceResolution(
    dimensions,
    selectedGenerationModel.value ?? form.family,
    form.pipeline,
  );
  const replaced = base64 !== previous.base64;
  if (caps.value.sourceImageMode !== "references") {
    const preserveReplacement = replaced && preservedSourceReplacement === base64;
    const nextResolution = resolveSourceCanvasTransition({
      source: resolution,
      automatic: automaticResolution,
      replaced,
      intent: canvasIntent.value,
      preserveReplacement,
    });
    if (replaced) {
      preservedSourceReplacement = "";
      if (preserveReplacement) canvasIntent.value = "manual";
      else if (canvasIntent.value !== "source-exact") canvasIntent.value = "source";
    }
    if (nextResolution) {
      form.width = nextResolution.width;
      form.height = nextResolution.height;
    }
  }
  return { base64, resolution, automaticResolution };
}

watch(
  [
    () =>
      caps.value.sourceImageMode !== "single"
        ? (form.imageAttachments[0] ?? null)
        : form.sourceImage,
    () => selectedGenerationModel.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => selectedGenerationModel.value?.generation_profile?.profile_hash ?? null,
    () => selectedGenerationModel.value?.max_pixels ?? null,
    () => selectedGenerationModel.value?.max_axis_pixels ?? null,
    () => selectedGenerationModel.value?.dimension_alignment ?? null,
    () =>
      selectedGenerationModel.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  ([base64]) => {
    if (isSequence.value) return;
    const replaced = Boolean(base64 && base64 !== previousStillSource);
    const next = applyMobileSourceResolution(
      base64,
      {
        base64: previousStillSource,
        resolution: previousStillResolution,
        automaticResolution: previousStillAutomaticResolution,
      },
      (width, height) => {
        form.sourceImageWidth = width;
        form.sourceImageHeight = height;
      },
    );
    previousStillSource = next.base64;
    previousStillResolution = next.resolution;
    previousStillAutomaticResolution = next.automaticResolution;
    if (replaced && caps.value.sourceImageMode === "single") {
      form.sourceFit = { mode: "lanczos-resize" };
    }
  },
  { immediate: true },
);

watch(
  [
    () => draft.openingImage?.base64 ?? null,
    () => selectedGenerationModel.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => selectedGenerationModel.value?.generation_profile?.profile_hash ?? null,
    () => selectedGenerationModel.value?.max_pixels ?? null,
    () => selectedGenerationModel.value?.max_axis_pixels ?? null,
    () => selectedGenerationModel.value?.dimension_alignment ?? null,
    () =>
      selectedGenerationModel.value?.recommended_dimensions
        ?.map(({ width, height }) => `${width}x${height}`)
        .join("|") ?? "",
  ],
  ([base64]) => {
    if (!isSequence.value) return;
    const next = applyMobileSourceResolution(
      base64,
      {
        base64: previousOpeningSource,
        resolution: previousOpeningResolution,
        automaticResolution: previousOpeningAutomaticResolution,
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
    previousOpeningAutomaticResolution = next.automaticResolution;
  },
  { immediate: true },
);
const sourceControlsValid = computed(() => !caps.value.supportsImg2img || sourceValid.value);
/**
 * The per-model source-image contract (#772) and wan's first/last-frame
 * pairing (#779), repeated beside Develop so a disabled button is never a
 * dead end. It also covers the case the well cannot report at all: an
 * advertised text-to-video checkpoint hides the well entirely.
 */
const sourceConditioningError = computed(() => sourceConditioningValidationError(form));
const fixedRecipeControls = computed(() =>
  fixedRecipeControlOverrides(
    effectiveGenerationRecipe(selectedGenerationModel.value, form.pipeline),
  ),
);
/**
 * A fixed recipe control is authority the moment the recipe is known, and
 * the gates below read the LIVE form: `stepsError` disables Develop through
 * `developBlockerReason`, and `basicParametersValid` returns `generate()`
 * early — both before the submit-time snap in `prepareGenerationRequest`
 * can run. So a stale value would strand Develop behind an error on a
 * control the user cannot edit.
 *
 * `applyModelDefaults` already reconciles a model *pick*; what it cannot
 * cover is a later write straight into the form — gallery reuse restoring a
 * print saved before the envelope was pinned goes through
 * `applyMetadataToForm`, which can leave the model alone and only move
 * `steps`. So this watches the VALUES, not just the recipe identity, and
 * re-asserts only the fields that disagree (assigning an equal value would
 * re-trigger it for nothing). Shared policy with desktop's
 * `reconcileModelCapabilities`.
 */
watch(
  () =>
    [
      fixedRecipeControls.value,
      form.steps,
      form.guidance,
      form.width,
      form.height,
      form.frames,
    ] as const,
  () => {
    const fixed = fixedRecipeControls.value;
    if (fixed.steps !== undefined && form.steps !== fixed.steps) form.steps = fixed.steps;
    if (fixed.guidance !== undefined && form.guidance !== fixed.guidance) {
      form.guidance = fixed.guidance;
    }
    if (fixed.width !== undefined && form.width !== fixed.width) form.width = fixed.width;
    if (fixed.height !== undefined && form.height !== fixed.height) form.height = fixed.height;
    if (fixed.frames !== undefined && form.frames !== fixed.frames) form.frames = fixed.frames;
  },
  { immediate: true },
);
const stepsError = computed(() =>
  profileStepsValidationError(form.steps, selectedGenerationModel.value, form.pipeline),
);
const guidanceError = computed(() =>
  profileGuidanceValidationError(
    caps.value.fixedGuidance ?? form.guidance,
    selectedGenerationModel.value,
    form.pipeline,
  ),
);
const basicParametersValid = computed(() => !stepsError.value && !guidanceError.value);
const mobileMediaBudgetError = computed(() => mobileMediaBudgetValidationError(form));
// Desktop parity: a conditioned LTX-2 render may go out undescribed, so the
// Develop button and the pre-submit guard both stop demanding a prompt the
// host would accept. The placeholder carries the same news.
const promptMissing = computed(() => promptRequired(form) && !form.prompt.trim());
const promptFieldPlaceholder = computed(() => promptPlaceholder(form, "Describe the print…"));
const developBlockerReason = computed<string | null>(() => {
  // An empty prompt is self-evident beside the composer and does not warrant
  // a persistent banner. Everything outside the visible composer names the
  // exact correction beside the pinned action.
  if (!selectedModelAvailable.value) return "Choose an installed model before generating.";
  if (quickExpansionSnapshot.value && quickStaleReasons.value.length > 0) {
    return "The prepared rewrite no longer matches these settings. Use a recovery action above.";
  }
  if (!seedValid.value) return "Enter a valid whole-number seed, or choose Random.";
  // Only malformed width/height blocks; off-profile sizes are advisories and
  // the server's own refusal is the authority.
  if (!resolutionValid.value) return "Enter whole-number width and height.";
  if (stepsError.value) return stepsError.value;
  if (guidanceError.value) return guidanceError.value;
  if (mobileMediaBudgetError.value) return mobileMediaBudgetError.value;
  if (sourceConditioningError.value) return sourceConditioningError.value;
  if (h3AuthoringError.value) return h3AuthoringError.value;
  if (!sourceControlsValid.value) return "Correct the source image settings above.";
  if (!parameterValid.value) return "Open Advanced and correct the highlighted settings.";
  return null;
});
const developDisabled = computed(() => promptMissing.value || developBlockerReason.value !== null);
const estimateRequest = computed(() => {
  if (!form.model) return null;
  return buildGenerationEstimateRequest(buildRequest(form), form.family);
});
const queuedJobs = computed(() => railOrder(generation.pending));
const printJobsByKey = computed(
  () => new Map(generation.pending.map((job) => [`print:${job.clientId}`, job])),
);
/** Live dispatch order across every connected host, folded from the queue
 *  listings the activity poll already reads. */
const queueStatusIndex = computed(() =>
  buildQueueStatusIndex(
    Object.entries(liveQueues.value).map(([hostId, listing]) => ({
      hostId,
      entries: listing.entries,
      plan: listing.plan,
    })),
  ),
);
const printActivity = computed<ActivityJobVM[]>(() => {
  const ordered = queuedJobs.value;
  // `mergeActivity` sorts active work by RECENCY, but a print queue is FIFO.
  // Re-express each rail position as a descending timestamp so the merge can
  // interleave sequences without ever reversing the queue the user submitted.
  const newest = ordered.reduce((max, job) => Math.max(max, job.clientId), 0);
  return ordered.map((job, index) => {
    const vm: PrintActivityVM = {
      kind: "print" as const,
      key: `print:${job.clientId}`,
      hostId: job.hostId ?? selectedHostId.value,
      hostLabel: job.hostLabel ?? "",
      model: job.model,
      prompt: job.prompt,
      phase: job.status === "queued" ? ("queued" as const) : ("running" as const),
      progress: null,
      chain: null,
      actions: ["cancel" as const],
      createdAtMs: newest - index,
      // The iPhone queue renders `generation.pending` only, so nothing settled
      // ever reaches this list — the strip's attention/expiry rules are a
      // desktop and web concern for now.
      settledAtMs: job.settledAtMs,
      error: job.error,
    };
    return withLiveQueueStatus(vm, queueStatusIndex.value, job.id);
  });
});
const sequenceActivity = computed<ActivityJobVM[]>(() => {
  const route = sequenceRoute.value;
  const job = sequenceJob.value;
  if (!route || !job) return [];
  return [
    sequenceToVM(
      job,
      { hostId: route.hostId, hostLabel: sequenceHostName(route.hostId) },
      sequenceProgress.value,
    ),
  ];
});
/** ONE queue: single prints and durable sequences in the same list. */
const activityRows = computed<ActivityRow[]>(() =>
  mergeActivity(printActivity.value, sequenceActivity.value).flatMap((vm): ActivityRow[] => {
    if (vm.kind === "sequence") return [{ key: vm.key, sequence: vm, print: null }];
    const print = printJobsByKey.value.get(vm.key);
    return print
      ? [
          {
            key: vm.key,
            sequence: null,
            print,
            queuePosition: vm.queuePosition ?? null,
            blockedReason: vm.blockedReason ?? null,
          },
        ]
      : [];
  }),
);

/**
 * Status code for one queue row, in this list's compact uppercase idiom. A
 * queued print reads its place from the live `/api/queue` listing rather than
 * the one-shot SSE frame it was born with, so the number counts down; only a
 * job the scheduler genuinely parked says why instead. The vocabulary itself
 * is the shared one — web and desktop resolve the same waiting row the same
 * way and only the casing is local.
 */
function activityRowStatus(row: ActivityRow): string {
  if (!row.print) return "";
  if (row.print.status !== "queued") return jobStatusCode(row.print);
  return queueWaitCode(
    resolveQueueWait({
      position: row.queuePosition ?? row.print.queuePosition,
      blockedReason: row.blockedReason,
    }),
  );
}
const sharedMobileActivity = computed(() => {
  const local = new Set(
    generation.jobs.flatMap((job) =>
      job.id ? [`${job.hostId ?? selectedHostId.value}:generation:${job.id}`] : [],
    ),
  );
  if (sequenceJob.value && sequenceRoute.value) {
    local.add(`${sequenceRoute.value.hostId}:sequence:${sequenceJob.value.id}`);
  }
  return mergeFleetActivity(Object.values(liveActivityHosts.value)).filter(
    (row) => !local.has(row.key),
  );
});
const expandedQueueFailures = ref(new Set<string>());

function toggleQueueFailure(key: string): void {
  const next = new Set(expandedQueueFailures.value);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  expandedQueueFailures.value = next;
}

let mobilePrintSelectionEpoch = 0;

async function selectMobilePrint(job: Job): Promise<void> {
  const epoch = ++mobilePrintSelectionEpoch;
  generation.select(job.clientId);
  if (job.hostId && hosts.value.some((host) => host.id === job.hostId)) {
    await selectHost(job.hostId);
  }
  if (epoch !== mobilePrintSelectionEpoch) return;
  const request = job.request;
  if (request) {
    if (request.source_image || request.edit_images?.length) {
      preserveRestoredSourceCanvas(request.edit_images?.[0] ?? request.source_image ?? "");
    }
    applyRequestToForm(form, request, generationModels.value);
    void restoreRunningJobSource(request, epoch);
  }
  draft.stopEditing();
  draft.output = "single";
  latestResultClientId.value = job.status === "complete" ? job.clientId : null;
  tab.value = "generate";
}

async function restoreRunningJobSource(request: GenerateRequest, epoch: number): Promise<void> {
  if (!request.source_image) return;
  const effective = request.source_image;
  const restored = await sha256HexOfBase64(effective)
    .then((sha256) => restoreGenerationSourceMedia(sha256))
    .catch(() => null);
  if (
    !restored ||
    epoch !== mobilePrintSelectionEpoch ||
    form.sourceImage !== effective ||
    form.model !== request.model
  )
    return;
  preserveRestoredSourceCanvas(restored.base64);
  form.sourceImage = restored.base64;
  form.sourceImageName = restored.filename;
  await nextTick();
  if (
    epoch !== mobilePrintSelectionEpoch ||
    form.sourceImage !== restored.base64 ||
    form.model !== request.model
  )
    return;
  form.sourceImageWidth = restored.width ?? null;
  form.sourceImageHeight = restored.height ?? null;
  form.width = request.width;
  form.height = request.height;
  const fit = parseSourceFitPolicy(request.source_fit);
  if (fit) form.sourceFit = fit;
}

function selectCurrentMobileSequence(): void {
  const route = sequenceRoute.value;
  const detail = sequenceJob.value;
  if (!route || !detail) return;
  loadMobileSequenceIntoCreate(detail);
}

function loadMobileSequenceIntoCreate(detail: ChainJobDetail): void {
  const script = normalizeServerChainScript(detail.script);
  if (script) {
    const loaded = chainScriptToClips(script);
    const shared = loaded.shared;
    if (shared.model) {
      const model = generationModels.value.find((entry) => entry.name === shared.model);
      if (model) applyModelDefaults(form, model);
      else form.model = shared.model;
    }
    if (shared.width != null) form.width = shared.width;
    if (shared.height != null) form.height = shared.height;
    if (shared.fps != null) form.fps = shared.fps;
    if (shared.steps != null) form.steps = shared.steps;
    if (shared.guidance != null) form.guidance = shared.guidance;
    if (shared.strength != null) form.strength = shared.strength;
    if (loaded.openingImage?.base64) {
      preserveRestoredSourceCanvas(loaded.openingImage.base64);
    }
    form.sourceFit = { mode: "crop-fill", alignX: "center", alignY: "center" };
    form.seed = shared.seed ?? "";
    draft.stopEditing();
    draft.output = "sequence";
    draft.clips.splice(0, draft.clips.length, ...loaded.clips);
    draft.activeClipId = loaded.clips[0]?.id ?? null;
    draft.enableAudio = loaded.enableAudio;
    draft.openingImage = loaded.openingImage;
  }
  tab.value = "generate";
}

let mobileLiveWorkSelectionEpoch = 0;

/** Restore server-owned work through its exact Keychain-authenticated route.
 *  This mirrors desktop's live-work selection while keeping iPhone remote-only:
 *  generation settings come from the authoritative queue row, and durable
 *  sequences come from their server-side script. */
async function openMobileLiveWork(row: FleetActiveWork): Promise<void> {
  const epoch = ++mobileLiveWorkSelectionEpoch;
  const host = connectedHosts.value.find((candidate) => candidate.id === row.hostId);
  if (!host) {
    setGenerationStatus("That machine is no longer connected.", true);
    return;
  }
  const target = mobileHostTarget(host);
  const snapshot = liveActivityHosts.value[row.hostId];

  try {
    if (!snapshot?.instanceId || snapshot.routeUrl !== target.baseUrl) {
      throw new Error("This queue item belongs to a machine route that has changed.");
    }
    const verifyAuthority = async (): Promise<void> => {
      const current = await apiJsonTo<ServerStatus>(target, "/api/status");
      if (current.instance_id !== snapshot.instanceId) {
        throw new Error("This queue item belongs to a different Mold server instance.");
      }
    };
    await verifyAuthority();
    if (epoch !== mobileLiveWorkSelectionEpoch) return;

    if (row.kind === "generation") {
      const queue = await listQueue(target);
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      await verifyAuthority();
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      const entry = queue.entries.find((candidate) => candidate.id === row.id);
      if (!entry?.metadata) {
        setGenerationStatus("This host cannot restore settings for that generation.", true);
        return;
      }
      await selectHost(host.id);
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      const metadata = entry.metadata as OutputMetadata;
      applyMetadataToForm(
        form,
        entry.seed_pinned === false
          ? ({ ...metadata, seed: null } as unknown as OutputMetadata)
          : metadata,
        generationModels.value,
      );
      draft.stopEditing();
      draft.output = "single";
      setGenerationStatus("Prompt settings restored");
      tab.value = "generate";
      void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
      return;
    }

    if (row.kind === "sequence") {
      const detail = await apiJsonTo<ChainJobDetail>(
        target,
        `/api/chain-jobs/${encodeURIComponent(row.id)}`,
      );
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      await verifyAuthority();
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      await selectHost(host.id);
      if (epoch !== mobileLiveWorkSelectionEpoch) return;
      if (!normalizeServerChainScript(detail.script)) {
        throw new Error("This sequence job has no restorable script.");
      }
      loadMobileSequenceIntoCreate(detail);
      setGenerationStatus("Sequence settings restored");
      return;
    }

    if (row.kind === "download") {
      openCatalog(host.id);
      return;
    }
    tab.value = "hosts";
    showHostDetail(host.id);
  } catch (error) {
    if (epoch !== mobileLiveWorkSelectionEpoch) return;
    setGenerationStatus(describeTransportError(error, host.name), true);
  }
}
/**
 * Running and waiting are counted apart. A settled sequence keeps its row (for
 * Resume / Dismiss) but is neither, and a queued print is NOT active work —
 * counting rows on screen is what made one rendering job plus four waiting
 * ones read "5 ACTIVE".
 */
const runningRowCount = computed(
  () =>
    activityRows.value.filter((row) =>
      row.print ? row.print.status !== "queued" : row.sequence.state === "running",
    ).length,
);
const waitingRowCount = computed(
  () =>
    activityRows.value.filter((row) =>
      row.print ? row.print.status === "queued" : row.sequence.state === "queued",
    ).length,
);
const activityCounts = computed(() => ({
  running: runningRowCount.value,
  waiting: waitingRowCount.value,
}));
const sequenceRowProgress = computed(() => {
  const progress = sequenceProgress.value;
  return progress && progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : null;
});
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
const generatedPreviewHost = computed(() => {
  const job = latestResultJob.value;
  return hosts.value.find((candidate) => candidate.id === job?.hostId) ?? null;
});
const generatedPreviewTarget = computed<ApiTarget>(() => {
  const host = generatedPreviewHost.value;
  return host ? mobileHostTarget(host) : { baseUrl: "", apiKey: null };
});
const resultPreviewError = computed(() => {
  const job = latestResultJob.value;
  return job?.resultError ? describeTransportError(job.resultError, job.hostLabel) : "";
});
const developButtonLabel = computed(() =>
  generationSubmissionPhase.value
    ? generationSubmissionPhase.value === "placement"
      ? "Cancel · Checking placement…"
      : "Cancel · Preparing source…"
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
const queueAnnouncement = computed(() => activityAnnouncement(activityCounts.value));
const generationStatus = computed(() => {
  const active = activeGeneration.value;
  if (!active) return progress.value;
  switch (active.status) {
    case "queued": {
      // Live listing over the one-shot SSE frame, so the number counts down.
      const live = queueStatusFor(
        queueStatusIndex.value,
        active.hostId ?? selectedHostId.value,
        active.id,
      );
      return queueWaitLabel(
        resolveQueueWait({
          position: live?.position ?? active.queuePosition,
          blockedReason: live?.blockedReason,
        }),
      );
    }
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
    referenceUploads: serverCapabilities[host.id]?.reference_uploads ?? null,
  };
}

function loadHosts(): MobileHost[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as MobileHost[];
    return raw.map((host) => ({
      ...host,
      connected: host.connected !== false,
      apiKey: "",
      online: false,
    }));
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
      connected: true,
      online: true,
    };
    if (existing) Object.assign(existing, saved);
    else hosts.value.push(saved);
    knownHostReachability.add(saved.id);
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

async function pairFromCode(code: () => Promise<string>): Promise<void> {
  if (pairing.value) return;
  pairing.value = true;
  hostError.value = "";
  try {
    const payload = parseMobilePairingPayload(await code());
    if (payload.expires_at !== null && payload.expires_at <= Math.floor(Date.now() / 1000)) {
      throw new Error("That pairing code expired. Create a new one in the host's Settings.");
    }
    const baseUrl = normalizeRemoteAddress(payload.base_url);
    const iPad =
      /iPad/i.test(navigator.userAgent) ||
      (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const claim = await claimPairingSession(baseUrl, payload.token, {
      name: iPad ? "Mold on iPad" : "Mold on iPhone",
      kind: iPad ? "ipad" : "iphone",
    });
    if (claim.instance_id !== payload.instance_id) {
      throw new Error("The pairing code was redeemed by a different Mold host.");
    }
    hostInput.name = claim.hostname ?? payload.name;
    hostInput.address = baseUrl;
    hostInput.apiKey = claim.api_key ?? "";
    await connectHost();
  } catch (error) {
    if (!pairingScannerCancelled) hostError.value = describeTransportError(error);
  } finally {
    pairingScannerCancelled = false;
    pairing.value = false;
  }
}

function scanPairingCode(): Promise<void> {
  return pairFromCode(async () => {
    let permission = await checkBarcodeScannerPermissions();
    if (permission === "prompt") {
      permission = await requestBarcodeScannerPermissions();
    }
    if (permission !== "granted") {
      throw new Error(
        "Camera access is required to scan a pairing code. Allow camera access in Settings and try again.",
      );
    }
    pairingScannerOpen.value = true;
    await nextTick();
    try {
      const result = await scan({
        cameraDirection: "back",
        formats: [Format.QRCode],
        windowed: true,
      });
      return result.content;
    } finally {
      pairingScannerOpen.value = false;
    }
  });
}

async function cancelPairingScan(): Promise<void> {
  if (!pairingScannerOpen.value) return;
  pairingScannerCancelled = true;
  try {
    await cancelBarcodeScanner();
  } catch (error) {
    pairingScannerCancelled = false;
    hostError.value = describeTransportError(error, "Pairing scanner");
  }
}

async function openPairingUrls(urls: string[]): Promise<void> {
  const pairingUrl = urls.find((url) => url.startsWith("mold://pair?"));
  if (pairingUrl) await pairFromCode(() => Promise.resolve(pairingUrl));
}

async function listenForPairingDeepLinks(): Promise<void> {
  stopPairingDeepLinks = await onOpenUrl((urls) => void openPairingUrls(urls));
  const current = await getCurrent();
  if (current) await openPairingUrls(current);
}

async function selectHost(id: string): Promise<void> {
  if (!connectedHosts.value.some((host) => host.id === id)) return;
  selectedHostId.value = id;
  await refreshModels();
  // A host restored from storage may only become ready during refreshModels.
  // Re-run the carry after its exact instance identity is known.
  if (selectedHostId.value === id) carryQuickTransformToHost(id);
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
    const priorCacheKey = host.instanceId?.trim() || host.id;
    const nextInstanceId = payload.status.instance_id ?? host.instanceId;
    const nextCacheKey = nextInstanceId?.trim() || host.id;
    if (priorCacheKey !== nextCacheKey) void clearCachedGalleryHosts([priorCacheKey]);
    host.version = payload.status.version;
    host.hostname = payload.status.hostname ?? undefined;
    host.instanceId = nextInstanceId;
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
  const wasKnownOffline = knownHostReachability.has(host.id) && !host.online;
  const timeout = setTimeout(() => controller.abort(), HOST_PROBE_TIMEOUT_MS);
  const probe = { epoch, controller, timeout };
  hostProbes.set(host.id, probe);
  try {
    const status = await apiJsonTo<ServerStatus>(mobileHostTarget(host), "/api/status", {
      signal: controller.signal,
    });
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    knownHostReachability.add(host.id);
    updateHostStatus({ id: host.id, status });
    if (wasKnownOffline && tab.value === "gallery") void refreshGallery();
  } catch {
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    knownHostReachability.add(host.id);
    updateHostStatus({ id: host.id, status: null });
  } finally {
    if (hostProbes.get(host.id)?.epoch === epoch) hostProbes.delete(host.id);
    clearTimeout(timeout);
  }
}

function probeHosts(): void {
  for (const host of connectedHosts.value) void probeHost(host);
}

function persistMobileActivity(): void {
  // The route contains the Keychain-supplied API key at runtime. Persist only
  // the safe snapshot/display fields; mobile API keys remain Keychain-only.
  const safe = Object.fromEntries(
    Object.entries(liveActivityHosts.value).map(([id, { target: _target, ...snapshot }]) => [
      id,
      snapshot,
    ]),
  );
  localStorage.setItem(LIVE_ACTIVITY_KEY, JSON.stringify(safe));
}

async function refreshMobileActivity(): Promise<void> {
  if (liveActivityRefreshing) return;
  liveActivityRefreshing = true;
  const configured = connectedHosts.value;
  try {
    await Promise.all(
      configured.map(async (host) => {
        const route = {
          hostId: host.id,
          hostLabel: host.name,
          target: mobileHostTarget(host),
        };
        const epoch = (liveActivityEpochs[host.id] ?? 0) + 1;
        liveActivityEpochs[host.id] = epoch;
        const previous = liveActivityHosts.value[host.id];
        const previousQueue = liveQueues.value[host.id] ?? null;
        // The queue read rides the same tick: it is what makes a queued row's
        // position count down, and losing it only costs the position.
        const [result, queue] = await Promise.all([
          (async () => {
            try {
              if (!host.online) throw new Error("Host is offline");
              return await listActiveWork(route.target);
            } catch (error) {
              return error instanceof Error ? error : new Error(String(error));
            }
          })(),
          (async (): Promise<QueueListing | null> => {
            try {
              if (!host.online) return null;
              return await listQueue(route.target);
            } catch {
              return previousQueue;
            }
          })(),
        ]);
        if (liveActivityEpochs[host.id] !== epoch) return;
        liveActivityHosts.value = {
          ...liveActivityHosts.value,
          [host.id]: reconcileActivityHost(route, previous, result),
        };
        const queues = { ...liveQueues.value };
        if (queue) queues[host.id] = queue;
        else delete queues[host.id];
        liveQueues.value = queues;
      }),
    );
    const ids = new Set(connectedHosts.value.map((host) => host.id));
    liveActivityHosts.value = Object.fromEntries(
      Object.entries(liveActivityHosts.value).filter(([id]) => ids.has(id)),
    );
    liveQueues.value = Object.fromEntries(
      Object.entries(liveQueues.value).filter(([id]) => ids.has(id)),
    );
    persistMobileActivity();
  } finally {
    liveActivityRefreshing = false;
  }
}

async function pollMobileActivity(): Promise<void> {
  await refreshMobileActivity();
  if (!unmounted) liveActivityTimer = setTimeout(pollMobileActivity, 5_000);
}

/** Drop a departed host's organization buckets so its tags and collections
 * stop feeding the merged Library, and release a tag filter that no longer
 * resolves against any connected host. */
function pruneHostOrganization(id: string): void {
  delete hostTags[id];
  delete hostCollections[id];
  const active = libraryFilters.tag;
  if (active && !mergedTags.value.some((tag) => tagKey(tag.name) === active)) {
    libraryFilters.tag = null;
  }
}

function disconnectHost(id: string): void {
  cancelHostProbe(id);
  const host = hosts.value.find((candidate) => candidate.id === id);
  if (!host) return;
  host.connected = false;
  host.online = false;
  delete hostTelemetry[id];
  pruneHostOrganization(id);
  if (selectedHostId.value === id) {
    selectedHostId.value = connectedHosts.value[0]?.id ?? "";
    models.value = [];
    modelsHostId.value = "";
    void refreshModels();
  }
  if (catalogHostId.value === id) catalogHostId.value = connectedHosts.value[0]?.id ?? "";
  persistHosts();
}

function reconnectHost(id: string): void {
  const host = hosts.value.find((candidate) => candidate.id === id);
  if (!host) return;
  host.connected = true;
  knownHostReachability.delete(id);
  persistHosts();
  void probeHost(host);
}

function removeHost(id: string): void {
  cancelHostProbe(id);
  knownHostReachability.delete(id);
  pruneHostOrganization(id);
  const removedSelectedHost = selectedHostId.value === id;
  const removedCatalogHost = catalogHostId.value === id;
  if (hostDetailId.value === id) hostDetailId.value = "";
  const removedHost = hosts.value.find((host) => host.id === id);
  hosts.value = hosts.value.filter((host) => host.id !== id);
  if (removedSelectedHost) {
    selectedHostId.value = connectedHosts.value[0]?.id ?? "";
    models.value = [];
    modelsHostId.value = "";
    void refreshModels();
  }
  if (removedCatalogHost) catalogHostId.value = connectedHosts.value[0]?.id ?? "";
  persistHosts();
  if (removedHost) {
    void clearCachedGalleryHosts([
      removedHost.instanceId?.trim() || removedHost.id,
      removedHost.id,
    ]);
  }
  void invoke("keychain_delete_api_key", { hostId: id });
}

function selectCatalogHost(id: string): void {
  if (connectedHosts.value.some((host) => host.id === id)) catalogHostId.value = id;
}

function openCatalog(id?: string, intent?: Omit<CatalogFilterIntent, "token">): void {
  if (id && connectedHosts.value.some((host) => host.id === id)) catalogHostId.value = id;
  else if (!connectedHosts.value.some((host) => host.id === catalogHostId.value)) {
    catalogHostId.value = selectedHostId.value || connectedHosts.value[0]?.id || "";
  }
  hostDetailId.value = "";
  if (intent) catalogFilterIntent.value = { ...intent, token: ++catalogIntentToken };
  tab.value = "catalog";
}

/** The sequence empty state must LAND on the filtered Discover shelf — a bare
 *  tab switch left the user to rediscover the Video + Models filters. */
function browseSequenceModels(): void {
  openCatalog(selectedHost.value?.id, { mediaType: "video", kind: "checkpoint" });
}

function catalogModelsChanged(hostId: string): void {
  if (hostId === selectedHostId.value) void refreshModels();
  else if (automaticRouting.value) void refreshRoutingModels();
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
      apiJsonTo<ServerCapabilities>(target, "/api/capabilities").catch(() => null),
    ]);
    if (unmounted || epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    knownHostReachability.add(hostId);
    host.online = true;
    host.version = status.version;
    host.hostname = status.hostname ?? undefined;
    const priorCacheKey = mobileGalleryCacheKey(host);
    host.instanceId = status.instance_id ?? host.instanceId;
    const nextCacheKey = mobileGalleryCacheKey(host);
    if (priorCacheKey !== nextCacheKey) await clearCachedGalleryHosts([priorCacheKey]);
    captureHostTelemetry(hostId, status);
    expandCapabilities[hostId] = capabilities?.expand;
    serverCapabilities[hostId] = capabilities;
    // Keep auxiliary entries for the Upscale and ControlNet pickers, while
    // the main Model select uses `generationModels` so those tools can never
    // become the active generation model.
    models.value = filterRestrictedModels(entries, capabilities);
    modelsHostId.value = hostId;
    modelsByHost.value = { ...modelsByHost.value, [hostId]: models.value };
    modelSnapshotIdentities.value = {
      ...modelSnapshotIdentities.value,
      [hostId]: {
        cacheKey: nextCacheKey,
        updatedAt: Date.now(),
        serverVersion: status.version ?? null,
      },
    };
    await storeCachedHostPresentation({
      hostId: nextCacheKey,
      updatedAt: Date.now(),
      instanceId: host.instanceId?.trim() || null,
      serverVersion: status.version ?? null,
      models: models.value,
      capabilities,
    });
    const selectedEntry = generationModels.value.find((model) => model.name === form.model);
    if (selectedEntry) {
      reconcileModelCapabilities(form, selectedEntry);
    } else if (generationModels.value[0]) {
      applyModelDefaults(form, generationModels.value[0]);
    }
    return true;
  } catch (error) {
    if (unmounted || epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    knownHostReachability.add(hostId);
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

/**
 * Read `/api/models` (and capabilities) from every reachable machine other
 * than the browsed one, so an automatic policy can see which machines hold the
 * model. A machine that refuses keeps its previous snapshot rather than
 * dropping out of routing on one bad poll; a forgotten machine is pruned.
 */
async function refreshRoutingModels(): Promise<void> {
  const peers = routingHosts.value.filter((host) => host.id !== selectedHostId.value);
  const snapshots = await Promise.all(
    peers.map(async (host) => {
      try {
        const target = mobileHostTarget(host);
        const cacheKey = mobileGalleryCacheKey(host);
        const serverVersion = host.version ?? null;
        const [entries, capabilities] = await Promise.all([
          apiJsonTo<ModelEntry[]>(target, "/api/models"),
          apiJsonTo<ServerCapabilities>(target, "/api/capabilities").catch(() => null),
        ]);
        const currentHost = hosts.value.find((candidate) => candidate.id === host.id);
        if (
          !currentHost ||
          mobileGalleryCacheKey(currentHost) !== cacheKey ||
          currentHost.baseUrl !== target.baseUrl ||
          (currentHost.apiKey || null) !== target.apiKey
        ) {
          return null;
        }
        if (capabilities !== null) {
          expandCapabilities[host.id] = capabilities.expand;
          serverCapabilities[host.id] = capabilities;
        }
        return [
          host.id,
          { cacheKey, updatedAt: Date.now(), serverVersion } satisfies ModelSnapshotIdentity,
          filterRestrictedModels(entries, capabilities),
        ] as const;
      } catch {
        return null;
      }
    }),
  );
  if (unmounted) return;
  const next: Record<string, ModelEntry[]> = {};
  const nextIdentities: Record<string, ModelSnapshotIdentity> = {};
  for (const [id, entries] of Object.entries(modelsByHost.value)) {
    const host = connectedHosts.value.find((candidate) => candidate.id === id);
    const identity = modelSnapshotIdentities.value[id];
    if (host && identity?.cacheKey === mobileGalleryCacheKey(host)) {
      next[id] = entries;
      nextIdentities[id] = identity;
    }
  }
  for (const snapshot of snapshots) {
    if (snapshot) {
      next[snapshot[0]] = snapshot[2];
      nextIdentities[snapshot[0]] = snapshot[1];
    }
  }
  modelsByHost.value = next;
  modelSnapshotIdentities.value = nextIdentities;
  // The union just grew: a machine other than the browsed one may hold the
  // only generation model in the fleet, and the picker must land on it the
  // same way `refreshModels` lands on the browsed machine's first model.
  if (!automaticRouting.value) return;
  const selectedEntry = generationModels.value.find((entry) => entry.name === form.model);
  if (selectedEntry) reconcileModelCapabilities(form, selectedEntry);
  else if (generationModels.value[0]) applyModelDefaults(form, generationModels.value[0]);
}

/**
 * A concrete policy IS the browsed machine. Restore flows (a queue row, a
 * print, a removed machine) move the browsed machine directly, so a pinned
 * target follows them; otherwise the picker could display one machine while
 * work went to another. An automatic policy is untouched by browsing.
 */
watch(selectedHostId, (id) => {
  if (!id || isAutomaticTarget(generateTargetPolicy.value)) return;
  if (generateTargetPolicy.value === id) return;
  generateTargetPolicy.value = id;
  saveMobileGenerateTarget(id);
});

/** The explicit "Use for generations" action: browse that machine and pin the
 *  generation target to it, whatever the previous policy was. */
async function useHostForGenerations(id: string): Promise<void> {
  await selectGenerateTarget(id);
}

/** Persist the generation-target policy. A concrete machine also becomes the
 *  browsed machine, which is what the pinned path has always meant. */
async function selectGenerateTarget(value: string): Promise<void> {
  generateTargetPolicy.value = value;
  saveMobileGenerateTarget(value);
  if (value === AUTO_TARGET_ID || value === CAPABLE_TARGET_ID) {
    await refreshRoutingModels();
    return;
  }
  await selectHost(value);
}

/** Retire the transport but KEEP the row: a settled job still offers Resume
 *  and Dismiss, and a settled job has nothing left to stream. */
function stopSequenceTransport(): void {
  sequenceWatch?.stop();
  sequenceWatch = null;
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

function sequenceHostName(hostId: string): string {
  return hosts.value.find((host) => host.id === hostId)?.name ?? "sequence host";
}

/** Optimistic row so an accepted job is visible before the first frame lands. */
function pendingSequenceJob(jobId: string, model: string, stageCount: number): ChainJobDetail {
  const now = Date.now();
  return {
    id: jobId,
    state: "queued",
    model,
    stage_count: stageCount,
    current_stage: 0,
    created_at_unix_ms: now,
    updated_at_unix_ms: now,
    error: null,
    ephemeral: false,
    stages: [],
  };
}

/**
 * Attach to a durable job on ONE immutable route. SSE is primary; the watcher
 * itself owns the 5s poll fallback and the wake re-sync (see sequenceWatch).
 */
function watchSequenceJob(
  hostId: string,
  target: ApiTarget,
  jobId: string,
  seed?: { model: string; stageCount: number },
): void {
  stopSequenceTransport();
  const route = { hostId, target: { ...target } };
  sequenceRoute.value = route;
  sequenceProgress.value = null;
  sequenceJob.value = pendingSequenceJob(jobId, seed?.model ?? "", seed?.stageCount ?? 0);
  sequenceWatch = watchChainJob({
    target: route.target,
    jobId,
    onUpdate: (live) => {
      if (sequenceRoute.value !== route) return;
      if (live.detail) sequenceJob.value = live.detail;
      const active = live.activeStage;
      sequenceProgress.value = active !== null ? (live.progress[active] ?? null) : null;
      sequenceError.value = "";
    },
    onError: (error) => {
      if (sequenceRoute.value !== route) return;
      sequenceError.value = describeTransportError(error, sequenceHostName(route.hostId));
    },
  });
}

// A settled durable job has nothing left to stream: drop the transport and the
// recovery record, but keep the row so Resume / Dismiss stay reachable.
watch(
  () => sequenceJob.value?.state,
  (state, previous) => {
    const route = sequenceRoute.value;
    if (!state || state === previous || !route) return;
    if (state === "queued" || state === "running") return;
    stopSequenceTransport();
    clearSequenceRecovery();
    generationAnnouncement.value =
      state === "completed"
        ? `Sequence completed on ${sequenceHostName(route.hostId)}.`
        : `Sequence ${state}. ${sequenceJob.value?.error ?? ""}`.trim();
    if (state === "completed" && tab.value === "gallery") void refreshGallery();
  },
);

/** One machine's answer to the automatic-routing fan-out. */
interface MobilePlacementProbe {
  host: MobileHost;
  /** The machine's exact route as it stood when the probe was ISSUED. A slow
   *  preview must never authorize one endpoint and then submit to another. */
  route: HostRoute;
  roundTripMs: number;
  preview: GenerationPlacementPreview | null;
  error: unknown;
  legacyUnsupported: boolean;
}

type MobileAutomaticRoute =
  | {
      kind: "route";
      host: MobileHost;
      route: HostRoute;
      placement: GenerationPlacementPreview | null;
      legacyUnsupported: boolean;
    }
  | { kind: "error"; message: string }
  | { kind: "abandoned" };

/**
 * The machines an automatic policy may dispatch to: reachable, allowed to run
 * the model, and — when any of them already holds it — narrowed to the owners,
 * which is what makes Auto model-aware. Owners spanning incompatible major
 * Mold versions stop automatic routing exactly as they do on desktop; the user
 * picks a machine instead.
 */
function automaticRoutingCandidates(
  model: string,
  family: string,
): { hosts: MobileHost[]; error: string | null } {
  const restrictions: string[] = [];
  const allowed = routingHosts.value.filter((host) => {
    const restriction = modelAccessRestrictionFor(serverCapabilities[host.id], {
      model,
      family,
      generation_profile_sha256: generationProfileHashForHost(host.id, model),
    });
    if (restriction) restrictions.push(restriction.message);
    return !restriction;
  });
  if (allowed.length === 0) {
    return {
      hosts: [],
      error:
        restrictions[0] ??
        "No connected machine is reachable right now. Reconnect a machine and try again. Nothing was queued.",
    };
  }
  const owners = modelHostIds(model);
  const candidates =
    owners.length > 0 ? allowed.filter((host) => owners.includes(host.id)) : allowed;
  // Owners exist but none of them survived the access filter. Say so instead
  // of handing the fan-out an empty candidate set.
  if (candidates.length === 0) {
    return {
      hosts: [],
      error:
        restrictions[0] ??
        `No connected machine can run ${model}. Choose another model or machine. Nothing was queued.`,
    };
  }
  const conflict = profileHashConflict(
    modelsByHost.value,
    model,
    candidates.map((host) => host.id),
    Object.fromEntries(candidates.map((host) => [host.id, host.version ?? null])),
  );
  if (conflict) {
    return {
      hosts: [],
      error: profileConflictMessage(
        conflict.hostIds.map((hostId) => ({
          label: hosts.value.find((host) => host.id === hostId)?.name ?? hostId,
          profileHash: conflict.hashesByHost[hostId] ?? null,
          version: hosts.value.find((host) => host.id === hostId)?.version ?? null,
        })),
      ),
    };
  }
  return { hosts: candidates, error: null };
}

/** The machine an automatic policy would pick from telemetry alone. Used for
 *  the provisional target of source preparation, before any machine has been
 *  asked for a placement plan. */
function provisionalAutomaticHost(model: string, family: string): MobileHost | null {
  const { hosts: candidates } = automaticRoutingCandidates(model, family);
  const views = (candidates.length > 0 ? candidates : routingHosts.value).map(routingHostView);
  const chosen =
    generateTarget.value === CAPABLE_TARGET_ID
      ? pickMostCapableHost(views, null, { lowestIdWins: true })
      : pickAutoHost(views, { lowestIdWins: true });
  return chosen?.host ?? null;
}

function mobileFleetPlacementFailure(
  probes: readonly MobilePlacementProbe[],
  subject: "print" | "sequence",
): string {
  if (probes.length === 1 && probes[0]!.preview) {
    return mobilePlacementFailure(probes[0]!.preview, probes[0]!.host.name, subject);
  }
  const detail = probes
    .map((probe) =>
      probe.preview
        ? mobilePlacementFailure(probe.preview, probe.host.name, subject).replace(
            " Nothing was queued.",
            "",
          )
        : `${probe.host.name} did not answer: ${describeTransportError(probe.error, probe.host.name)}`,
    )
    .join(" ");
  return `No connected machine could run this ${subject}. ${detail} Nothing was queued.`;
}

/**
 * Ask every candidate machine for a placement plan and choose one.
 *
 * Auto takes the soonest predicted completion (round trip included); Most
 * capable takes the strongest GPU among the machines that answered `planned`,
 * using each machine's own `gpu_info.backend`. The winner is returned as a
 * complete route so the caller can freeze it — host id, URL, Keychain key, and
 * instance id — exactly as the pinned path does.
 */
async function routeAutomaticGeneration(options: {
  request: Record<string, unknown>;
  chain: boolean;
  copies: number;
  model: string;
  family: string;
  subject: "print" | "sequence";
  requireAuthoritative: boolean;
  isCurrent?: () => boolean;
  signal?: AbortSignal;
}): Promise<MobileAutomaticRoute> {
  const isCurrent = options.isCurrent ?? (() => true);
  const { hosts: candidates, error } = automaticRoutingCandidates(options.model, options.family);
  if (error) return { kind: "error", message: error };
  const probes: MobilePlacementProbe[] = [];
  const controllers = candidates.map(() => new AbortController());
  let pending = candidates.length;
  let resolveAllSettled!: () => void;
  let resolveFirstPlanned!: () => void;
  const allSettled = new Promise<void>((resolve) => (resolveAllSettled = resolve));
  const firstPlanned = new Promise<void>((resolve) => (resolveFirstPlanned = resolve));
  candidates.forEach((host, index) => {
    void (async () => {
      const controller = controllers[index]!;
      const abortFromCaller = () => controller.abort(options.signal?.reason);
      if (options.signal?.aborted) abortFromCaller();
      else options.signal?.addEventListener("abort", abortFromCaller, { once: true });
      const started = performance.now();
      const elapsed = () => Math.max(0, performance.now() - started);
      const probeOptions = { signal: controller.signal };
      // Frozen before the request leaves: the winner carries this snapshot, so
      // a URL, key, or instance that changed mid-flight is caught by the
      // caller's connection fence instead of silently replacing the endpoint
      // the plan was authorized for.
      const route = routeForMobileHost(host);
      const probeTarget = { ...route.target };
      try {
        const preview = options.chain
          ? await previewChainPlacement(probeTarget, options.request, options.copies, probeOptions)
          : await previewGenerationPlacement(
              probeTarget,
              options.request,
              options.copies,
              probeOptions,
            );
        probes.push({
          host,
          route,
          roundTripMs: elapsed(),
          preview,
          error: null,
          legacyUnsupported: false,
        });
        if (classifyPlacementPreview(preview) === "planned") resolveFirstPlanned();
      } catch (probeError) {
        probes.push({
          host,
          route,
          roundTripMs: elapsed(),
          preview: null,
          error: probeError,
          legacyUnsupported:
            probeError instanceof ApiError &&
            (probeError.status === 404 || probeError.status === 405),
        });
      } finally {
        options.signal?.removeEventListener("abort", abortFromCaller);
        pending -= 1;
        if (pending === 0) resolveAllSettled();
      }
    })();
  });
  // Nothing to wait for: the settle promise is only ever resolved by a probe.
  if (pending === 0) resolveAllSettled();
  // A phone must not sit on a stalled machine once another one can run the
  // print — but it must not manufacture a dead end either, so the settle
  // window only starts once some machine has actually answered `planned`.
  await Promise.race([
    allSettled,
    ...(candidates.length > 1
      ? [
          firstPlanned.then(
            () => new Promise<void>((resolve) => setTimeout(resolve, MOBILE_PLACEMENT_SETTLE_MS)),
          ),
        ]
      : []),
  ]);
  if (pending > 0) for (const controller of controllers) controller.abort();
  if (unmounted || !isCurrent()) return { kind: "abandoned" };
  // Route on the snapshot that met the settle window; a late cancellation row
  // must not join the decision.
  const settledProbes = probes.slice();
  const planned = settledProbes.flatMap((probe) =>
    probe.preview && classifyPlacementPreview(probe.preview) === "planned"
      ? [{ host: routingHostView(probe.host), roundTripMs: probe.roundTripMs, probe }]
      : [],
  );
  const chosen = chooseRoutedHost(
    planned.map((entry) => ({
      host: entry.host,
      roundTripMs: entry.roundTripMs,
      preview: entry.probe.preview!,
    })),
    generateTarget.value,
    comparePlacementPreviews,
    { lowestIdWins: true },
  );
  if (chosen) {
    const winner = planned.find((entry) => entry.host.id === chosen.id)!;
    return {
      kind: "route",
      host: winner.probe.host,
      route: winner.probe.route,
      placement: winner.probe.preview,
      legacyUnsupported: false,
    };
  }
  // Servers that predate the authoritative preview still route, unless the
  // request itself requires that authority (reference media).
  const legacy = settledProbes.filter(
    (probe) => probe.legacyUnsupported || classifyPlacementPreview(probe.preview) === "unsupported",
  );
  if (!options.requireAuthoritative && legacy.length > 0) {
    const views = legacy.map((probe) => routingHostView(probe.host));
    const fallback =
      generateTarget.value === CAPABLE_TARGET_ID
        ? pickMostCapableHost(views, null, { lowestIdWins: true })
        : pickAutoHost(views, { lowestIdWins: true });
    if (fallback) {
      const probe = legacy.find((entry) => entry.host.id === fallback.id)!;
      return {
        kind: "route",
        host: probe.host,
        route: probe.route,
        placement: null,
        legacyUnsupported: true,
      };
    }
  }
  return { kind: "error", message: mobileFleetPlacementFailure(settledProbes, options.subject) };
}

async function submitMobileSequence(): Promise<void> {
  if (sequenceStarting.value) {
    cancelMobileSequenceSubmission();
    return;
  }
  const automatic = automaticRouting.value;
  // Under an automatic policy the machine is provisional until the placement
  // fan-out answers; source fitting only ever uses it for an optional upscale,
  // so the built request stays machine-independent.
  const initialHost = automatic
    ? provisionalAutomaticHost(form.model, form.family)
    : selectedHost.value;
  const restriction =
    initialHost && !automatic
      ? modelAccessRestrictionFor(serverCapabilities[initialHost.id], {
          model: form.model,
          family: form.family,
          generation_profile_sha256: generationProfileHashForHost(initialHost.id, form.model),
        })
      : null;
  if (restriction) {
    sequenceError.value = restriction.message;
    return;
  }
  const entry = selectedGenerationModel.value;
  if (!initialHost || !entry) return;
  let host: MobileHost = initialHost;
  let target = { ...mobileHostTarget(host) };
  let frozenRoute: HostRoute = {
    hostId: host.id,
    label: host.name,
    kind: "remote",
    target,
    instanceId: host.instanceId ?? null,
  };
  // Freeze all request-affecting values at the tap boundary. Source fitting
  // and placement preview are asynchronous; edits during either await belong
  // to the next submission.
  const requestForm = cloneGenerateForm(form);
  const clips = JSON.parse(JSON.stringify(draft.clips)) as typeof draft.clips;
  // The opening image obeys the checkpoint's own source-image contract: a
  // checkpoint that reads none shows no well, so a retained image is parked
  // out of the request rather than shipped as conditioning admission refuses.
  const openingImageSupported =
    parseSourceImageCapability(entry.source_image ?? form.sourceImageCapability) !== "unsupported";
  const openingSnapshot =
    openingImageSupported && draft.openingImage ? { ...draft.openingImage } : null;
  const enableAudio = draft.enableAudio;
  const motionTailFrames = sequenceMotionTail.value;
  sequenceStarting.value = true;
  sequenceCancellationRequest = null;
  const token = sequenceSubmissionGuard.begin();
  const signal = sequenceSubmissionGuard.signalFor(token);
  const isCurrent = () => sequenceSubmissionGuard.isCurrent(token) && !signal.aborted;
  sequenceError.value = "";
  try {
    // Stale limits would mis-gate audio and frame caps for the routed host.
    if (!chainLimits.value || chainLimits.value.model !== entry.name) await loadChainLimits();
    if (!isCurrent()) return;
    requestForm.sourceImage = openingSnapshot?.base64 ?? null;
    requestForm.maskImage = null;
    if (requestForm.sourceImage) {
      const result = await applySourceFitPreprocess(
        {
          source: requestForm.sourceImage,
          mask: null,
          policy: coerceSourceFitForMaskless(requestForm.sourceFit),
          target: { width: requestForm.width, height: requestForm.height },
        },
        {
          ops: domCanvasOps,
          cache: sourceFitCache,
          upscale: (image, model) => upscaleImage({ image, model, target, signal }),
          onStatus: setGenerationStatus,
        },
      );
      if (!isCurrent()) return;
      requestForm.sourceImage = result.source;
    }
    const openingImage = openingSnapshot
      ? { ...openingSnapshot, base64: requestForm.sourceImage }
      : null;
    const request = buildChainRequest(sequenceParams(requestForm, entry), clips, {
      motionTailFrames,
      enableAudio,
      openingImage,
    });
    let preview: GenerationPlacementPreview | null = null;
    let legacyUnsupported = false;
    if (automatic) {
      const routed = await routeAutomaticGeneration({
        request: request as unknown as Record<string, unknown>,
        chain: true,
        copies: 1,
        model: entry.name,
        family: form.family,
        subject: "sequence",
        requireAuthoritative: false,
        isCurrent,
        signal,
      });
      if (routed.kind === "abandoned") return;
      if (routed.kind === "error") throw new Error(routed.message);
      // Freeze the machine the fan-out chose: this exact route is what the
      // durable sequence is recovered against.
      host = routed.host;
      target = { ...mobileHostTarget(host) };
      frozenRoute = routed.route;
      preview = routed.placement;
      legacyUnsupported = routed.legacyUnsupported;
    } else {
      try {
        preview = await previewChainPlacement(
          target,
          request as unknown as Record<string, unknown>,
          1,
          { signal },
        );
      } catch (error) {
        if (error instanceof ApiError && (error.status === 404 || error.status === 405)) {
          legacyUnsupported = true;
        } else {
          throw error;
        }
      }
    }
    const classification: string = classifyPlacementPreview(preview);
    if (!isCurrent()) return;
    if (!legacyUnsupported && classification !== "unsupported" && classification !== "planned") {
      throw new Error(mobilePlacementFailure(preview, host.name, "sequence"));
    }
    const fenceHost = automatic
      ? hosts.value.find((candidate) => candidate.id === frozenRoute.hostId)
      : selectedHost.value;
    if (!sameFrozenHost(frozenRoute, fenceHost)) {
      throw new Error("The selected host changed while checking this sequence.");
    }
    const operationId = createUuid();
    sequenceCancellationRequest = () =>
      apiFetchTo(
        target,
        `/api/chain-jobs/${encodeURIComponent(operationId)}/operations/${encodeURIComponent(operationId)}/cancel`,
        { method: "POST", keepalive: true },
      );
    const response = await apiJsonTo<CreateChainJobResponse>(target, "/api/chain-jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-mold-operation-id": operationId,
      },
      body: JSON.stringify(request),
    });
    if (!isCurrent()) {
      await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(response.job_id)}/cancel`, {
        method: "POST",
      }).catch(() => {});
      return;
    }
    persistSequenceRecovery(host, response.job_id);
    watchSequenceJob(host.id, target, response.job_id, {
      model: entry.name,
      stageCount: clips.length,
    });
  } catch (error) {
    if (!isCurrent()) return;
    sequenceError.value = describeTransportError(error, host.name);
  } finally {
    if (sequenceSubmissionGuard.isCurrent(token)) {
      sequenceStarting.value = false;
      sequenceCancellationRequest = null;
    }
  }
}

function cancelMobileSequenceSubmission(): void {
  if (!sequenceStarting.value) return;
  sequenceSubmissionGuard.invalidate();
  const cancellation = sequenceCancellationRequest;
  sequenceCancellationRequest = null;
  sequenceStarting.value = false;
  sequenceError.value = "";
  if (!cancellation) {
    setGenerationStatus("Sequence preparation cancelled — nothing was queued");
    return;
  }
  setGenerationStatus("Cancelling sequence creation…");
  void confirmCancellation(cancellation)
    .then(() => setGenerationStatus("Sequence creation cancelled — nothing was queued"))
    .catch(() => {
      sequenceError.value = "Cancellation could not be confirmed. Check the queue before retrying.";
      setGenerationStatus(sequenceError.value);
    });
}

async function cancelMobileSequence(): Promise<void> {
  const route = sequenceRoute.value;
  const job = sequenceJob.value;
  if (!route || !job) return;
  sequenceError.value = "";
  try {
    await apiFetchTo(route.target, `/api/chain-jobs/${encodeURIComponent(job.id)}/cancel`, {
      method: "POST",
    });
    await sequenceWatch?.refresh();
  } catch (error) {
    sequenceError.value = describeTransportError(error, sequenceHostName(route.hostId));
  }
}

async function resumeMobileSequence(): Promise<void> {
  const route = sequenceRoute.value;
  const job = sequenceJob.value;
  if (!route || !job) return;
  sequenceError.value = "";
  try {
    await apiFetchTo(route.target, `/api/chain-jobs/${encodeURIComponent(job.id)}/resume`, {
      method: "POST",
    });
    const host = hosts.value.find((candidate) => candidate.id === route.hostId);
    if (host) persistSequenceRecovery(host, job.id);
    watchSequenceJob(route.hostId, route.target, job.id, {
      model: job.model,
      stageCount: job.stage_count,
    });
  } catch (error) {
    sequenceError.value = describeTransportError(error, sequenceHostName(route.hostId));
  }
}

function dismissMobileSequence(): void {
  stopSequenceTransport();
  clearSequenceRecovery();
  sequenceRoute.value = null;
  sequenceJob.value = null;
  sequenceProgress.value = null;
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
  if (host?.connected === false) {
    sequenceError.value = `Reconnect ${host.name} in Machines to resume this saved sequence.`;
    return;
  }
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

/**
 * Output switching. A non-chain-capable pick is remembered and swapped for
 * the first capable model; switching back restores it. Clips are PARKED in
 * both directions — the draft store never erases them.
 */
function setOutputMode(mode: string | number): void {
  const next: OutputMode = mode === "sequence" ? "sequence" : "single";
  if (next === draft.output) return;
  if (next === "sequence") {
    const current = selectedGenerationModel.value;
    if (!current || !sequenceModels.value.some((entry) => entry.name === current.name)) {
      draft.lastSingleModel = form.model || null;
      const pick = sequenceModels.value[0];
      if (pick) applyModelDefaults(form, pick);
    }
  } else if (draft.lastSingleModel) {
    const restored = generationModels.value.find((entry) => entry.name === draft.lastSingleModel);
    if (restored) applyModelDefaults(form, restored);
    draft.lastSingleModel = null;
  }
  draft.setOutput(
    next,
    { getPrompt: () => form.prompt, setPrompt: (value) => (form.prompt = value) },
    sequenceDefaultFrames.value,
  );
}

let chainLimitsFetch = 0;
async function loadChainLimits(): Promise<void> {
  const version = ++chainLimitsFetch;
  const host = generationTargetHost.value;
  const entry = selectedGenerationModel.value;
  if (!host || !entry) {
    chainLimits.value = null;
    return;
  }
  try {
    const limits = await apiJsonTo<ChainLimits>(
      mobileHostTarget(host),
      `/api/capabilities/chain-limits?model=${encodeURIComponent(entry.name)}&fps=${encodeURIComponent(Math.max(1, Math.floor(form.fps)))}`,
    );
    if (version !== chainLimitsFetch) return;
    chainLimits.value = limits;
    if (!limits.supports_audio) draft.enableAudio = false;
    if (!draft.editing) {
      const frames = defaultClipFrames(entry, limits, sequenceMotionTail.value);
      draft.adoptSequenceModel(entry.name, frames);
    }
  } catch {
    if (version === chainLimitsFetch) chainLimits.value = null;
  }
}

// Chain limits are per model AND per host — refetch when either moves, and
// keep the two-clip floor stocked once Sequence is the active output.
watch(
  () => form.model,
  (next, previous) => {
    if (isSequence.value && previous && next !== previous) {
      if (draft.editing) draft.stopEditing();
    }
  },
);
watch(
  [isSequence, () => form.model, () => form.fps, selectedHostId],
  () => {
    if (!isSequence.value) return;
    draft.ensureClips(sequenceDefaultFrames.value);
    if (form.model) void loadChainLimits();
  },
  { immediate: true },
);

function changeModel(): void {
  const model = generationModels.value.find((entry) => entry.name === form.model);
  if (model) applyModelDefaults(form, model);
}

function clearHostScopedGenerationSelections(): void {
  form.loras = [];
  form.upscaleModel = "";
  form.controlModel = "";
  form.cameraControl = null;
  form.icLoraControl = null;
  if (form.sourceFit.mode === "upscale-then-fit") {
    form.sourceFit = { ...form.sourceFit, upscalerModel: "" };
  }
}

function appendPromptWord(word: string): void {
  const trimmed = word.trim();
  if (!trimmed) return;
  onPromptAuthored(form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed);
}

function onPromptAuthored(prompt: string): void {
  applyAuthoredPrompt(form, prompt, quickExpansionSnapshot.value !== null);
}

let templateLoadEpoch = 0;
async function loadTemplate(template: GenerationTemplate): Promise<void> {
  const epoch = ++templateLoadEpoch;
  const hydrated = await hydrateGenerationTemplate(template);
  if (epoch !== templateLoadEpoch) return;
  // A pre-#787 template lacking `negativePromptDefault` is normalized first
  // so its empty negative reads as "untouched", not the explicit "" opt-out.
  Object.assign(form, normalizeLegacyNegativeSnapshot(hydrated.form, generationModels.value));
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
  const mediaMessage = hydrated.missingMediaReferences.length
    ? ` Re-add ${formatTemplateMediaReferences(hydrated.missingMediaReferences)}.`
    : "";
  generationAnnouncement.value = sameHost
    ? `Template loaded.${mediaMessage}`
    : `Template loaded. Re-add host-specific LoRAs and auxiliary models.${mediaMessage}`;
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
    selectedHostPolicy: selectedHostId.value || null,
  };
}

function mobileRemixRecoveryPayload(
  requestedSource: RemixSourceKind = remixSource.value,
): MobileRemixRecoveryPayload {
  const request = buildRequest(form);
  const source = promptSource(form.prompt, form.originalPrompt, requestedSource);
  return {
    sourcePrompt: source.prompt,
    ...(source.rootPrompt ? { rootPrompt: source.rootPrompt } : {}),
    sourceKind: source.kind,
    dimensions: [...remixDimensions.value],
    conditioningFingerprint: conditioningFingerprint(request),
  };
}

function remixInputs(sourceKind: RemixSourceKind = remixSource.value): {
  prepared: PreparedExpansionInputs;
  remix: MobileRemixRecoveryPayload;
  visiblePrompt: string;
} {
  const remix = mobileRemixRecoveryPayload(sourceKind);
  return {
    prepared: {
      kind: "remix",
      sourcePrompt: remix.sourcePrompt,
      ...(remix.rootPrompt ? { rootPrompt: remix.rootPrompt } : {}),
      sourceKind: remix.sourceKind,
      dimensions: [...remix.dimensions],
      conditioningFingerprint: remix.conditioningFingerprint,
      model: form.model,
      family: form.family,
      task: currentExpansionTask.value,
      requestedCount: DEFAULT_REMIX_VARIATIONS,
      stylePreset: form.stylePreset || null,
      selectedHostPolicy: selectedHostId.value || null,
    },
    remix,
    visiblePrompt: form.prompt,
  };
}

function sameFrozenHost(route: HostRoute, host: MobileHost | undefined): boolean {
  return mobileHostMatchesRoute(route, host);
}

/**
 * The host prompt expansion runs on (shared policy, issue #1162 §5).
 *
 * iPhone pins exactly ONE machine, so the candidate list is that machine and
 * the answer is always the generation route — including when it lacks the
 * expander, where the existing 422 → pull recovery still owns the outcome.
 * The call is shaped so mobile Auto / Most capable (#1163) can hand a real
 * candidate list and ranker here without moving the policy.
 */
function expansionRouteFor(route: HostRoute): HostRoute {
  const capability = expandCapabilities[route.hostId];
  const decision = resolveExpansionRoute(
    { kind: "pinned", hostId: route.hostId },
    { hostId: route.hostId },
    [
      {
        hostId: route.hostId,
        ready: true,
        ...(capability
          ? { modelPresent: capability.model_present, configured: capability.configured }
          : {}),
      },
    ],
    () => null,
  );
  if (decision.kind !== "reroute") return route;
  const host = hosts.value.find((candidate) => candidate.id === decision.hostId);
  return host ? routeForMobileHost(host) : route;
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
  remix: MobileRemixRecoveryPayload | null = null,
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
      remix,
    });
  }
  return missingModel
    ? `The expansion model ${missingModel} isn't installed on ${route.label}.`
    : `Expansion failed on ${route.label}: ${describeTransportError(error, route.label)}`;
}

function recoveryStaleReason(recovery: MobileExpansionRecoveryRecord): string | null {
  const currentRemix = recovery.remix
    ? mobileRemixRecoveryPayload(recovery.remix.sourceKind)
    : null;
  return mobileExpansionRecoveryStaleReason(recovery, {
    inputs: expansionInputs(recovery.inputs.requestedCount),
    currentHost: hosts.value.find((host) => host.id === recovery.route.hostId),
    tokenCurrent:
      !unmounted &&
      expansionRecovery.value?.id === recovery.id &&
      preparationGuard.isCurrent(recovery.requestToken),
    remix: currentRemix,
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
    remixUndo.value = null;
    appliedRemix.value = null;
    quickExpansionOriginal.value = inputs.sourcePrompt;
    form.prompt = prompts[0]!;
    form.originalPrompt = inputs.sourcePrompt;
    quickExpansionSnapshot.value = {
      requestToken,
      originalPrompt: inputs.sourcePrompt,
      expandedPrompt: prompts[0]!,
      model: inputs.model,
      family: inputs.family,
      task: inputs.task,
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
  remixUndo.value = null;
  appliedRemix.value = null;
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
  const expandOn = expansionRouteFor(route);

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
        task: inputs.task,
        ...(styleDirective ? { style: styleDirective } : {}),
      },
      expandOn.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const prompts = validateExpandedPrompts(response.expanded, count);
    const currentHost = hosts.value.find((candidate) => candidate.id === route.hostId);
    const current = expansionInputs(count);
    if (
      current.sourcePrompt !== inputs.sourcePrompt ||
      current.model !== inputs.model ||
      current.family !== inputs.family ||
      current.task !== inputs.task ||
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

function commitRemixReview(
  prepared: PreparedExpansionInputs,
  remix: MobileRemixRecoveryPayload,
  visiblePrompt: string,
  route: HostRoute,
  variants: ReturnType<typeof validateRemixVariants>,
  requestToken: number,
): void {
  remixReview.value = {
    sourcePrompt: remix.sourcePrompt,
    ...(remix.rootPrompt ? { rootPrompt: remix.rootPrompt } : {}),
    sourceKind: remix.sourceKind,
    visiblePrompt,
    model: prepared.model,
    family: prepared.family,
    task: prepared.task,
    stylePreset: prepared.stylePreset,
    dimensions: [...remix.dimensions],
    conditioningFingerprint: remix.conditioningFingerprint,
    selectedHostPolicy: prepared.selectedHostPolicy,
    route: { ...route, target: { ...route.target } },
    variants: variants.map((variant, index) => ({
      id: `remix-${requestToken}-${index + 1}`,
      prompt: variant.prompt,
      dimensions: [...variant.dimensions],
      selected: false,
    })),
    requestToken,
  };
  clearExpansionRecovery(false);
}

async function remixCurrent(
  routeOverride: HostRoute | null = null,
  replacePrepared = false,
): Promise<void> {
  const { prepared, remix, visiblePrompt } = remixInputs();
  const route = routeOverride ?? selectedRoute.value;
  const host = route ? hosts.value.find((candidate) => candidate.id === route.hostId) : undefined;
  if (
    !prepared.sourcePrompt ||
    !prepared.model ||
    !route ||
    !host ||
    remix.dimensions.length === 0 ||
    expansionRunning.value ||
    (preparedBatch.value && !replacePrepared)
  ) {
    return;
  }
  if (!sameFrozenHost(route, host)) {
    expansionError.value = `${route.label} isn't reachable with the frozen connection. Remix will not fall back.`;
    return;
  }
  if (expandCapabilities[route.hostId]?.configured === false) {
    expansionError.value = `Prompt tools aren't configured on ${route.label}. Configure that host before retrying.`;
    clearExpansionRecovery();
    return;
  }

  clearExpansionRecovery();
  submissionGuard.invalidate();
  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = "";
  try {
    const styleDirective = styleHint(prepared.stylePreset ?? "");
    const response = await remixPrompt(
      {
        source_prompt: remix.sourcePrompt,
        ...(remix.rootPrompt ? { root_prompt: remix.rootPrompt } : {}),
        source_kind: remix.sourceKind,
        model_family: prepared.family,
        variations: DEFAULT_REMIX_VARIATIONS,
        task: prepared.task,
        ...(styleDirective ? { style: styleDirective } : {}),
        dimensions: [...remix.dimensions],
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    if (
      response.source_prompt !== remix.sourcePrompt ||
      response.source_kind !== remix.sourceKind ||
      (response.root_prompt ?? undefined) !== remix.rootPrompt
    ) {
      throw new Error("The host returned Remix provenance for a different source prompt.");
    }
    const variants = validateRemixVariants(response.variants, DEFAULT_REMIX_VARIATIONS);
    const current = remixInputs(remix.sourceKind);
    const currentHost = hosts.value.find((candidate) => candidate.id === route.hostId);
    if (
      JSON.stringify(current.prepared) !== JSON.stringify(prepared) ||
      JSON.stringify(current.remix) !== JSON.stringify(remix) ||
      !sameFrozenHost(route, currentHost)
    ) {
      expansionError.value =
        "The prompt, model, conditioning, dimensions, style, or host changed while Remix was running. Remix again with the current inputs.";
      return;
    }
    commitRemixReview(prepared, remix, visiblePrompt, route, variants, token);
    if (replacePrepared) preparedBatch.value = null;
  } catch (error) {
    if (!preparationGuard.isCurrent(token)) return;
    expansionError.value = setExpansionFailure(
      error,
      prepared,
      route,
      token,
      replacePrepared,
      remix,
    )
      .replaceAll("Expansion", "Remix")
      .replaceAll("expansion", "remix");
  } finally {
    if (!unmounted && preparationGuard.isCurrent(token)) expansionRunning.value = false;
  }
}

function replacePreparedPrompts(useFrozenRoute: boolean): void {
  const batch = preparedBatch.value;
  if (!batch) return;
  if (batch.kind !== "remix") {
    void expandForCurrentBatch(true, useFrozenRoute ? batch.route : null);
    return;
  }
  remixSource.value = batch.sourceKind ?? "current";
  remixDimensions.value = [
    ...(batch.dimensions ?? defaultRemixDimensions(batch.task, Boolean(batch.stylePreset))),
  ];
  void remixCurrent(useFrozenRoute ? batch.route : null, true);
}

function toggleRemixVariant(id: string): void {
  const variant = remixReview.value?.variants.find((candidate) => candidate.id === id);
  if (variant && remixStaleReasons.value.length === 0 && !expansionRunning.value) {
    variant.selected = !variant.selected;
  }
}

function editRemixVariant(payload: { id: string; prompt: string }): void {
  const variant = remixReview.value?.variants.find((candidate) => candidate.id === payload.id);
  if (variant && remixStaleReasons.value.length === 0 && !expansionRunning.value) {
    variant.prompt = payload.prompt;
  }
}

function rememberRemixUndo(): void {
  remixUndo.value = {
    prompt: form.prompt,
    originalPrompt: form.originalPrompt,
    stylePreset: form.stylePreset,
  };
}

function applyRemixSelection(): void {
  const review = remixReview.value;
  if (!review || remixStaleReasons.value.length > 0) return;
  const selected = review.variants.filter((variant) => variant.selected && variant.prompt.trim());
  if (selected.length === 0) return;
  if (selected.length === 1) {
    rememberRemixUndo();
    preparationGuard.invalidate();
    submissionGuard.invalidate();
    form.prompt = selected[0]!.prompt.trim();
    form.originalPrompt = review.rootPrompt ?? review.sourcePrompt;
    bakeStyleNegative(review.stylePreset ?? "", review.family);
    form.stylePreset = "";
    quickExpansionOriginal.value = null;
    quickExpansionSnapshot.value = {
      requestToken: review.requestToken,
      originalPrompt: review.sourcePrompt,
      expandedPrompt: selected[0]!.prompt.trim(),
      model: review.model,
      family: review.family,
      task: review.task,
      stylePreset: review.stylePreset,
      selectedHostPolicy: review.selectedHostPolicy,
      route: { ...review.route, target: { ...review.route.target } },
    };
    appliedRemix.value = {
      prompt: selected[0]!.prompt.trim(),
      ...(review.rootPrompt ? { rootPrompt: review.rootPrompt } : {}),
      sourcePrompt: review.sourcePrompt,
      sourceKind: review.sourceKind,
      task: review.task,
      dimensions: [...selected[0]!.dimensions],
    };
    remixReview.value = null;
    expansionError.value = "";
    clearExpansionRecovery();
    return;
  }
  const inputs: PreparedExpansionInputs = {
    kind: "remix",
    sourcePrompt: review.sourcePrompt,
    ...(review.rootPrompt ? { rootPrompt: review.rootPrompt } : {}),
    sourceKind: review.sourceKind,
    dimensions: [...review.dimensions],
    conditioningFingerprint: review.conditioningFingerprint,
    model: review.model,
    family: review.family,
    task: review.task,
    requestedCount: selected.length,
    stylePreset: review.stylePreset,
    selectedHostPolicy: review.selectedHostPolicy,
  };
  preparedBatch.value = createPreparedExpansionBatch(
    inputs,
    review.route,
    selected.map((variant) => variant.prompt.trim()),
    review.requestToken,
  );
  Object.assign(preparedBatch.value, {
    remixVariantDimensions: selected.map((variant) => [...variant.dimensions]),
  });
  remixUndo.value = null;
  appliedRemix.value = null;
  quickExpansionOriginal.value = null;
  quickExpansionSnapshot.value = null;
  remixReview.value = null;
  expansionError.value = "";
}

function restoreRemixSource(): void {
  const review = remixReview.value;
  if (!review) return;
  rememberRemixUndo();
  form.prompt = review.sourcePrompt;
  form.originalPrompt = review.sourceKind === "original" ? null : (review.rootPrompt ?? null);
  appliedRemix.value = null;
  quickExpansionSnapshot.value = null;
  remixReview.value = null;
  preparationGuard.invalidate();
  clearExpansionRecovery();
  expansionError.value = "";
}

function discardRemixReview(): void {
  preparationGuard.invalidate();
  remixReview.value = null;
  clearExpansionRecovery();
  expansionError.value = "";
}

function undoPromptPreparation(): void {
  const snapshot = remixUndo.value;
  if (!snapshot) {
    restoreQuickExpansion();
    return;
  }
  submissionGuard.invalidate();
  preparationGuard.invalidate();
  form.prompt = snapshot.prompt;
  form.originalPrompt = snapshot.originalPrompt;
  form.stylePreset = snapshot.stylePreset;
  remixUndo.value = null;
  appliedRemix.value = null;
  quickExpansionSnapshot.value = null;
  remixReview.value = null;
  expansionError.value = "";
  clearExpansionRecovery();
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
  generationSubmissionPhase.value = null;
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

async function recoverQuickPromptTransform(): Promise<void> {
  if (!appliedRemix.value) {
    await reexpandAndDevelop();
    return;
  }
  undoPromptPreparation();
  await nextTick();
  await remixCurrent();
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
  generationSubmissionPhase.value = null;
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
  generationSubmissionPhase.value = null;
  expansionError.value = "";
  clearExpansionRecovery();
  if (restoreFocus) {
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  }
}

function preparedRemixDimensions(
  batch: PreparedExpansionBatchState,
  index: number,
): RemixDimension[] {
  const perVariant = (
    batch as PreparedExpansionBatchState & {
      remixVariantDimensions?: readonly (readonly RemixDimension[])[];
    }
  ).remixVariantDimensions;
  return [...(perVariant?.[index] ?? batch.dimensions ?? [])];
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
    const response = recovery.remix
      ? await remixPrompt(
          {
            source_prompt: recovery.remix.sourcePrompt,
            ...(recovery.remix.rootPrompt ? { root_prompt: recovery.remix.rootPrompt } : {}),
            source_kind: recovery.remix.sourceKind,
            model_family: recovery.inputs.family,
            variations: DEFAULT_REMIX_VARIATIONS,
            task: recovery.inputs.task,
            ...(styleDirective ? { style: styleDirective } : {}),
            dimensions: [...recovery.remix.dimensions],
          },
          recovery.route.target,
        )
      : await expandPrompt(
          recovery.inputs.sourcePrompt,
          {
            variations: recovery.inputs.requestedCount,
            ...(recovery.inputs.family ? { modelFamily: recovery.inputs.family } : {}),
            task: recovery.inputs.task,
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
    if (recovery.remix) {
      if (!("variants" in response))
        throw new Error("The host returned an invalid Remix response.");
      const variants = validateRemixVariants(response.variants, DEFAULT_REMIX_VARIATIONS);
      commitRemixReview(
        { ...recovery.inputs },
        {
          ...recovery.remix,
          dimensions: [...recovery.remix.dimensions],
        },
        form.prompt,
        { ...recovery.route, target: { ...recovery.route.target } },
        variants,
        recovery.requestToken,
      );
      if (recovery.replacePrepared) preparedBatch.value = null;
    } else {
      if (!("expanded" in response))
        throw new Error("The host returned an invalid expansion response.");
      const prompts = validateExpandedPrompts(response.expanded, recovery.inputs.requestedCount);
      commitExpandedPrompts(
        { ...recovery.inputs },
        { ...recovery.route, target: { ...recovery.route.target } },
        prompts,
        recovery.requestToken,
        recovery.replacePrepared,
        focus,
      );
    }
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
  if (!url.startsWith("blob:")) return;
  URL.revokeObjectURL(url);
  objectUrls.delete(url);
}

const sourceFitCache = new SourceFitPreprocessCache();

async function prepareGenerationRequest(
  target: ApiTarget,
  draft: GenerateForm,
  isCurrent: () => boolean = () => true,
  signal?: AbortSignal,
) {
  // Fixed recipe controls are not user choices: a stale draft value (restored
  // before the recipe landed, model swapped under it) snaps to what the
  // disabled control displays instead of queueing a shape the host refuses.
  // It runs before the source fits below so their target is the canvas that
  // actually renders. Shared with desktop and web.
  Object.assign(
    draft,
    fixedRecipeControlOverrides(
      effectiveGenerationRecipe(selectedGenerationModel.value, draft.pipeline),
    ),
  );
  const draftCaps = generationCapabilitiesForFamily(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
  );
  if (draftCaps.sourceImageMode === "h3-boundaries") {
    // H3 FL2VA boundaries take the same client-side fit, coerced maskless.
    draft.h3Authoring =
      (await applyH3BoundaryFit(
        draft.h3Authoring,
        draft.sourceFit,
        { width: draft.width, height: draft.height },
        {
          ops: domCanvasOps,
          cache: sourceFitCache,
          upscale: (image, model) =>
            upscaleImage({
              image,
              model,
              target,
              ...(signal ? { signal } : {}),
              onProgress: (message) => {
                if (isCurrent()) setGenerationStatus(message);
              },
            }),
          onStatus: (message) => {
            if (isCurrent()) setGenerationStatus(message);
          },
        },
      )) ?? emptyMinimaxH3AuthoringState();
  } else if (draftCaps.sourceImageMode === "qwen-edit" && draft.imageAttachments[0]) {
    const sourceTarget = resolveSourceConditioningTarget(
      { width: draft.width, height: draft.height },
      selectedGenerationModel.value ?? draft.family,
      draft.pipeline,
    );
    const result = await applySourceFitPreprocess(
      {
        source: draft.imageAttachments[0],
        mask: null,
        policy: coerceSourceFitForMaskless(draft.sourceFit),
        target: sourceTarget,
      },
      {
        ops: domCanvasOps,
        cache: sourceFitCache,
        upscale: (image, model) =>
          upscaleImage({
            image,
            model,
            target,
            ...(signal ? { signal } : {}),
            onProgress: (message) => {
              if (isCurrent()) setGenerationStatus(message);
            },
          }),
        onStatus: (message) => {
          if (isCurrent()) setGenerationStatus(message);
        },
      },
    );
    if (result.source) draft.imageAttachments[0] = result.source;
  } else if (
    draftCaps.supportsImg2img &&
    draftCaps.sourceImageMode === "single" &&
    draft.sourceImage
  ) {
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
            ...(signal ? { signal } : {}),
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
  return withPrintTitle(buildRequest(draft));
}

/** Stamp the Create title onto a mobile-built request (additive `title`;
 * absent when blank so the server's own default naming applies). Batch
 * siblings and prepared Batch N spread this request, so they inherit it. */
function withPrintTitle(request: GenerateRequest): GenerateRequest {
  const title = requestTitle(printTitle.value);
  if (!title.ok) throw new Error(title.reason);
  if (!title.title) return request;
  return { ...request, title: title.title } as GenerateRequest;
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
        originalPrompt: prepared.rootPrompt ?? prepared.sourcePrompt,
        promptTransforms:
          prepared.kind === "remix"
            ? prepared.prompts.map((_, index): PromptTransformProvenance => ({
                operation: "remix",
                ...(prepared.rootPrompt ? { root_prompt: prepared.rootPrompt } : {}),
                source_prompt: prepared.sourcePrompt,
                source_kind: prepared.sourceKind ?? "current",
                task: prepared.task,
                dimensions: preparedRemixDimensions(prepared, index),
              }))
            : undefined,
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
  // Prepared and quick work keeps the machine it was frozen on. An ordinary
  // submission under Auto / Most capable starts on a provisional machine — the
  // placement fan-out below replaces it before anything is queued.
  const automaticOrdinary = automaticRouting.value && !preparedSubmission && !quickSubmission;
  const provisionalHost = automaticOrdinary
    ? provisionalAutomaticHost(form.model, form.family)
    : null;
  const host = preparedSubmission
    ? hosts.value.find((candidate) => candidate.id === preparedSubmission.route.hostId)
    : quickSubmission
      ? hosts.value.find((candidate) => candidate.id === quickSubmission.route.hostId)
      : automaticOrdinary
        ? provisionalHost
        : selectedHost.value;
  const initialRoute =
    preparedSubmission?.route ??
    quickSubmission?.route ??
    (automaticOrdinary
      ? provisionalHost
        ? routeForMobileHost(provisionalHost)
        : null
      : selectedRoute.value);
  const target = initialRoute?.target ?? null;
  // Automatic routing filters restricted machines out of its candidate set, so
  // only an explicit machine is checked up front.
  const restriction =
    initialRoute && !automaticOrdinary
      ? modelAccessRestrictionFor(serverCapabilities[initialRoute.hostId], {
          model: form.model,
          family: form.family,
          generation_profile_sha256: generationProfileHashForHost(initialRoute.hostId, form.model),
        })
      : null;
  if (restriction) {
    setGenerationStatus(restriction.message, true);
    generationAnnouncement.value = `${restriction.message} Nothing was queued.`;
    return;
  }
  if (
    !host ||
    !initialRoute ||
    !target ||
    promptMissing.value ||
    !selectedModelAvailable.value ||
    !seedValid.value ||
    !parameterValid.value ||
    !sourceControlsValid.value ||
    !resolutionValid.value ||
    !basicParametersValid.value ||
    !!mobileMediaBudgetError.value ||
    !!sourceConditioningError.value ||
    !!h3AuthoringError.value ||
    preparingGeneration.value
  )
    return;

  // Replaced by the fan-out winner under Auto / Most capable; frozen from here
  // on for every other path.
  let route: HostRoute = initialRoute;
  const draft = cloneGenerateForm(form);
  const originalSource = draft.sourceImage
    ? {
        base64: draft.sourceImage,
        filename: draft.sourceImageName ?? "Source image",
        width: draft.sourceImageWidth,
        height: draft.sourceImageHeight,
        sourceFit: parseSourceFitPolicy(draft.sourceFit) ?? { mode: "pad-repaint" },
      }
    : draft.h3Authoring?.firstFrame?.data
      ? {
          base64: draft.h3Authoring.firstFrame.data,
          filename: draft.h3Authoring.firstFrame.filename,
          width: draft.h3Authoring.firstFrame.width,
          height: draft.h3Authoring.firstFrame.height,
          mime: draft.h3Authoring.firstFrame.mimeType,
          sourceFit: parseSourceFitPolicy(draft.sourceFit) ?? { mode: "pad-repaint" },
        }
      : null;
  const draftCaps = generationCapabilitiesForFamily(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
  );
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
  const submitSignal = submissionGuard.signalFor(token);
  const uiId = ++submissionUiId;
  const ownsPreparedSubmission = () =>
    !unmounted &&
    uiId === submissionUiId &&
    submissionGuard.isCurrent(token) &&
    (!preparedSubmission || preparedBatch.value?.batchId === preparedSubmission.batchId);
  const releasePreparedSubmission = () => {
    if (!unmounted && uiId === submissionUiId) {
      preparingGeneration.value = false;
      generationSubmissionPhase.value = null;
    }
    if (ownsPreparedSubmission()) preparedSubmitting.value = false;
  };
  let request: GenerateRequest;
  preparingGeneration.value = true;
  generationSubmissionPhase.value = "preparing";
  preparedSubmitting.value = !!preparedSubmission;
  try {
    request = await prepareGenerationRequest(
      target,
      draft,
      () => submissionGuard.isCurrent(token),
      submitSignal,
    );
    if (request.source_image && originalSource) {
      void persistGenerationSourceMedia(request.source_image, originalSource);
    }
    if (appliedRemix.value && appliedRemix.value.prompt === form.prompt) {
      request.prompt_transform = {
        operation: "remix",
        ...(appliedRemix.value.rootPrompt ? { root_prompt: appliedRemix.value.rootPrompt } : {}),
        source_prompt: appliedRemix.value.sourcePrompt,
        source_kind: appliedRemix.value.sourceKind,
        task: appliedRemix.value.task,
        dimensions: [...appliedRemix.value.dimensions],
      };
    }
  } catch (error) {
    if (!ownsPreparedSubmission()) {
      releasePreparedSubmission();
      return;
    }
    setGenerationStatus(describeTransportError(error, route.label), true);
    generationAnnouncement.value = `Couldn’t prepare the source image. ${progress.value}`;
    releasePreparedSubmission();
    return;
  }

  if (!submissionGuard.isCurrent(token)) {
    releasePreparedSubmission();
    return;
  }
  if (guardedSubmission && JSON.stringify(cloneGenerateForm(form)) !== liveFormIdentity) {
    setGenerationStatus("The generation inputs changed while the source was being prepared.");
    generationAnnouncement.value = `${progress.value} Review the current inputs before developing.`;
    releasePreparedSubmission();
    return;
  }
  if (
    // The provisional machine of an automatic submission is not frozen work —
    // it is replaced by the fan-out winner a few lines below.
    !automaticOrdinary &&
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
  if (!unmounted && uiId === submissionUiId) generationSubmissionPhase.value = "placement";
  const requireAuthoritativePlacement = requiresAuthoritativePlacement(
    request as unknown as Record<string, unknown>,
  );
  let placement: GenerationPlacementPreview | null = null;
  let legacyUnsupported = false;
  const previewRequest =
    chainRouting.kind === "chain"
      ? previewRequestForSiblingFanout(
          buildAutoChainRequest(request, chainRouting) as unknown as Record<string, unknown>,
          batchSize,
        )
      : previewRequestForSiblingFanout(request as unknown as Record<string, unknown>, batchSize);
  if (automaticOrdinary) {
    const routed = await routeAutomaticGeneration({
      request: previewRequest,
      chain: chainRouting.kind === "chain",
      copies: batchSize,
      model: request.model,
      family: draft.family,
      subject: "print",
      requireAuthoritative: requireAuthoritativePlacement,
      isCurrent: () => submissionGuard.isCurrent(token),
      signal: submitSignal,
    });
    if (routed.kind === "abandoned") {
      releasePreparedSubmission();
      return;
    }
    if (routed.kind === "error") {
      setGenerationStatus(routed.message, true);
      generationAnnouncement.value = routed.message;
      releasePreparedSubmission();
      return;
    }
    // The chosen machine is frozen here: every later step — submission,
    // recovery, and the connection fence — reads this exact route.
    route = routed.route;
    placement = routed.placement;
    legacyUnsupported = routed.legacyUnsupported;
  } else {
    try {
      placement =
        chainRouting.kind === "chain"
          ? await previewChainPlacement(target, previewRequest, batchSize, {
              signal: submitSignal,
            })
          : await previewGenerationPlacement(target, previewRequest, batchSize, {
              signal: submitSignal,
            });
    } catch (error) {
      if (!submissionGuard.isCurrent(token)) {
        releasePreparedSubmission();
        return;
      }
      if (error instanceof ApiError && (error.status === 404 || error.status === 405)) {
        if (requireAuthoritativePlacement) {
          setGenerationStatus(
            `${route.label} does not provide the authoritative placement preview required for reference media. Nothing was queued.`,
            true,
          );
          releasePreparedSubmission();
          return;
        }
        legacyUnsupported = true;
      } else {
        setGenerationStatus(describeTransportError(error, route.label), true);
        releasePreparedSubmission();
        return;
      }
    }
  }
  if (!submissionGuard.isCurrent(token)) {
    releasePreparedSubmission();
    return;
  }
  const classification: string = classifyPlacementPreview(placement);
  if (requireAuthoritativePlacement && classification === "unsupported") {
    setGenerationStatus(
      `${route.label} does not provide the authoritative placement preview required for reference media. Nothing was queued.`,
      true,
    );
    releasePreparedSubmission();
    return;
  }
  if (!legacyUnsupported && classification !== "unsupported" && classification !== "planned") {
    setGenerationStatus(mobilePlacementFailure(placement, route.label, "print"), true);
    releasePreparedSubmission();
    return;
  }
  if (
    !sameFrozenHost(
      route,
      hosts.value.find((candidate) => candidate.id === route.hostId),
    )
  ) {
    expansionError.value = `${route.label}'s connection details changed while checking placement.`;
    releasePreparedSubmission();
    return;
  }

  const requestOptions = preparedSubmission
    ? {
        prompts: preparedSubmission.prompts,
        originalPrompt: preparedSubmission.originalPrompt,
        ...(preparedSubmission.promptTransforms
          ? { promptTransforms: preparedSubmission.promptTransforms }
          : {}),
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
      instanceId: route.instanceId ?? null,
      referenceUploads: route.referenceUploads ?? null,
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

function cancelGenerationSubmission(): void {
  if (!preparingGeneration.value) return;
  submissionGuard.invalidate();
  submissionUiId += 1;
  preparingGeneration.value = false;
  preparedSubmitting.value = false;
  generationSubmissionPhase.value = null;
  setGenerationStatus("Cancelled before generation started");
  generationAnnouncement.value = "Generation planning cancelled. Nothing was queued.";
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
  if (resultMediaRecoveryTimer !== null) {
    clearTimeout(resultMediaRecoveryTimer);
    resultMediaRecoveryTimer = null;
  }
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
}

function recoverGeneratedMedia(): void {
  const job = latestResultJob.value;
  if (!job || job.resultUrlLoading) return;
  if (resultMediaRecoveryClientId !== job.clientId) {
    if (resultMediaRecoveryTimer !== null) {
      clearTimeout(resultMediaRecoveryTimer);
      resultMediaRecoveryTimer = null;
    }
    resultMediaRecoveryClientId = job.clientId;
    resultMediaRecoveryAttempts = 0;
  }
  if (resultMediaRecoveryTimer !== null) return;

  const retryDelays = job.result?.format === "mp4" ? GENERATED_VIDEO_RECOVERY_DELAYS_MS : [0];
  const delay = retryDelays[resultMediaRecoveryAttempts];
  if (delay !== undefined) {
    resultMediaRecoveryAttempts += 1;
    if (delay === 0) {
      renewGeneratedResult(true);
      return;
    }
    const clientId = job.clientId;
    resultMediaRecoveryTimer = setTimeout(() => {
      resultMediaRecoveryTimer = null;
      if (latestResultClientId.value === clientId) renewGeneratedResult(true);
    }, delay);
    return;
  }

  if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
  job.resultUrl = null;
  job.resultUrlIsObjectUrl = false;
  job.resultUrlExpiresAt = null;
  job.resultError = "Couldn’t load this generated print from the host.";
}

function retryGeneratedPreview(): void {
  if (resultMediaRecoveryTimer !== null) {
    clearTimeout(resultMediaRecoveryTimer);
    resultMediaRecoveryTimer = null;
  }
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
  renewGeneratedResult(true);
}

async function thumbnailUrl(target: ApiTarget, hostId: string, filename: string): Promise<string> {
  let blob = await loadCachedGalleryMedia(hostId, filename, "thumbnail");
  if (!blob) {
    const response = await apiFetchTo(target, galleryMediaPath(filename, "host", true));
    blob = await response.blob();
    void storeCachedGalleryMedia(hostId, filename, "thumbnail", blob);
  }
  // WKWebView's native image context menu forwards an object URL as text to
  // Share extensions. Give iOS an inline image resource so Share, Copy, and
  // Save to Photos receive image data rather than a process-local `blob:` URL.
  // Thumbnails are bounded to 256 px by the server, so the base64 expansion
  // stays small; browser development keeps cheaper revocable object URLs.
  if (isNativeIOSRuntime()) {
    const mimeType = blob.type.startsWith("image/") ? blob.type : "image/png";
    return `data:${mimeType};base64,${await blobToBase64(blob)}`;
  }
  const url = URL.createObjectURL(blob);
  objectUrls.add(url);
  return url;
}

function mobileGalleryCacheKey(host: MobileHost): string {
  return host.instanceId?.trim() || host.id;
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
  clearGallerySelection();
  galleryLoading.value = true;
  galleryError.value = "";
  const prior = gallery.value;
  gallery.value = [];
  for (const item of prior) revokeObjectUrl(item.thumbnailUrl);
  const allHosts = connectedHosts.value;
  const cachedResults = await Promise.all(
    allHosts.map(async (host) => ({
      host,
      cacheKey: mobileGalleryCacheKey(host),
      prints: await loadCachedGallery(mobileGalleryCacheKey(host)),
    })),
  );
  const cachedByHost = new Map(
    cachedResults.map(({ host, cacheKey, prints }) => [host.id, { cacheKey, prints }]),
  );
  const cachedCopies = cachedResults.flatMap(({ host, cacheKey, prints }) => {
    const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
    return prints.map((print) => ({
      ...print,
      hostId: host.id,
      cacheKey,
      hostName: host.name,
      target,
    }));
  });
  if (cachedCopies.length > 0) {
    galleryCopies = cachedCopies.sort((a, b) => b.timestamp - a.timestamp);
    rebuildGalleryOrganization();
    pendingGallery = visibleRepresentatives();
    await loadMoreGalleryPage();
  }

  const galleryHosts = allHosts.filter(
    (host) => host.online || !knownHostReachability.has(host.id),
  );
  const knownOffline = allHosts.length - galleryHosts.length;
  const results = await Promise.allSettled(
    galleryHosts.map(async (host) => {
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
      const cacheKey = mobileGalleryCacheKey(host);
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), GALLERY_HOST_TIMEOUT_MS);
      let prints: MobileGalleryImage[];
      let capabilities: ServerCapabilities | undefined;
      // Capabilities ride along so the Library can gate organization and the
      // trash per host; a refusal keeps whatever this host last advertised.
      // BOTH reads settle under the shared deadline: a stalled capabilities
      // endpoint must never leave this host's promise pending forever (the
      // outer allSettled would wait on it and keep the whole Library
      // loading). The abort race guarantees settlement at the deadline even
      // if a transport ignores the signal.
      const capabilitiesRead = Promise.race<ServerCapabilities | undefined>([
        apiJsonTo<ServerCapabilities>(target, "/api/capabilities", {
          signal: controller.signal,
        }).catch(() => undefined),
        new Promise<undefined>((resolve) =>
          controller.signal.addEventListener("abort", () => resolve(undefined), { once: true }),
        ),
      ]);
      try {
        [prints, capabilities] = await Promise.all([
          apiJsonTo<MobileGalleryImage[]>(target, "/api/gallery", {
            signal: controller.signal,
          }),
          capabilitiesRead,
        ]);
      } finally {
        clearTimeout(timeout);
      }
      const currentHost = hosts.value.find((candidate) => candidate.id === host.id);
      if (!currentHost || mobileGalleryCacheKey(currentHost) !== cacheKey) {
        throw new Error("Gallery host identity changed while refreshing");
      }
      if (capabilities !== undefined) serverCapabilities[host.id] = capabilities;
      await storeCachedGallery(cacheKey, prints);
      return { host, cacheKey, target, prints };
    }),
  );
  // Collections and tags for every host that can organize (read after the
  // capabilities above so a just-upgraded host is included this pass).
  await refreshHostOrganization(
    results.flatMap((result) => (result.status === "fulfilled" ? [result.value.host.id] : [])),
  );
  const refreshedByHost = new Map(
    results.flatMap((result) =>
      result.status === "fulfilled" ? [[result.value.host.id, result.value] as const] : [],
    ),
  );
  const refreshedCopies = connectedHosts.value
    .flatMap((host) => {
      const refreshed = refreshedByHost.get(host.id);
      const cacheKey = refreshed?.cacheKey ?? mobileGalleryCacheKey(host);
      const target = refreshed?.target ?? { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
      const cached = cachedByHost.get(host.id);
      const prints = refreshed?.prints ?? (cached?.cacheKey === cacheKey ? cached.prints : []);
      return prints.map((print) => ({
        ...print,
        hostId: host.id,
        cacheKey,
        hostName: host.name,
        target,
      }));
    })
    .sort((a, b) => b.timestamp - a.timestamp);
  const failed = knownOffline + results.filter((result) => result.status === "rejected").length;
  if (selectedPrint.value) {
    galleryRefreshDeferred = true;
    if (failed) {
      galleryError.value = `${failed} host${failed === 1 ? "" : "s"} unavailable${
        cachedCopies.length > 0 ? " · Showing saved Library" : ""
      }`;
    }
    galleryLoading.value = false;
    return;
  }
  galleryCopies = refreshedCopies;
  for (const item of gallery.value) revokeObjectUrl(item.thumbnailUrl);
  gallery.value = [];
  rebuildGalleryOrganization();
  // The Trash listing is refetched on its own schedule; a live refresh only
  // marks it stale so the next Trash visit re-reads the host.
  if (libraryScope.value !== "trash") trashLoaded.value = false;
  pendingGallery = visibleRepresentatives();
  if (failed) {
    galleryError.value = `${failed} host${failed === 1 ? "" : "s"} unavailable${
      cachedCopies.length > 0 ? " · Showing saved Library" : ""
    }`;
  }
  await loadMoreGalleryPage();
  void pruneCachedGalleryMedia();
  galleryLoading.value = false;
  if (libraryScope.value === "trash") void refreshTrash();
}

function loadMoreGallery(): Promise<void> {
  if (
    galleryLoadMoreQueued ||
    galleryLoading.value ||
    galleryLoadingMore.value ||
    galleryRemaining.value === 0
  ) {
    return Promise.resolve();
  }
  galleryLoadMoreQueued = true;
  return enqueueGalleryOperation(async () => {
    // The serialized operation is now running; `galleryLoadingMore` guards
    // observer callbacks until the page settles.
    galleryLoadMoreQueued = false;
    await loadMoreGalleryPage();
  });
}

async function loadMoreGalleryPage(): Promise<void> {
  galleryLoadingMore.value = true;
  const page = pendingGallery.splice(0, 40);
  for (let offset = 0; offset < page.length; offset += 4) {
    const batch = await Promise.allSettled(
      page.slice(offset, offset + 4).map(async ({ target, ...print }) => ({
        ...print,
        target,
        thumbnailUrl: await thumbnailUrl(target, print.cacheKey, print.filename),
      })),
    );
    gallery.value.push(
      ...batch.flatMap((result) => (result.status === "fulfilled" ? [result.value] : [])),
    );
  }
  markMobileLibrarySeen(galleryCopies);
  galleryRemaining.value = pendingGallery.length;
  galleryLoadingMore.value = false;
}

async function reusePrint(print: GalleryPrint): Promise<void> {
  if (reusingPrint.value || print.metadata_synthetic) return;
  const epoch = ++reusePrintEpoch;
  reusePrintController?.abort();
  const controller = new AbortController();
  reusePrintController = controller;
  reusingPrint.value = true;
  reusePrintError.value = "";
  try {
    if (selectedHostId.value !== print.hostId) {
      selectedHostId.value = print.hostId;
    }
    const printHost = hosts.value.find((candidate) => candidate.id === print.hostId);
    const inMemoryIdentity = modelSnapshotIdentities.value[print.hostId];
    const inMemoryModels =
      inMemoryIdentity?.cacheKey === print.cacheKey &&
      (!printHost?.version ||
        !inMemoryIdentity.serverVersion ||
        inMemoryIdentity.serverVersion === printHost.version)
        ? modelsByHost.value[print.hostId]
        : undefined;
    let reuseModels =
      inMemoryModels?.filter((model) => model.downloaded && isGenerationModel(model)) ?? [];
    let usedCachedPresentationAt: number | null =
      inMemoryModels && printHost?.online !== true ? (inMemoryIdentity?.updatedAt ?? null) : null;
    const cachedPresentation = await loadCachedHostPresentation(print.cacheKey);
    if (cancelledReuse(epoch, controller)) return;
    const cacheMatchesHost =
      cachedPresentation !== null &&
      (!cachedPresentation.instanceId || cachedPresentation.instanceId === print.cacheKey) &&
      (!printHost?.version ||
        !cachedPresentation.serverVersion ||
        cachedPresentation.serverVersion === printHost.version);
    const validCachedPresentation = cacheMatchesHost ? cachedPresentation : null;
    if (reuseModels.length === 0 && validCachedPresentation) {
      reuseModels = validCachedPresentation.models.filter(
        (model) => model.downloaded && isGenerationModel(model),
      );
      if (reuseModels.length > 0) usedCachedPresentationAt = validCachedPresentation.updatedAt;
    }
    if (reuseModels.length === 0) {
      const presentation = await readReusePresentation(print, controller.signal);
      if (cancelledReuse(epoch, controller)) return;
      reuseModels = presentation.models.filter(
        (model) => model.downloaded && isGenerationModel(model),
      );
    } else {
      // Saved state is authoritative for this interaction. Refresh it for the
      // next one without holding the viewer or mutating this reuse attempt.
      void readReusePresentation(print, new AbortController().signal).catch(() => undefined);
    }
    if (reuseModels.length === 0) {
      reusePrintError.value = `${print.hostName} has no downloaded models available.`;
      return;
    }
    const reuse = applyMobileGalleryMetadata(form, print.metadata, reuseModels);
    if (reuse.sequenceUnsupportedReason) {
      reusePrintError.value = reuse.sequenceUnsupportedReason;
      return;
    }
    // The print's own saved title comes back with its settings; the Library
    // title (a later rename) wins over the metadata stamp when it exists.
    printTitle.value = organizationOf(print)?.title ?? reuse.title;
    const sourceRestoreNotice = !reuse.sequence
      ? await restoreOrdinaryReusedSource(
          print,
          controller.signal,
          () => !cancelledReuse(epoch, controller),
        )
      : null;
    if (cancelledReuse(epoch, controller)) return;
    if (reuse.sequence) {
      // A sequence print reloads the clip rail as a NEW draft: no edit
      // session, nothing cached (iPhone has no chain-detail recovery route,
      // so Edit sequence stays a desktop/web action for now).
      draft.stopEditing();
      draft.output = "sequence";
      draft.clips.splice(0, draft.clips.length, ...reuse.sequence.clips);
      draft.activeClipId = reuse.sequence.clips[0]?.id ?? null;
      draft.enableAudio = print.metadata.enable_audio === true;
      draft.bindSequenceModel(form.model);
    } else {
      // Reuse follows the mode the print was authored in. In particular, a
      // One shot may carry internal chain provenance after automatic long-
      // video routing; that must never strand iPhone in its persisted
      // Sequence mode.
      draft.setOutput(
        "single",
        {
          getPrompt: () => form.prompt,
          setPrompt: (prompt) => (form.prompt = prompt),
        },
        25,
      );
      draft.stopEditing();
      draft.lastSingleModel = null;
    }
    const notes: string[] = [];
    if (usedCachedPresentationAt !== null) {
      const ageMinutes = Math.max(0, Math.floor((Date.now() - usedCachedPresentationAt) / 60_000));
      notes.push(
        ageMinutes < 1
          ? "Saved model information was used and is refreshing in the background."
          : `Model information saved ${ageMinutes} min ago was used and is refreshing in the background.`,
      );
    }
    if (reuse.substitutedModel) {
      notes.push(
        `The original model isn’t installed on ${print.hostName}; using ${reuse.modelName}.`,
      );
    }
    if (reuse.sequence) {
      notes.push(sequenceReuseNote(reuse.sequence.clips.length, reuse.sequence.lossy));
      if (reuse.sequence.raised > 0) {
        notes.push(sequenceReuseClampNote(modelDisplayNameForId(form.model, reuseModels)));
      }
    } else if (notes.length === 0) {
      notes.push("Prompt settings restored");
    }
    if (sourceRestoreNotice) notes.push(sourceRestoreNotice);
    // A first/last-frame print restores every knob except its closing still:
    // saved metadata records each keyframe's name and digest, never the bytes
    // (`applyMetadataToForm` already cleared `form.endFrame`). Say so, the same
    // way an unrestorable source video is reported, rather than letting Develop
    // look ready to reproduce the render.
    const endFrameNotice = firstLastFrameRestoreNotice(
      caps.value.supportsEndFrame,
      print.metadata.keyframes,
      // A first/last print carries its opening frame only in keyframes[0];
      // without source provenance both endpoints need reattaching.
      Boolean(print.metadata.source_image_sha256 ?? print.metadata.source_image_name),
    );
    if (endFrameNotice) notes.push(endFrameNotice);
    setGenerationStatus(notes.join(" · "), !!endFrameNotice || !!sourceRestoreNotice);
    // FL2VA reuse leaves bytes-less boundary descriptors; when the original
    // was a gallery image its bytes are still on the print's host — fetch
    // them so the wells fill instead of demanding a reattach.
    void restoreReusedH3BoundaryMedia(print);
    selectedPrint.value = null;
    // The next Gallery visit performs its normal refresh; do not refetch the
    // grid while navigating directly to the restored prompt.
    galleryRefreshDeferred = false;
    tab.value = "generate";
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  } catch (error) {
    if (!cancelledReuse(epoch, controller)) {
      reusePrintError.value =
        error instanceof Error && error.message.includes("timed out")
          ? error.message
          : `Couldn’t load models from ${print.hostName}. ${describeTransportError(error, print.hostName)}`;
    }
  } finally {
    if (epoch === reusePrintEpoch) {
      reusingPrint.value = false;
      reusePrintController = null;
    }
  }
}

function cancelledReuse(epoch: number, controller: AbortController): boolean {
  return unmounted || controller.signal.aborted || epoch !== reusePrintEpoch;
}

async function readReusePresentation(
  print: GalleryPrint,
  signal: AbortSignal,
): Promise<CachedHostPresentation> {
  const host = hosts.value.find((candidate) => candidate.id === print.hostId);
  if (!host) throw new Error("This machine is no longer configured.");
  const target = mobileHostTarget(host);
  const cacheFence = captureCachedHostFence(print.cacheKey);
  const timeoutController = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let rejectCancellation: ((reason: Error) => void) | null = null;
  const cancelled = new Promise<never>((_resolve, reject) => {
    rejectCancellation = reject;
  });
  const abort = () => {
    timeoutController.abort();
    rejectCancellation?.(new DOMException("Reuse settings was cancelled", "AbortError"));
  };
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) abort();
  timeout = setTimeout(() => {
    timeoutController.abort();
    rejectCancellation?.(
      new Error(`Loading saved settings from ${print.hostName} timed out. Try again.`),
    );
  }, REUSE_PRESENTATION_TIMEOUT_MS);
  try {
    const [status, entries, capabilities] = await Promise.race([
      Promise.all([
        apiJsonTo<ServerStatus>(target, "/api/status", { signal: timeoutController.signal }),
        apiJsonTo<ModelEntry[]>(target, "/api/models", { signal: timeoutController.signal }),
        apiJsonTo<ServerCapabilities>(target, "/api/capabilities", {
          signal: timeoutController.signal,
        }).catch(() => null),
      ]),
      cancelled,
    ]);
    const instanceId = status.instance_id?.trim() || null;
    const expectedInstanceId =
      print.cacheKey !== print.hostId ? print.cacheKey : host.instanceId?.trim();
    if (expectedInstanceId && instanceId !== expectedInstanceId) {
      throw new Error("This machine now reports a different server identity.");
    }
    const currentHost = hosts.value.find((candidate) => candidate.id === print.hostId);
    if (!currentHost || mobileGalleryCacheKey(currentHost) !== print.cacheKey) {
      throw new Error("This machine changed while its saved settings were refreshing.");
    }
    const presentation: CachedHostPresentation = {
      hostId: print.cacheKey,
      updatedAt: Date.now(),
      instanceId,
      serverVersion: status.version ?? null,
      models: filterRestrictedModels(entries, capabilities),
      capabilities,
    };
    await storeCachedHostPresentation(presentation, cacheFence);
    return presentation;
  } finally {
    if (timeout !== null) clearTimeout(timeout);
    signal.removeEventListener("abort", abort);
    rejectCancellation = null;
  }
}

async function restoreOrdinaryReusedSource(
  print: GalleryPrint,
  signal: AbortSignal,
  isCurrent: () => boolean,
): Promise<string | null> {
  if (caps.value.sourceImageMode !== "single") return null;
  const restoredSourceFit = parseSourceFitPolicy(print.metadata.source_fit);
  const stored = await restoreGenerationSourceMedia(print.metadata.source_image_sha256).catch(
    () => null,
  );
  if (!isCurrent()) return null;
  if (stored) {
    preserveRestoredSourceCanvas(stored.base64);
    form.sourceImage = stored.base64;
    form.sourceImageName = stored.filename;
    // The normal source-change watcher selects Resize for a newly attached
    // image. Let that watcher settle, then reassert provenance attributes:
    // Library reuse is restoration, not a new pick.
    await nextTick();
    if (!isCurrent()) return null;
    form.sourceImageWidth = stored.width ?? null;
    form.sourceImageHeight = stored.height ?? null;
    form.width = print.metadata.generation_width ?? print.metadata.width;
    form.height = print.metadata.generation_height ?? print.metadata.height;
    if (restoredSourceFit) form.sourceFit = restoredSourceFit;
    return null;
  }

  const filename = print.metadata.source_image_name;
  if (!filename) return null;
  const candidates = new Map<string, ApiTarget>([[print.hostId, print.target]]);
  for (const entry of gallery.value) {
    if (entry.filename === filename && !candidates.has(entry.hostId)) {
      candidates.set(entry.hostId, entry.target);
    }
  }
  for (const target of candidates.values()) {
    try {
      const source = await readReusedSourceCandidate(target, filename, signal);
      if (!isCurrent()) return null;
      if (!source) continue;
      preserveRestoredSourceCanvas(source.base64);
      form.sourceImage = source.base64;
      form.sourceImageName = filename;
      await nextTick();
      if (!isCurrent()) return null;
      form.sourceImageWidth = source.dimensions?.width ?? null;
      form.sourceImageHeight = source.dimensions?.height ?? null;
      form.width = print.metadata.generation_width ?? print.metadata.width;
      form.height = print.metadata.generation_height ?? print.metadata.height;
      if (restoredSourceFit) form.sourceFit = restoredSourceFit;
      return null;
    } catch {
      // Continue through every host that advertises the named source.
    }
  }
  if (!isCurrent()) return null;
  return "The original source image is unavailable. Reattach it before developing.";
}

async function readReusedSourceCandidate(
  target: ApiTarget,
  filename: string,
  signal: AbortSignal,
): Promise<{ base64: string; dimensions: SourceDimensions | null } | null> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let rejectDeadline: ((reason: Error) => void) | null = null;
  const deadline = new Promise<never>((_resolve, reject) => {
    rejectDeadline = reject;
  });
  const abort = () => {
    controller.abort();
    rejectDeadline?.(new DOMException("Reuse source restoration was cancelled", "AbortError"));
  };
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) abort();
  timeout = setTimeout(() => {
    controller.abort();
    rejectDeadline?.(new Error("Source restoration timed out"));
  }, REUSE_PRESENTATION_TIMEOUT_MS);
  try {
    return await Promise.race([
      (async () => {
        const response = await apiFetchTo(target, galleryMediaPath(filename, "host"), {
          signal: controller.signal,
        });
        if (response.ok === false) return null;
        const blob = await response.blob();
        if (!blob.size) return null;
        const base64 = await blobToBase64(blob);
        return { base64, dimensions: imageDimensionsFromBase64(base64) };
      })(),
      deadline,
    ]);
  } finally {
    if (timeout !== null) clearTimeout(timeout);
    signal.removeEventListener("abort", abort);
    rejectDeadline = null;
  }
}

/** Fetch bytes for the reuse-restored FL2VA boundary descriptors, within the
 * mobile media budget. The print's own host is tried first; a source frame
 * picked on another machine (auto-routing rendered elsewhere) resolves
 * through the merged gallery's per-print targets. Failures leave the
 * existing reattach affordance in place. */
async function restoreReusedH3BoundaryMedia(print: {
  hostId: string;
  target: ApiTarget;
  filename: string;
}): Promise<void> {
  if (minimaxH3TaskForModel(form.model) !== "fl2va") return;
  const wanted = h3BoundariesNeedingMedia(form.h3Authoring);
  if (wanted.length === 0) return;
  const modelAtStart = form.model;
  for (const slot of wanted) {
    // Candidate routes: origin host first, then any host whose merged
    // gallery lists the named file — deduped by host id.
    const candidates = new Map<string, ApiTarget>([[print.hostId, print.target]]);
    for (const entry of gallery.value) {
      if (entry.filename === slot.filename && !candidates.has(entry.hostId)) {
        candidates.set(entry.hostId, entry.target);
      }
    }
    for (const target of candidates.values()) {
      try {
        const response = await apiFetchTo(target, galleryMediaPath(slot.filename, "host"));
        const existingBytes = inlineGenerationMediaBytes(form, null);
        const declaredBytes = Number(response.headers?.get("content-length") ?? Number.NaN);
        if (
          Number.isFinite(declaredBytes) &&
          declaredBytes >= 0 &&
          existingBytes + declaredBytes > MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES
        ) {
          break; // over budget on every host — the file is what it is
        }
        const blob = await response.blob();
        if (blob.size === 0) continue;
        if (existingBytes + blob.size > MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES) break;
        const base64 = await blobToBase64(blob);
        if (form.model !== modelAtStart) return;
        const live = form.h3Authoring?.[slot.endpoint];
        if (!live || live.data || live.filename !== slot.filename) break;
        const result = setMinimaxH3PickedImageBoundary(form.h3Authoring, slot.endpoint, {
          filename: slot.filename,
          base64,
        });
        if (result.ok) form.h3Authoring = result.state;
        break;
      } catch {
        // Not on this host — try the next candidate; the reattach hint
        // remains if none of them has it.
      }
    }
  }
}

async function useSelectedPrintAsSource(): Promise<void> {
  const print = selectedPrint.value;
  if (!print || !canUseSelectedPrintAsSource.value || usingPrintAsSource.value) return;
  const epoch = ++sourceUseEpoch;
  sourceUseController?.abort();
  const controller = new AbortController();
  sourceUseController = controller;
  const isCurrent = () => !unmounted && !controller.signal.aborted && epoch === sourceUseEpoch;
  usingPrintAsSource.value = true;
  reusePrintError.value = "";
  try {
    const response = await apiFetchTo(print.target, galleryMediaPath(print.filename, "host"), {
      signal: controller.signal,
    });
    if (!isCurrent()) return;
    const h3Task = minimaxH3TaskForModel(form.model);
    const attachmentMode = caps.value.sourceImageMode !== "single";
    const existingBytes = inlineGenerationMediaBytes(
      form,
      h3Task === "fl2va" ? "h3FirstFrame" : attachmentMode ? null : "sourceImage",
    );
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
    if (!isCurrent()) return;
    if (blob.size === 0) throw new Error("That gallery image is empty.");
    if (exceedsBudget(blob.size)) throw new Error(MOBILE_MEDIA_BUDGET_ERROR);
    const base64 = await blobToBase64(blob);
    if (!isCurrent()) return;
    if (h3Task) {
      const dimensions = imageDimensionsFromBase64(base64) ?? {
        width: print.metadata.width,
        height: print.metadata.height,
      };
      const image = {
        filename: print.filename,
        mimeType: galleryImageMimeType(print, blob.type),
        width: dimensions.width,
        height: dimensions.height,
        data: base64,
      };
      const result =
        h3Task === "ref2va"
          ? appendMinimaxH3GalleryImageReference(form.h3Authoring, image)
          : setMinimaxH3GalleryImageFirstFrame(form.h3Authoring, image);
      if (!result.ok) throw new Error(result.error);
      form.h3Authoring = result.state;
      setGenerationStatus(
        h3Task === "ref2va"
          ? `Added gallery print as reference ${result.reference}`
          : "Gallery print selected as first frame",
      );
    } else if (isMinimaxH3Identity(form.family, form.model)) {
      throw new Error(
        "Choose an explicit MiniMax H3 FL2VA or Ref2VA model before adding a source.",
      );
    } else if (attachmentMode) {
      form.imageAttachments = [base64, ...form.imageAttachments].slice(
        0,
        isFlux2DevModel(form.model) ? 4 : undefined,
      );
      setGenerationStatus(
        isFlux2DevModel(form.model)
          ? "Added gallery print as reference 1"
          : "Added gallery print as the edit target",
      );
    } else {
      form.sourceImage = base64;
      form.sourceImageName = print.filename;
      setGenerationStatus("Gallery print selected as source");
    }
    selectedPrint.value = null;
    galleryRefreshDeferred = false;
    tab.value = "generate";
  } catch (error) {
    if (isCurrent()) reusePrintError.value = describeTransportError(error, print.hostName);
  } finally {
    if (epoch === sourceUseEpoch) {
      usingPrintAsSource.value = false;
      sourceUseController = null;
    }
  }
}

function galleryImageMimeType(print: GalleryImage, declared: string): string {
  const mime = declared.split(";", 1)[0]!.trim().toLowerCase();
  if (mime.startsWith("image/")) return mime;
  const format = (print.format ?? print.filename.split(".").pop() ?? "")
    .toLowerCase()
    .replace("jpg", "jpeg");
  return format ? `image/${format}` : "application/octet-stream";
}

function openPrint(print: GalleryPrint): void {
  reusePrintError.value = "";
  selectedPrint.value = print;
}

const galleryPrintKey = (print: Pick<GalleryPrint, "hostId" | "filename">) =>
  `${print.hostId}|${print.filename}`;

function allGalleryPrints(): Array<GalleryPrint | PendingGalleryPrint> {
  return [...gallery.value, ...pendingGallery];
}

// ── Library organization: support, merges, filters ──────────────────────────

const librarySupport = computed(() =>
  libraryOrganizationSupport(connectedHosts.value, serverCapabilities),
);
const libraryOrganizeEnabled = computed(() => librarySupport.value.organize);
const libraryTrashEnabled = computed(() => librarySupport.value.trash);
const libraryScopes = computed(() =>
  MOBILE_LIBRARY_SCOPES.filter(
    (scope) =>
      scope === "prints" ||
      (scope === "collections" && libraryOrganizeEnabled.value) ||
      (scope === "trash" && libraryTrashEnabled.value),
  ),
);
const hostNamesById = computed(() =>
  Object.fromEntries(connectedHosts.value.map((host) => [host.id, host.name])),
);
const libraryCollectionCards = computed<MobileCollectionCard[]>(() =>
  collectionCards(mergedCollectionsFor(hostCollections, connectedHosts.value), hostNamesById.value),
);
// Scoped to connected hosts so a disconnected or forgotten machine's
// retained bucket leaves no ghost chips (its bucket is also pruned below).
const mergedTags = computed(() =>
  mergeHostTags(
    hostTags,
    connectedHosts.value.map((host) => host.id),
  ),
);
const libraryTagChips = computed(() => tagChipPlan(mergedTags.value, libraryFilters.tag));
const libraryHostChips = computed(() =>
  connectedHosts.value.length > 1
    ? connectedHosts.value.map((host) => ({
        id: host.id,
        name: host.name,
        count: libraryHostCounts.value[host.id] ?? 0,
      }))
    : [],
);
const libraryChipRowVisible = computed(
  () =>
    libraryScope.value === "prints" &&
    (libraryOrganizeEnabled.value || libraryHostChips.value.length > 0),
);
const activeCollection = computed(
  () =>
    libraryCollectionCards.value.find((card) => card.slug === libraryFilters.collectionSlug) ??
    null,
);
const libraryScopeCounts = computed<Record<MobileLibraryScope, number>>(() => ({
  prints: libraryPrintCount.value,
  collections: libraryCollectionCards.value.length,
  trash: trashCount.value,
}));
const trashRetention = computed(() =>
  trashRetentionSummary(trashRetentionHosts(connectedHosts.value, librarySupport.value)),
);
const libraryEmptyCopy = computed(() => {
  if (libraryScope.value === "trash") return "Trash is empty.";
  if (libraryScope.value === "collections" && activeCollection.value) {
    return "No prints in this collection yet. Add some from Select.";
  }
  if (libraryFilters.favoritesOnly) return "No favorites yet. Tap ♥ on a print to keep it close.";
  if (libraryFilters.tag) return "No prints carry this tag.";
  if (libraryFilters.hostId) return "No prints on this host.";
  return "No prints found.";
});
const selectedGalleryKeysList = computed(() => [...gallerySelection.value]);
/** Whether every selected print is already a favorite (♥ toggles off then). */
const selectedAllFavorite = computed(() => {
  const keys = selectedGalleryKeysList.value;
  return keys.length > 0 && keys.every((key) => galleryOrganization.value.get(key)?.favorite);
});
const selectedDeleteKind = computed<"trash" | "delete" | "delete-forever">(() => {
  if (libraryScope.value === "trash") return "delete-forever";
  const hostIds = selectedPhysicalCopies().map((copy) => copy.hostId);
  return selectionDeleteKind(hostIds, librarySupport.value);
});
const galleryDeleteCopy = computed(() =>
  deleteActionCopy(
    selectedDeleteKind.value,
    gallerySelection.value.size,
    galleryDeleteConfirming.value,
    galleryDeleting.value,
  ),
);
const printTitleError = computed(() => {
  const result = requestTitle(printTitle.value);
  return result.ok ? "" : result.reason;
});
const selectedPrintOrganization = computed(() =>
  selectedPrint.value ? organizationOf(selectedPrint.value) : undefined,
);
const selectedPrintTrashed = computed(
  () =>
    libraryScope.value === "trash" || (selectedPrintOrganization.value?.trashedAt ?? null) !== null,
);

function organizationOf(
  print: Pick<GalleryPrint, "hostId" | "filename">,
): OrganizationUnion | undefined {
  return galleryOrganization.value.get(galleryPrintKey(print));
}

function printCaption(print: Pick<GalleryPrint, "hostId" | "filename" | "metadata">): string {
  return displayTitle({
    title: organizationOf(print)?.title ?? null,
    metadata: print.metadata,
    filename: print.filename,
  });
}

function tileLabel(print: GalleryPrint, action: string): string {
  return `${action} ${printCaption(print)} from ${print.hostName}`;
}

function purgeChipFor(
  print: Pick<GalleryPrint, "hostId" | "filename" | "purge_at">,
): string | null {
  const purgeAt = organizationOf(print)?.purgeAt ?? print.purge_at ?? null;
  return purgeChipLabel(purgeAt, Date.now());
}

function scopeCopies(): PendingGalleryPrint[] {
  return libraryScope.value === "trash" ? trashCopies : galleryCopies;
}

function rebuildGalleryOrganization(): void {
  const resolver = collectionSlugResolver(
    connectedHosts.value.map((host) => ({
      hostId: host.id,
      collections: hostCollections[host.id] ?? [],
    })),
  );
  galleryOrganization.value = buildOrganizationIndex(
    [...galleryCopies, ...trashCopies],
    resolver,
    selectedHostId.value || null,
  );
  const counts: Record<string, number> = {};
  for (const copy of galleryCopies) counts[copy.hostId] = (counts[copy.hostId] ?? 0) + 1;
  libraryHostCounts.value = counts;
  libraryPrintCount.value = groupLogicalGalleryPrints(galleryCopies).length;
  trashCount.value = groupLogicalGalleryPrints(trashCopies).length;
}

/** The representatives the grid pages through for the current scope + chips. */
function visibleRepresentatives(): PendingGalleryPrint[] {
  const copies = scopeCopies();
  const representatives = groupLogicalGalleryPrints(copies).map((group) => group.representative);
  const filters =
    libraryScope.value === "prints"
      ? { ...libraryFilters, collectionSlug: null }
      : libraryScope.value === "collections"
        ? { ...EMPTY_LIBRARY_FILTERS, collectionSlug: libraryFilters.collectionSlug }
        : { ...EMPTY_LIBRARY_FILTERS };
  return filterLibraryPrints(representatives, filters, organizationOf, (print) =>
    logicalCopiesOf(copies, print),
  );
}

/** Rebuild the paged grid from the current scope, filters, and organization. */
async function resetGalleryPaging(): Promise<void> {
  for (const item of gallery.value) revokeObjectUrl(item.thumbnailUrl);
  gallery.value = [];
  pendingGallery = visibleRepresentatives();
  await loadMoreGalleryPage();
}

function requeueGallery(): Promise<void> {
  return enqueueGalleryOperation(resetGalleryPaging);
}

async function refreshHostOrganization(hostIds: readonly string[]): Promise<void> {
  const support = librarySupport.value;
  await Promise.all(
    hostIds.map(async (hostId) => {
      const host = connectedHosts.value.find((candidate) => candidate.id === hostId);
      if (!host || !support.organizeHostIds.has(hostId)) return;
      const target = mobileHostTarget(host);
      const [collections, tags] = await Promise.allSettled([
        listHostCollections(target),
        listHostTags(target),
      ]);
      if (collections.status === "fulfilled") hostCollections[hostId] = collections.value;
      if (tags.status === "fulfilled") hostTags[hostId] = tags.value;
    }),
  );
}

function setLibraryScope(scope: MobileLibraryScope): void {
  if (libraryScope.value === scope) return;
  setGallerySelectMode(false);
  libraryScope.value = scope;
  emptyTrashConfirming.value = false;
  collectionMenuSlug.value = null;
  collectionDeleteConfirmSlug.value = null;
  if (scope !== "collections") libraryFilters.collectionSlug = null;
  if (scope === "trash" && !trashLoaded.value) void refreshTrash();
  else void requeueGallery();
}

function toggleFavoritesFilter(): void {
  libraryFilters.favoritesOnly = !libraryFilters.favoritesOnly;
  void requeueGallery();
}

function setTagFilter(name: string | null): void {
  const key = name ? tagKey(name) : null;
  libraryFilters.tag = libraryFilters.tag === key ? null : key;
  librarySheet.value = null;
  void requeueGallery();
}

function setHostFilter(hostId: string | null): void {
  libraryFilters.hostId = libraryFilters.hostId === hostId ? null : hostId;
  void requeueGallery();
}

function openCollection(slug: string): void {
  setGallerySelectMode(false);
  collectionMenuSlug.value = null;
  libraryFilters.collectionSlug = slug;
  void requeueGallery();
}

function closeCollection(): void {
  setGallerySelectMode(false);
  libraryFilters.collectionSlug = null;
  void requeueGallery();
}

/** Cover thumbnail for a collection card: a loaded tile when the cover print
 * is on screen, otherwise fetched once through the cached media pipeline. */
function collectionCoverUrl(card: MobileCollectionCard): string {
  const loaded = gallery.value.find((print) =>
    (organizationOf(print)?.collections ?? []).includes(card.slug),
  );
  if (loaded) return loaded.thumbnailUrl;
  const cached = collectionCovers[card.slug];
  if (cached) return cached;
  const cover = card.cover;
  if (!cover) return "";
  const host = connectedHosts.value.find((candidate) => candidate.id === cover.hostId);
  if (!host) return "";
  collectionCovers[card.slug] = "";
  void thumbnailUrl(mobileHostTarget(host), mobileGalleryCacheKey(host), cover.filename)
    .then((url) => {
      collectionCovers[card.slug] = url;
    })
    .catch(() => {
      delete collectionCovers[card.slug];
    });
  return "";
}

// ── Library organization: mutations ─────────────────────────────────────────

function selectedRepresentatives(): Array<GalleryPrint | PendingGalleryPrint> {
  return allGalleryPrints().filter((print) => gallerySelection.value.has(galleryPrintKey(print)));
}

/** Every physical copy behind the selected logical prints. */
function selectedPhysicalCopies(): PendingGalleryPrint[] {
  const copies = scopeCopies();
  const seen = new Map<string, PendingGalleryPrint>();
  for (const print of selectedRepresentatives()) {
    for (const copy of logicalCopiesOf(copies, print)) seen.set(galleryPrintKey(copy), copy);
  }
  return [...seen.values()];
}

function fanoutHosts(): Record<
  string,
  { id: string; name: string; target: ApiTarget; collections: Collection[] }
> {
  return Object.fromEntries(
    connectedHosts.value.map((host) => [
      host.id,
      {
        id: host.id,
        name: host.name,
        target: mobileHostTarget(host),
        collections: hostCollections[host.id] ?? [],
      },
    ]),
  );
}

interface LibraryMutationOutcome {
  failedHostIds: Set<string>;
  ok: boolean;
}

/**
 * Fan one mutation out to every physical copy on its own Keychain-
 * authenticated host; failures land in the inline banner, never a toast.
 */
async function runLibraryMutation(
  copies: readonly PendingGalleryPrint[],
  mutation: OrganizationMutation,
  action: string,
): Promise<LibraryMutationOutcome> {
  organizationError.value = "";
  organizationBusy.value = true;
  try {
    const ops = planOrganizationFanout(copies, mutation);
    const result = await runOrganizationFanout(ops, fanoutHosts(), undefined, {
      trashHostIds: librarySupport.value.trashHostIds,
    });
    if (result.createdCollections.length > 0) {
      await refreshHostOrganization(result.createdCollections.map((entry) => entry.hostId));
    }
    const failedHostIds = new Set(result.failures.map((failure) => failure.hostId));
    if (result.failures.length > 0) {
      organizationError.value = fanoutFailureMessage(action, result.failures, (error, name) =>
        describeTransportError(error, name),
      );
    }
    return { failedHostIds, ok: result.failures.length === 0 };
  } finally {
    organizationBusy.value = false;
  }
}

/** Apply a local organization patch to every matching copy (pending, loaded,
 * trash, and the IndexedDB cache), then re-derive the merged index. */
async function applyOrganizationPatch(
  copies: readonly PendingGalleryPrint[],
  patch: (copy: PendingGalleryPrint) => Partial<MobileGalleryImage>,
  skipHostIds: ReadonlySet<string> = new Set(),
): Promise<void> {
  const patches = new Map<string, Partial<MobileGalleryImage>>();
  for (const copy of copies) {
    if (skipHostIds.has(copy.hostId)) continue;
    patches.set(galleryPrintKey(copy), patch(copy));
  }
  const apply = <T extends PendingGalleryPrint>(list: T[]): T[] =>
    list.map((entry) => {
      const next = patches.get(galleryPrintKey(entry));
      return next ? { ...entry, ...next } : entry;
    });
  galleryCopies = apply(galleryCopies);
  trashCopies = apply(trashCopies);
  pendingGallery = apply(pendingGallery);
  gallery.value = apply(gallery.value);
  if (selectedPrint.value) {
    const next = patches.get(galleryPrintKey(selectedPrint.value));
    if (next) selectedPrint.value = { ...selectedPrint.value, ...next };
  }
  rebuildGalleryOrganization();
  const byCacheKey = new Map<
    string,
    Array<{ filename: string; patch: Partial<MobileGalleryImage> }>
  >();
  for (const copy of copies) {
    const next = patches.get(galleryPrintKey(copy));
    if (!next) continue;
    const list = byCacheKey.get(copy.cacheKey) ?? [];
    list.push({ filename: copy.filename, patch: next });
    byCacheKey.set(copy.cacheKey, list);
  }
  await Promise.all(
    [...byCacheKey].map(([cacheKey, list]) => patchCachedGalleryPrints(cacheKey, list)),
  );
}

async function setFavoriteFor(
  copies: readonly PendingGalleryPrint[],
  favorite: boolean,
): Promise<void> {
  const outcome = await runLibraryMutation(
    copies,
    { kind: "setFavorite", favorite },
    favorite ? "favorite these prints" : "unfavorite these prints",
  );
  await applyOrganizationPatch(copies, () => ({ favorite }), outcome.failedHostIds);
  // The mutation changed the active filter's predicate: rebuild the paged
  // grid so a print that no longer matches leaves immediately.
  if (libraryFilters.favoritesOnly) await requeueGallery();
}

async function setTitleFor(
  copies: readonly PendingGalleryPrint[],
  title: string | null,
): Promise<void> {
  const outcome = await runLibraryMutation(
    copies,
    { kind: "setTitle", title },
    "rename this print",
  );
  await applyOrganizationPatch(copies, () => ({ title }), outcome.failedHostIds);
}

async function addTagsFor(copies: readonly PendingGalleryPrint[], tags: string[]): Promise<void> {
  if (tags.length === 0) return;
  const outcome = await runLibraryMutation(copies, { kind: "addTags", tags }, "tag these prints");
  await applyOrganizationPatch(
    copies,
    (copy) => {
      const existing = copy.tags ?? [];
      const keys = new Set(existing.map(tagKey));
      return { tags: [...existing, ...tags.filter((tag) => !keys.has(tagKey(tag)))] };
    },
    outcome.failedHostIds,
  );
  await refreshHostOrganization([...new Set(copies.map((copy) => copy.hostId))]);
  await requeueIfTagFilterAffected(tags);
}

/** Rebuild the paged grid when a tag mutation touched the active tag filter,
 * so prints that stopped (or started) matching move immediately. */
async function requeueIfTagFilterAffected(tags: readonly string[]): Promise<void> {
  const active = libraryFilters.tag;
  if (active && tags.some((tag) => tagKey(tag) === active)) await requeueGallery();
}

async function removeTagsFor(
  copies: readonly PendingGalleryPrint[],
  tags: string[],
): Promise<void> {
  if (tags.length === 0) return;
  const outcome = await runLibraryMutation(
    copies,
    { kind: "removeTags", tags },
    "remove the tag from these prints",
  );
  const removed = new Set(tags.map(tagKey));
  await applyOrganizationPatch(
    copies,
    (copy) => ({ tags: (copy.tags ?? []).filter((tag) => !removed.has(tagKey(tag))) }),
    outcome.failedHostIds,
  );
  await refreshHostOrganization([...new Set(copies.map((copy) => copy.hostId))]);
  await requeueIfTagFilterAffected(tags);
}

async function setCollectionMembershipFor(
  copies: readonly PendingGalleryPrint[],
  collection: { slug: string; name: string },
  member: boolean,
): Promise<void> {
  const outcome = await runLibraryMutation(
    copies,
    member
      ? { kind: "addToCollection", name: collection.name, slug: collection.slug }
      : { kind: "removeFromCollection", slug: collection.slug },
    member ? `add to “${collection.name}”` : `remove from “${collection.name}”`,
  );
  await refreshHostOrganization([...new Set(copies.map((copy) => copy.hostId))]);
  await applyOrganizationPatch(
    copies,
    (copy) => {
      const hostCollection = collectionOnHost(hostCollections, copy.hostId, collection.slug);
      const ids = (copy.collections ?? []).filter((id) => id !== hostCollection?.id);
      if (member && hostCollection) ids.push(hostCollection.id);
      return { collections: ids };
    },
    outcome.failedHostIds,
  );
  if (libraryScope.value === "collections") await requeueGallery();
}

// ── Select-mode bulk actions ───────────────────────────────────────────────

async function favoriteSelected(): Promise<void> {
  if (organizationBusy.value || gallerySelection.value.size === 0) return;
  galleryDeleteConfirming.value = false;
  await setFavoriteFor(selectedPhysicalCopies(), !selectedAllFavorite.value);
}

function openLibrarySheet(sheet: LibrarySheet): void {
  galleryDeleteConfirming.value = false;
  librarySheetInput.value = sheet.kind === "rename-collection" ? sheet.name : "";
  librarySheetError.value = "";
  librarySheet.value = sheet;
}

function closeLibrarySheet(): void {
  librarySheet.value = null;
  librarySheetInput.value = "";
  librarySheetError.value = "";
}

/** Tags every selected print carries (for the × chips in the tag editor). */
const selectedTags = computed(() => {
  const counts = new Map<string, { name: string; count: number }>();
  for (const key of selectedGalleryKeysList.value) {
    for (const tag of galleryOrganization.value.get(key)?.tags ?? []) {
      const entry = counts.get(tagKey(tag)) ?? { name: tag, count: 0 };
      entry.count += 1;
      counts.set(tagKey(tag), entry);
    }
  }
  return [...counts.values()].sort((a, b) => a.name.localeCompare(b.name));
});
const tagSuggestions = computed(() => {
  const present = new Set(selectedTags.value.map((tag) => tagKey(tag.name)));
  return mergedTags.value.filter((tag) => !present.has(tagKey(tag.name))).slice(0, 12);
});

async function addTagToSelected(raw: string): Promise<void> {
  const name = raw
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^#+\s*/, "");
  if (!name) return;
  librarySheetBusy.value = true;
  try {
    await addTagsFor(selectedPhysicalCopies(), [name]);
    librarySheetInput.value = "";
  } finally {
    librarySheetBusy.value = false;
  }
}

async function removeTagFromSelected(name: string): Promise<void> {
  librarySheetBusy.value = true;
  try {
    await removeTagsFor(selectedPhysicalCopies(), [name]);
  } finally {
    librarySheetBusy.value = false;
  }
}

/** Checklist state for the collection sheet: every selected print is in it. */
function selectedInCollection(slug: string): boolean {
  const keys = selectedGalleryKeysList.value;
  return (
    keys.length > 0 &&
    keys.every((key) => (galleryOrganization.value.get(key)?.collections ?? []).includes(slug))
  );
}

async function toggleSelectedCollection(card: { slug: string; name: string }): Promise<void> {
  librarySheetBusy.value = true;
  try {
    await setCollectionMembershipFor(
      selectedPhysicalCopies(),
      card,
      !selectedInCollection(card.slug),
    );
  } finally {
    librarySheetBusy.value = false;
  }
}

async function removeSelectedFromCollection(): Promise<void> {
  const collection = activeCollection.value;
  if (!collection || organizationBusy.value) return;
  const copies = selectedPhysicalCopies();
  await setCollectionMembershipFor(copies, collection, false);
  gallerySelection.value = new Set();
  setGallerySelectMode(false);
}

/** Create a collection by name. With a selection (collection sheet) the
 * selected prints join it; from the Collections scope it is created empty on
 * every host that can organize. */
async function createCollectionFromSheet(): Promise<void> {
  const validation = validateCollectionName(librarySheetInput.value);
  if (!validation.ok) {
    librarySheetError.value = validation.reason ?? "";
    return;
  }
  librarySheetBusy.value = true;
  librarySheetError.value = "";
  organizationError.value = "";
  try {
    if (librarySheet.value?.kind === "collections" && gallerySelection.value.size > 0) {
      await setCollectionMembershipFor(
        selectedPhysicalCopies(),
        { slug: collectionSlug(validation.value), name: validation.value },
        true,
      );
      librarySheetInput.value = "";
      return;
    }
    const hosts = connectedHosts.value.filter((host) =>
      librarySupport.value.organizeHostIds.has(host.id),
    );
    const results = await Promise.allSettled(
      hosts.map((host) => createHostCollection(mobileHostTarget(host), { name: validation.value })),
    );
    const failures = results.flatMap((result, index) =>
      result.status === "rejected"
        ? [{ hostId: hosts[index]!.id, hostName: hosts[index]!.name, error: result.reason }]
        : [],
    );
    if (failures.length > 0) {
      organizationError.value = fanoutFailureMessage(
        `create “${validation.value}”`,
        failures,
        (error, name) => describeTransportError(error, name),
      );
    }
    await refreshHostOrganization(hosts.map((host) => host.id));
    rebuildGalleryOrganization();
    closeLibrarySheet();
  } finally {
    librarySheetBusy.value = false;
  }
}

function openCollectionMenu(slug: string): void {
  collectionDeleteConfirmSlug.value = null;
  collectionMenuSlug.value = collectionMenuSlug.value === slug ? null : slug;
}

async function renameCollectionFromSheet(): Promise<void> {
  const sheet = librarySheet.value;
  if (sheet?.kind !== "rename-collection") return;
  const validation = validateCollectionName(librarySheetInput.value);
  if (!validation.ok) {
    librarySheetError.value = validation.reason ?? "";
    return;
  }
  librarySheetBusy.value = true;
  organizationError.value = "";
  try {
    const hosts = connectedHosts.value.filter((host) =>
      collectionOnHost(hostCollections, host.id, sheet.slug),
    );
    const results = await Promise.allSettled(
      hosts.map((host) =>
        updateHostCollection(
          mobileHostTarget(host),
          collectionOnHost(hostCollections, host.id, sheet.slug)!.id,
          { name: validation.value },
        ),
      ),
    );
    const failures = results.flatMap((result, index) =>
      result.status === "rejected"
        ? [{ hostId: hosts[index]!.id, hostName: hosts[index]!.name, error: result.reason }]
        : [],
    );
    if (failures.length > 0) {
      organizationError.value = fanoutFailureMessage(
        `rename “${sheet.name}”`,
        failures,
        (error, name) => describeTransportError(error, name),
      );
    }
    await refreshHostOrganization(hosts.map((host) => host.id));
    rebuildGalleryOrganization();
    collectionMenuSlug.value = null;
    closeLibrarySheet();
  } finally {
    librarySheetBusy.value = false;
  }
}

/** Two-step: first tap arms, second deletes the collection on every host
 * that has it. Prints are never touched (D7). */
async function deleteCollection(card: MobileCollectionCard): Promise<void> {
  if (collectionDeleteConfirmSlug.value !== card.slug) {
    collectionDeleteConfirmSlug.value = card.slug;
    return;
  }
  collectionDeleteConfirmSlug.value = null;
  organizationBusy.value = true;
  organizationError.value = "";
  try {
    const hosts = connectedHosts.value.filter((host) =>
      collectionOnHost(hostCollections, host.id, card.slug),
    );
    const results = await Promise.allSettled(
      hosts.map((host) =>
        deleteHostCollection(
          mobileHostTarget(host),
          collectionOnHost(hostCollections, host.id, card.slug)!.id,
        ),
      ),
    );
    const failures = results.flatMap((result, index) =>
      result.status === "rejected"
        ? [{ hostId: hosts[index]!.id, hostName: hosts[index]!.name, error: result.reason }]
        : [],
    );
    if (failures.length > 0) {
      organizationError.value = fanoutFailureMessage(
        `delete “${card.name}”`,
        failures,
        (error, name) => describeTransportError(error, name),
      );
    }
    await refreshHostOrganization(hosts.map((host) => host.id));
    rebuildGalleryOrganization();
    collectionMenuSlug.value = null;
    if (libraryFilters.collectionSlug === card.slug) closeCollection();
  } finally {
    organizationBusy.value = false;
  }
}

// ── Trash ───────────────────────────────────────────────────────────────────

function refreshTrash(): Promise<void> {
  return enqueueGalleryOperation(async () => {
    trashLoading.value = true;
    trashError.value = "";
    const trashCapable = connectedHosts.value.filter((host) =>
      librarySupport.value.trashHostIds.has(host.id),
    );
    // A known-offline trash-capable host is skipped, not silently dropped:
    // it counts against completeness so the snapshot stays retry-eligible.
    const hosts = trashCapable.filter((host) => host.online);
    const skippedHosts = trashCapable.length - hosts.length;
    const results = await Promise.allSettled(
      hosts.map(async (host) => {
        const target = mobileHostTarget(host);
        const prints = await listHostTrash<MobileGalleryImage>(target);
        return { host, target, prints };
      }),
    );
    const refreshed: PendingGalleryPrint[] = [];
    const refreshedHostIds = new Set<string>();
    let rejectedHosts = 0;
    results.forEach((result) => {
      if (result.status !== "fulfilled") {
        rejectedHosts += 1;
        return;
      }
      const { host, target, prints } = result.value;
      refreshedHostIds.add(host.id);
      for (const print of prints) {
        refreshed.push({
          ...print,
          hostId: host.id,
          cacheKey: mobileGalleryCacheKey(host),
          hostName: host.name,
          target,
        });
      }
    });
    const outcome = mergeTrashSnapshot({
      previous: trashCopies,
      refreshed,
      refreshedHostIds,
      trashCapableHostIds: new Set(trashCapable.map((host) => host.id)),
      rejectedHosts,
      skippedHosts,
    });
    trashCopies = outcome.copies;
    // An incomplete pass is never the authoritative snapshot: leaving
    // `trashLoaded` false keeps the scope retry-eligible so re-entering
    // Trash refetches the hosts this pass could not read.
    trashLoaded.value = outcome.complete;
    if (outcome.failedHosts > 0) {
      trashError.value = `${outcome.failedHosts} host${
        outcome.failedHosts === 1 ? "" : "s"
      } unavailable · Trash may be incomplete`;
    }
    rebuildGalleryOrganization();
    if (libraryScope.value === "trash") await resetGalleryPaging();
    trashLoading.value = false;
  });
}

/** Drop copies from the scope lists and the grid (after a trash / restore /
 * purge succeeded on their host). */
async function dropCopiesFromLibrary(
  keys: ReadonlySet<string>,
  options: { purgeThumbnails?: boolean } = {},
): Promise<void> {
  const removed = [...galleryCopies, ...trashCopies].filter((copy) =>
    keys.has(galleryPrintKey(copy)),
  );
  if (options.purgeThumbnails) {
    await removeCachedGalleryPrints(
      removed.map((copy) => ({ hostId: copy.cacheKey, filename: copy.filename })),
    );
  }
  galleryCopies = galleryCopies.filter((copy) => !keys.has(galleryPrintKey(copy)));
  trashCopies = trashCopies.filter((copy) => !keys.has(galleryPrintKey(copy)));
  rebuildGalleryOrganization();
  await resetGalleryPaging();
  gallerySelection.value = new Set(
    [...gallerySelection.value].filter((key) =>
      allGalleryPrints().some((print) => galleryPrintKey(print) === key),
    ),
  );
  if (gallerySelection.value.size === 0) setGallerySelectMode(false);
}

async function restoreSelectedGalleryPrints(): Promise<void> {
  if (galleryRestoring.value || gallerySelection.value.size === 0) return;
  galleryDeleteConfirming.value = false;
  galleryRestoring.value = true;
  try {
    const copies = selectedPhysicalCopies();
    const outcome = await runLibraryMutation(copies, { kind: "restore" }, "restore these prints");
    const restored = copies.filter((copy) => !outcome.failedHostIds.has(copy.hostId));
    const keys = new Set(restored.map(galleryPrintKey));
    // Restored copies rejoin the live Library locally; the next refresh
    // confirms them against the host.
    galleryCopies = [
      ...galleryCopies,
      ...restored.map(({ trashed_at: _trashedAt, purge_at: _purgeAt, ...copy }) => copy),
    ].sort((a, b) => b.timestamp - a.timestamp);
    trashCopies = trashCopies.filter((copy) => !keys.has(galleryPrintKey(copy)));
    rebuildGalleryOrganization();
    await resetGalleryPaging();
    gallerySelection.value = new Set(
      [...gallerySelection.value].filter((key) =>
        allGalleryPrints().some((print) => galleryPrintKey(print) === key),
      ),
    );
    if (gallerySelection.value.size === 0) setGallerySelectMode(false);
  } finally {
    galleryRestoring.value = false;
  }
}

async function emptyTrash(): Promise<void> {
  if (emptyingTrash.value) return;
  if (!emptyTrashConfirming.value) {
    emptyTrashConfirming.value = true;
    return;
  }
  emptyTrashConfirming.value = false;
  emptyingTrash.value = true;
  organizationError.value = "";
  try {
    const hosts = connectedHosts.value.filter((host) =>
      librarySupport.value.trashHostIds.has(host.id),
    );
    const results = await Promise.allSettled(
      hosts.map((host) => emptyHostTrash(mobileHostTarget(host))),
    );
    const failures = results.flatMap((result, index) =>
      result.status === "rejected"
        ? [{ hostId: hosts[index]!.id, hostName: hosts[index]!.name, error: result.reason }]
        : [],
    );
    if (failures.length > 0) {
      organizationError.value = fanoutFailureMessage("empty the trash", failures, (error, name) =>
        describeTransportError(error, name),
      );
    }
    const failedHostIds = new Set(failures.map((failure) => failure.hostId));
    const purged = new Set(
      trashCopies.filter((copy) => !failedHostIds.has(copy.hostId)).map(galleryPrintKey),
    );
    await dropCopiesFromLibrary(purged, { purgeThumbnails: true });
  } finally {
    emptyingTrash.value = false;
  }
}

// ── Viewer (info sheet) handlers ────────────────────────────────────────────

function viewerCopies(): PendingGalleryPrint[] {
  const print = selectedPrint.value;
  return print ? logicalCopiesOf(scopeCopies(), print) : [];
}

async function renameSelectedPrint(title: string | null): Promise<void> {
  await setTitleFor(viewerCopies(), title);
}

async function favoriteSelectedPrint(favorite: boolean): Promise<void> {
  await setFavoriteFor(viewerCopies(), favorite);
}

async function tagSelectedPrint(change: { add?: string[]; remove?: string[] }): Promise<void> {
  const copies = viewerCopies();
  if (change.add?.length) await addTagsFor(copies, change.add);
  if (change.remove?.length) await removeTagsFor(copies, change.remove);
}

async function collectSelectedPrint(change: {
  slug: string;
  name: string;
  member: boolean;
}): Promise<void> {
  await setCollectionMembershipFor(viewerCopies(), change, change.member);
}

async function restoreSelectedPrint(): Promise<void> {
  const copies = viewerCopies();
  const outcome = await runLibraryMutation(copies, { kind: "restore" }, "restore this print");
  const restored = copies.filter((copy) => !outcome.failedHostIds.has(copy.hostId));
  if (restored.length === 0) return;
  const keys = new Set(restored.map(galleryPrintKey));
  galleryCopies = [
    ...galleryCopies,
    ...restored.map(({ trashed_at: _trashedAt, purge_at: _purgeAt, ...copy }) => copy),
  ].sort((a, b) => b.timestamp - a.timestamp);
  trashCopies = trashCopies.filter((copy) => !keys.has(galleryPrintKey(copy)));
  selectedPrint.value = null;
  galleryRefreshDeferred = false;
  rebuildGalleryOrganization();
  await requeueGallery();
}

async function deleteSelectedPrintForever(): Promise<void> {
  const copies = viewerCopies();
  const outcome = await runLibraryMutation(
    copies,
    { kind: "deleteForever" },
    "delete this print forever",
  );
  const keys = new Set(
    copies.filter((copy) => !outcome.failedHostIds.has(copy.hostId)).map(galleryPrintKey),
  );
  if (keys.size === 0) return;
  selectedPrint.value = null;
  galleryRefreshDeferred = false;
  await enqueueGalleryOperation(() => dropCopiesFromLibrary(keys, { purgeThumbnails: true }));
}

function setGallerySelectMode(next: boolean): void {
  if (!next) {
    finishGallerySelectionDrag();
    galleryDragPendingClicks = 0;
  }
  // Entering or leaving the mode is a deliberate tap boundary: a click still
  // owed to an earlier pinch is stale and must not swallow the next real one.
  // The gesture itself is deliberately NOT reset here — an SSE-driven gallery
  // refresh reaches this function, and killing a live pinch from it would drop
  // the user's fingers mid-resize.
  galleryPinchPendingClicks = 0;
  gallerySelectMode.value = next;
  galleryDeleteConfirming.value = false;
  if (!next) gallerySelection.value = new Set();
}

function clearGallerySelection(): void {
  setGallerySelectMode(false);
}

function clearSelectedGalleryPrints(): void {
  gallerySelection.value = new Set();
  galleryDeleteConfirming.value = false;
}

function toggleGallerySelection(print: GalleryPrint): void {
  const key = galleryPrintKey(print);
  const next = new Set(gallerySelection.value);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  gallerySelection.value = next;
  galleryDeleteConfirming.value = false;
}

function applyGalleryDragSelection(print: GalleryPrint): void {
  const key = galleryPrintKey(print);
  if (galleryDragVisited.has(key)) return;
  galleryDragVisited.add(key);
  const next = new Set(gallerySelection.value);
  if (galleryDragSelect) next.add(key);
  else next.delete(key);
  gallerySelection.value = next;
  galleryDeleteConfirming.value = false;
}

function galleryPrintAtPoint(x: number, y: number): GalleryPrint | null {
  // Keep tiles discoverable beneath the sticky selection toolbar while edge
  // auto-scroll moves the grid behind it.
  const elements = document.elementsFromPoint?.(x, y) ?? [document.elementFromPoint(x, y)];
  const tile = elements
    .map((element) => element?.closest<HTMLElement>("[data-gallery-print-key]") ?? null)
    .find((element) => element !== null);
  const key = tile?.dataset.galleryPrintKey;
  return key ? (gallery.value.find((print) => galleryPrintKey(print) === key) ?? null) : null;
}

function applyGalleryDragAtPoint(): void {
  const print = galleryPrintAtPoint(galleryDragClientX, galleryDragClientY);
  if (print) applyGalleryDragSelection(print);
}

function applyGalleryDragSegment(fromX: number, fromY: number, toX: number, toY: number): void {
  const distance = Math.hypot(toX - fromX, toY - fromY);
  // This stride is well below the 44pt minimum tile target. Sampling the
  // whole segment prevents a fast native swipe from jumping over a column.
  const steps = Math.max(1, Math.ceil(distance / 12));
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps;
    const print = galleryPrintAtPoint(
      fromX + (toX - fromX) * progress,
      fromY + (toY - fromY) * progress,
    );
    if (print) applyGalleryDragSelection(print);
  }
}

function runGalleryDragFrame(): void {
  galleryDragFrame = null;
  if (galleryDragPointerId === null || !galleryDragActive || !gallerySelectMode.value) return;
  const scroller = mobileContent.value;
  if (scroller) {
    const bounds = scroller.getBoundingClientRect();
    const topDepth = Math.max(0, bounds.top + GALLERY_DRAG_SCROLL_EDGE - galleryDragClientY);
    const bottomDepth = Math.max(
      0,
      galleryDragClientY - (bounds.bottom - GALLERY_DRAG_SCROLL_EDGE),
    );
    const direction = bottomDepth > 0 ? 1 : topDepth > 0 ? -1 : 0;
    const depth = Math.max(topDepth, bottomDepth);
    if (direction && depth) {
      const speed = Math.min(
        GALLERY_DRAG_SCROLL_MAX,
        Math.max(2, (depth / GALLERY_DRAG_SCROLL_EDGE) * GALLERY_DRAG_SCROLL_MAX),
      );
      scroller.scrollTop += direction * speed;
      applyGalleryDragAtPoint();
    }
  }
  galleryDragFrame = requestAnimationFrame(runGalleryDragFrame);
}

function beginGallerySelectionDrag(event: PointerEvent, print: GalleryPrint): void {
  if (
    !gallerySelectMode.value ||
    event.isPrimary === false ||
    (event.pointerType === "mouse" && event.button !== 0)
  ) {
    return;
  }
  galleryDragPointerId = event.pointerId;
  galleryDragActive = event.pointerType === "mouse";
  galleryDragSelectionBaseline = new Set(gallerySelection.value);
  galleryDragSelect = !gallerySelection.value.has(galleryPrintKey(print));
  galleryDragStartX = event.clientX;
  galleryDragStartY = event.clientY;
  galleryDragClientX = event.clientX;
  galleryDragClientY = event.clientY;
  galleryDragVisited.clear();
  // Mouse has no native vertical-pan gesture to preserve, so it can paint on
  // pointerdown. Touch waits for movement intent: vertical remains native
  // scrolling, while horizontal/diagonal movement claims drag-selection.
  if (galleryDragActive) {
    event.preventDefault();
    applyGalleryDragSelection(print);
    (event.currentTarget as HTMLElement | null)?.setPointerCapture?.(event.pointerId);
    if (galleryDragFrame === null) galleryDragFrame = requestAnimationFrame(runGalleryDragFrame);
  }
}

function moveGallerySelectionDrag(event: PointerEvent): void {
  if (event.pointerId !== galleryDragPointerId) return;
  const points = [...(event.getCoalescedEvents?.() ?? []), event];
  for (const point of points) {
    if (!galleryDragActive) {
      const deltaX = point.clientX - galleryDragStartX;
      const deltaY = point.clientY - galleryDragStartY;
      if (Math.hypot(deltaX, deltaY) < GALLERY_DRAG_INTENT_THRESHOLD) continue;
      if (Math.abs(deltaY) > Math.abs(deltaX)) {
        // Do not prevent this event: WebKit keeps ownership and scrolls the
        // Library with native momentum. No tile was painted speculatively.
        finishGallerySelectionDrag();
        return;
      }
      galleryDragActive = true;
      const startingPrint = galleryPrintAtPoint(galleryDragStartX, galleryDragStartY);
      if (startingPrint) applyGalleryDragSelection(startingPrint);
      if (galleryDragFrame === null) galleryDragFrame = requestAnimationFrame(runGalleryDragFrame);
    }
    event.preventDefault();
    applyGalleryDragSegment(galleryDragClientX, galleryDragClientY, point.clientX, point.clientY);
    galleryDragClientX = point.clientX;
    galleryDragClientY = point.clientY;
  }
}

function finishGallerySelectionDrag(event?: PointerEvent): void {
  if (event && event.pointerId !== galleryDragPointerId) return;
  if (event?.type === "pointerup" && galleryDragActive) galleryDragPendingClicks += 1;
  galleryDragPointerId = null;
  galleryDragActive = false;
  galleryDragSelectionBaseline = null;
  galleryDragVisited.clear();
  if (galleryDragFrame !== null) cancelAnimationFrame(galleryDragFrame);
  galleryDragFrame = null;
}

/**
 * Pinch over the Library grid to resize thumbnails, the iPhone counterpart to
 * the web/desktop thumbnail-size slider. It deliberately shares no state with
 * the gallery viewer's own gestures and owns the whole print area, including
 * the unused space below a short final row.
 */
function beginGalleryPinch(event: PointerEvent): void {
  if (event.pointerType === "mouse") return;
  if (galleryZoom.points.size === 0) {
    // A fresh touch sequence. WKWebView dispatches its compatibility click
    // before any new finger lands, so a claim still outstanding here belongs to
    // a click that never came and must not swallow this deliberate tap.
    galleryPinchPendingClicks = 0;
  }
  pinchPointerDown(galleryZoom, event);
  if (!isPinching(galleryZoom)) return;
  // A second finger means a pinch, never a selection: undo the tile the first
  // finger speculatively painted, then abandon the drag. `pan-y` would let the
  // UA claim a drifting two-finger gesture as a scroll and cancel our pointers,
  // so the grid holds `none` for as long as the pinch owns the fingers.
  if (galleryDragPointerId !== null) {
    if (galleryDragSelectionBaseline) gallerySelection.value = galleryDragSelectionBaseline;
    finishGallerySelectionDrag();
  }
  galleryPinchSurface.value?.style.setProperty("touch-action", "none");
  event.preventDefault();
}

function moveGalleryPinch(event: PointerEvent): void {
  const next = pinchPointerMove(galleryZoom, event);
  if (isPinching(galleryZoom)) event.preventDefault();
  if (next === null) return;
  galleryColumns.value = next;
  saveMobileGalleryColumns(next);
  const bound =
    next === MOBILE_GALLERY_COLUMNS_MIN
      ? " Largest."
      : next === MOBILE_GALLERY_COLUMNS_MAX
        ? " Smallest."
        : "";
  galleryZoomAnnouncement.value = `Thumbnails ${next} across.${bound}`;
}

function endGalleryPinch(event: PointerEvent): void {
  if (!tracksPointer(galleryZoom, event.pointerId)) return;
  // A finger lifted from a pinch can still earn a WKWebView compatibility click
  // on whatever tile it rested on, which would open that print or flip its
  // selection. Claim exactly one, and only for the finger that ends the pinch;
  // `pointercancel` claims none, since no compatibility click follows it. The
  // claim is provisional — WebKit usually synthesizes no click at all once a
  // second touch lands, so the next fresh touch sequence discards it.
  if (event.type === "pointerup" && isPinching(galleryZoom)) galleryPinchPendingClicks = 1;
  pinchPointerUp(galleryZoom, event.pointerId);
  if (galleryZoom.points.size === 0)
    galleryPinchSurface.value?.style.removeProperty("touch-action");
}

function selectAllGalleryPrints(): void {
  gallerySelection.value = new Set(gallery.value.map(galleryPrintKey));
  galleryDeleteConfirming.value = false;
}

function handleGalleryTileClick(event: MouseEvent, print: GalleryPrint): void {
  if (galleryPinchPendingClicks > 0 && event.detail !== 0) {
    galleryPinchPendingClicks -= 1;
    return;
  }
  // WKWebView may delay compatibility clicks until after another tap has
  // already completed. Consume one click for every pointer gesture instead
  // of keeping a single flag that the first delayed click can clear.
  if (gallerySelectMode.value && galleryDragPendingClicks > 0 && event.detail !== 0) {
    galleryDragPendingClicks -= 1;
    return;
  }
  if (gallerySelectMode.value) toggleGallerySelection(print);
  else openPrint(print);
}

async function deleteSelectedGalleryPrints(): Promise<void> {
  if (galleryDeleting.value || gallerySelection.value.size === 0) return;
  if (!galleryDeleteConfirming.value) {
    galleryDeleteConfirming.value = true;
    return;
  }
  galleryDeleteConfirming.value = false;
  galleryDeleting.value = true;
  galleryError.value = "";
  organizationError.value = "";

  const kind = selectedDeleteKind.value;
  const selected = selectedRepresentatives();
  const copies = selectedPhysicalCopies();
  // One op per host: the trash on hosts that have one, today's hard
  // `DELETE` elsewhere, `?permanent=true` only from the Trash scope.
  const result = await runOrganizationFanout(
    planOrganizationFanout(
      copies,
      kind === "delete-forever" ? { kind: "deleteForever" } : { kind: "trash" },
    ),
    fanoutHosts(),
    undefined,
    { trashHostIds: librarySupport.value.trashHostIds },
  );
  const failedHostIds = new Set(result.failures.map((failure) => failure.hostId));
  const removedKeys = new Set(
    copies.filter((copy) => !failedHostIds.has(copy.hostId)).map(galleryPrintKey),
  );
  const removedCopies = copies.filter((copy) => removedKeys.has(galleryPrintKey(copy)));
  // Removed copies leave the offline cache (metadata and thumbnail bytes);
  // the Trash grid repaints trashed prints from the host's own listing.
  await removeCachedGalleryPrints(
    removedCopies.map((copy) => ({ hostId: copy.cacheKey, filename: copy.filename })),
  );
  const trashedCopies =
    kind === "trash"
      ? removedCopies.filter((copy) => librarySupport.value.trashHostIds.has(copy.hostId))
      : [];
  for (const print of gallery.value) revokeObjectUrl(print.thumbnailUrl);
  galleryCopies = galleryCopies.filter((print) => !removedKeys.has(galleryPrintKey(print)));
  trashCopies = trashCopies.filter((print) => !removedKeys.has(galleryPrintKey(print)));
  rebuildGalleryOrganization();
  if (trashedCopies.length > 0) {
    // The Trash listing is host authority (it carries the purge stamps): mark
    // it stale so the next Trash visit refetches rather than guessing.
    trashLoaded.value = false;
    trashCount.value += groupLogicalGalleryPrints(trashedCopies).length;
  }
  gallery.value = [];
  pendingGallery = visibleRepresentatives();
  await loadMoreGalleryPage();

  const failedPrints = selected.filter((print) =>
    logicalCopiesOf(copies, print).some((copy) => failedHostIds.has(copy.hostId)),
  ).length;
  const donePrints = selected.length - failedPrints;
  if (failedPrints > 0) {
    const verb =
      kind === "trash"
        ? `Moved ${donePrints} of ${selected.length} prints to the trash.`
        : kind === "delete-forever"
          ? `Deleted ${donePrints} of ${selected.length} prints forever.`
          : `Deleted ${donePrints} of ${selected.length} prints everywhere.`;
    galleryError.value = `${verb} ${failedPrints} still have a copy on an unavailable device.`;
  }
  gallerySelection.value = new Set(
    [...gallerySelection.value].filter((key) =>
      allGalleryPrints().some((print) => galleryPrintKey(print) === key),
    ),
  );
  if (gallerySelection.value.size === 0) setGallerySelectMode(false);
  galleryDeleting.value = false;
}

function isFreshMobilePrint(print: GalleryPrint): boolean {
  const seenAt = librarySeenAtBaseline[print.hostId];
  return libraryPreviouslyVisited && seenAt != null && print.timestamp > seenAt;
}

function markMobileLibrarySeen(prints: Array<GalleryPrint | PendingGalleryPrint>): void {
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
  reusePrintEpoch += 1;
  reusePrintController?.abort();
  reusePrintController = null;
  reusingPrint.value = false;
  sourceUseEpoch += 1;
  sourceUseController?.abort();
  sourceUseController = null;
  usingPrintAsSource.value = false;
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

watch(gallerySentinel, (sentinel) => {
  gallerySentinelObserver?.disconnect();
  gallerySentinelObserver = null;
  gallerySentinelVisible.value = false;
  if (!sentinel) return;
  gallerySentinelObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) gallerySentinelVisible.value = entry.isIntersecting;
      if (gallerySentinelVisible.value) {
        galleryChainedFetches = 0;
        void loadMoreGallery();
      }
    },
    { root: mobileContent.value, rootMargin: "600px 0px" },
  );
  gallerySentinelObserver.observe(sentinel);
});

// Failed thumbnails can leave the sentinel in view without producing another
// observer event. Advance a bounded number of pages so valid older prints
// behind that failed page still become reachable.
watch(galleryLoadingMore, (loading) => {
  if (loading || !gallerySentinelVisible.value || galleryRemaining.value === 0) return;
  if (galleryChainedFetches >= MAX_GALLERY_CHAINED_FETCHES) return;
  galleryChainedFetches += 1;
  void loadMoreGallery();
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
  // Suspending mid-touch never delivers the matching pointerup, and a phantom
  // finger would make the next single touch read as a pinch — resizing the grid
  // from a plain scroll. Nothing can still be held after a resume, so drop them.
  galleryPinchPendingClicks = 0;
  resetPinch(galleryZoom, galleryColumns.value);
  galleryPinchSurface.value?.style.removeProperty("touch-action");
  if ("__TAURI_INTERNALS__" in window) {
    void invoke("restore_mobile_viewport").catch(() => undefined);
  }
  probeHosts();
  void refreshMobileActivity();
  if (modelLoadError.value && !loadingModels.value) void refreshModels();
  renewGeneratedResult(false);
  if (tab.value === "gallery") void refreshGallery();
}

function usesSoftwareKeyboard(target: EventTarget | null): target is HTMLElement {
  if (!(target instanceof HTMLElement)) return false;
  if (target.matches("textarea, select, [contenteditable='true']")) return true;
  if (!(target instanceof HTMLInputElement)) return false;
  return !NON_KEYBOARD_INPUT_TYPES.has(target.type.toLowerCase());
}

function syncVisualViewportOffset(): void {
  const pageTop = window.visualViewport?.pageTop ?? window.scrollY;
  document.documentElement.style.setProperty(
    "--mobile-visual-viewport-page-top",
    `${Math.max(0, pageTop)}px`,
  );
}

function restoreNativeViewport(): void {
  if (!("__TAURI_INTERNALS__" in window) || unmounted) return;
  // Keyboard avoidance may scroll WebKit's document layer despite Mold's
  // non-scrolling root. Reset it without disturbing .mobile-content, whose
  // independent scroll position belongs to the user.
  window.scrollTo(0, 0);
  syncVisualViewportOffset();
  void invoke("restore_mobile_viewport").catch(() => undefined);
}

function cancelKeyboardViewportRestore(): void {
  if (!keyboardViewportRestoreTimer) return;
  clearTimeout(keyboardViewportRestoreTimer);
  keyboardViewportRestoreTimer = null;
}

function handleKeyboardFocusIn(event: FocusEvent): void {
  if (!usesSoftwareKeyboard(event.target)) return;
  const editor = event.target;
  cancelKeyboardViewportRestore();
  // iOS pans WKWebView after focus to reveal an editor. Re-anchor after the
  // focus turn and once more after the keyboard animation; the Create body's
  // own overflow remains the only scrollable app layer.
  queueMicrotask(() => {
    if (unmounted || document.activeElement !== editor) return;
    restoreNativeViewport();
    keyboardViewportRestoreTimer = setTimeout(() => {
      keyboardViewportRestoreTimer = null;
      if (document.activeElement !== editor) return;
      restoreNativeViewport();
    }, KEYBOARD_VIEWPORT_SETTLE_MS);
  });
}

/**
 * WKWebView can keep the keyboard-reduced presentation frame after an editor
 * blurs. Repair once after focus has moved, then again after UIKit's keyboard
 * dismissal animation so its final layout pass cannot reinstate the gap.
 */
function handleKeyboardFocusOut(event: FocusEvent): void {
  if (!usesSoftwareKeyboard(event.target)) return;
  queueMicrotask(() => {
    if (unmounted || usesSoftwareKeyboard(document.activeElement)) return;
    restoreNativeViewport();
    cancelKeyboardViewportRestore();
    keyboardViewportRestoreTimer = setTimeout(() => {
      keyboardViewportRestoreTimer = null;
      if (usesSoftwareKeyboard(document.activeElement)) return;
      restoreNativeViewport();
    }, KEYBOARD_VIEWPORT_SETTLE_MS);
  });
}

watch(resultPreviewError, (error) => {
  if (!error) return;
  generationAnnouncement.value = `Generation completed, but its preview is unavailable. ${error}`;
});

onMounted(async () => {
  document.addEventListener("focusin", handleKeyboardFocusIn, true);
  document.addEventListener("focusout", handleKeyboardFocusOut, true);
  window.addEventListener("scroll", syncVisualViewportOffset, true);
  window.addEventListener("pointermove", moveGallerySelectionDrag, { passive: false });
  window.addEventListener("pointerup", finishGallerySelectionDrag);
  window.addEventListener("pointercancel", finishGallerySelectionDrag);
  // The pinch tracks globally so a finger that slides off the grid mid-gesture
  // still reports, and so a lift outside the grid always ends it.
  window.addEventListener("pointermove", moveGalleryPinch, { passive: false });
  window.addEventListener("pointerup", endGalleryPinch);
  window.addEventListener("pointercancel", endGalleryPinch);
  window.visualViewport?.addEventListener("resize", syncVisualViewportOffset);
  window.visualViewport?.addEventListener("scroll", syncVisualViewportOffset);
  syncVisualViewportOffset();
  if ("__TAURI_INTERNALS__" in window) {
    void invoke("restore_mobile_viewport").catch(() => undefined);
    void import("@tauri-apps/api/app")
      .then(({ getVersion }) => getVersion())
      .then((version) => {
        appVersion.value = version;
      })
      .catch(() => {});
  }
  await hydrateApiKeys();
  if (unmounted) return;
  if ("__TAURI_INTERNALS__" in window) {
    try {
      await listenForPairingDeepLinks();
    } catch (error) {
      hostError.value = describeTransportError(error, "Mobile pairing");
    }
  }
  if (unmounted) return;
  recoverMobileSequence();
  // Start the cadence before awaiting individual tailnet hosts. One slow host
  // must not prevent every other saved host from being probed on schedule.
  hostProbeTimer = setInterval(probeHosts, 10_000);
  liveActivityTimer = setTimeout(pollMobileActivity, 0);
  document.addEventListener("visibilitychange", handleForegroundResume);
  window.addEventListener("pageshow", handleForegroundResume);
  if (selectedHost.value) {
    await Promise.all([
      refreshModels(),
      ...connectedHosts.value
        .filter((host) => host.id !== selectedHostId.value)
        .map((host) => probeHost(host)),
    ]);
    // Peers answer /api/models only after their probe lands, which is what
    // makes the automatic policies model-aware on the first Develop.
    if (automaticRouting.value) void refreshRoutingModels();
    void refreshMobileActivity();
  } else {
    tab.value = "hosts";
  }
});

onBeforeUnmount(() => {
  unmounted = true;
  reusePrintEpoch += 1;
  reusePrintController?.abort();
  sourceUseEpoch += 1;
  sourceUseController?.abort();
  if (pairingScannerOpen.value) {
    pairingScannerCancelled = true;
    void cancelBarcodeScanner().catch(() => undefined);
  }
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  sequenceSubmissionGuard.invalidate();
  const sequenceCancellation = sequenceCancellationRequest;
  sequenceCancellationRequest = null;
  if (sequenceCancellation) {
    void confirmCancellation(sequenceCancellation).catch(() => {});
  }
  submissionUiId += 1;
  recoveryRetryId += 1;
  expansionPullRequestId += 1;
  clearExpansionRecovery();
  document.removeEventListener("visibilitychange", handleForegroundResume);
  document.removeEventListener("focusin", handleKeyboardFocusIn, true);
  document.removeEventListener("focusout", handleKeyboardFocusOut, true);
  window.removeEventListener("scroll", syncVisualViewportOffset, true);
  window.removeEventListener("pointermove", moveGallerySelectionDrag);
  window.removeEventListener("pointerup", finishGallerySelectionDrag);
  window.removeEventListener("pointercancel", finishGallerySelectionDrag);
  finishGallerySelectionDrag();
  window.removeEventListener("pointermove", moveGalleryPinch);
  window.removeEventListener("pointerup", endGalleryPinch);
  window.removeEventListener("pointercancel", endGalleryPinch);
  resetPinch(galleryZoom, galleryColumns.value);
  window.visualViewport?.removeEventListener("resize", syncVisualViewportOffset);
  window.visualViewport?.removeEventListener("scroll", syncVisualViewportOffset);
  document.documentElement.style.removeProperty("--mobile-visual-viewport-page-top");
  window.removeEventListener("pageshow", handleForegroundResume);
  cancelKeyboardViewportRestore();
  if (resultMediaRecoveryTimer !== null) {
    clearTimeout(resultMediaRecoveryTimer);
    resultMediaRecoveryTimer = null;
  }
  stopPairingDeepLinks?.();
  stopPairingDeepLinks = null;
  if (hostProbeTimer) clearInterval(hostProbeTimer);
  hostProbeTimer = null;
  if (liveActivityTimer) clearInterval(liveActivityTimer);
  liveActivityTimer = null;
  gallerySentinelObserver?.disconnect();
  gallerySentinelObserver = null;
  stopSequenceTransport();
  for (const id of [...hostProbes.keys()]) cancelHostProbe(id);
  generation.resetJobs();
  for (const url of objectUrls) URL.revokeObjectURL(url);
});
</script>

<template>
  <main
    class="mobile-shell"
    :class="{ 'is-settings-open': settingsOpen, 'is-pair-scanning': pairingScannerOpen }"
  >
    <section
      v-if="pairingScannerOpen"
      class="mobile-pair-scanner"
      role="dialog"
      aria-modal="true"
      aria-labelledby="mobile-pair-scanner-title"
      data-test="mobile-pair-scanner"
    >
      <header class="mobile-pair-scanner-head">
        <div>
          <span class="mobile-pair-scanner-kicker">Mold pairing</span>
          <h1 id="mobile-pair-scanner-title">Scan pairing code</h1>
        </div>
        <button
          class="mobile-pair-scanner-cancel"
          type="button"
          data-test="mobile-pair-scanner-cancel"
          @click="cancelPairingScan"
        >
          Cancel
        </button>
      </header>
      <div class="mobile-pair-scanner-stage" aria-hidden="true">
        <div class="mobile-pair-scanner-frame">
          <span class="mobile-pair-scanner-corner corner-top-left" />
          <span class="mobile-pair-scanner-corner corner-top-right" />
          <span class="mobile-pair-scanner-corner corner-bottom-left" />
          <span class="mobile-pair-scanner-corner corner-bottom-right" />
        </div>
      </div>
      <footer class="mobile-pair-scanner-foot">
        <strong>Point your camera at the QR code</strong>
        <span>On your host, open Settings → Mobile pairing.</span>
      </footer>
    </section>
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
        <div class="host-chip">{{ headerTargetLabel }}</div>
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
    <p class="sr-only" aria-live="polite" aria-atomic="true" data-test="mobile-gallery-zoom-status">
      {{ galleryZoomAnnouncement }}
    </p>

    <section
      ref="mobileContent"
      class="mobile-content"
      :class="{ 'is-library': !settingsOpen && tab === 'gallery' }"
    >
      <MobileSettingsView
        v-if="settingsOpen"
        :settings="mobileSettings"
        :host-count="hosts.length"
        :app-version="appVersion"
        :host="selectedHost ?? null"
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
          <div class="mobile-create-head">
            <h1 class="section-title">Create</h1>
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
          <p class="section-note">{{ developOnNote }}</p>
          <label v-if="connectedHosts.length > 1" class="field">
            <span>Host</span>
            <select
              class="control"
              :value="generateTarget"
              data-test="mobile-generate-host"
              @change="selectGenerateTarget(($event.target as HTMLSelectElement).value)"
            >
              <!-- Automatic policies appear only with two or more reachable
                   machines; with one there is nothing to choose between. -->
              <option v-if="autoRoutingAvailable" :value="AUTO_TARGET_ID">Auto</option>
              <option v-if="autoRoutingAvailable" :value="CAPABLE_TARGET_ID">Most capable</option>
              <option v-for="host in connectedHosts" :key="host.id" :value="host.id">
                {{ host.name }}{{ host.online ? "" : " · offline" }}
              </option>
            </select>
          </label>
          <p
            v-if="autoRoutingAvailable && automaticRouting"
            class="section-note"
            data-test="mobile-routing-hint"
          >
            {{ routingHint }}
          </p>
          <!-- Output is a FIELD of the Create form, not a mode pair pinned
               above it: One shot and Sequence share this whole stack. -->
          <div class="mobile-output-mode" data-test="mobile-output-mode">
            <span class="mobile-output-mode-label">Output</span>
            <SegmentedControl
              :model-value="draft.output"
              :options="OUTPUT_OPTIONS"
              label="Output"
              @update:model-value="setOutputMode"
            />
            <p v-if="isSequence" class="mobile-output-mode-hint">a sequence renders one timeline</p>
          </div>
          <label class="field">
            <span>{{ isSequence ? "Video model" : "Model" }}</span>
            <!-- Keyed on the output: the option universe changes wholesale
                 between One shot and Sequence, and a patched-in-place select
                 can keep the previous mode's selection highlighted. -->
            <select
              :key="`model-${draft.output}`"
              v-model="form.model"
              class="control"
              :disabled="loadingModels || pickerModels.length === 0"
              @change="changeModel"
            >
              <option v-if="!form.model" value="" disabled>
                {{ loadingModels ? "Loading models…" : "No generation models available" }}
              </option>
              <option v-if="form.model && !selectedModelAvailable" :value="form.model" disabled>
                {{ modelLabel(form.model) }} · not installed
              </option>
              <option v-for="model in pickerModels" :key="model.name" :value="model.name">
                {{ modelDisplayName(model)
                }}{{
                  modelAvailabilityTag(model.name) ? ` · ${modelAvailabilityTag(model.name)}` : ""
                }}
              </option>
            </select>
          </label>
          <ErrorNotice
            v-if="modelLoadError"
            class="mobile-model-state is-error"
            data-test="mobile-model-error"
            :message="modelLoadError"
          >
            <template #actions>
              <button
                class="secondary-button"
                type="button"
                data-test="mobile-model-retry"
                @click="refreshModels"
              >
                Retry
              </button>
            </template>
          </ErrorNotice>
          <div
            v-else-if="!loadingModels && generationModels.length === 0"
            class="mobile-model-state"
            data-test="mobile-model-empty"
          >
            <p>No downloaded generation model is available on {{ modelScopeLabel }}.</p>
            <button class="secondary-button" type="button" @click="openCatalog(selectedHost.id)">
              Open Catalog
            </button>
          </div>

          <template v-if="isSequence">
            <!-- The chain wire has no title slot (`ChainRequestWire`): the
                 Title field is hidden here instead of silently dropped. -->
            <p class="mobile-empty-note" data-test="mobile-sequence-title-note">
              Sequences don't carry a title — rename the stitched print in the Library.
            </p>
            <div
              v-if="sequenceModels.length === 0"
              class="mobile-sequence-empty"
              data-test="mobile-sequence-empty"
            >
              <strong>Sequences need a video model</strong>
              <p>
                Pull a chain-capable LTX Video or distilled LTX-2 checkpoint on
                {{ modelScopeLabel }}.
              </p>
              <button
                class="secondary-button"
                type="button"
                data-test="mobile-sequence-browse"
                @click="browseSequenceModels"
              >
                Browse video models
              </button>
            </div>
            <MobileSequenceComposer
              v-else
              :selected-model="selectedGenerationModel"
              :form="form"
              :upscalers="upscalers"
              :chain-limits="chainLimits"
              :target="generationTarget"
              :shared="sequenceParams(form, selectedGenerationModel)"
              :fps="form.fps"
              :submitting="sequenceStarting"
              :error="sequenceError"
              :settings-summary="sequenceSettingsSummary"
              :camera-controls="cameraControls"
              :camera-controls-loaded="cameraControlsLoaded"
              :camera-unsupported-reason="cameraUnsupportedReason"
              @submit="submitMobileSequence"
              @cancel="cancelMobileSequenceSubmission"
            >
              <template #settings>
                <MobileSharedParams
                  :form="form"
                  :model="selectedGenerationModel"
                  :last-seed="generation.lastSeedUsed"
                  :disabled="loadingModels"
                  show-fps
                  :steps-error="stepsError"
                  :guidance-error="guidanceError"
                  :canvas-intent="canvasIntent"
                  @resolution-validity="resolutionValid = $event"
                  @seed-validity="seedValid = $event"
                  @canvas-intent="setCanvasIntent"
                />
              </template>
            </MobileSequenceComposer>
          </template>
          <template v-else>
            <label class="field">
              <span>Title</span>
              <input
                id="mobile-print-title"
                v-model="printTitle"
                class="control"
                autocomplete="off"
                enterkeyhint="next"
                placeholder="Untitled print"
                data-test="mobile-create-title"
              />
            </label>
            <p
              v-if="printTitleError"
              class="status-line error-text"
              role="alert"
              data-test="mobile-create-title-error"
            >
              {{ printTitleError }}
            </p>
            <label class="field">
              <span>Prompt</span>
              <textarea
                id="mobile-prompt"
                v-model="form.prompt"
                class="control"
                :placeholder="promptFieldPlaceholder"
                @input="onPromptAuthored(($event.target as HTMLTextAreaElement).value)"
              />
            </label>
            <MobileStyleChips v-model="form.stylePreset" />
            <MobilePromptTools
              v-if="selectedTarget"
              :form="form"
              :model="selectedGenerationModel"
              :target="selectedTarget"
              :running="expansionRunning"
              :can-undo="quickExpansionOriginal !== null || remixUndo !== null"
              :blocked="!!preparedBatch || !!remixReview"
              :models="generationModels"
              :remix-source="remixSource"
              :remix-dimensions="remixDimensions"
              :task="currentExpansionTask"
              @expand="expandForCurrentBatch()"
              @remix="remixCurrent()"
              @undo="undoPromptPreparation"
              @update:remix-source="remixSource = $event"
              @update:remix-dimensions="remixDimensions = $event"
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
                  @click="recoverQuickPromptTransform"
                >
                  {{ appliedRemix ? "Re-remix" : "Re-expand and Develop" }}
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
                  @click="undoPromptPreparation"
                >
                  Restore original
                </button>
              </div>
            </div>
            <div
              v-if="expansionError && !expansionMissingModel && !preparedBatch && !remixReview"
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
            <MobileRemixReview
              v-if="remixReview"
              :source-kind="remixReview.sourceKind"
              :source-prompt="remixReview.sourcePrompt"
              :variants="remixReview.variants"
              :host-label="remixReview.route.label"
              :stale-reasons="remixStaleReasons"
              :running="expansionRunning"
              :error="expansionMissingModel ? '' : expansionError"
              @toggle="toggleRemixVariant"
              @edit="editRemixVariant"
              @reremix="remixCurrent()"
              @apply="applyRemixSelection"
              @restore="restoreRemixSource"
              @discard="discardRemixReview"
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
              @regenerate="replacePreparedPrompts(true)"
              @refresh="replacePreparedPrompts(false)"
              @discard="discardPreparedBatch"
              @generate="generate"
              @cancel="cancelGenerationSubmission"
            />
            <p
              v-if="mobileMediaBudgetError"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-media-budget-error"
            >
              {{ mobileMediaBudgetError }}
            </p>
            <p
              v-if="sourceConditioningError"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-source-conditioning-gate"
            >
              {{ sourceConditioningError }}
            </p>
            <p
              v-if="h3AuthoringError"
              class="mobile-generate-validation"
              role="alert"
              data-test="mobile-h3-authoring-error"
            >
              {{ h3AuthoringError }}
            </p>

            <!-- Source media in the primary form: the model dictates whether
                 (and how) it renders, exactly like resolutions. H3 FL2VA
                 boundaries render through the same component. -->
            <details
              v-if="form.model && showSourceMedia"
              class="mobile-native-disclosure"
              :open="
                !!(
                  form.sourceImage ||
                  form.endFrame ||
                  form.controlImage ||
                  form.imageAttachments.length
                ) ||
                caps.requiresSourceImage ||
                caps.sourceImageMode !== 'single'
              "
              data-test="mobile-source-disclosure"
            >
              <summary>
                <span>{{ sourceSectionTitle }}</span>
                <small>{{ sourceSectionSummary }}</small>
              </summary>
              <MobileSourceControls
                :form="form"
                :model="selectedGenerationModel"
                :target="generationTarget"
                :control-models="controlModels"
                :upscalers="upscalers"
                @validity-change="sourceValid = $event"
              />
            </details>

            <MobileSharedParams
              :form="form"
              :model="selectedGenerationModel"
              :duration-model="selectedGenerationModel"
              :last-seed="generation.lastSeedUsed"
              :disabled="loadingModels"
              :steps-error="stepsError"
              :guidance-error="guidanceError"
              :canvas-intent="canvasIntent"
              @resolution-validity="resolutionValid = $event"
              @seed-validity="seedValid = $event"
              @canvas-intent="setCanvasIntent"
            />

            <label
              v-if="caps.supportsAudio && !isMinimaxH3Identity(form.family, form.model)"
              class="mobile-generate-toggle-row"
              data-test="mobile-generate-audio-control"
            >
              <span>
                <strong>Generate audio</strong>
                <small>Include a synchronized soundtrack when the model supports it.</small>
              </span>
              <input
                v-model="form.enableAudio"
                type="checkbox"
                :disabled="selectedGenerationModel?.supports_audio === false"
                data-test="mobile-enable-audio"
              />
            </label>
            <p
              v-if="
                caps.supportsAudio &&
                !isMinimaxH3Identity(form.family, form.model) &&
                selectedGenerationModel?.supports_audio === false
              "
              class="mobile-generate-validation"
            >
              Audio assets are not included with this checkpoint. Video generation remains
              available.
            </p>

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
                <span>Advanced (sampler, LoRA, format)</span>
                <span
                  v-if="advancedActiveCount > 0"
                  class="mobile-advanced-trigger-badge"
                  data-test="mobile-advanced-trigger-count"
                  >{{ advancedActiveCount }}</span
                >
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
                :target="generationTarget"
                :upscalers="upscalers"
                :selected-model="selectedGenerationModel"
                :control-adapters="controlAdapters"
                :camera-controls="cameraControls"
                :camera-controls-loaded="cameraControlsLoaded"
                :camera-unsupported-reason="cameraUnsupportedReason"
                @validity-change="parameterValid = $event"
              />
              <label
                v-if="form.model && (caps.supportsNegativePrompt || form.negativePrompt.trim())"
                class="field"
              >
                <span>Negative prompt</span>
                <input
                  v-model="form.negativePrompt"
                  class="control"
                  placeholder="Optional"
                  :disabled="!caps.supportsNegativePrompt"
                />
                <small
                  v-if="!caps.supportsNegativePrompt"
                  data-test="mobile-negative-unavailable-hint"
                >
                  Saved for reuse, but this distilled recipe fixes CFG and does not use
                  negative-prompt guidance. Choose a Dev checkpoint with Auto or a guided pipeline
                  to enable it.
                </small>
              </label>
              <MobileLoraControls
                v-if="generationTarget"
                :form="form"
                :target="generationTarget"
                :model="selectedGenerationModel"
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
            <div v-if="generationStatusIsError" data-test="mobile-generation-summary">
              <ErrorNotice :message="generationStatus" data-test="mobile-generation-error" />
            </div>
            <div v-else class="status-line" data-test="mobile-generation-summary">
              {{ generationStatus }}
            </div>
            <ErrorNotice
              v-if="resultPreviewError"
              class="result-preview-error"
              :message="resultPreviewError"
            >
              <template #actions>
                <button class="secondary-button" type="button" @click="retryGeneratedPreview">
                  Try preview again
                </button>
              </template>
            </ErrorNotice>
            <video
              v-if="resultUrl && resultIsVideo"
              :key="`${latestResultJob?.clientId}:${resultMediaLoadKey}`"
              class="result-media"
              :src="resultUrl"
              controls
              playsinline
              preload="metadata"
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

          <!-- ONE queue for both outputs: single prints and durable sequences
               land in the same list (mockup 1c). -->
          <section
            v-if="activityRows.length || sharedMobileActivity.length"
            class="mobile-generation-queue"
            aria-label="Generation queue"
            data-test="mobile-generation-queue"
          >
            <div class="mobile-generation-queue-head">
              <h2>Queue</h2>
              <span data-test="mobile-queue-count">{{ activityCountLabel(activityCounts) }}</span>
            </div>
            <ol>
              <li
                v-for="row in activityRows"
                :key="row.key"
                class="mobile-generation-job"
                :data-test="row.print ? 'mobile-generation-job' : 'mobile-sequence-job'"
                role="button"
                tabindex="0"
                @click="row.print ? selectMobilePrint(row.print) : selectCurrentMobileSequence()"
                @keydown.enter.prevent="
                  row.print ? selectMobilePrint(row.print) : selectCurrentMobileSequence()
                "
              >
                <template v-if="row.print">
                  <div class="mobile-generation-job-copy">
                    <p>{{ row.print.prompt }}</p>
                    <span>{{ modelLabel(row.print.model) }} · {{ row.print.hostLabel }}</span>
                  </div>
                  <div class="mobile-generation-job-action">
                    <span data-test="mobile-generation-status">{{ activityRowStatus(row) }}</span>
                    <button
                      class="mobile-generation-cancel"
                      type="button"
                      :aria-label="
                        row.print.cancelling
                          ? `Cancelling ${row.print.prompt}`
                          : `Cancel ${row.print.prompt}`
                      "
                      data-test="mobile-generation-cancel"
                      :disabled="row.print.cancelling"
                      @click.stop="cancelGeneration(row.print)"
                    >
                      {{ row.print.cancelling ? "Cancelling…" : "Cancel" }}
                    </button>
                  </div>
                </template>
                <template v-else-if="row.sequence">
                  <div class="mobile-generation-job-copy">
                    <p>
                      {{ modelLabel(row.sequence.model) || "Sequence" }} ·
                      {{ row.sequence.stageCount }} clips
                    </p>
                    <span>
                      {{ row.sequence.phase ?? row.sequence.state }} · clip
                      {{ Math.min(row.sequence.currentStage + 1, row.sequence.stageCount) }}/{{
                        row.sequence.stageCount
                      }}
                      <template v-if="sequenceRowProgress !== null">
                        · {{ sequenceRowProgress }}%
                      </template>
                    </span>
                    <button
                      v-if="row.sequence.error"
                      type="button"
                      class="mobile-sequence-row-error"
                      :class="{
                        'mobile-sequence-row-error--expanded': expandedQueueFailures.has(row.key),
                      }"
                      data-test="mobile-sequence-error-disclosure"
                      :aria-expanded="expandedQueueFailures.has(row.key)"
                      @click.stop="toggleQueueFailure(row.key)"
                    >
                      <span>{{ row.sequence.error }}</span>
                      <span aria-hidden="true">
                        {{ expandedQueueFailures.has(row.key) ? "Less" : "Details" }}
                      </span>
                    </button>
                  </div>
                  <div class="mobile-generation-job-action">
                    <span data-test="mobile-sequence-status">{{ row.sequence.hostLabel }}</span>
                    <button
                      v-if="row.sequence.actions.includes('cancel')"
                      class="mobile-generation-cancel"
                      type="button"
                      data-test="mobile-sequence-cancel"
                      @click.stop="cancelMobileSequence"
                    >
                      Cancel
                    </button>
                    <button
                      v-else-if="row.sequence.actions.includes('resume')"
                      class="mobile-generation-cancel mobile-sequence-resume"
                      type="button"
                      data-test="mobile-sequence-resume"
                      @click.stop="resumeMobileSequence"
                    >
                      Resume
                    </button>
                    <button
                      v-if="row.sequence.actions.includes('delete')"
                      class="mobile-generation-cancel mobile-sequence-dismiss"
                      type="button"
                      data-test="mobile-sequence-dismiss"
                      @click.stop="dismissMobileSequence"
                    >
                      Dismiss
                    </button>
                  </div>
                </template>
              </li>
            </ol>
            <LiveActivityList
              :rows="sharedMobileActivity"
              interactive
              @select="openMobileLiveWork"
            />
          </section>
          <p
            v-if="sequenceError && sequenceRoute && !isSequence"
            class="status-line error-text"
            role="alert"
            data-test="mobile-sequence-route-error"
          >
            {{ sequenceError }}
          </p>
        </template>
      </template>

      <template v-else-if="tab === 'gallery'">
        <div class="mobile-library-heading">
          <div>
            <h1 class="section-title">Library</h1>
            <p class="section-note" data-test="mobile-library-note">
              {{
                gallerySelectMode
                  ? `${gallerySelection.size} selected`
                  : libraryScope === "trash"
                    ? `${trashCount} in trash · Restore or delete forever from Select`
                    : libraryScope === "collections"
                      ? `${libraryCollectionCards.length} collection${libraryCollectionCards.length === 1 ? "" : "s"} across your hosts`
                      : "Prints from every connected host · Pinch to resize · Tap Select for multiple"
              }}
            </p>
          </div>
          <div class="mobile-library-heading-actions">
            <button
              v-if="libraryScope === 'trash' && !gallerySelectMode"
              class="secondary-button mobile-library-select mobile-library-empty-trash"
              type="button"
              :class="{ 'is-armed': emptyTrashConfirming }"
              :disabled="emptyingTrash || trashCount === 0"
              data-test="mobile-library-empty-trash"
              @click="emptyTrash"
              @blur="emptyTrashConfirming = false"
            >
              {{ emptyingTrash ? "Emptying…" : emptyTrashConfirming ? "Confirm" : "Empty trash" }}
            </button>
            <button
              v-if="libraryScope === 'collections' && !activeCollection && !gallerySelectMode"
              class="secondary-button mobile-library-select"
              type="button"
              aria-label="New collection"
              data-test="mobile-library-new-collection"
              @click="openLibrarySheet({ kind: 'new-collection' })"
            >
              <span aria-hidden="true">＋</span>
            </button>
            <button
              v-if="libraryScope !== 'collections' || activeCollection"
              class="secondary-button mobile-library-select"
              type="button"
              :aria-pressed="gallerySelectMode"
              data-test="mobile-gallery-select"
              @click="setGallerySelectMode(!gallerySelectMode)"
            >
              {{ gallerySelectMode ? "Done" : "Select" }}
            </button>
          </div>
        </div>
        <p v-if="emptyTrashConfirming" class="status-line" data-test="mobile-library-empty-prompt">
          Delete everything in the trash forever?
        </p>
        <div
          v-if="libraryScopes.length > 1"
          class="mobile-library-scope"
          role="radiogroup"
          aria-label="Library scope"
          data-test="mobile-library-scope"
        >
          <button
            v-for="scope in libraryScopes"
            :key="scope"
            type="button"
            role="radio"
            :aria-checked="libraryScope === scope"
            :data-on="libraryScope === scope ? 'true' : undefined"
            :data-test="`mobile-library-scope-${scope}`"
            @click="setLibraryScope(scope)"
          >
            <span>{{ MOBILE_LIBRARY_SCOPE_LABELS[scope] }}</span>
            <span class="mobile-library-scope-count">{{ libraryScopeCounts[scope] }}</span>
          </button>
        </div>
        <p v-if="galleryError" class="status-line error-text">{{ galleryError }}</p>
        <p
          v-if="organizationError"
          class="status-line error-text"
          role="alert"
          data-test="mobile-library-organization-error"
        >
          {{ organizationError }}
        </p>
        <div
          v-if="libraryChipRowVisible"
          class="mobile-library-chips"
          role="group"
          aria-label="Library filters"
          data-test="mobile-library-chips"
        >
          <button
            v-if="libraryOrganizeEnabled"
            class="mobile-library-chip"
            type="button"
            :aria-pressed="libraryFilters.favoritesOnly"
            :data-on="libraryFilters.favoritesOnly ? 'true' : undefined"
            data-test="mobile-library-chip-favorites"
            @click="toggleFavoritesFilter"
          >
            <span class="mobile-library-chip-heart" aria-hidden="true">♥</span>Favorites
          </button>
          <button
            v-for="tag in libraryTagChips.visible"
            :key="`tag-${tag.name}`"
            class="mobile-library-chip"
            type="button"
            :aria-pressed="libraryFilters.tag === tagKey(tag.name)"
            :data-on="libraryFilters.tag === tagKey(tag.name) ? 'true' : undefined"
            data-test="mobile-library-chip-tag"
            @click="setTagFilter(tag.name)"
          >
            {{ tag.name }}<span class="mobile-library-chip-count">{{ tag.count }}</span>
          </button>
          <button
            v-if="libraryTagChips.overflow.length"
            class="mobile-library-chip is-more"
            type="button"
            data-test="mobile-library-chip-more"
            @click="openLibrarySheet({ kind: 'more-tags' })"
          >
            More…
          </button>
          <button
            v-for="host in libraryHostChips"
            :key="`host-${host.id}`"
            class="mobile-library-chip is-host"
            type="button"
            :aria-pressed="libraryFilters.hostId === host.id"
            :data-on="libraryFilters.hostId === host.id ? 'true' : undefined"
            data-test="mobile-library-chip-host"
            @click="setHostFilter(host.id)"
          >
            {{ host.name }}<span class="mobile-library-chip-count">{{ host.count }}</span>
          </button>
        </div>

        <template v-if="libraryScope === 'collections' && !activeCollection">
          <ul
            v-if="libraryCollectionCards.length"
            class="mobile-collection-list"
            data-test="mobile-collection-list"
          >
            <li
              v-for="card in libraryCollectionCards"
              :key="card.slug"
              class="mobile-collection-item"
              :data-test="`mobile-collection-${card.slug}`"
            >
              <button
                class="mobile-collection-row"
                type="button"
                :aria-label="`Open ${card.name}, ${card.count} prints`"
                data-test="mobile-collection-open"
                @click="openCollection(card.slug)"
                @contextmenu.prevent="openCollectionMenu(card.slug)"
              >
                <span class="mobile-collection-cover" aria-hidden="true">
                  <img v-if="collectionCoverUrl(card)" :src="collectionCoverUrl(card)" alt="" />
                </span>
                <span class="mobile-collection-copy">
                  <strong>{{ card.name }}</strong>
                  <span
                    ><span class="mobile-collection-count">{{ card.count }}</span>
                    <template v-if="card.hostsLabel"> · {{ card.hostsLabel }}</template></span
                  >
                </span>
                <span class="mobile-collection-chevron" aria-hidden="true">›</span>
              </button>
              <button
                class="mobile-collection-menu-button"
                type="button"
                :aria-label="`More actions for ${card.name}`"
                :aria-expanded="collectionMenuSlug === card.slug"
                data-test="mobile-collection-menu"
                @click="openCollectionMenu(card.slug)"
              >
                <span aria-hidden="true">…</span>
              </button>
              <div
                v-if="collectionMenuSlug === card.slug"
                class="mobile-collection-actions"
                data-test="mobile-collection-actions"
              >
                <p v-if="collectionDeleteConfirmSlug === card.slug" class="status-line">
                  Delete collection “{{ card.name }}”? Its prints stay in the Library.
                </p>
                <div class="row-actions">
                  <button
                    class="secondary-button"
                    type="button"
                    data-test="mobile-collection-rename"
                    @click="
                      openLibrarySheet({
                        kind: 'rename-collection',
                        slug: card.slug,
                        name: card.name,
                      })
                    "
                  >
                    Rename
                  </button>
                  <button
                    class="secondary-button mobile-inline-danger"
                    type="button"
                    :disabled="organizationBusy"
                    data-test="mobile-collection-delete"
                    @click="deleteCollection(card)"
                  >
                    {{
                      collectionDeleteConfirmSlug === card.slug ? "Confirm" : "Delete collection"
                    }}
                  </button>
                  <button
                    v-if="collectionDeleteConfirmSlug === card.slug"
                    class="secondary-button"
                    type="button"
                    @click="collectionDeleteConfirmSlug = null"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </li>
            <li class="mobile-collection-item">
              <button
                class="mobile-collection-row is-new"
                type="button"
                data-test="mobile-collection-new-row"
                @click="openLibrarySheet({ kind: 'new-collection' })"
              >
                <span class="mobile-collection-cover is-new" aria-hidden="true">＋</span>
                <span class="mobile-collection-copy">
                  <strong>New collection</strong>
                  <span>Name it, then add prints from Select</span>
                </span>
              </button>
            </li>
          </ul>
          <div v-else class="empty-state" data-test="mobile-collections-empty">
            <p>No collections yet.</p>
            <button
              class="secondary-button"
              type="button"
              data-test="mobile-collection-new-row"
              @click="openLibrarySheet({ kind: 'new-collection' })"
            >
              New collection
            </button>
          </div>
        </template>
        <template v-else>
          <div
            v-if="libraryScope === 'collections' && activeCollection"
            class="mobile-collection-drillin"
            data-test="mobile-collection-drillin"
          >
            <button
              class="mobile-back-button"
              type="button"
              data-test="mobile-collection-back"
              @click="closeCollection"
            >
              <span aria-hidden="true">‹</span> Collections
            </button>
            <div class="mobile-collection-drillin-title">
              <strong>{{ activeCollection.name }}</strong>
              <span
                ><span class="mobile-collection-count">{{ activeCollection.count }}</span>
                <template v-if="activeCollection.hostsLabel">
                  · {{ activeCollection.hostsLabel }}</template
                ></span
              >
            </div>
          </div>
          <div
            v-if="libraryScope === 'trash' && trashRetention.segments.length"
            class="mobile-library-banner"
            data-test="mobile-library-trash-banner"
          >
            <p>
              <template v-for="(segment, index) in trashRetention.segments" :key="index">
                <span v-if="segment.mono" class="mobile-library-mono">{{ segment.text }}</span>
                <template v-else>{{ segment.text }}</template>
              </template>
            </p>
            <button
              class="mobile-library-banner-link"
              type="button"
              data-test="mobile-library-retention-link"
              @click="tab = 'hosts'"
            >
              Change · Machines
            </button>
          </div>
          <p
            v-if="libraryScope === 'trash' && trashError"
            class="status-line error-text"
            data-test="mobile-library-trash-error"
          >
            {{ trashError }}
          </p>
          <div
            ref="galleryPinchSurface"
            class="mobile-gallery-pinch-surface"
            data-test="mobile-gallery-pinch-surface"
            @pointerdown="beginGalleryPinch"
          >
            <div
              v-if="galleryLoading || (libraryScope === 'trash' && trashLoading && !gallery.length)"
              class="empty-state"
            >
              {{ libraryScope === "trash" ? "Loading trash…" : "Loading prints…" }}
            </div>
            <div
              v-else-if="gallery.length"
              class="gallery-grid"
              :class="{ 'is-selecting': gallerySelectMode }"
              :style="{ '--mobile-gallery-columns': galleryColumns }"
              :aria-label="`Prints, ${galleryColumns} across. Pinch to resize.`"
              :data-gallery-columns="galleryColumns"
              role="group"
              data-test="mobile-gallery-grid"
            >
              <button
                v-for="print in gallery"
                :key="`${print.hostId}:${print.filename}`"
                class="gallery-item"
                :class="{ 'gallery-item-selected': gallerySelection.has(galleryPrintKey(print)) }"
                type="button"
                :aria-label="
                  gallerySelectMode
                    ? tileLabel(
                        print,
                        gallerySelection.has(galleryPrintKey(print)) ? 'Deselect' : 'Select',
                      )
                    : tileLabel(print, 'Open')
                "
                :aria-pressed="
                  gallerySelectMode ? gallerySelection.has(galleryPrintKey(print)) : undefined
                "
                :data-gallery-print-key="galleryPrintKey(print)"
                data-test="gallery-item"
                @pointerdown="beginGallerySelectionDrag($event, print)"
                @click="handleGalleryTileClick($event, print)"
              >
                <img
                  :src="print.thumbnailUrl"
                  :alt="print.metadata.prompt || print.filename"
                  loading="lazy"
                />
                <span
                  v-if="isVideoItem(print) || isAudioItem(print)"
                  class="gallery-video-badge"
                  aria-hidden="true"
                  >{{ isAudioItem(print) ? "♪" : "▶" }}</span
                >
                <span
                  v-if="!gallerySelectMode && libraryScope !== 'trash' && isFreshMobilePrint(print)"
                  class="gallery-new-badge"
                  data-test="new-badge"
                >
                  New
                </span>
                <span
                  v-if="isUpscaledImage(print)"
                  class="gallery-upscaled-badge"
                  data-test="upscaled-badge"
                >
                  Upscaled
                </span>
                <span
                  v-if="!gallerySelectMode && organizationOf(print)?.favorite"
                  class="gallery-favorite-badge"
                  data-test="favorite-badge"
                  aria-label="Favorite"
                  >♥</span
                >
                <span
                  v-if="libraryScope === 'trash' && purgeChipFor(print)"
                  class="gallery-purge-chip"
                  data-test="purge-chip"
                  >{{ purgeChipFor(print) }}</span
                >
                <span
                  v-if="gallerySelectMode"
                  class="gallery-selection-indicator"
                  data-test="mobile-gallery-selection-indicator"
                  aria-hidden="true"
                >
                  {{ gallerySelection.has(galleryPrintKey(print)) ? "✓" : "" }}
                </span>
              </button>
            </div>
            <div v-else class="empty-state" data-test="mobile-library-empty">
              {{ libraryEmptyCopy }}
            </div>
            <div
              v-if="!galleryLoading && galleryRemaining"
              ref="gallerySentinel"
              class="gallery-scroll-sentinel"
              data-test="mobile-gallery-sentinel"
              aria-live="polite"
            >
              {{ galleryLoadingMore ? "Loading older prints…" : "" }}
            </div>
          </div>
        </template>
        <div
          v-if="gallerySelectMode"
          class="mobile-gallery-actions"
          role="toolbar"
          aria-label="Library selection actions"
          data-test="mobile-gallery-actions"
        >
          <span>{{ galleryDeleteCopy.status }}</span>
          <button
            v-if="!galleryDeleteConfirming"
            type="button"
            :disabled="gallery.length === 0"
            @click="selectAllGalleryPrints"
          >
            All
          </button>
          <button
            v-if="!galleryDeleteConfirming"
            type="button"
            :disabled="gallerySelection.size === 0"
            @click="clearSelectedGalleryPrints"
          >
            Clear
          </button>
          <template
            v-if="!galleryDeleteConfirming && libraryScope !== 'trash' && libraryOrganizeEnabled"
          >
            <button
              type="button"
              :disabled="gallerySelection.size === 0 || organizationBusy"
              :aria-pressed="selectedAllFavorite"
              :aria-label="selectedAllFavorite ? 'Remove favorite' : 'Favorite'"
              data-test="mobile-gallery-favorite"
              @click="favoriteSelected"
            >
              <span aria-hidden="true">{{ selectedAllFavorite ? "♥" : "♡" }}</span>
            </button>
            <button
              type="button"
              :disabled="gallerySelection.size === 0"
              data-test="mobile-gallery-tag"
              @click="openLibrarySheet({ kind: 'tags' })"
            >
              Tag
            </button>
            <button
              type="button"
              :disabled="gallerySelection.size === 0"
              data-test="mobile-gallery-collect"
              @click="openLibrarySheet({ kind: 'collections' })"
            >
              Add to collection
            </button>
            <button
              v-if="activeCollection"
              type="button"
              :disabled="gallerySelection.size === 0 || organizationBusy"
              data-test="mobile-gallery-remove-from-collection"
              @click="removeSelectedFromCollection"
            >
              Remove
            </button>
          </template>
          <button
            v-if="!galleryDeleteConfirming && libraryScope === 'trash'"
            type="button"
            class="is-primary"
            :disabled="gallerySelection.size === 0 || galleryRestoring || organizationBusy"
            data-test="mobile-gallery-restore"
            @click="restoreSelectedGalleryPrints"
          >
            {{ galleryRestoring ? "Restoring…" : "Restore" }}
          </button>
          <button
            v-if="galleryDeleteConfirming"
            type="button"
            :disabled="galleryDeleting"
            @click="galleryDeleteConfirming = false"
          >
            Cancel
          </button>
          <button
            class="danger"
            type="button"
            :disabled="gallerySelection.size === 0 || galleryDeleting"
            data-test="mobile-gallery-delete"
            @click="deleteSelectedGalleryPrints"
          >
            {{ galleryDeleteCopy.button }}
          </button>
        </div>

        <MobileLibrarySheet
          :open="librarySheet?.kind === 'tags'"
          title="Tags"
          test-id="mobile-tag-sheet"
          @close="closeLibrarySheet"
        >
          <div class="mobile-library-tag-list" data-test="mobile-tag-sheet-current">
            <span v-if="selectedTags.length === 0" class="mobile-empty-note">No tags yet.</span>
            <span v-for="tag in selectedTags" :key="tag.name" class="mobile-library-tag">
              <span>{{ tag.name }}</span>
              <button
                type="button"
                :aria-label="`Remove tag ${tag.name}`"
                :disabled="librarySheetBusy"
                data-test="mobile-tag-sheet-remove"
                @click="removeTagFromSelected(tag.name)"
              >
                ×
              </button>
            </span>
          </div>
          <form
            class="mobile-library-sheet-form"
            @submit.prevent="addTagToSelected(librarySheetInput)"
          >
            <label class="field">
              <span>Add a tag</span>
              <input
                v-model="librarySheetInput"
                class="control"
                autocomplete="off"
                autocapitalize="off"
                enterkeyhint="done"
                placeholder="smurf"
                data-test="mobile-tag-sheet-input"
              />
            </label>
            <button
              class="primary-button"
              type="submit"
              :disabled="librarySheetBusy || !librarySheetInput.trim()"
              data-test="mobile-tag-sheet-add"
            >
              {{ librarySheetBusy ? "Saving…" : "Add" }}
            </button>
          </form>
          <div
            v-if="tagSuggestions.length"
            class="mobile-library-tag-list"
            data-test="mobile-tag-sheet-suggestions"
          >
            <button
              v-for="tag in tagSuggestions"
              :key="`suggest-${tag.name}`"
              class="mobile-library-chip"
              type="button"
              :disabled="librarySheetBusy"
              @click="addTagToSelected(tag.name)"
            >
              {{ tag.name }}<span class="mobile-library-chip-count">{{ tag.count }}</span>
            </button>
          </div>
        </MobileLibrarySheet>

        <MobileLibrarySheet
          :open="librarySheet?.kind === 'collections'"
          title="Add to collection"
          test-id="mobile-collection-sheet"
          @close="closeLibrarySheet"
        >
          <ul class="mobile-library-checklist" data-test="mobile-collection-sheet-list">
            <li v-for="card in libraryCollectionCards" :key="card.slug">
              <button
                type="button"
                role="checkbox"
                :aria-checked="selectedInCollection(card.slug)"
                :disabled="librarySheetBusy"
                data-test="mobile-collection-sheet-option"
                @click="toggleSelectedCollection(card)"
              >
                <span class="mobile-library-check" aria-hidden="true">{{
                  selectedInCollection(card.slug) ? "✓" : ""
                }}</span>
                <span class="mobile-collection-copy">
                  <strong>{{ card.name }}</strong>
                  <span
                    ><span class="mobile-collection-count">{{ card.count }}</span>
                    <template v-if="card.hostsLabel"> · {{ card.hostsLabel }}</template></span
                  >
                </span>
              </button>
            </li>
            <li v-if="libraryCollectionCards.length === 0" class="mobile-empty-note">
              No collections yet — name one below.
            </li>
          </ul>
          <form class="mobile-library-sheet-form" @submit.prevent="createCollectionFromSheet">
            <label class="field">
              <span>New collection</span>
              <input
                v-model="librarySheetInput"
                class="control"
                autocomplete="off"
                enterkeyhint="done"
                placeholder="Collection name"
                data-test="mobile-collection-sheet-input"
              />
            </label>
            <button
              class="primary-button"
              type="submit"
              :disabled="librarySheetBusy || !librarySheetInput.trim()"
              data-test="mobile-collection-sheet-create"
            >
              {{ librarySheetBusy ? "Saving…" : "New" }}
            </button>
          </form>
          <p v-if="librarySheetError" class="status-line error-text" role="alert">
            {{ librarySheetError }}
          </p>
        </MobileLibrarySheet>

        <MobileLibrarySheet
          :open="
            librarySheet?.kind === 'new-collection' || librarySheet?.kind === 'rename-collection'
          "
          :title="
            librarySheet?.kind === 'rename-collection' ? 'Rename collection' : 'New collection'
          "
          done-label="Cancel"
          test-id="mobile-collection-name-sheet"
          @close="closeLibrarySheet"
        >
          <form
            class="mobile-library-sheet-form"
            @submit.prevent="
              librarySheet?.kind === 'rename-collection'
                ? renameCollectionFromSheet()
                : createCollectionFromSheet()
            "
          >
            <label class="field">
              <span>Name</span>
              <input
                v-model="librarySheetInput"
                class="control"
                autocomplete="off"
                enterkeyhint="done"
                placeholder="Collection name"
                data-test="mobile-collection-name-input"
              />
            </label>
            <button
              class="primary-button"
              type="submit"
              :disabled="librarySheetBusy || !librarySheetInput.trim()"
              data-test="mobile-collection-name-submit"
            >
              {{
                librarySheetBusy
                  ? "Saving…"
                  : librarySheet?.kind === "rename-collection"
                    ? "Save"
                    : "Create"
              }}
            </button>
          </form>
          <p v-if="librarySheetError" class="status-line error-text" role="alert">
            {{ librarySheetError }}
          </p>
        </MobileLibrarySheet>

        <MobileLibrarySheet
          :open="librarySheet?.kind === 'more-tags'"
          title="All tags"
          test-id="mobile-more-tags-sheet"
          @close="closeLibrarySheet"
        >
          <div class="mobile-library-tag-list">
            <button
              v-for="tag in mergedTags"
              :key="`all-${tag.name}`"
              class="mobile-library-chip"
              type="button"
              :data-on="libraryFilters.tag === tagKey(tag.name) ? 'true' : undefined"
              data-test="mobile-more-tags-option"
              @click="setTagFilter(tag.name)"
            >
              {{ tag.name }}<span class="mobile-library-chip-count">{{ tag.count }}</span>
            </button>
          </div>
        </MobileLibrarySheet>
      </template>

      <template v-else-if="tab === 'hosts'">
        <MobileHostDetail
          v-if="hostDetail"
          :host="hostDetail"
          :active="hostDetail.id === selectedHostId"
          @back="hostDetailId = ''"
          @select="useHostForGenerations"
          @rename="renameHost"
          @disconnect="disconnectHost"
          @reconnect="reconnectHost"
          @forget="removeHost"
          @catalog="openCatalog"
          @status="updateHostStatus"
        />
        <template v-else>
          <h1 class="section-title">Machines</h1>
          <p class="section-note">LAN discovery, Tailscale MagicDNS, or an address</p>
          <!-- One line each: the Create picker offers these only while two or
               more machines are reachable. -->
          <p v-if="autoRoutingAvailable" class="section-note" data-test="mobile-machines-auto-hint">
            {{ MOBILE_AUTO_ROUTING_HINT }}
          </p>
          <p
            v-if="autoRoutingAvailable"
            class="section-note"
            data-test="mobile-machines-capable-hint"
          >
            {{ MOBILE_CAPABLE_ROUTING_HINT }}
          </p>
          <button
            class="primary-button mobile-pair-button"
            type="button"
            :disabled="pairing"
            data-test="mobile-scan-pairing"
            @click="scanPairingCode"
          >
            <span aria-hidden="true">▦</span>
            {{ pairing ? "Opening camera…" : "Scan pairing code" }}
          </button>
          <p class="mobile-pair-note">On your host, open Settings → Mobile pairing.</p>
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
                  <span
                    class="status-dot"
                    :class="host.connected !== false ? (host.online ? 'is-ready' : 'is-error') : ''"
                  />
                  <span class="host-chip">{{
                    host.connected === false
                      ? "disconnected"
                      : host.online
                        ? `v${host.version ?? ""}`
                        : "offline"
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
                :disabled="host.connected === false || host.id === selectedHostId"
                @click="useHostForGenerations(host.id)"
              >
                {{
                  host.connected === false
                    ? "Disconnected"
                    : host.id === selectedHostId
                      ? "Active"
                      : "Use host"
                }}
              </button>
            </div>
          </div>
        </template>
      </template>

      <KeepAlive>
        <MobileCatalogView
          v-if="!settingsOpen && tab === 'catalog'"
          :hosts="connectedHosts"
          :selected-host-id="catalogHostId"
          :filter-intent="catalogFilterIntent"
          @select-host="selectCatalogHost"
          @models-changed="catalogModelsChanged"
        />
      </KeepAlive>
    </section>

    <div
      v-if="!settingsOpen && tab === 'generate' && selectedHost && !isSequence && !preparedBatch"
      class="mobile-create-action"
      data-test="mobile-create-action"
    >
      <ActionBlocker
        v-if="developBlockerReason"
        :reason="developBlockerReason"
        compact
        data-test="mobile-develop-blocker"
      />
      <div class="mobile-estimate">
        <EstimateBadge :request="estimateRequest" :target="generationTarget" />
      </div>
      <button
        class="primary-button"
        type="button"
        :disabled="developDisabled"
        data-test="mobile-develop-button"
        @click="preparingGeneration ? cancelGenerationSubmission() : generate()"
      >
        {{ developButtonLabel }}
      </button>
    </div>

    <MobileGalleryViewer
      v-if="selectedPrint"
      :item="selectedPrint"
      :target="selectedPrint.target"
      :cache-key="selectedPrint.cacheKey"
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
      :organization="selectedPrintOrganization ?? null"
      :organize-enabled="libraryOrganizeEnabled"
      :trashed="selectedPrintTrashed"
      :organizing="organizationBusy"
      :organization-error="organizationError"
      :tag-suggestions="mergedTags"
      :collections="libraryCollectionCards"
      @close="closePrint"
      @reuse="reuseSelectedPrint"
      @use-source="useSelectedPrintAsSource"
      @previous="navigateSelectedPrint(-1)"
      @next="navigateSelectedPrint(1)"
      @rename="renameSelectedPrint"
      @favorite="favoriteSelectedPrint"
      @tags="tagSelectedPrint"
      @collection="collectSelectedPrint"
      @restore="restoreSelectedPrint"
      @delete-forever="deleteSelectedPrintForever"
    />

    <MobileGalleryViewer
      v-if="generatedViewerOpen && generatedPreviewItem && resultUrl"
      :item="generatedPreviewItem"
      :target="generatedPreviewTarget"
      :cache-key="latestResultJob?.hostId ?? 'generated'"
      :host-name="latestResultJob?.hostLabel ?? selectedHost?.name ?? 'Mold host'"
      :thumbnail-url="resultUrl"
      :media-url-override="resultUrl"
      :export-enabled="generatedPreviewHost !== null"
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

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  dropTargetAtPosition,
  isSupportedDroppedImage,
  reduceNativeImageDrag,
} from "../lib/nativeImageDrop";
import { useRoute, useRouter } from "vue-router";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "../lib/generateModels";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import Icon from "@ui/components/Icon.vue";
import ProgressRing from "@ui/components/ProgressRing.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import type { ClipRailMedia } from "@ui/components/types";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import ExpansionPullStatus from "../components/generate/ExpansionPullStatus.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import GenerateErrorNotice from "../components/generate/GenerateErrorNotice.vue";
import MissingModelDialog from "../components/generate/MissingModelDialog.vue";
import DownloadTargetDialog from "../components/models/DownloadTargetDialog.vue";
import { useLicenseAcceptance } from "@studio/composables/useLicenseAcceptance";
import { licenseRequirements } from "@studio/lib/licenseAcceptance";
import {
  classifyMissingModelHold,
  type GenerationPlacementPreview,
} from "@studio/api/generationPlacement";
import {
  watchSelectedQueuePreview,
  type QueueJobProgress,
  type SelectedQueuePreviewSource,
} from "@studio/api/generationSelection";
import CreateHeader from "../components/create/CreateHeader.vue";
import ClipModeStrip from "../components/create/ClipModeStrip.vue";
import { type InspectorTab } from "../components/create/inspectorTabs";
import ComposerCard from "../components/create/ComposerCard.vue";
import StylePicker from "../components/create/StylePicker.vue";
import InspectorPanel from "../components/create/InspectorPanel.vue";
import SequenceComposer from "../components/create/SequenceComposer.vue";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import { SEQUENCE_NEEDS_STYLE, type SequenceConfirmation } from "../lib/sequenceTimeline";
import {
  BENCH_RESIZER_HEIGHT,
  COMPOSER_FALLBACK_HEIGHT,
  DEFAULT_SEQUENCE_BENCH_HEIGHT,
  MIN_BENCH_HEIGHT,
  MIN_SEQUENCE_BENCH_HEIGHT,
  MIN_SEQUENCE_CANVAS_HEIGHT,
  MIN_STILL_CANVAS_HEIGHT,
  benchHeightCeiling,
  clampBenchHeight as clampBenchHeightWithin,
} from "../lib/benchLayout";
import { useSequenceDraftStore, type ClipMode } from "@studio/stores/sequenceDraft";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import { outputKindFor, outputKindForModel } from "../composables/useCreateOutputKind";
import { useOutputKindDoor } from "../composables/useOutputKindDoor";
import { filterRestrictedModels } from "@studio/lib/modelAccess";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { profileConflictMessage } from "@studio/lib/profileFleet";
import { useLiveActivityStore } from "../stores/liveActivity";
import { useJobsStore } from "../stores/jobs";
import {
  buildQueueStatusIndex,
  queueStatusFor,
  queueWaitLabel,
  resolveQueueWait,
} from "@studio/lib/queuePosition";
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
import { useChainJobsStore } from "../stores/chainJobs";
import { buildChainRequest } from "@studio/lib/sequenceForm";
import { chainScriptToClips } from "@studio/lib/sequenceForm";
import {
  defaultClipFrames,
  friendlySequenceError,
  modelSupportsSequence,
  modelsForOutput,
  sequenceClipFrameCap,
  sequenceFrameOptions,
  sequenceMotionTailFrames,
} from "@studio/lib/sequence";
import { promptGuidance, promptRequired } from "@studio/lib/promptRequirement";
import { promptInputForForm, promptRecipeFromForm } from "../lib/promptRecipe";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import { isMeshArtifact } from "@studio/lib/meshCompletion";
import { meshStatsLabel } from "@studio/lib/meshControls";
import MeshViewer from "@studio/components/MeshViewer.vue";
import {
  applyAuthoredPrompt,
  quickTransformSurvivesAuthoring,
  type PromptAuthoringSource,
} from "@studio/lib/promptProvenance";
import {
  applyMinimaxH3ReferenceCrops,
  emptyMinimaxH3AuthoringState,
  minimaxH3AuthoringError,
  setMinimaxH3PickedImageBoundary,
} from "@studio/lib/minimaxH3Authoring";
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
import {
  normalizeTargetHost,
  pickAutoHost,
  pickMostCapableHost,
  readyHostSignature,
} from "../lib/hosts";
import {
  expandModelId,
  expansionPolicyForSelection,
  resolveExpansionRoute,
  type ExpansionCandidate,
} from "@studio/lib/expansionRouting";
import { useAppPrefsStore } from "../stores/appPrefs";
import { createUuid } from "@studio/lib/id";
import { confirmCancellation } from "@studio/lib/cancellationRetry";
import { useHostModelsStore } from "../stores/hostModels";
import { strongestRoutableGpu, useHostsStore, type FeasibleRouteResult } from "../stores/hosts";
import { useConnectionStore } from "../stores/connection";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  jobProgressCopy,
  needsHostRoute,
  suggestOutputFilename,
  type BatchRequestOptions,
  type GalleryPrintOnCanvas,
  type Job,
} from "../stores/generation";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { copyBase64ImageToClipboard, copyImageBytesToClipboard } from "../lib/clipboard";
import { suggestedSaveName } from "../lib/gallery/saveName";
import { copyLocalOutputPath } from "../lib/localOutputPath";
import { useUiStore, type CreateIntent } from "../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import {
  generationCapabilitiesForFamily,
  generationCapabilitiesForForm,
} from "../lib/capabilities";
import { batchLockedForForm, batchLockedForRequest, canRepeatPrint } from "../lib/variations";
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
  chainFilingFields,
  cloneGenerateForm,
  keepingPrintIdentity,
  loraBindingMatchesRoute,
  loraHostBinding,
  MAX_BATCH_SIZE,
  normalizeLegacyNegativeSnapshot,
  reconcileModelCapabilities,
} from "../lib/generateForm";
import { emptyMeshForm } from "@studio/lib/meshControls";
import { videoFramesError, type VideoFrameContract } from "@studio/lib/videoDuration";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  identityConditioningValidationError,
  meshTargetFacesValidationError,
  profileGuidanceValidationError,
  profileStepsValidationError,
  resolutionValidationError,
  resolutionValidationWarning,
  sourceConditioningValidationError,
  wanRecipeValidationError,
} from "../lib/generateValidation";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { applyH3BoundaryFit, applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  parseSourceFitPolicy,
} from "@studio/lib/sourceFit";
import { expansionContextForRequest, expansionTaskForRequest } from "@studio/lib/expandTask";
import { domCanvasOps } from "@studio/lib/sourceFitCanvas";
import { conditioningForRequest } from "@studio/lib/sourceMediaPlan";
import { upscaleImage } from "../lib/api/upscale";
import { expandPrompt } from "../lib/api/expand";
import { remixPrompt } from "../lib/api/remix";
import {
  conditioningFingerprint,
  defaultRemixDimensions,
  promptSource,
  promptTransformBlockedReason,
  validateRemixVariants,
} from "@studio/lib/promptTransform";
import type {
  HostFeasibilityFailure,
  HostPlacementFailure,
  HostRoute,
  HostView,
} from "../stores/hosts";
import { planModelInstall, type ModelInstallTarget } from "@studio/lib/modelInstallTargets";
import {
  formatTemplateMediaReferences,
  hydrateGenerationTemplate,
  type GenerationTemplate,
} from "../lib/generationTemplates";
import { fetchHistoryFrom } from "../lib/api/history";
import {
  availablePromptHistoryStorage,
  PromptHistoryCoordinator,
  promptHistoryHostSignature,
  recordPromptHistoryCache,
} from "@studio/lib/promptHistoryCache";
import { randomSeed } from "../stores/generation";
import type {
  ChainCreateRequest,
  CompleteEvent,
  GenerateRequest,
  OutputMetadata,
} from "../lib/api/types";
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
  restoreH3Boundaries,
  restoreSourceImage,
  sha256HexOfBase64,
  type SourceRestoreDeps,
} from "../lib/sourceRestore";
import {
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
} from "@studio/lib/generationSourceMedia";
import { relayRetainedSourceMedia } from "@studio/api/gallerySourceMedia";
import { persistIdentityPhoto, restoreIdentityPhoto } from "@studio/lib/identityConditioning";
import { isMissingModelError } from "../lib/generateErrors";
import { copyableError, describeTransportError } from "../lib/api/errors";
import { startCatalogDownload } from "../lib/api/catalog";
import { computeEtaSeconds, useDownloadsStore, type DownloadsState } from "../stores/downloads";
import { usePullResumeStore } from "../stores/pullResume";
import { modelDisplayNameForId } from "../lib/models";
import {
  authedMediaUrl,
  fetchGalleryMediaBytes,
  fullSizeMediaUrl,
  galleryMediaPath,
  localMediaPath,
  mediaMimeType,
  mediaPath,
  nativeBytes,
  thumbnailPath,
} from "../lib/gallery/media";
import { applyGalleryEntryAsSource, canUseGalleryEntryAsSource } from "../lib/gallery/useAsSource";
import { readGalleryMediaBase64 } from "../lib/gallery/sourceMedia";
import { useReuseStillPrint } from "../composables/useReuseStillPrint";
import { upscaleLibraryImage } from "@studio/api/videoUpscale";
import { defaultUpscaler } from "@studio/lib/upscale";
import UpscaleDialog from "@ui/components/UpscaleDialog.vue";
import { sequenceStageMediaUrl } from "../lib/sequenceMedia";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import { ApiError, apiFetch, apiFetchTo } from "../lib/api/client";
import { blobToBase64 } from "../lib/image";
import { ipc } from "../lib/ipc";
import type { ApiTarget } from "../lib/api/client";
import { applyDesktopImageDrop } from "../lib/desktopImageDrop";
import type { DropTarget } from "@studio/lib/imageDropRouting";
import { isAudioCompletion } from "@studio/lib/ltx2Pipeline";
import { useGalleryStore, type MergedPrint } from "../stores/gallery";
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
const reuseStillPrint = useReuseStillPrint();
const downloads = useDownloadsStore();
const pullResume = usePullResumeStore();
const licenseAcceptance = useLicenseAcceptance();
const liveActivity = useLiveActivityStore();
const jobsStore = useJobsStore();

function placementFailureMessage(result: Exclude<FeasibleRouteResult, { kind: "route" }>): string {
  if (result.kind === "profile_mismatch") {
    return profileConflictMessage(result.perHost);
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
  /** False when the frozen request still has to be finalized against the
   *  chosen machine — download only, never a resume promise. */
  resumeAfterPull?: boolean;
  /** Set when the machine is already HOLDING this work: resume retries that
   *  exact child instead of queueing a second print. */
  retryClientId?: number | null;
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

/** Freeze the exact request a resumed submit will send, including the quick
 * expansion's prompt provenance (`submit()` applies it after routing). */
function pullResumeRequest(request: GenerateRequest): GenerateRequest {
  const transform = quickExpansionSnapshot.value?.promptTransform;
  return transform ? { ...request, prompt_transform: transform } : request;
}

/** The work a missing-model pull would resume, frozen before the dialog opens. */
interface MissingModelSubmission {
  model: string;
  modelFamily: string;
  request: GenerateRequest;
  batch: number;
  chainRouting: ChainRoutingDecision | null;
  requestOptions: BatchRequestOptions;
  /**
   * False when the frozen request is not yet the one that would render —
   * a source that still has to be fitted against the chosen machine
   * (`upscale-then-fit`). The download is still offered; the promise to
   * generate is not, because resuming would use different conditioning.
   */
  resumeAfterPull?: boolean;
  /** Set when the machine is already HOLDING this work: resume retries that
   *  exact child instead of queueing a second print. */
  retryClientId?: number | null;
}

function freezeModelFamily(route: HostRoute | null, modelFamily: string): HostRoute | null {
  const family = modelFamily.trim();
  return route
    ? {
        ...route,
        target: { ...route.target },
        ...(family ? { modelFamily: family } : {}),
      }
    : null;
}
let missingModelNotificationId = 0;

/** An open machine picker for a pre-submit pull (more than one candidate). */
const missingModelTargets = ref<{
  model: string;
  targets: ModelInstallTarget<HostView>[];
  submission: MissingModelSubmission;
  /** Exact route per machine, preferring what the placement probe proved. */
  routeFor: (hostId: string) => HostRoute | null;
} | null>(null);

/**
 * Machines that refused ONLY for want of the model. Auto / Most capable must
 * never dead-end (#1162): when nothing can route because nobody has the
 * model, a pull is the recovery — but a machine that cannot fit the print, or
 * that a policy refuses, is never a pull target.
 */
function missingModelFailures(
  result: Exclude<FeasibleRouteResult, { kind: "route" }>,
): HostPlacementFailure[] {
  if (result.kind !== "infeasible" && result.kind !== "mixed") return [];
  const perHost: HostFeasibilityFailure[] = result.perHost;
  return perHost.filter(
    (failure): failure is HostPlacementFailure =>
      failure.kind === "infeasible" && failure.missingModel !== null,
  );
}

/**
 * Offer the pull instead of a toast. One candidate skips the machine picker
 * (the same rule `chooseInstallTarget` uses); several open the shared
 * `DownloadTargetDialog`, defaulting to whichever machine the current routing
 * policy would have picked. Returns false when there is nothing to offer, so
 * the caller keeps its existing failure message.
 */
function offerMissingModelPull(
  result: Exclude<FeasibleRouteResult, { kind: "route" }>,
  submission: MissingModelSubmission,
): boolean {
  const failures = missingModelFailures(result);
  if (failures.length === 0) return false;
  const preferredId =
    hosts.resolveRoute(appPrefs.settings?.generateTargetHost ?? null)?.hostId ?? null;
  const ordered = [...failures].sort(
    (left, right) => Number(right.hostId === preferredId) - Number(left.hostId === preferredId),
  );
  if (ordered.length === 1) {
    missingModel.value = {
      ...submission,
      route: freezeModelFamily(ordered[0]!.route, submission.modelFamily),
    };
    return true;
  }
  const candidateHosts = ordered.flatMap((failure) => {
    const host = hosts.all.find((entry) => entry.id === failure.hostId);
    return host ? [host] : [];
  });
  const targets = planModelInstall(candidateHosts, hostModels.hostsFor(submission.model), {
    // The placement preview IS the positive knowledge that these machines
    // lack the model — stronger evidence than the inventory poll, which may
    // not have read them yet.
    inventoryKnown: () => true,
  }).targets;
  return presentMissingModelPull(
    targets,
    submission,
    (hostId) =>
      ordered.find((failure) => failure.hostId === hostId)?.route ?? hosts.resolveRoute(hostId),
  );
}

/**
 * One machine skips the picker; several open the shared host picker.
 * `routeFor` prefers the route the placement probe already proved, falling
 * back to the store for a machine that was never probed.
 */
function presentMissingModelPull(
  targets: ModelInstallTarget<HostView>[],
  submission: MissingModelSubmission,
  routeFor: (hostId: string) => HostRoute | null,
): boolean {
  if (targets.length === 0) return false;
  if (targets.length === 1) {
    const route = freezeModelFamily(routeFor(targets[0]!.host.id), submission.modelFamily);
    if (!route) return false;
    missingModel.value = { ...submission, route };
    return true;
  }
  missingModelTargets.value = {
    model: submission.model,
    targets,
    submission,
    routeFor,
  };
  return true;
}

/**
 * A print is admitted BEFORE the machine resolves its model, so "nobody has
 * this model" now arrives as a HELD child carrying the machine's own reason
 * rather than as an infeasible placement preview. Same offer, same policy —
 * only the machine that parked the work is a pull target, and the resume
 * retries that exact child instead of queueing a second print.
 */
const offeredMissingModelHolds = new Set<number>();
function offerHeldMissingModelPull(job: Job): void {
  const request = job.request;
  if (!request) return;
  const missing = classifyMissingModelHold(job.holdCode, job.model);
  if (!missing) {
    // The hold ended (or was never about the model): a later hold on the
    // same print is a new offer.
    offeredMissingModelHolds.delete(job.clientId);
    return;
  }
  if (offeredMissingModelHolds.has(job.clientId)) return;
  offeredMissingModelHolds.add(job.clientId);
  // Only the machine that parked the job is a pull target; a job with no
  // known host has nobody to pull on.
  const hostId = job.hostId ?? null;
  const host = hostId ? hosts.all.find((entry) => entry.id === hostId) : null;
  const targets = planModelInstall(host ? [host] : [], hostModels.hostsFor(missing.model), {
    // The machine's own hold IS the positive knowledge that it lacks the
    // model — stronger evidence than an inventory poll that may be stale.
    inventoryKnown: () => true,
  }).targets;
  presentMissingModelPull(
    targets,
    {
      model: missing.model,
      modelFamily: form.family,
      // Nothing is resubmitted on this path; the child is already queued.
      request,
      batch: 1,
      chainRouting: null,
      requestOptions: {},
      retryClientId: job.clientId,
    },
    (candidateId) => hosts.resolveRoute(candidateId),
  );
}

watch(
  () =>
    generation.jobs
      .filter((job) => job.holdError)
      .map((job) => `${job.clientId}:${job.holdCode ?? ""}:${job.holdError}`)
      .join("|"),
  () => {
    for (const job of generation.jobs) {
      if (job.holdError) offerHeldMissingModelPull(job);
    }
  },
);

/**
 * The Create picker's "Not installed" row. There is no placement evidence
 * here, so the machines come from the shared install-target policy: every
 * reachable machine whose inventory has been read and does not have it.
 */
async function offerPullForSelectedModel(model: string) {
  if (!model) return;
  // Self-heal first: another client may already have pulled it.
  await hostModels.refresh(true);
  if (hostModels.hostsFor(model).length > 0) return;
  const blocker = generationInputBlockerReason.value;
  if (blocker) {
    toasts.push(blocker, "error");
    return;
  }
  const readyHosts = hosts.all.filter((host) => host.status === "ready" && host.baseUrl);
  const targets = planModelInstall(readyHosts, hostModels.hostsFor(model), {
    inventoryKnown: (host) => (hostModels.byHost[host.id]?.fetchedAt ?? 0) > 0,
  }).targets;
  const caps = generationCapabilitiesForFamily(form.family, form.model);
  const submission: MissingModelSubmission = {
    model,
    modelFamily: form.family,
    request: pullResumeRequest(buildRequest(cloneGenerateForm(form))),
    // Same rule Generate uses: Batch N submits N ordinary siblings.
    batch: caps.forcesBatchSizeOne ? 1 : form.batchSize,
    chainRouting: null,
    requestOptions: {},
  };
  if (!presentMissingModelPull(targets, submission, (hostId) => hosts.resolveRoute(hostId))) {
    toasts.push(
      `No connected machine can download ${model} right now. Connect a machine under Machines, then try again.`,
      "error",
    );
  }
}

/** The user picked a machine in the pre-submit picker: confirm the pull there. */
function chooseMissingModelHost(host: HostView) {
  const pending = missingModelTargets.value;
  missingModelTargets.value = null;
  if (!pending) return;
  const route = pending.routeFor(host.id);
  if (!route) {
    toasts.push(`${host.label} is no longer reachable. Nothing was queued.`, "error");
    return;
  }
  missingModel.value = {
    ...pending.submission,
    route: freezeModelFamily(route, pending.submission.modelFamily),
  };
}

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
    retryClientId: info.retryClientId ?? null,
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
  const resumes = info.resumeAfterPull !== false;
  const notificationOwner = `create-model:${++missingModelNotificationId}`;
  downloads.armNotificationAction(info.model, bucketId, notificationOwner, { kind: "create" });
  try {
    // Watch the EXACT job the server enqueues; a stale completed pull of the
    // same model in history can then never trigger a premature resume.
    const jobId = await startCatalogDownload(info.model, route?.target, route?.kind === "remote");
    downloads.refineNotificationAction(info.model, bucketId, notificationOwner, jobId);
    if (resumes) pullResume.arm({ ...armed, jobId });
    toasts.push(
      resumes
        ? `Pulling ${info.model} on ${label} — generation starts when it's ready`
        : `Pulling ${info.model} on ${label} — press Generate again once it's ready.`,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      // Already downloading (another client or an earlier click) — watch by
      // model; the running job is live, not terminal, so it can't be stale.
      if (resumes) pullResume.arm({ ...armed, jobId: null });
      toasts.push(
        resumes
          ? `${info.model} is already downloading on ${label} — will generate when ready`
          : `${info.model} is already downloading on ${label} — press Generate again once it's ready.`,
      );
    } else if (/unknown model/i.test(String(err))) {
      downloads.clearNotificationAction(info.model, bucketId, notificationOwner);
      toasts.push(
        `${label} can't pull ${info.model} by name — pull it from the Catalog there, then generate again.`,
        "error",
      );
    } else if (routeRequired.value) {
      downloads.clearNotificationAction(info.model, bucketId, notificationOwner);
      toasts.push(String(err), "error");
    } else {
      downloads.clearNotificationAction(info.model, bucketId, notificationOwner);
    }
  }
}

// Store-backed so the model, prompt, and params survive navigating away and
// back — this view unmounts on every route change.
const formStore = useGenerateFormStore();
const form = formStore.form;

const composerRef = ref<InstanceType<typeof ComposerCard> | null>(null);
const workbenchRef = ref<HTMLDivElement | null>(null);
const inspectorTab = ref<InspectorTab>("settings");
const inspectorRef = ref<InstanceType<typeof InspectorPanel> | null>(null);
/** Recent prompts for the composer's ↑/↓ history cycling. */
const promptHistory = ref<string[]>([]);
const nativeImageDragOver = ref(false);
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
    : DEFAULT_SEQUENCE_BENCH_HEIGHT,
);
let benchResizeStartY = 0;
let benchResizeStartHeight = 0;

/**
 * The composer's real height, measured. It is the piece `overflow-hidden`
 * eats when the reservation is short, and it is the one piece whose height
 * the user changes (a long prompt grows the textarea), so the bench ceiling
 * has to follow it rather than assume it.
 */
const composerHeight = ref(COMPOSER_FALLBACK_HEIGHT);
let composerObserver: ResizeObserver | null = null;

/** The canvas's own floor — bound to its `min-height`, so the CSS and the
 *  clamp can never disagree the way they did when one said 320 and the
 *  other reserved 144. */
const canvasFloor = computed(() =>
  isSequence.value ? MIN_SEQUENCE_CANVAS_HEIGHT : MIN_STILL_CANVAS_HEIGHT,
);

function minBenchHeight(): number {
  return isSequence.value ? MIN_SEQUENCE_BENCH_HEIGHT : MIN_BENCH_HEIGHT;
}

function benchClampInput(requested: number) {
  return {
    requested,
    available: workbenchRef.value?.clientHeight ?? window.innerHeight,
    minBench: minBenchHeight(),
    canvasFloor: canvasFloor.value,
    resizerHeight: BENCH_RESIZER_HEIGHT,
    composerHeight: composerHeight.value,
  };
}

function clampBenchHeight(height: number): number {
  return clampBenchHeightWithin(benchClampInput(height));
}

/** The resizer's `aria-valuemax`: the same ceiling the clamp enforces. */
function maxBenchHeight(): number {
  return benchHeightCeiling(benchClampInput(benchHeight.value));
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

/**
 * Watch the composer's box. A prompt that grows to several lines takes those
 * pixels from the same column the bench sits in, so the ceiling has to move
 * with it — otherwise a long prompt pushes Generate out under
 * `overflow-hidden` again.
 */
function observeComposerHeight() {
  const el = composerRef.value?.$el;
  if (!(el instanceof HTMLElement) || typeof ResizeObserver === "undefined") return;
  composerHeight.value = el.offsetHeight || COMPOSER_FALLBACK_HEIGHT;
  clampBenchToViewport();
  composerObserver = new ResizeObserver(() => {
    const measured = el.offsetHeight;
    if (!measured || measured === composerHeight.value) return;
    composerHeight.value = measured;
    clampBenchToViewport();
  });
  composerObserver.observe(el);
}
const preparedBatch = ref<PreparedExpansionBatchState | null>(null);
const remixSource = ref<"original" | "current">("original");
const expansionRunning = ref(false);
const expansionError = ref<string | null>(null);
const expansionMissingModel = ref<{ model: string; route: HostRoute } | null>(null);
interface ExpansionPullAttempt {
  id: number;
  notificationOwner: string;
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
const preparedSubmitting = ref(false);
const submissionPlanning = ref(false);
const preparationGuard = new PreparationRequestGuard();
const submissionGuard = new PreparationRequestGuard();
const sequenceSubmissionGuard = new PreparationRequestGuard();
// Completion is detached from authoring once the store accepts a batch. Only
// the newest accepted submission may publish mutable recovery authority:
// older batches can finish later, but must not replace a newer pull/resume
// decision.
let latestAcceptedSubmissionId = 0;

let stopNativeImageDrop: (() => void) | null = null;
let nativeImageDropUnmounted = false;
let nativeImageDragCandidate = false;

/**
 * One-line disclosure when a restore lands on a model no machine has, mirroring
 * the template loader's notice. The id stays in the form (the picker shows it
 * with a Not installed tag) — silence here read as "the model was dropped".
 */
function discloseMissingRestoredModel() {
  const name = form.model;
  if (!name || findInstalledModel(installedModels.value, name)) return;
  toasts.push(
    `${modelDisplayNameForId(name, installedModels.value)} isn't installed — open the model list to download it.`,
  );
}

/** What each well calls the image it just received, for the toast. */
const DROP_TARGET_LABELS: Record<string, string> = {
  source: "Source image attached.",
  references: "Reference image added.",
  end: "End frame attached.",
  identity: "Identity photo attached.",
  opening: "Opening image attached.",
  "h3-first": "First frame attached.",
  "h3-last": "Last frame attached.",
  "h3-reference": "Reference added.",
};

async function importDroppedImage(path: string, hovered: string | null) {
  try {
    const image = await ipc.importSourceImage(path);
    // The well under the cursor decides, not the model: `resolveDropTarget`
    // takes the hovered `data-drop-target` when the plan renders it and the
    // plan's own default otherwise.
    const result = await applyDesktopImageDrop(
      form,
      image,
      installedModels.value,
      hovered as DropTarget | null,
      {
        identityVisible: !isSequence.value && form.identitySupported === true,
        openingVisible: isSequence.value,
        sequenceDraft: isSequence.value ? draft : null,
      },
    );
    if (result.metadataApplied) discloseMissingRestoredModel();
    const attachedMessage =
      (result.target && DROP_TARGET_LABELS[result.target]) || "Image attached.";
    if (result.metadataApplied && result.attached) {
      toasts.push(`Loaded generation settings. ${attachedMessage}`);
    } else if (result.metadataApplied) {
      toasts.push(`Loaded generation settings; ${result.refused ?? "nothing was attached."}`);
    } else if (result.attached) {
      toasts.push(attachedMessage);
    } else {
      toasts.push(result.refused ?? "The selected model doesn't accept a source image.", "error");
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
    // Hit-test BEFORE the async import: the cursor position is the only
    // record of which well the user aimed at.
    const hovered = dropTargetAtPosition(payload.position);
    if (path) void importDroppedImage(path, hovered);
    else toasts.push("Drop a PNG or JPEG image.", "error");
  });
  if (nativeImageDropUnmounted) unlisten();
  else stopNativeImageDrop = unlisten;
}

/** The Recent tab's rows: the pictures already made, newest first. */
const RECENT_PRINTS_CAP = 24;
const recentPrints = computed(() => hostGallery.merged.slice(0, RECENT_PRINTS_CAP));

/**
 * Recent tab: a row restores the whole recipe, exactly as the Lightbox's
 * "Use these settings" does — full metadata through `applyPrefillToForm`, plus
 * the print's own retained-source authority when its machine kept one. The
 * door that opens this tab promises those settings, so it must be the same
 * path, not a prompt-only shortcut.
 */
function reuseRecentPrint(entry: MergedPrint) {
  // A stitched clip restores as a clip, exactly as the Lightbox restores it:
  // a metadata prefill would force `single` and keep one scene's words.
  if (planSequenceReuse(entry.item.metadata)) {
    composer.setSequence({ kind: "reuse", metadata: entry.item.metadata });
    return;
  }
  reuseStillPrint(entry);
  inspectorTab.value = "settings";
  void nextTick(() => composerRef.value?.focus?.());
}

const selectedQueueRender = ref<{
  source: SelectedQueuePreviewSource;
  width: number;
  height: number;
  model: string;
  preview: QueueJobProgress | null;
} | null>(null);
let stopSelectedQueuePreview: (() => void) | null = null;

function clearSelectedQueueRender() {
  stopSelectedQueuePreview?.();
  stopSelectedQueuePreview = null;
  selectedQueueRender.value = null;
}

function inspectSelectedQueueRender(source: SelectedQueuePreviewSource | undefined) {
  clearSelectedQueueRender();
  if (!source?.running) return;
  const host = hosts.all.find((candidate) => candidate.id === source.hostId);
  if (!host?.baseUrl) return;
  selectedQueueRender.value = {
    source,
    width: form.width,
    height: form.height,
    model: form.model,
    preview: null,
  };
  stopSelectedQueuePreview = watchSelectedQueuePreview(
    { baseUrl: host.baseUrl, apiKey: host.apiKey },
    source.jobId,
    (preview) => {
      if (
        selectedQueueRender.value?.source.hostId === source.hostId &&
        selectedQueueRender.value.source.jobId === source.jobId
      ) {
        selectedQueueRender.value = { ...selectedQueueRender.value, preview };
      }
    },
    750,
    () => {
      if (selectedQueueRender.value?.source.jobId === source.jobId) {
        clearSelectedQueueRender();
      }
    },
  );
}

const job = computed(() => (selectedQueueRender.value ? null : generation.active));
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
    effectiveGenerationRecipe(contractEntry.value, form.pipeline),
  ),
);
const formValidationError = computed(
  () =>
    // A canvasless recipe (a 3-D mesh) renders at 0 × 0 by contract, so the
    // whole-number canvas check would block every mesh submit on the exact
    // size the recipe itself advertises. Read from the form's own snapshot,
    // the same authority `buildRequest` submits under.
    ((form.recipeCapabilities?.canvasless ?? isMeshFamily(form.family))
      ? null
      : resolutionValidationError(
          form.width,
          form.height,
          contractEntry.value ?? null,
          form.pipeline,
        )) ??
    profileStepsValidationError(form.steps, contractEntry.value, form.pipeline) ??
    profileGuidanceValidationError(
      caps.value.fixedGuidance ?? form.guidance,
      contractEntry.value,
      form.pipeline,
    ) ??
    meshTargetFacesValidationError(form) ??
    (caps.value.supportsVideo
      ? videoFramesError(form.frames, contractEntry.value ?? { family: form.family })
      : null) ??
    (caps.value.supportsVideo ? fpsValidationError(form.fps) : null) ??
    cameraControlValidationError(form) ??
    audioOutputValidationError(form) ??
    sourceConditioningValidationError(form, {
      ignoreUnsupportedStagedSource: true,
    }) ??
    identityConditioningValidationError(form) ??
    advancedVideoValidationError(form) ??
    wanRecipeValidationError(form),
);
/** Exact submit blocker for request-shape and automatic-chain validation. */
const chainValidationError = computed<string | null>(() => {
  if (formValidationError.value) return formValidationError.value;
  if (!caps.value.supportsVideo) return null;
  const request = buildRequest(form);
  const decision = decideGenerateRequestRouting(request, form.family, selectedEntry.value);
  if (decision.kind === "reject") return decision.reason;
  const unsupported = decision.kind === "chain" ? unsupportedAutoChainFields(request) : [];
  return unsupported.length
    ? "This long video uses options automatic sequencing cannot preserve. Reduce Duration or remove the incompatible Advanced options."
    : null;
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
  return buildGenerationEstimateRequest(buildRequest(form), form.family, selectedEntry.value);
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

/** Which conditioning this form's request will carry — the same shared
 * decision `buildRequest` makes, so the batch cap and the canvas watcher
 * cannot disagree with the wire. */
const requestConditioning = computed(() =>
  conditioningForRequest(caps.value.sourceImageMode, {
    hasSource: Boolean(form.sourceImage),
    referenceCount: form.imageAttachments.length,
    lastWrite: form.exclusiveWell ?? null,
  }),
);

/** The recipe renders one at a time (an edit model, or a request that
 * carries references): the Make chip reads 1 and locks. */
const batchLocked = computed(() => batchLockedForForm(form, caps.value));
const effectiveBatchSize = computed(() =>
  batchLocked.value ? 1 : Math.max(1, Math.floor(form.batchSize)),
);

/** Where the print itself would go. Expansion starts here and only leaves it
 *  for a machine that positively has the expand model. */
const generationRoute = computed<HostRoute | null>(() =>
  hosts.resolveRoute(stickyTarget.value, form.model || null),
);

const expansionPolicy = computed(() => expansionPolicyForSelection(stickyTarget.value));

/** One row per machine: readiness plus what its `/api/capabilities.expand`
 *  says. An unread host contributes no `modelPresent` — unknown, not absent. */
const expansionCandidates = computed<ExpansionCandidate[]>(() =>
  hosts.all.map((host) => {
    const expand = hosts.capabilities[host.id]?.expand;
    return {
      hostId: host.id,
      ready: host.status === "ready",
      ...(expand ? { modelPresent: expand.model_present, configured: expand.configured } : {}),
    };
  }),
);

/** Rank an eligible subset with the generation router's own ordering. */
function rankExpansionHosts(hostIds: readonly string[]): string | null {
  const pool = hosts.all.filter((host) => hostIds.includes(host.id));
  if (stickyTarget.value === "capable") {
    return (
      pickMostCapableHost(
        pool.map((host) => ({ ...host, gpu: strongestRoutableGpu(hosts.telemetry[host.id]) })),
        null,
      )?.id ?? null
    );
  }
  return pickAutoHost(pool)?.id ?? null;
}

const expansionRouteDecision = computed(() =>
  resolveExpansionRoute(
    expansionPolicy.value,
    generationRoute.value,
    expansionCandidates.value,
    rankExpansionHosts,
  ),
);

/** Expansion always resolves a concrete host, even in the one-host case. */
const currentExpansionRoute = computed<HostRoute | null>(() => {
  const decision = expansionRouteDecision.value;
  if (decision.kind === "reroute") {
    return hosts.resolveRoute(decision.hostId) ?? generationRoute.value;
  }
  return generationRoute.value;
});
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
          dimensions: defaultRemixDimensions(expansionTaskForRequest(form.family, request), false),
          conditioningFingerprint: conditioningFingerprint(request),
        }
      : {}),
    requestedCount: effectiveBatchSize.value,
    // This surface has no style preset to freeze or to go stale on.
    stylePreset: null,
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
const lastUsed = useLastUsedStylesStore();
const chains = useChainJobsStore();
const isSequence = computed(() => draft.output === "sequence");
const selectedEntry = computed(() =>
  hostModels.installedEntryForTarget(form.model, stickyTarget.value),
);
/**
 * The row that answers for the CHECKPOINT'S CONTRACT, resolved from whichever
 * machine has it — the same authority the inspector reads, so the canvas, the
 * submit blocker and the settings panel cannot disagree about the checkpoint
 * the moment Create is aimed at a machine that must download it first.
 * `selectedEntry` stays the answer for ROUTING and for the memory estimate,
 * which really are about the machine that will run the job.
 */
const contractEntry = computed(() =>
  hostModels.contractEntryForTarget(form.model, stickyTarget.value),
);

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

function applyDecodedSourceResolution(
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
    contractEntry.value ?? form.family,
    form.pipeline,
  );
  const automaticResolution = resolveDefaultSourceResolution(
    dimensions,
    contractEntry.value ?? form.family,
    form.pipeline,
  );
  const replaced = base64 !== previous.base64;
  // A reference strip is not a canvas: the render's size comes from the model,
  // not from whichever picture happens to sit first in the order. Qwen edit is
  // the exception the profile names — `primary_is_target` means image 0 IS the
  // picture being edited, so its canvas stays source-driven.
  if (
    requestConditioning.value !== "references" ||
    caps.value.referenceImages?.primaryIsTarget === true
  ) {
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
    // A canvasless recipe (a 3-D mesh) renders no pixel canvas: its zero
    // canvas is the recipe's own answer, and a source's size is not a
    // resolution to fit it to. The snapshot is the authority; the family is
    // the fallback for a form restored before it landed (as `buildRequest`).
    const canvasless = form.recipeCapabilities?.canvasless ?? isMeshFamily(form.family);
    if (nextResolution && !canvasless) {
      form.width = nextResolution.width;
      form.height = nextResolution.height;
    }
  }
  return { base64, resolution, automaticResolution };
}

watch(
  [
    () =>
      requestConditioning.value === "references"
        ? (form.imageAttachments[0] ?? null)
        : form.sourceImage,
    () => contractEntry.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => contractEntry.value?.generation_profile?.profile_hash ?? null,
    () => contractEntry.value?.max_pixels ?? null,
    () => contractEntry.value?.max_axis_pixels ?? null,
    () => contractEntry.value?.dimension_alignment ?? null,
    () =>
      contractEntry.value?.recommended_dimensions
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
  },
  { immediate: true },
);

watch(
  [
    () => draft.openingImage?.base64 ?? null,
    () => contractEntry.value?.name ?? form.model,
    () => form.pipeline ?? null,
    () => contractEntry.value?.generation_profile?.profile_hash ?? null,
    () => contractEntry.value?.max_pixels ?? null,
    () => contractEntry.value?.max_axis_pixels ?? null,
    () => contractEntry.value?.dimension_alignment ?? null,
    () =>
      contractEntry.value?.recommended_dimensions
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
let sequenceAmendInFlight = false;
let sequenceCancellationRequest: (() => Promise<void>) | null = null;
/** Snapshot of the shared params at edit-load time — drives chainLevelDirty. */
const editSharedBaseline = ref<string | null>(null);
/** What a My images reuse could NOT restore, said once and quietly beneath the
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
    const pick = lastUsed.pick("clip", sequenceCapableModels.value);
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
      const pick = lastUsed.pick("clip", sequenceCapableModels.value);
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
  const version = ++chainLimitsFetch;
  const entry = selectedSequenceEntry.value;
  if (!entry) {
    chainLimits.value = null;
    return;
  }
  try {
    const target = routeForModel(entry)?.target ?? null;
    const limits = (await fetchChainLimits(entry.name, target, form.fps)) as ChainLimits;
    if (version !== chainLimitsFetch) return;
    chainLimits.value = limits;
    if (!limits.supports_audio) draft.enableAudio = false;
    if (!draft.editing) {
      const frames = defaultClipFrames(entry, limits, sequenceMotionTail.value);
      // A model switch adopts that model's audio answer — on wherever the
      // chain renders sound, matching a one-shot of the same model. A
      // re-fetch for the SAME model returns false from `adoptSequenceModel`
      // and leaves the user's own choice alone.
      draft.adoptSequenceModel(entry.name, frames, limits.supports_audio);
    }
  } catch {
    if (version === chainLimitsFetch) chainLimits.value = null;
  }
}

// Chain limits are per model AND per host — refetch when either moves.
watch(
  () => form.model,
  (next, previous) => {
    if (isSequence.value && previous && next !== previous) {
      if (draft.editing) {
        draft.stopEditing();
        editSharedBaseline.value = null;
      }
    }
  },
);
watch(
  [isSequence, () => form.model, () => form.fps, stickyTarget, () => readyHostSignature(hosts.all)],
  () => {
    if (isSequence.value && form.model) void loadChainLimits();
  },
  { immediate: true },
);

// The queue rail lists every connected host's durable jobs.
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

/** How far the playing scene's own player has run. The timeline adds the
 *  scenes before it to put the transport's needle on the whole clip. */
const sequenceClipElapsed = ref(0);
watch(sequencePlaybackSrc, () => (sequenceClipElapsed.value = 0));
function onSequenceTimeUpdate(event: Event) {
  sequenceClipElapsed.value = (event.target as HTMLVideoElement).currentTime;
}

function returnToLiveSequence() {
  playingSequenceStage.value = null;
}

/**
 * The watched job once it has settled. The Create strip no longer keeps a
 * settled row, so the canvas is what holds the result: the finished video with
 * Edit clip / Show in My images, or the failure with Resume. Settling must
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
  if (sequenceSubmitting.value) {
    cancelSequenceSubmission();
    return;
  }
  clearSelectedQueueRender();
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
  // Sequence stage images predate the additive contract. Preserve compatible
  // behavior for absent/unknown fields and park them only when this exact
  // checkpoint explicitly advertises `unsupported`.
  const supportsSourceImages = entry.source_image !== "unsupported";
  // Sequence media is parked just like the one-shot source. Strip only the
  // frozen request copy for an unsupported checkpoint; the shared draft stays
  // intact so switching back restores every opening/per-clip image.
  const openingSnapshot =
    supportsSourceImages && draft.openingImage ? { ...draft.openingImage } : null;
  if (!supportsSourceImages) {
    for (const clip of clips) clip.sourceImage = null;
  }
  const enableAudio = draft.enableAudio;
  const motionTailFrames = sequenceMotionTail.value;
  const hostRoute = editing ? hosts.resolveRoute(editing.hostId, entry.name) : routeForModel(entry);
  if (!hostRoute) {
    toasts.push("The selected host isn't reachable. Pick another host.", "error");
    return;
  }
  sequenceSubmitting.value = true;
  const token = sequenceSubmissionGuard.begin();
  const signal = sequenceSubmissionGuard.signalFor(token);
  const isCurrent = () => sequenceSubmissionGuard.isCurrent(token) && !signal.aborted;
  try {
    // Refetch stale limits so frames caps/audio gating match the routed host.
    if (!chainLimits.value || chainLimits.value.model !== entry.name) {
      await loadChainLimits();
      if (!isCurrent()) return;
    }
    requestForm.sourceImage = openingSnapshot?.base64 ?? null;
    requestForm.maskImage = null;
    if (!(await preprocessSourceFit(hostRoute, requestForm, signal)) || !isCurrent()) return;
    const openingImage = openingSnapshot
      ? { ...openingSnapshot, base64: requestForm.sourceImage }
      : null;
    // The stitched print is the only artifact a sequence puts in the gallery,
    // so it is what carries the header title and the File under choice.
    const request: ChainCreateRequest = {
      ...buildChainRequest(sequenceParams(requestForm, entry), clips, {
        motionTailFrames,
        enableAudio,
        openingImage,
      }),
      ...chainFilingFields(requestForm),
    };
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
    const feasibility = await hosts.resolveFeasible(hostRoute.hostId, request, 1, { signal });
    if (!isCurrent()) return;
    if (
      feasibility.kind !== "route" ||
      feasibility.route.hostId !== hostRoute.hostId ||
      feasibility.route.target.baseUrl !== hostRoute.target.baseUrl ||
      feasibility.route.target.apiKey !== hostRoute.target.apiKey
    ) {
      toasts.push(
        feasibility.kind === "route"
          ? "The sequence machine changed while checking dependencies. Nothing was queued."
          : placementFailureMessage(feasibility),
        "error",
      );
      return;
    }
    const { accepted } = await licenseAcceptance.request({
      hostLabel: hostRoute.label,
      target: hostRoute.target,
      requirements: licenseRequirements(feasibility.preview?.pending_downloads),
    });
    if (!accepted || !isCurrent()) return;
    if (editing) {
      const operationId = createUuid();
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
        sequenceAmendInFlight = true;
        sequenceCancellationRequest = () =>
          chains.cancelMutation(editing.hostId, editing.jobId, operationId, hostRoute.target);
        const outcome = await chains.amend(
          editing.hostId,
          editing.jobId,
          amend,
          hostRoute.target,
          operationId,
        );
        if (!isCurrent()) {
          await chains.cancel(editing.hostId, editing.jobId, hostRoute.target).catch(() => {});
          return;
        }
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
      const operationId = createUuid();
      sequenceCancellationRequest = () =>
        chains.cancelMutation(hostRoute.hostId, operationId, operationId, hostRoute.target);
      const jobId = await chains.create(hostRoute.hostId, request, hostRoute.target, operationId);
      if (!isCurrent()) {
        await chains.cancel(hostRoute.hostId, jobId, hostRoute.target).catch(() => {});
        return;
      }
      sequenceStageClipIdsByJob.set(
        `${hostRoute.hostId}:${jobId}`,
        clips.map((clip) => clip.id),
      );
      toasts.push("Sequence queued");
    }
    // The caveat described the handoff, not the submitted job.
    sequenceReuseNotice.value = null;
  } catch (err) {
    if (!isCurrent()) return;
    toasts.push(String(err), "error");
  } finally {
    if (sequenceSubmissionGuard.isCurrent(token)) {
      sequenceSubmitting.value = false;
      sequenceAmendInFlight = false;
      sequenceCancellationRequest = null;
    }
  }
}

function cancelSequenceSubmission() {
  if (!sequenceSubmitting.value) return;
  sequenceSubmissionGuard.invalidate();
  const cancellation = sequenceCancellationRequest;
  const cancellingAmendment = sequenceAmendInFlight;
  sequenceCancellationRequest = null;
  sequenceAmendInFlight = false;
  sequenceSubmitting.value = false;
  preprocessingStatus.value = null;
  if (!cancellation) {
    toasts.push("Sequence preparation cancelled — nothing was queued");
    return;
  }
  toasts.push(
    cancellingAmendment ? "Cancelling the sequence update…" : "Cancelling sequence creation…",
  );
  void confirmCancellation(cancellation)
    .then(() =>
      toasts.push(
        cancellingAmendment
          ? "Sequence update cancelled"
          : "Sequence creation cancelled — nothing was queued",
      ),
    )
    .catch(() =>
      toasts.push("Cancellation could not be confirmed. Check Activity before retrying.", "error"),
    );
}

/** Edit session: submit the current clips as a brand-new job instead. */
async function duplicateSequenceAsNew() {
  draft.stopEditing();
  editSharedBaseline.value = null;
  await generateSequence();
}

/** Edit sequence: load a durable job's effective script into an edit
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
    if (loaded.openingImage?.base64) {
      preserveRestoredSourceCanvas(loaded.openingImage.base64);
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
    if (form.model) draft.bindSequenceModel(form.model);
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

/** Amending an existing clip is not a new print: the one Generate button says
 *  what it will actually do to the clip already on the timeline. */
// "Update clip" only while the button really amends: an edit session parked
// behind the Simple toggle is kept, but Simple's Generate makes a new print.
const buttonLabel = computed(() =>
  composerSubmitting.value
    ? "Cancel"
    : isSequence.value && draft.editing
      ? "Update clip"
      : "Generate",
);
/** The queue depth rides beside the button, never inside its one word. */
const queuedNote = computed(() =>
  generation.pending.length > 0 ? `+${generation.pending.length} queued` : null,
);
const submissionStatus = computed(() =>
  submissionPlanning.value
    ? (preprocessingStatus.value ?? "Checking machine fit and generation route…")
    : preprocessingStatus.value,
);

function cancelSubmissionPlanning() {
  if (!submissionPlanning.value) return;
  submissionGuard.invalidate();
  sequenceSubmissionGuard.invalidate();
  submissionPlanning.value = false;
  preparedSubmitting.value = false;
}

/**
 * The first candidate that can actually shape a frame. A canvasless recipe
 * (a 3-D mesh) requests 0 × 0 by contract, and `0 / 0` is invalid CSS that
 * both rules below would be dropped for — collapsing the frame to nothing.
 * A mesh print's poster dimensions answer instead, and a square is the last
 * resort while one is still developing.
 */
function framedDimension(...candidates: (number | null | undefined)[]): number {
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return 1;
}
const previewWidth = computed(() =>
  framedDimension(
    selectedQueueRender.value?.width,
    job.value?.width,
    job.value?.result?.width,
    form.width,
  ),
);
const previewHeight = computed(() =>
  framedDimension(
    selectedQueueRender.value?.height,
    job.value?.height,
    job.value?.result?.height,
    form.height,
  ),
);
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

/** Fraction of the selected print's denoise the host has reported, or null
 * before it reports a step counter. */
const selectedQueuePreviewFraction = computed(() => {
  const preview = selectedQueueRender.value?.preview;
  return preview && preview.step !== null && preview.total !== null
    ? preview.step / preview.total
    : null;
});
const selectedQueuePreviewSrc = computed(() =>
  selectedQueueRender.value?.preview?.preview_image
    ? `data:image/png;base64,${selectedQueueRender.value.preview.preview_image}`
    : null,
);

const queueStatus = computed(() =>
  buildQueueStatusIndex(
    Object.values(jobsStore.queues).map((snapshot) => ({
      hostId: snapshot.hostId,
      entries: snapshot.entries,
      plan: snapshot.plan,
      paused: snapshot.paused,
    })),
  ),
);

const liveGenerationStatus = computed(() => {
  const selected = selectedQueueRender.value;
  if (selected) {
    const preview = selected.preview;
    if (preview && preview.step !== null && preview.total !== null) {
      return `Developing ${preview.step}/${preview.total}`;
    }
    return preview?.stage ? `${preview.stage}…` : "Preparing selected print…";
  }
  const j = job.value;
  if (!j || j.status === "complete" || j.status === "error") return "";
  if (j.status === "queued") {
    return queueWaitLabel(
      resolveQueueWait(queueStatusFor(queueStatus.value, j.hostId ?? hosts.primaryHost?.id, j.id)),
    );
  }
  if (j.status === "loading") return `${j.stage ?? "Preparing"}…`;
  const copy = jobProgressCopy(j);
  return j.status === "finishing" ? `${copy}…` : copy;
});

/** The caption's left slot: the file this print landed as, or — while it is
 * still developing, and for an inline completion the host never named — the
 * style that is making it. */
const edgeCode = computed(() => {
  const selected = selectedQueueRender.value;
  if (selected) return modelDisplayNameForId(selected.model, installedModels.value);
  const j = job.value;
  if (!j) return "";
  return j.result?.filename ?? modelDisplayNameForId(j.model, installedModels.value);
});

/** The caption's right slot: the print's size and how long it took. */
const captionMeta = computed(() => {
  const selected = selectedQueueRender.value;
  if (selected) {
    const preview = selected.preview;
    return preview && preview.step !== null && preview.total !== null
      ? `${preview.step}/${preview.total}`
      : (preview?.stage ?? "waiting for progress");
  }
  const j = job.value;
  if (!j) return "";
  // A mesh has no pixels to describe — `width`/`height` are its poster's, not
  // the print's — so its geometry is the provenance: the one shared caption
  // every surface writes under a 3-D print.
  // A durable child reports no counts at all, so the viewer's own `ready`
  // facts answer for it. Neither authority may fall back to `width × height`:
  // a canvasless recipe requests 0 × 0 by contract, and printing that reads
  // as a failed render rather than as "this print has no pixels".
  const size = isMeshArtifact(j.result)
    ? meshStatsLabel(
        j.result?.mesh_vertices ?? viewerMeshStats.value?.vertexCount,
        j.result?.mesh_faces ?? viewerMeshStats.value?.triangleCount,
        j.result?.mesh_bounds_min ?? viewerMeshStats.value?.bounds,
        j.result?.mesh_bounds_max,
      )
    : j.result
      ? squareSizeLabel(j.result.width, j.result.height)
      : squareSizeLabel(j.width, j.height);
  const time = j.result ? `${(j.result.generation_time_ms / 1000).toFixed(1)}s` : "";
  return [size, time].filter(Boolean).join(" · ");
});

/** `1024²` for a square canvas, `1216×704` otherwise. */
function squareSizeLabel(width: number, height: number): string {
  return width === height ? `${width}²` : `${width}×${height}`;
}

let templateLoadEpoch = 0;
async function loadTemplate(template: GenerationTemplate) {
  const epoch = ++templateLoadEpoch;
  const hydrated = await hydrateGenerationTemplate(template);
  if (epoch !== templateLoadEpoch) return;
  // buildRequest's pruneRequestForFamily still guards anything the (possibly
  // different) family can't use after the source snapshot is restored. A
  // pre-#787 template lacking `negativePromptDefault` is normalized first so
  // its empty negative reads as "untouched", not the explicit "" opt-out.
  //
  // A template is a set of PARAMETERS. Its snapshot carries whatever title and
  // filing the print it was saved from happened to have (`stripTemplateForm`
  // only strips media), so applying it wholesale would rename and re-file the
  // print in progress and flip the Settings ▸ Library auto-tag mirror.
  keepingPrintIdentity(form, () =>
    Object.assign(form, normalizeLegacyNegativeSnapshot(hydrated.form, installedModels.value)),
  );
  // A template saved before the redesign carries a style preset, and this
  // surface has no control that would show one. Restoring it would put back an
  // invisible prompt rewriter: the words on screen would not be the words sent.
  form.stylePreset = "";
  // A template is PARAMETERS, not a capability snapshot. One saved before the
  // snapshot existed (or on another host) carries none, and applying it over
  // a Hunyuan3D form left the mesh recipe's `glb` pin and zero canvas in
  // place. The installed model's advertised recipe is the authority; nothing
  // installed means nothing may answer for it.
  const templateModel = form.model ? findInstalledModel(installedModels.value, form.model) : null;
  if (templateModel) {
    reconcileModelCapabilities(form, templateModel);
  } else {
    form.recipeCapabilities = null;
    form.mesh = emptyMeshForm();
  }
  if (form.model && !templateModel) {
    toasts.push(`Model "${form.model}" isn't installed — settings applied anyway.`);
  }
  if (hydrated.missingMediaReferences.length > 0) {
    toasts.push(`Re-add media: ${formatTemplateMediaReferences(hydrated.missingMediaReferences)}.`);
  }
}

function siblingDot(s: Job): string {
  if (s.status === "complete") return "text-fg"; // ◉ developed
  if (s.status === "error") return s.outcomeUnknown ? "text-fg-2" : "text-error";
  return "text-fg-dim"; // ◎ pending
}

/** Whether this job produced a WAV rather than a picture or a clip. */
function isAudioResult(job: { result: CompleteEvent | null } | null): boolean {
  return isAudioCompletion(job?.result);
}

/**
 * Whether this job produced a 3-D mesh. Probed BEFORE the audio and video
 * questions everywhere it is asked: a mesh carries neither samples nor
 * frames, so a wider test running first classifies it as a still and draws
 * binary glTF through an `<img>`.
 */
function isMeshResult(job: { result: CompleteEvent | null } | null): boolean {
  return isMeshArtifact(job?.result);
}

/** The rendered PNG a mesh print carries — its only raster, and the viewer's
 * poster while the geometry loads (and forever, if it cannot). A completion
 * synthesized from a durable batch child carries no poster inline, so the
 * host's own thumbnail (the poster it rendered at save time) stands in. */
const resultMeshPoster = computed(() => {
  const result = job.value?.result;
  if (!isMeshArtifact(result)) return "";
  if (result?.mesh_poster) return `data:image/png;base64,${result.mesh_poster}`;
  return durableMeshPoster.value;
});

/** The host the finished print lives on: the frozen generation target, or
 * the primary connection for a job whose target the store no longer holds. */
function resultHostTarget(clientId: number): ApiTarget | null {
  const frozen = generation.targetForJob(clientId);
  if (frozen) return frozen;
  const info = conn.info;
  return info?.baseUrl ? { baseUrl: info.baseUrl, apiKey: info.apiKey ?? null } : null;
}

/**
 * A durable completion names a FILE on its host. The Library opens that same
 * file through the native media bridge (bytes from the Tauri side, wrapped in
 * a blob URL) and has always rendered it; the canvas used to hand the viewer
 * the host's raw http URL and let the webview fetch it itself, which is the
 * one place in the app that did so — and the one place a finished mesh came
 * back as "the 3-D view couldn't start". Load it the Library's way, keep the
 * ticketed/direct URL as the fallback, and fetch the poster beside it.
 */
const durableMeshSrc = ref("");
const durableMeshPoster = ref("");
const durableMeshAttempt = ref(0);
let durableMeshEpoch = 0;
watch(
  () =>
    [
      job.value?.clientId ?? null,
      job.value?.result?.filename ?? null,
      isMeshArtifact(job.value?.result) && !job.value?.result?.image,
      durableMeshAttempt.value,
    ] as const,
  async ([clientId, filename, durableMesh]) => {
    const epoch = ++durableMeshEpoch;
    durableMeshSrc.value = "";
    durableMeshPoster.value = "";
    if (!durableMesh || clientId === null || !filename) return;
    const target = resultHostTarget(clientId);
    if (!target) return;
    const cacheKey = job.value?.hostId ?? `job-${clientId}`;
    const options = { target, cacheKey };
    const [src, poster] = await Promise.all([
      fullSizeMediaUrl(galleryMediaPath(filename, "host"), {
        ...options,
        // A mesh is fetched whole by the viewer, exactly as the Library's
        // AuthedMedia loads it: never the legacy image blob path.
        allowLegacyBlob: false,
      }).catch(() => ""),
      authedMediaUrl(thumbnailPath(filename), options).catch(() => ""),
    ]);
    if (epoch !== durableMeshEpoch) return;
    durableMeshSrc.value = src;
    durableMeshPoster.value = poster;
  },
  { immediate: true },
);

/** One renewed attempt per print: a ticket that expired while the print sat
 * in the rail is re-minted and the viewer remounted; a second failure stays
 * on the poster with the viewer's own note. */
const durableMeshRetried = new Set<number>();
function onResultMeshFail(): void {
  const j = job.value;
  if (!j || !isMeshArtifact(j.result) || j.result?.image) return;
  if (durableMeshRetried.has(j.clientId)) return;
  durableMeshRetried.add(j.clientId);
  void generation
    .refreshRemoteResultUrl(j.clientId, true)
    .catch(() => undefined)
    .finally(() => {
      if (job.value?.clientId === j.clientId) durableMeshAttempt.value += 1;
    });
}

/**
 * The GLB the viewer loads from INLINE bytes, as an object URL.
 *
 * A `data:model/gltf-binary` URL would re-encode tens of megabytes of
 * geometry into the DOM on every render; a Blob keeps the bytes out of the
 * document and is revoked the moment the canvas moves on, so a session of
 * meshes cannot leak them.
 */
const inlineMeshSrc = ref("");
let revokeResultMesh: (() => void) | null = null;
watch(
  () => {
    const result = job.value?.result;
    return isMeshArtifact(result) && result?.image ? result.image : "";
  },
  (base64) => {
    revokeResultMesh?.();
    revokeResultMesh = null;
    if (!base64) {
      inlineMeshSrc.value = "";
      return;
    }
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
    inlineMeshSrc.value = url;
    revokeResultMesh = () => URL.revokeObjectURL(url);
  },
  { immediate: true },
);
onBeforeUnmount(() => {
  revokeResultMesh?.();
  revokeResultMesh = null;
});

/**
 * What the viewer actually loads.
 *
 * A durable batch child names a FILE, not bytes: `applyDurableCompletion`
 * synthesizes the completion from that name and the media arrives separately
 * as `job.resultUrl`, already authorized for this host (a media ticket or a
 * blob the app fetched). Since every Create submit goes through the durable
 * route, that is the ordinary case, and keying the viewer on inline bytes
 * alone left the canvas to the `<img>` arm — the broken-resource icon over
 * black that the acceptance pass saw after every finished 3-D print. The
 * Library's own mesh viewer has always been handed a URL exactly like this
 * one, which is why the same file opened there correctly.
 */
const resultMeshSrc = computed(() => {
  if (inlineMeshSrc.value) return inlineMeshSrc.value;
  if (durableMeshSrc.value) return durableMeshSrc.value;
  const j = job.value;
  return isMeshArtifact(j?.result) && j?.resultUrl ? j.resultUrl : "";
});

/**
 * The geometry the mounted viewer parsed, for a print whose completion
 * reported none. The server publishes the counts only on the live SSE
 * completion; the viewer has the file open either way, so its own `ready`
 * facts are the caption's second authority rather than the canvasless
 * recipe's zero canvas. Cleared whenever the canvas moves to another print.
 */
type ViewerMeshStats = {
  vertexCount: number;
  triangleCount: number;
  bounds: { min: readonly number[]; max: readonly number[] };
};
const viewerMeshStats = ref<ViewerMeshStats | null>(null);
watch(
  () => resultMeshSrc.value,
  () => {
    viewerMeshStats.value = null;
  },
);
function onResultMeshReady(stats: ViewerMeshStats): void {
  viewerMeshStats.value = stats;
}

/** Inline completion bytes as a Blob, so both deliveries reach the one rule. */
function base64Blob(base64: string, mime: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

/**
 * The finished print on the canvas, shaped as the gallery row it already is
 * on the host.
 *
 * Every ordinary desktop submission settles through `applyDurableCompletion`,
 * which records the FILE the host saved and sets `image` to the empty string.
 * So what the print IS — still, clip, mesh, audio — comes from its filename
 * and format, exactly as the Library reads it, and the bytes are fetched from
 * the host on demand. A completion that DID carry bytes inline (a legacy or
 * non-durable stream) and has no filename gets the name the Save action would
 * suggest, so it is still a print with a name.
 */
function canvasPrintEntry(j: Job): MergedPrint | null {
  const r = j.result;
  if (!r) return null;
  const filename =
    r.filename ??
    (r.image ? suggestOutputFilename(r.model, r.seed_used, r.format, j.submittedAtUnixMs) : null);
  if (!filename) return null;
  return {
    item: {
      filename,
      timestamp: Math.floor((j.settledAtMs ?? j.submittedAtUnixMs) / 1000),
      format: r.format,
      metadata: {
        prompt: j.prompt,
        model: r.model,
        seed: r.seed_used,
        steps: j.total,
        guidance: j.guidance,
        width: r.width,
        height: r.height,
      },
    },
    sourceKey: j.hostId ?? "local",
    hostLabel: j.hostLabel ?? "",
    availableOn: [],
  };
}

/**
 * "Use as source" on the canvas — the SAME rule the Library and the History
 * drawer run, so a clip becomes source video and an H3 model gets a first
 * frame or an ordered reference here too. Only the bytes reader differs: an
 * inline completion already holds them, a durable one reads the file back
 * from the machine that rendered it.
 */
async function useCanvasResultAsSource(j: Job, entry: MergedPrint | null): Promise<void> {
  if (!entry) return;
  const outcome = await applyGalleryEntryAsSource(entry, form, async (print) => {
    const inline = j.result?.image;
    if (inline) return base64Blob(inline, mediaMimeType(print.item.filename));
    const target = resultHostTarget(j.clientId);
    if (!target) throw new Error("This print's machine is no longer connected.");
    const bytes = await fetchGalleryMediaBytes(
      galleryMediaPath(print.item.filename, "host"),
      target,
    );
    return new Blob([nativeBytes(bytes)], { type: mediaMimeType(print.item.filename) });
  });
  if (!outcome.ok) {
    toasts.push(outcome.error, "error");
    return;
  }
  toasts.push(outcome.message);
}

/**
 * Make 4 variations: the print that is on the canvas, made again as a batch.
 * Nothing else changes — the words, the style, the size and the seed policy
 * are whatever produced this picture, which is the whole point of the action.
 * The count rides this ONE submission: writing it to the form would leave
 * every later Generate making four.
 *
 * It is offered only for a finished still on a recipe that can repeat: a clip,
 * a mesh and a sound have no batch, and a batch-locked recipe coerces the
 * count to one, so the button would promise four and make exactly one.
 */
const VARIATION_BATCH = 4;

/**
 * The recipe that answers for a PRINT — the checkpoint's own contract, read
 * from whichever machine holds it, exactly as the composer reads the form's.
 * The form is irrelevant here: it may have moved on to another style entirely
 * since this picture was made.
 */
function capabilitiesForPrint(request: GenerateRequest, hostId: string | null) {
  const entry = hostModels.contractEntryForTarget(request.model, hostId);
  const pipeline = request.pipeline ?? null;
  return generationCapabilitiesForFamily(
    entry?.family ?? "",
    request.model,
    pipeline,
    undefined,
    undefined,
    effectiveGenerationRecipe(entry, pipeline),
  );
}

function canMakeVariations(candidate: Job | null): boolean {
  const request = candidate?.request;
  if (isSequence.value || !request) return false;
  return canRepeatPrint(
    candidate,
    batchLockedForRequest(request, capabilitiesForPrint(request, candidate.hostId ?? null)),
  );
}
function makeVariations(count = VARIATION_BATCH) {
  const candidate = job.value;
  if (!candidate || !canMakeVariations(candidate)) return;
  void repeatPrint(candidate, Math.min(Math.max(count, 1), MAX_BATCH_SIZE));
}

/**
 * The print, made again as a batch of `count`.
 *
 * It submits the print's OWN saved request rather than rebuilding one from the
 * live form: the request already carries the fitted source bytes, the baked
 * style, the seed policy and the filing this picture was made with, and every
 * one of those was persisted and stashed at the first submit. So the whole
 * composer half of `generate()` — the prepared-batch guards, the style
 * composition, source preprocessing, the retained-source relay, the media
 * stash — has nothing to do here, and running it would make four of whatever
 * the composer is holding now.
 *
 * What it keeps is everything about WHERE the four will run: the chain-routing
 * refusal, placement (four prints is a new question, and the style may have
 * been evicted since), the licence gate and the missing-model pull offer. An
 * empty composer, or one holding an unrenderable form, does not block it: that
 * refusal is the form's, not this picture's.
 * The host is the print's own — a picture made on
 * another machine re-rendered here is a different picture — and a pinned
 * machine that has gone away errors through the same placement message rather
 * than silently rerouting.
 */
async function repeatPrint(candidate: Job, count: number) {
  const source = candidate.request;
  if (!source || preparedSubmitting.value || submissionPlanning.value) return;
  const submitToken = submissionGuard.begin();
  const submitSignal = submissionGuard.signalFor(submitToken);
  submissionPlanning.value = true;
  try {
    const request: GenerateRequest = { ...source, batch_size: count };
    // The print's own machine, always named: to placement a null selection is
    // AUTOMATIC (any ready machine), and a print this device made carries
    // hostId null, so the fallback is the primary rather than the fleet.
    const printHostId = candidate.hostId ?? hosts.primaryHost?.id ?? null;
    const entry = hostModels.contractEntryForTarget(request.model, printHostId);
    const family = entry?.family ?? "";
    const routingModel = entry
      ? { default_frames: entry.default_frames, source_image: entry.source_image }
      : null;
    const chainRouting = decideGenerateRequestRouting(request, family, routingModel);
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
    const planningRequest =
      chainRouting.kind === "chain" ? buildAutoChainRequest(request, chainRouting) : request;
    // No LoRA provenance check: a saved request carries the look's PATH, not
    // the machine that supplied it, and the resubmission goes to the print's
    // own host, where those paths are exactly as valid as they were when it
    // was made. A host that has gone away fails placement below with the
    // sentence that names the machine, which is the same refusal.
    const feasibility = await hosts.resolveFeasible(printHostId, planningRequest, count, {
      signal: submitSignal,
    });
    if (!submissionGuard.isCurrent(submitToken)) return;
    if (feasibility.kind !== "route") {
      const offered = offerMissingModelPull(feasibility, {
        model: request.model,
        modelFamily: family,
        // The print's own request, verbatim — never the composer's quick
        // rewrite, which belongs to a different, unsubmitted print.
        request,
        batch: count,
        chainRouting: chainRouting.kind === "chain" ? chainRouting : null,
        requestOptions: {},
        // The request is already final — nothing is fitted or composed after
        // the pull, so the resumed submission is byte-for-byte this one.
        resumeAfterPull: true,
      });
      if (!offered) toasts.push(placementFailureMessage(feasibility), "error");
      return;
    }
    const route = freezeModelFamily(feasibility.route, family)!;
    const { accepted } = await licenseAcceptance.request({
      hostLabel: route.label,
      target: route.target,
      requirements: licenseRequirements(feasibility.preview?.pending_downloads),
    });
    if (!accepted || !submissionGuard.isCurrent(submitToken)) return;
    let settled: ReturnType<typeof generation.submitBatch>["settled"];
    try {
      ({ settled } = generation.submitBatch(request, count, route, chainRouting, {}));
    } catch (error) {
      toasts.push(error instanceof Error ? error.message : String(error), "error");
      return;
    }
    missingModel.value = null;
    void settled.then((done) => {
      for (const warning of new Set(done.flatMap((finished) => finished.requestWarnings))) {
        toasts.push(warning, "warning");
      }
      const ok = done.filter((finished) => finished.status === "complete").length;
      const failedCount = done.filter(
        (finished) => finished.status === "error" && !finished.outcomeUnknown,
      ).length;
      if (ok > 0) {
        toasts.push(
          failedCount > 0
            ? `Generated ${ok} of ${done.length} variations. ${failedCount} failed; the rest were saved to My images.`
            : ok === 1
              ? "Generated — saved to My images"
              : `Generated ${ok} pictures — saved to My images`,
        );
      }
    });
  } finally {
    if (submissionGuard.isCurrent(submitToken)) submissionPlanning.value = false;
  }
}

// ── Make bigger (the caption's upscale door) ────────────────────────────────
// One print, one machine: the canvas result has exactly one origin, so this
// carries neither the Library's authority picker nor its framewise video
// recovery. A host that can upscale a gallery file in place does; anything
// older streams the bytes back and saves the result beside them.
const upscaleEntry = ref<MergedPrint | null>(null);
const upscaleModel = ref("");
const upscaleBusy = ref(false);
const upscaleError = ref("");

function canUpscaleCanvasResult(j: Job): boolean {
  return (
    j.status === "complete" &&
    !j.result?.video_frames &&
    !isAudioResult(j) &&
    !isMeshResult(j) &&
    !!canvasPrintEntry(j)
  );
}

function openCanvasUpscale(j: Job) {
  const entry = canvasPrintEntry(j);
  if (!entry) return;
  upscaleEntry.value = entry;
  upscaleError.value = "";
  upscaleModel.value = defaultUpscaler(models.upscalers);
}

function closeCanvasUpscale() {
  upscaleEntry.value = null;
  upscaleBusy.value = false;
  upscaleError.value = "";
}

async function startCanvasUpscale() {
  const entry = upscaleEntry.value;
  const target = entry ? hostGallery.targetOf(entry.sourceKey) : null;
  if (!entry || !target || upscaleBusy.value) return;
  upscaleBusy.value = true;
  upscaleError.value = "";
  try {
    if (hosts.capabilities[entry.sourceKey]?.video_upscale?.gallery_image === true) {
      const result = await upscaleLibraryImage(target, entry.item.filename, upscaleModel.value);
      toasts.push(`Made bigger — ${result.filename}`);
    } else {
      const upscaled = await upscaleImage({
        model: upscaleModel.value,
        image: await readGalleryMediaBase64(entry, hostGallery),
        target,
      });
      const stem = entry.item.filename.replace(/\.[^.]+$/, "");
      const saved = await ipc.saveOutputBytes(`${stem}-upscaled.png`, upscaled);
      toasts.push(`Made bigger — saved as ${saved}`);
    }
    void hostGallery.refreshHost(entry.sourceKey);
    closeCanvasUpscale();
  } catch (error) {
    upscaleError.value = error instanceof Error ? error.message : String(error);
  } finally {
    upscaleBusy.value = false;
  }
}

/**
 * Whether the canvas result can be written out as a file right now.
 *
 * Not "does it carry bytes": every ordinary desktop generation settles through
 * `applyDurableCompletion`, which names the host's file and leaves `image`
 * empty, so gating on inline bytes hid Save from every print the app makes.
 * A named file on a reachable machine is saveable; inline bytes (a legacy or
 * non-durable stream) still are too.
 */
function canSaveCanvasResult(j: Job): boolean {
  const result = j.result;
  if (!result || result.video_frames || isAudioResult(j)) return false;
  if (result.image) return true;
  return !!result.filename && !!resultHostTarget(j.clientId);
}

function saveCanvasResult(j: Job) {
  const result = j.result;
  if (!result) return;
  if (result.image) {
    const filename =
      result.filename ??
      suggestOutputFilename(result.model, result.seed_used, result.format, j.submittedAtUnixMs);
    void ipc
      .saveMediaBytes(filename, result.image)
      .then((saved) => showSavedMediaToast(toasts, saved))
      .catch((error) =>
        toasts.push(error instanceof Error ? error.message : String(error), "error"),
      );
    return;
  }
  const entry = canvasPrintEntry(j);
  const target = resultHostTarget(j.clientId);
  if (!entry || !result.filename) return;
  if (!target) {
    toasts.push("This print’s machine is no longer connected.", "error");
    return;
  }
  // The same road the Library and the Lightbox take, so one print saved from
  // two places lands under one name.
  void saveGalleryMedia(target, result.filename, suggestedSaveName(entry.item))
    .then((saved) => showSavedMediaToast(toasts, saved))
    .catch((error) => toasts.push(error instanceof Error ? error.message : String(error), "error"));
}

/**
 * Copy image. A durable completion has no inline bytes, so the copy takes the
 * picture the canvas is ALREADY showing — `resultUrl`, the media the native
 * bridge (or the host) resolved for the viewer — rather than the empty field.
 */
async function copyCanvasImage(j: Job): Promise<void> {
  const result = j.result;
  if (!result) return;
  const mime = result.format === "jpeg" ? "image/jpeg" : `image/${result.format}`;
  if (result.image) {
    await copyBase64ImageToClipboard(result.image, mime);
    return;
  }
  const url = j.resultUrl;
  if (!url) throw new Error("This print’s picture hasn’t loaded yet.");
  await copyImageBytesToClipboard(url, {
    fetchImage: async (path) => new Uint8Array(await (await fetch(path)).arrayBuffer()),
    mimeType: mime,
  });
}

/** Whether there is a picture to copy: raster only, and bytes from somewhere. */
function canCopyCanvasImage(j: Job): boolean {
  const result = j.result;
  if (!result || result.video_frames || isAudioResult(j) || isMeshResult(j)) return false;
  return !!result.image || !!j.resultUrl;
}

function canvasMenu(): MenuEntry[] {
  const j = job.value;
  if (!j) return [];
  const live = j.status !== "complete" && j.status !== "error";
  const sourceEntry = canvasPrintEntry(j);
  return [
    {
      label: "Cancel",
      danger: true,
      disabled: !live,
      action: () =>
        void generation
          .cancel(j.clientId)
          .then((cancelled) => {
            if (cancelled) toasts.push("Cancelled");
          })
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          ),
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
      disabled: !canCopyCanvasImage(j),
      action: () => {
        void copyCanvasImage(j)
          .then(() => toasts.push("Image copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      },
    },
    {
      label: isMeshResult(j) ? "Save mesh" : "Save image",
      disabled: !canSaveCanvasResult(j),
      action: () => saveCanvasResult(j),
    },
    {
      // Also the caption's own button. The caption collapses its word actions
      // on a narrow frame (a portrait print), so every one of them has to be
      // reachable from here or hiding it would remove it from the app.
      label: "Make 4 variations",
      disabled: !canMakeVariations(j),
      action: () => makeVariations(),
    },
    {
      label: "Make bigger",
      disabled: !canUpscaleCanvasResult(j),
      action: () => openCanvasUpscale(j),
    },
    {
      label: "Start from this photo",
      // Whether this print CAN be a source is a question about the print, not
      // about which delivery the completion used: a mesh is geometry and an
      // audio print has no pixels, both answered from the filename the host
      // saved. Never gate on inline bytes — every ordinary submission settles
      // through `applyDurableCompletion`, which carries none.
      disabled:
        j.status !== "complete" || !sourceEntry || !canUseGalleryEntryAsSource(sourceEntry.item),
      action: () => void useCanvasResultAsSource(j, sourceEntry),
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
      label: "Show in My images",
      disabled: j.status !== "complete",
      action: () => void router.push("/library"),
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

/**
 * Why the selected recipe refuses a prompt rewrite, or `null` when it reads
 * one. `capabilities.prompt.mode: "ignored"` (Hunyuan3D has no text encoder
 * anywhere in the family) means a rewrite changes nothing about the render,
 * and the host answers such a transform with ONE advisory result rather than
 * the requested variations — so no request is worth sending and the
 * missing-expander pull is never worth offering. The composer derives the
 * same answer from the same form, so the disabled control and this refusal
 * necessarily carry one sentence.
 */
const promptTransformBlocked = computed(() =>
  promptTransformBlockedReason(form.recipeCapabilities?.promptMode),
);
/** True when the intent was refused; the caller returns. */
function refusePromptTransform(): boolean {
  const reason = promptTransformBlocked.value;
  if (!reason) return false;
  toasts.push(reason);
  return true;
}

function expansionInputs(count: number): PreparedExpansionInputs {
  const request = buildRequest(form);
  return {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    task: expansionTaskForRequest(form.family, request),
    context: expansionContextForRequest(form.family, request, promptRecipeFromForm(form)),
    requestedCount: count,
    stylePreset: null,
    selectedHostPolicy: stickyTarget.value,
  };
}

async function remixForCurrentPrompt(replacePrepared = false) {
  if (refusePromptTransform()) return;
  if (!form.prompt.trim() || !form.model || expansionRunning.value) return;
  submissionGuard.invalidate();
  sequenceSubmissionGuard.invalidate();
  const route = currentExpansionRoute.value;
  // Frozen before the request: an Auto rerank mid-flight must not move the
  // print's machine out from under the reviewed set.
  const printRoute = generationRoute.value;
  if (!route) {
    expansionError.value = unavailableExpansionHostMessage();
    return;
  }
  const request = buildRequest(form);
  const task = expansionTaskForRequest(form.family, request);
  const source = promptSource(form.prompt, form.originalPrompt, remixSource.value);
  const dimensions = defaultRemixDimensions(task, false);
  // Match the Batch value the composer actually presents. Capability/source
  // constraints can force the effective value to one while preserving the
  // user's saved raw preference for a later compatible model.
  const requestedCount = effectiveBatchSize.value;
  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = null;
  expansionAttemptHostLabel.value = route.label;
  try {
    const response = await remixPrompt(
      {
        source_prompt: source.prompt,
        ...(source.rootPrompt ? { root_prompt: source.rootPrompt } : {}),
        source_kind: source.kind,
        model_family: form.family,
        variations: requestedCount,
        task,
        context: expansionContextForRequest(form.family, request, promptRecipeFromForm(form)),
        dimensions,
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const variants = validateRemixVariants(response.variants, requestedCount, {
      promptIgnored: !!promptTransformBlocked.value,
    });
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
      quickExpansionSnapshot.value = {
        requestToken: token,
        originalPrompt: response.root_prompt ?? response.source_prompt,
        expandedPrompt: selected.prompt,
        model: form.model,
        family: form.family,
        task,
        stylePreset: null,
        selectedHostPolicy: stickyTarget.value,
        route: frozenGenerationRoute(printRoute, route),
        ...expansionRouteProvenance(printRoute, route),
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
        stylePreset: null,
        selectedHostPolicy: stickyTarget.value,
      },
      frozenGenerationRoute(printRoute, route),
      variants.map((variant) => variant.prompt),
      token,
    );
    Object.assign(batch, expansionRouteProvenance(printRoute, route));
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

/**
 * The route prepared/quick work freezes for the PRINT. Expansion may run on a
 * peer that has the expander; the generation itself never follows it there.
 *
 * `captured` is read BEFORE the request, never after: telemetry can rerank
 * Auto / Most capable while expansion is in flight, and freezing whatever
 * `generationRoute` says on return would hand the reviewed set to a machine
 * the user never resolved.
 */
function frozenGenerationRoute(captured: HostRoute | null, expansionRoute: HostRoute): HostRoute {
  const frozen = captured ?? expansionRoute;
  return { ...frozen, target: { ...frozen.target } };
}

/** Record the expansion host only when it left the captured generation route. */
function expansionRouteProvenance(
  captured: HostRoute | null,
  expansionRoute: HostRoute,
): { expansionRoute?: HostRoute } {
  if (!captured || captured.hostId === expansionRoute.hostId) return {};
  return { expansionRoute: { ...expansionRoute, target: { ...expansionRoute.target } } };
}

/**
 * Where to pull a missing expander. Every reachable machine is a legitimate
 * target (`planModelInstall` is the one policy), but a prepared batch must
 * freeze ONE route, so the machine the expansion just tried wins whenever it
 * is a target at all.
 */
function expansionPullRoute(attempted: HostRoute): HostRoute {
  const reachable = hosts.all.filter((host) => host.status === "ready" && host.baseUrl);
  const owners = reachable
    .filter((host) => hosts.capabilities[host.id]?.expand?.model_present === true)
    .map((host) => host.id);
  const plan = planModelInstall(reachable, owners, {
    inventoryKnown: (host) => hosts.capabilities[host.id]?.expand != null,
  });
  const preferred =
    plan.targets.find((target) => target.host.id === attempted.hostId) ?? plan.targets[0];
  if (!preferred) return attempted;
  return hosts.resolveRoute(preferred.host.id) ?? attempted;
}

function offerExpansionPull(model: string, attempted: HostRoute): string {
  const route = expansionPullRoute(attempted);
  expansionMissingModel.value = { model, route };
  expansionPullAttempt.value = null;
  return `The expansion model ${model} isn't installed on ${route.label}.`;
}

function describeExpansionError(error: unknown, route: HostRoute): string {
  const message = error instanceof Error ? error.message : String(error);
  const missingModel = parseMissingExpandModel(message);
  if (missingModel) return offerExpansionPull(missingModel, route);
  expansionMissingModel.value = null;
  expansionPullAttempt.value = null;
  return `Expansion failed on ${route.label}: ${message}`;
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
  if (refusePromptTransform()) return;
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
  // Frozen before the request, for the same reason as remix above.
  const printRoute = generationRoute.value;
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
  // No eligible machine has the expander and this one is KNOWN to lack it:
  // offer the pull instead of a request the host has already said it refuses.
  // A retry after a pull (`routeOverride`) always goes to the wire — the
  // capability snapshot lags the download that just completed.
  if (
    !routeOverride &&
    expansionRouteDecision.value.kind === "missing" &&
    capability?.model_present === false
  ) {
    expansionError.value = offerExpansionPull(expandModelId(capability), route);
    return;
  }

  const token = preparationGuard.begin();
  expansionRunning.value = true;
  expansionError.value = null;
  expansionMissingModel.value = null;
  try {
    const response = await expandPrompt(
      inputs.sourcePrompt,
      {
        variations: count,
        ...(inputs.family ? { modelFamily: inputs.family } : {}),
        task: inputs.task,
        ...(inputs.context ? { context: inputs.context } : {}),
      },
      route.target,
    );
    if (!preparationGuard.isCurrent(token)) return;
    const prompts = validateExpandedPrompts(response.expanded, count, {
      promptIgnored: !!promptTransformBlocked.value,
    });
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
        current.selectedHostPolicy !== inputs.selectedHostPolicy ||
        !hostStillReady
      ) {
        expansionError.value =
          "The prompt, model, or generation host changed while expansion was running. Expand again to use the current inputs.";
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
        route: frozenGenerationRoute(printRoute, route),
        ...expansionRouteProvenance(printRoute, route),
      };
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
    preparedBatch.value = Object.assign(
      createPreparedExpansionBatch(
        inputs,
        frozenGenerationRoute(printRoute, route),
        prompts,
        token,
      ),
      expansionRouteProvenance(printRoute, route),
    );
    quickExpansionSnapshot.value = null;
  } catch (error) {
    if (!preparationGuard.isCurrent(token)) return;
    expansionError.value = describeExpansionError(error, route);
  } finally {
    if (preparationGuard.isCurrent(token)) expansionRunning.value = false;
  }
}

/** True while a quick expansion still owns the composer: its frozen route or
 * its undo. */
function quickExpansionActive(): boolean {
  return quickExpansionSnapshot.value !== null || quickExpansionOriginal.value !== null;
}

/**
 * Drop every trace of a quick expansion without touching the prompt text: the
 * frozen route snapshot, the undo and the provenance. Undo re-arms the
 * pre-expansion state through here and then puts the original prompt back;
 * a history recall goes through here and then installs the recalled prompt.
 */
function releaseQuickExpansion() {
  if (!quickExpansionActive()) return;
  submissionGuard.invalidate();
  form.originalPrompt = null;
  quickExpansionOriginal.value = null;
  quickExpansionSnapshot.value = null;
  expansionError.value = null;
}

function restoreQuickExpansion() {
  const original = quickExpansionOriginal.value;
  if (original === null) return;
  releaseQuickExpansion();
  form.prompt = original;
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
  const attemptId = ++expansionPullRequestId;
  const attempt: ExpansionPullAttempt = {
    id: attemptId,
    notificationOwner: `create-expansion:${attemptId}`,
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
  const notificationHostId = downloadNotificationHostId(route);
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
    downloads.armNotificationAction(missing.model, notificationHostId, attempt.notificationOwner, {
      kind: "create",
    });
    const jobId = await startCatalogDownload(missing.model, route.target, route.kind === "remote");
    if (expansionPullAttempt.value?.id !== attempt.id) {
      downloads.clearNotificationAction(
        missing.model,
        notificationHostId,
        attempt.notificationOwner,
      );
      return;
    }
    expansionPullAttempt.value.jobId = jobId;
    downloads.refineNotificationAction(
      missing.model,
      notificationHostId,
      attempt.notificationOwner,
      jobId,
    );
  } catch (error) {
    if (expansionPullAttempt.value?.id !== attempt.id) {
      downloads.clearNotificationAction(
        missing.model,
        notificationHostId,
        attempt.notificationOwner,
      );
      return;
    }
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
    downloads.clearNotificationAction(missing.model, notificationHostId, attempt.notificationOwner);
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

function downloadNotificationHostId(route: HostRoute): string | null {
  return route.kind === "local" || route.hostId === downloads.primaryHostId ? null : route.hostId;
}

const expansionPullBucket = computed<DownloadsState>(() => {
  const route = expansionPullAttempt.value?.route ?? expansionMissingModel.value?.route;
  return route ? downloadBucketForRoute(route) : { activeJobs: [], queued: [], history: [] };
});

watch(expansionMissingModel, (missing) => {
  if (!missing && expansionPullAttempt.value) {
    const attempt = expansionPullAttempt.value;
    downloads.clearNotificationAction(
      attempt.model,
      downloadNotificationHostId(attempt.route),
      attempt.notificationOwner,
      attempt.jobId,
    );
    expansionPullAttempt.value = null;
  }
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
  onPromptAuthored(form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed);
}

function onPromptAuthored(prompt: string, source: PromptAuthoringSource = "typed") {
  // Clip mode's words belong to the selected scene, so neither the one-shot
  // prompt's provenance nor its quick rewrite has anything to say about them.
  if (isSequence.value) return;
  // A ↑/↓ recall replaces the whole prompt, so the prepared rewrite has
  // nothing left to describe: release it instead of raising the stale banner
  // whose recovery actions would re-expand a prompt no longer on screen.
  if (!quickTransformSurvivesAuthoring(source)) releaseQuickExpansion();
  applyAuthoredPrompt(form, prompt, quickExpansionSnapshot.value !== null, source);
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
  signal?: AbortSignal,
): Promise<boolean> {
  // The frozen draft's own recipe snapshot decides the reference contract, so
  // an exclusive (Klein) recipe fits the well the request will actually carry
  // rather than the family heuristic's guess at it.
  const draftCaps = generationCapabilitiesForForm(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
    draft.recipeCapabilities,
  );
  // A canvasless recipe (a 3-D mesh) has no canvas to fit onto: its own
  // advertised target is 0 × 0, so every fit below would resize the source to
  // nothing. The engine reads the source at its native resolution.
  const canvasless = draft.recipeCapabilities
    ? draft.recipeCapabilities.canvasless
    : isMeshFamily(draft.family);
  if (canvasless) return true;
  // H3 FL2VA boundaries take the same client-side fit as an ordinary source,
  // coerced maskless (H3 has no repaint mask).
  if (draftCaps.sourceImageMode === "h3-boundaries") {
    try {
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
                model,
                image,
                ...(route ? { target: route.target } : {}),
                ...(signal ? { signal } : {}),
                onProgress: (message) => (preprocessingStatus.value = message),
              }),
            onStatus: (message) => (preprocessingStatus.value = message),
          },
        )) ?? emptyMinimaxH3AuthoringState();
      return true;
    } catch (error) {
      if (signal?.aborted) return false;
      const message = error instanceof Error ? error.message : String(error);
      toasts.push(`Source preprocessing failed: ${message}`, "error");
      return false;
    } finally {
      preprocessingStatus.value = null;
    }
  }
  // Ref2VA: pending image crops are applied at the original resolution on the
  // frozen draft, before the placement preview and any upload conversion.
  if (draftCaps.sourceImageMode === "ordered-references" && draft.h3Authoring) {
    try {
      draft.h3Authoring = await applyMinimaxH3ReferenceCrops(draft.h3Authoring, domCanvasOps);
      return true;
    } catch (error) {
      if (signal?.aborted) return false;
      const message = error instanceof Error ? error.message : String(error);
      toasts.push(`Reference preprocessing failed: ${message}`, "error");
      return false;
    }
  }
  if (draftCaps.sourceImageMode === "qwen-edit" && draft.imageAttachments[0]) {
    try {
      const target = resolveSourceConditioningTarget(
        { width: draft.width, height: draft.height },
        selectedEntry.value ?? draft.family,
        draft.pipeline,
      );
      const result = await applySourceFitPreprocess(
        {
          source: draft.imageAttachments[0],
          mask: null,
          policy: coerceSourceFitForMaskless(draft.sourceFit),
          target,
        },
        {
          ops: domCanvasOps,
          cache: sourceFitCache,
          upscale: (image, model) =>
            upscaleImage({
              model,
              image,
              ...(route ? { target: route.target } : {}),
              ...(signal ? { signal } : {}),
              onProgress: (message) => (preprocessingStatus.value = message),
            }),
          onStatus: (message) => (preprocessingStatus.value = message),
        },
      );
      if (result.source) draft.imageAttachments[0] = result.source;
      return true;
    } catch (error) {
      if (signal?.aborted) return false;
      const message = error instanceof Error ? error.message : String(error);
      toasts.push(`Source preprocessing failed: ${message}`, "error");
      return false;
    } finally {
      preprocessingStatus.value = null;
    }
  }
  // Source fit applies to the image the request will actually carry — on an
  // exclusive (Klein) recipe that is the source well only while it is the
  // active one; a parked source is never preprocessed.
  const draftConditioning = conditioningForRequest(draftCaps.sourceImageMode, {
    hasSource: Boolean(draft.sourceImage),
    referenceCount: draft.imageAttachments.length,
    lastWrite: draft.exclusiveWell ?? null,
  });
  if (!draftCaps.supportsImg2img || draftConditioning !== "source") return true;
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
            ...(signal ? { signal } : {}),
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
    if (signal?.aborted) return false;
    const message = error instanceof Error ? error.message : String(error);
    toasts.push(`Source preprocessing failed: ${message}`, "error");
    return false;
  } finally {
    preprocessingStatus.value = null;
  }
}

function sourcePreprocessingNeedsRoute(draft: ReturnType<typeof cloneGenerateForm>): boolean {
  const draftCaps = generationCapabilitiesForForm(
    draft.family,
    draft.model,
    draft.pipeline,
    draft.guidanceCapabilities,
    draft.sourceImageCapability,
    draft.recipeCapabilities,
  );
  const conditioning = conditioningForRequest(draftCaps.sourceImageMode, {
    hasSource: Boolean(draft.sourceImage),
    referenceCount: draft.imageAttachments.length,
    lastWrite: draft.exclusiveWell ?? null,
  });
  return (
    draftCaps.supportsImg2img &&
    ((conditioning === "source" && Boolean(draft.sourceImage)) ||
      (draftCaps.sourceImageMode === "qwen-edit" && Boolean(draft.imageAttachments[0]))) &&
    draft.sourceFit.mode === "upscale-then-fit" &&
    Boolean(draft.sourceFit.upscalerModel)
  );
}

// Moves in lockstep with ComposerCard's `promptMissing`: one is the disabled
// Generate button, the other this silent early return, and an enabled control
// that quietly does nothing is exactly the dead end the prepared-expansion
// invariant forbids.
const promptMissing = computed(
  () => promptRequired(promptInputForForm(form)) && !form.prompt.trim(),
);
const h3RequireFirstFrame = computed(
  () =>
    effectiveGenerationRecipe(selectedEntry.value, form.pipeline)?.capabilities.source_image ===
      "required" ||
    selectedEntry.value?.source_image === "required" ||
    form.sourceImageCapability === "required",
);
const h3AuthoringError = computed(() =>
  minimaxH3AuthoringError(form.family, form.model, form.h3Authoring, h3RequireFirstFrame.value),
);

/**
 * One visible blocker authority drives both the button and the submit guard.
 * Keep transient in-flight states on the button label; this list is only for
 * conditions the user can correct or explicitly resolve.
 */
const generationInputBlockerReason = computed<string | null>(() => {
  // Required media is the first actionable step for a conditioned model. The
  // ordinary empty-prompt blocker stays visually quiet, but must not hide the
  // H3 opening-frame correction until the user starts typing.
  if (h3AuthoringError.value) return h3AuthoringError.value;
  if (promptMissing.value) return "Add a prompt before generating.";
  if (!form.model) return "Choose a style first.";
  if (chainValidationError.value) return chainValidationError.value;
  if (quickStaleReasons.value.length > 0) {
    return "The prepared rewrite no longer matches the prompt, model, or machine. Choose a recovery action above.";
  }
  if (expansionRunning.value) return "Wait for prompt preparation to finish.";
  return null;
});

const composerBlockerReason = computed<string | null>(() => {
  if (promptMissing.value && !h3AuthoringError.value) return null;
  if (generationInputBlockerReason.value) return generationInputBlockerReason.value;
  if (preparedBatch.value) {
    return "Use the reviewed variations panel to generate this prepared batch, or discard it to return to one-shot generation.";
  }
  return null;
});

const composerDisabled = computed(
  () => generationInputBlockerReason.value !== null || preparedBatch.value !== null,
);

/** Non-blocking pre-generate advisory: an off-profile custom size submits
 * anyway (the server is the authority), but the inspector can be scrolled
 * away from the Generate button, so the advisory rides the composer too. */
const composerWarningReason = computed<string | null>(() =>
  composerBlockerReason.value
    ? null
    : resolutionValidationWarning(
        form.width,
        form.height,
        selectedEntry.value ?? null,
        form.pipeline,
      ),
);

/*
 * One composer, two modes. In clip mode it carries the SELECTED SCENE's words,
 * the timeline above owns the refusal, and Generate submits the whole chain —
 * so every binding below asks `isSequence` once and nothing downstream has to.
 */
const activeScene = computed(
  () => draft.clips.find((clip) => clip.id === draft.activeClipId) ?? draft.clips[0] ?? null,
);
const activeSceneIndex = computed(() => {
  const index = draft.clips.findIndex((clip) => clip.id === activeScene.value?.id);
  return index >= 0 ? index : 0;
});
const composerPrompt = computed(() =>
  isSequence.value ? (activeScene.value?.prompt ?? "") : null,
);
/** A one-shot on a clip style: the plain render the Simple sub-mode makes.
 *  Read from the one section authority, never a second family list. */
const isSimpleClip = computed(
  () => !isSequence.value && outputKindFor(draft.output, form.family) === "clip",
);
const composerPlaceholder = computed(() => {
  if (isSimpleClip.value) return "Describe the clip";
  if (!isSequence.value) return null;
  return activeSceneIndex.value === 0
    ? "Scene 1 — describe how the clip opens"
    : `Scene ${activeSceneIndex.value + 1} — describe what happens next`;
});

/**
 * The Length chip's contract, which is the SAME row the inspector's Clip card
 * reads — both write `form.frames`, so the chip and the slider are one control
 * shown twice. Absent outside the simple clip: a sequence's lengths are
 * per-scene and a still has none.
 */
const composerLengthContract = computed<VideoFrameContract | null>(() => {
  if (!isSimpleClip.value || form.predictDuration) return null;
  const entry = contractEntry.value;
  return {
    ...(entry ?? {}),
    family: entry?.family ?? form.family,
    name: entry?.name ?? form.model,
    source_image: entry?.source_image ?? form.sourceImageCapability,
  };
});
function writeScenePrompt(value: string) {
  const scene = activeScene.value;
  if (scene) scene.prompt = value;
}
/**
 * What Generate must refuse for in clip mode. The timeline raises its own
 * refusal while it is mounted, but it is `v-else`'d away entirely when no
 * installed style can make a clip — so the view answers for it there, and the
 * one composer is locked whether or not the timeline exists.
 */
const timelineBlockedReason = ref<string | null>(SEQUENCE_NEEDS_STYLE);
const sequenceBlockedReason = computed(() =>
  showSequenceEmpty.value ? SEQUENCE_NEEDS_STYLE : timelineBlockedReason.value,
);
const composerRefusal = computed(() =>
  isSequence.value ? sequenceBlockedReason.value : composerBlockerReason.value,
);
const composerLocked = computed(() =>
  isSequence.value ? sequenceBlockedReason.value !== null : composerDisabled.value,
);
const composerSubmitting = computed(() =>
  isSequence.value ? sequenceSubmitting.value : submissionPlanning.value,
);

/**
 * The timeline's destructive questions, asked over the whole workbench. The
 * bench strip is a `container-type: size` box, which makes it the containing
 * block for an absolutely positioned dialog — one rendered inside it would be
 * centred and clipped in a 320px strip.
 */
const sequenceConfirmation = ref<SequenceConfirmation | null>(null);
function resolveSequenceConfirmation(answer: "confirm" | "cancel") {
  const pending = sequenceConfirmation.value;
  if (pending) pending[answer]();
}

function composerGenerate() {
  if (isSequence.value) void generateSequence();
  else void generate();
}
function composerCancel() {
  if (isSequence.value) cancelSequenceSubmission();
  else cancelSubmissionPlanning();
}

// The shared rule picks the sentence: the surface's own wording while the
// prompt is required, the optional wording once conditioning makes it
// optional, and the image-preparation wording for a recipe that never reads
// it (a mesh model has no text encoder, so nothing here "animates").
const emptyCanvasGuidance = computed(() =>
  promptGuidance(
    promptInputForForm(form),
    "Describe an image below, pick a look, and press Generate. Everything runs on your own machine.",
  ),
);

/** `batchOverride` submits a one-off count — Make 4 variations — without
 *  touching the batch size the form keeps for every later Generate. */
async function generate(batchOverride: number | null = null) {
  if (generationInputBlockerReason.value || preparedSubmitting.value || submissionPlanning.value)
    return;
  clearSelectedQueueRender();
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
  const submitSignal = submissionGuard.signalFor(submitToken);
  preparedSubmitting.value = preparedSubmission !== null;
  submissionPlanning.value = true;
  try {
    const draft = cloneGenerateForm(form);
    if (batchOverride !== null) draft.batchSize = batchOverride;
    const routingModel = selectedEntry.value
      ? {
          default_frames: selectedEntry.value.default_frames,
          source_image: selectedEntry.value.source_image,
        }
      : null;
    const originalSource = draft.sourceImage
      ? {
          base64: draft.sourceImage,
          filename: draft.sourceImageName ?? "Source image",
          width: draft.sourceImageWidth,
          height: draft.sourceImageHeight,
          sourceFit: parseSourceFitPolicy(draft.sourceFit) ?? defaultSourceFitPolicy(),
        }
      : draft.h3Authoring?.firstFrame?.data
        ? {
            base64: draft.h3Authoring.firstFrame.data,
            filename: draft.h3Authoring.firstFrame.filename,
            width: draft.h3Authoring.firstFrame.width,
            height: draft.h3Authoring.firstFrame.height,
            mime: draft.h3Authoring.firstFrame.mimeType,
            sourceFit: parseSourceFitPolicy(draft.sourceFit) ?? defaultSourceFitPolicy(),
          }
        : null;
    const draftCaps = generationCapabilitiesForFamily(draft.family, draft.model);
    // No style bake here: the desktop composer has no preset strip, and a
    // preset surviving in the form (an old template, a persisted draft) would
    // rewrite the prompt invisibly — the words on screen are the words sent.
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
    let placementPreview: GenerationPlacementPreview | null = null;
    const routeRequiredForPreprocessing = sourcePreprocessingNeedsRoute(draft);
    let routeResolvedAgainstFinalRequest = false;
    let sourcePreprocessed = false;
    if (!route && !routeRequiredForPreprocessing) {
      if (!(await preprocessSourceFit(null, draft, submitSignal))) return;
      if (!submissionGuard.isCurrent(submitToken)) return;
      sourcePreprocessed = true;
    }
    const preliminaryRequest = buildRequest(draft);
    const preliminaryRouting = decideGenerateRequestRouting(
      preliminaryRequest,
      draft.family,
      routingModel,
    );
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
    const loraBinding = loraHostBinding(draft.loras);
    if (loraBinding.kind === "conflict") {
      toasts.push(
        "The selected LoRAs have mixed or missing machine provenance. Remove them and choose the stack again from one machine.",
        "error",
      );
      return;
    }
    if (loraBinding.kind === "bound") {
      const boundRoute = hosts.resolveRoute(loraBinding.hostId, draft.model);
      if (!boundRoute || !loraBindingMatchesRoute(loraBinding, boundRoute)) {
        toasts.push(
          "The machine that supplied the selected LoRA is no longer the same server. Reconnect it or remove the LoRA.",
          "error",
        );
        return;
      }
      if (route && !loraBindingMatchesRoute(loraBinding, route)) {
        toasts.push(
          "The prepared print is frozen to a different machine than the selected LoRA. Re-expand it or remove the LoRA.",
          "error",
        );
        return;
      }
    }
    const generateTarget =
      loraBinding.kind === "bound"
        ? loraBinding.hostId
        : (appPrefs.settings?.generateTargetHost ?? null);
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
      const feasibility = await hosts.resolveFeasible(generateTarget, planningRequest, batch, {
        signal: submitSignal,
      });
      if (!submissionGuard.isCurrent(submitToken)) return;
      if (feasibility.kind !== "route") {
        // Nothing can run this print. When the only thing in the way is the
        // model itself, offer the pull instead of a dead-end toast — but only
        // once the source is final, so the resumed request is the exact one
        // the user submitted.
        const offered = offerMissingModelPull(feasibility, {
          model: preliminaryRequest.model,
          modelFamily: draft.family,
          request: pullResumeRequest(preliminaryRequest),
          batch,
          chainRouting: preliminaryRouting.kind === "chain" ? preliminaryRouting : null,
          requestOptions: {},
          // The source still has to be fitted against the machine that runs
          // it, so this request is not the one that would render.
          resumeAfterPull: sourcePreprocessed,
        });
        if (!offered) toasts.push(placementFailureMessage(feasibility), "error");
        return;
      }
      route = feasibility.route;
      placementPreview = feasibility.preview ?? null;
      routeResolvedAgainstFinalRequest = sourcePreprocessed;
    }
    if (!sourcePreprocessed) {
      if (!(await preprocessSourceFit(route, draft, submitSignal))) return;
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
    let request = buildRequest(draft);
    if (quickExpansionSnapshot.value?.promptTransform) {
      request.prompt_transform = quickExpansionSnapshot.value.promptTransform;
    }
    let retainedSourceOption: BatchRequestOptions["retainedSource"];
    const retained = composer.retainedSource;
    const retainedVersion = retained ? composer.retainedSourceVersion : null;
    if (retained?.inventory.availability === "available" && retained.inventory.members.length > 0) {
      const fieldForRole: Record<string, keyof GenerateRequest> = {
        source_image: "source_image",
        identity_image: "id_image",
        identity_images: "id_images",
        edit_images: "edit_images",
        mask_image: "mask_image",
        control_image: "control_image",
        audio_file: "audio_file",
        audio_file_path: "audio_file",
        source_video: "source_video",
        source_video_path: "source_video",
        extend_video: "extend_video",
        extend_video_path: "extend_video",
        keyframes: "keyframes",
        references: "references",
      };
      const wanted = retained.inventory.members.filter((member) => {
        const field = fieldForRole[member.role];
        if (!field) return false;
        // Descriptor-only H3 references are topology, not byte authority; the
        // server session hydrates those descriptors in place.
        if (field === "references") {
          return (
            request.references?.every((reference) => reference.media.authority === "descriptor") ===
            true
          );
        }
        return request[field] == null;
      });
      if (wanted.length > 0) {
        const sameHost =
          route?.target.baseUrl === retained.origin.baseUrl &&
          route.target.apiKey === retained.origin.apiKey;
        if (sameHost && batch === 1) {
          retainedSourceOption = {
            filename: retained.filename,
            memberIds: wanted.map((member) => member.member_id),
          };
        } else {
          try {
            request = await relayRetainedSourceMedia(
              retained.filename,
              wanted,
              request,
              retained.origin,
            );
          } catch (error) {
            toasts.push(
              `Couldn’t restore retained source media: ${error instanceof Error ? error.message : String(error)}`,
              "error",
            );
            return;
          }
          if (retainedVersion !== null && !composer.isRetainedSourceCurrent(retainedVersion))
            return;
        }
      }
    }
    if (request.source_image && originalSource) {
      void persistGenerationSourceMedia(request.source_image, originalSource);
    }
    if (request.id_image) {
      // Saved metadata records `id_image_sha256`, never the face bytes, so the
      // photo has to be kept locally under the digest of exactly what shipped
      // or Reuse settings has nothing to look up. Best-effort: a failed write
      // costs a reattach later, never this print.
      const decoded = imageDimensionsFromBase64(request.id_image);
      void persistIdentityPhoto(request.id_image, {
        filename: request.id_image_name ?? "identity photo",
        width: decoded?.width ?? null,
        height: decoded?.height ?? null,
      });
    }
    const chainRouting = decideGenerateRequestRouting(request, draft.family, routingModel);
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
    // Submission options are a pure derivation of the prepared batch; resolve
    // them before the finalized check so a pull offered there resumes the
    // exact ordered prompts.
    const requestOptions: BatchRequestOptions = preparedSubmission
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
    if (retainedSourceOption) requestOptions.retainedSource = retainedSourceOption;
    if (route && !routeResolvedAgainstFinalRequest) {
      const finalized = await hosts.resolveFeasible(route.hostId, finalizedPlanningRequest, batch, {
        signal: submitSignal,
      });
      if (!submissionGuard.isCurrent(submitToken)) return;
      const finalizedRoute = finalized.kind === "route" ? finalized.route : null;
      if (
        !finalizedRoute ||
        finalizedRoute.hostId !== route.hostId ||
        finalizedRoute.target.baseUrl !== route.target.baseUrl ||
        finalizedRoute.target.apiKey !== route.target.apiKey ||
        (finalizedRoute.instanceId ?? null) !== (route.instanceId ?? null)
      ) {
        if (
          finalized.kind !== "route" &&
          offerMissingModelPull(finalized, {
            model: request.model,
            modelFamily: draft.family,
            request,
            batch,
            chainRouting: chainRouting.kind === "chain" ? chainRouting : null,
            requestOptions,
          })
        ) {
          return;
        }
        toasts.push(
          finalized.kind === "route"
            ? "The selected machine changed while the finalized source request was being checked. Nothing was queued."
            : placementFailureMessage(finalized),
          "error",
        );
        return;
      }
      route = finalizedRoute;
      placementPreview = finalized.kind === "route" ? (finalized.preview ?? null) : null;
    }
    route = freezeModelFamily(route, draft.family)!;
    const { accepted } = await licenseAcceptance.request({
      hostLabel: route.label,
      target: route.target,
      requirements: licenseRequirements(placementPreview?.pending_downloads),
    });
    if (!accepted || !submissionGuard.isCurrent(submitToken)) return;
    if (retainedVersion !== null && !composer.isRetainedSourceCurrent(retainedVersion)) return;
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
    // A machine that cannot carry this print refuses it by name and queues
    // nothing; there is no second submission path to fall through to.
    let settled: ReturnType<typeof generation.submitBatch>["settled"];
    try {
      ({ settled } = generation.submitBatch(request, batch, route, chainRouting, requestOptions));
    } catch (error) {
      toasts.push(error instanceof Error ? error.message : String(error), "error");
      preparedSubmitting.value = false;
      return;
    }
    composer.invalidateRetainedSource();
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
    const recordedPrompt = preparedSubmission?.originalPrompt ?? request.prompt;
    composerRef.value?.record?.(recordedPrompt);
    recordPromptHistoryCache(
      availablePromptHistoryStorage(),
      hosts.all.map((host) => ({ hostId: host.id, hostLabel: host.label })),
      route?.hostId ?? "local",
      { prompt: recordedPrompt, model: request.model, used_at: Date.now() },
    );
    void settled.then((done) => {
      void loadPromptHistory();
      for (const warning of new Set(done.flatMap((job) => job.requestWarnings))) {
        toasts.push(warning, "warning");
      }
      const ok = done.filter((s) => s.status === "complete").length;
      // An unknown outcome is advisory, never counted as a failure.
      const failedCount = done.filter((s) => s.status === "error" && !s.outcomeUnknown).length;
      const failed = done.find((s) => s.status === "error" && !s.outcomeUnknown);
      if (ok > 0) {
        if (failedCount > 0) {
          toasts.push(
            `Generated ${ok} of ${done.length} variations. ${failedCount} failed; the rest were saved to My images.`,
            "error",
          );
        } else {
          toasts.push(
            ok === 1
              ? "Generated — saved to My images"
              : `Generated ${ok} pictures — saved to My images`,
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
    if (submissionGuard.isCurrent(submitToken)) {
      preparedSubmitting.value = false;
      submissionPlanning.value = false;
    }
  }
}

const promptHistoryCoordinator = new PromptHistoryCoordinator();
async function loadPromptHistory() {
  try {
    const history = await promptHistoryCoordinator.load(
      availablePromptHistoryStorage(),
      hosts.all.map((host) => ({
        hostId: host.id,
        hostLabel: host.label,
        fetchable: host.status === "ready" && Boolean(host.baseUrl),
        source: { baseUrl: host.baseUrl ?? "", apiKey: host.apiKey },
      })),
      (target) => fetchHistoryFrom(target),
    );
    if (history) promptHistory.value = history.map((entry) => entry.prompt);
  } catch {
    // No history API (older engine / DB off) — arrows just move the caret.
  }
}

// Auto-select a model only when none is set. With the form persisted in a
// store, the `!form.model` guard means "the first visit this launch" — a
// remount after the user chose a model leaves their choice untouched.
//
// That first visit lands on the style the person LEFT: the last-used memory
// names the style and the section, and the form takes it as soon as any
// machine reports it. While a ready machine is still reporting, a remembered
// style that has not appeared yet is waited for rather than replaced — the
// first inventory to arrive is this device's, and picking FLUX from it would
// lock the form before the machine that holds the style could answer. Only
// once the whole fleet has reported without it does the usual pick apply.
const installedInventorySettled = computed(
  () => !models.loading && !hostModels.loading && hostModels.allReadyHostsFetched,
);
watch(
  [installedModels, installedInventorySettled],
  ([installed, settled]) => {
    if (form.model || installed.length === 0) return;
    // In Scenes the sequence watcher above owns the pick (memory-aware too)
    // and may leave the form EMPTY on purpose when the pinned machine has no
    // style that can join scenes; refilling it here would fight that.
    if (isSequence.value) return;
    const section = lastUsed.lastSection;
    const rememberedName = section ? lastUsed.bySection[section] : null;
    const remembered = rememberedName ? findInstalledModel(installed, rememberedName) : null;
    if (remembered) {
      formStore.applyModel(remembered);
      return;
    }
    if (rememberedName && !settled) return;
    const preferred = preferredInstalledModel(installed);
    if (preferred) formStore.applyModel(preferred);
  },
  { immediate: true },
);

// Whatever style the form holds is the one its section was last used with.
// One writer for every path that sets a model — the picker, the doors, Use
// these settings, an edit session — so the memory can never disagree with
// the screen.
watch(
  () => [form.model, form.family] as const,
  ([model, family]) => {
    if (!model) return;
    lastUsed.remember(outputKindForModel({ family: family ?? "" }), model);
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

// Re-scope on every configured-host and reachability transition. Watching only
// ready ids leaves a forgotten offline host's cached prompts behind forever.
watch(
  () => promptHistoryHostSignature(hosts.all),
  () => {
    void loadPromptHistory();
  },
);

/** Monotonic token: only the latest prefill's async source restore may touch
 *  the form — a superseded restore (newer prefill, ⌘N, user edits) is
 *  dropped silently. Bumped by every prefill and by ⌘N. */
let restoreEpoch = 0;
let authoritativeReuseApply = false;
function invalidateRetainedRestore(): void {
  restoreEpoch += 1;
  composer.invalidateRetainedSource();
}

/**
 * "Use these settings again" shows the picture too. The restored recipe is
 * already in the form, so the canvas job carries the request that recipe
 * builds — which is what Make 4 variations and Make bigger read — and the
 * print's own media, fetched from its bucket. A recipe the form cannot yet
 * build (a model no machine has, an unrestorable source) still shows the
 * picture; the request falls back to the print's own words and model.
 */
function showRestoredPrint(print: GalleryPrintOnCanvas): void {
  let request: GenerateRequest;
  try {
    request = buildRequest(form);
  } catch {
    request = {
      prompt: print.metadata.prompt,
      model: print.metadata.model,
      width: print.metadata.width,
      height: print.metadata.height,
      steps: print.metadata.steps,
      seed: print.metadata.seed,
    };
  }
  generation.showGalleryPrint(print, request);
}

function applyPrefill() {
  const prefill = composer.take();
  if (!prefill) return;
  const pendingRetainedVersion = composer.takePendingRetainedSourceApplyVersion();
  if (
    pendingRetainedVersion !== null &&
    composer.isRetainedSourceCurrent(pendingRetainedVersion) &&
    "metadata" in prefill
  ) {
    authoritativeReuseApply = true;
    void nextTick(() => {
      authoritativeReuseApply = false;
    });
  }
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
  if (
    "request" in prefill &&
    (prefill.request?.source_image || prefill.request?.edit_images?.length)
  ) {
    preserveRestoredSourceCanvas(
      prefill.request.edit_images?.[0] ?? prefill.request.source_image ?? "",
    );
  }
  applyPrefillToForm(form, prefill, installedModels.value);
  inspectSelectedQueueRender("metadata" in prefill ? prefill.queueSelection : undefined);
  if ("metadata" in prefill && prefill.print) showRestoredPrint(prefill.print);
  discloseMissingRestoredModel();
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
    // Independent of the source restore above: identity is its own partition,
    // and a print may carry a face photo on a checkpoint that takes no source
    // image at all — the source restore's own early-outs must not skip it.
    void restorePrefillIdentityPhoto(prefill.metadata, restoreEpoch);
  } else if ("request" in prefill && prefill.request) {
    void restoreRequestSource(prefill.request, restoreEpoch);
  }
  void nextTick(() => composerRef.value?.focus?.());
}

/**
 * Fill in the bytes behind a reused print's identity photo.
 *
 * `applyMetadataToForm` restores the recorded knobs and a bytes-less reattach
 * descriptor; the photo itself lives only in the local content-addressed
 * stash, keyed by the digest of exactly what shipped. A miss is deliberately
 * silent here — the well already renders `IDENTITY_PHOTO_UNAVAILABLE` for a
 * descriptor with no bytes, and rendering a different face would be worse
 * than saying the original is gone.
 */
async function restorePrefillIdentityPhoto(metadata: OutputMetadata, epoch: number) {
  const wanted = form.identityImage;
  if (!wanted || wanted.base64) return;
  const restored = await restoreIdentityPhoto(metadata.id_image_sha256).catch(() => null);
  if (!restored || epoch !== restoreEpoch) return;
  const slot = form.identityImage;
  // Cleared, reattached, or replaced while the lookup ran — the user wins.
  if (!slot || slot.base64 || slot.filename !== wanted.filename) return;
  form.identityImage = { filename: restored.filename || slot.filename, base64: restored.base64 };
}

async function restoreRequestSource(request: GenerateRequest, epoch: number) {
  if (!request.source_image) return;
  const effective = request.source_image;
  const restored = await sha256HexOfBase64(effective)
    .then((sha256) => restoreGenerationSourceMedia(sha256))
    .catch(() => null);
  if (
    !restored ||
    epoch !== restoreEpoch ||
    form.model !== request.model ||
    form.sourceImage !== effective
  )
    return;
  preserveRestoredSourceCanvas(restored.base64);
  form.sourceImage = restored.base64;
  form.sourceImageName = restored.filename;
  await nextTick();
  if (epoch !== restoreEpoch || form.sourceImage !== restored.base64) return;
  form.sourceImageWidth = restored.width ?? null;
  form.sourceImageHeight = restored.height ?? null;
  form.width = request.width;
  form.height = request.height;
  const fit = parseSourceFitPolicy(request.source_fit);
  if (fit) form.sourceFit = fit;
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
  // Which well this PRINT used. An exclusive (Klein) recipe has two, so the
  // print's own recorded conditioning decides — not the layout.
  const attachmentMode =
    caps.value.sourceImageMode === "single-or-references"
      ? (metadata.edit_image_sha256s?.length ?? 0) > 0
      : caps.value.sourceImageMode !== "single";
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
  if (caps.value.sourceImageMode === "h3-boundaries") {
    // FL2VA restores its endpoints into the boundary slots, not the single
    // source well. Reuse leaves bytes-less reattach descriptors there; each
    // one keyed with provenance resolves stash-first, then by gallery
    // filename across every connected host.
    // Snapshot the descriptors this restore is answering, so a slot the user
    // cleared or reattached while the fetch was in flight is never clobbered.
    const wantedFirst =
      form.h3Authoring?.firstFrame && !form.h3Authoring.firstFrame.data
        ? form.h3Authoring.firstFrame
        : null;
    const wantedLast =
      form.h3Authoring?.lastFrame && !form.h3Authoring.lastFrame.data
        ? form.h3Authoring.lastFrame
        : null;
    if (!wantedFirst && !wantedLast) return;
    const boundaries = await restoreH3Boundaries(metadata, deps);
    if (epoch !== restoreEpoch || form.model !== modelAtStart) return;
    if (caps.value.sourceImageMode !== "h3-boundaries") return;
    let failed = 0;
    const commit = (
      endpoint: "firstFrame" | "lastFrame",
      restored: { base64: string; filename: string | null } | null,
      wanted: { filename: string } | null,
    ) => {
      if (!wanted) return;
      const slot = form.h3Authoring?.[endpoint];
      // Cleared (null), reattached (data), or replaced with a different
      // descriptor while we fetched — the user's action wins.
      if (!slot || slot.data || slot.filename !== wanted.filename) return;
      if (!restored) {
        failed += 1;
        return;
      }
      const result = setMinimaxH3PickedImageBoundary(form.h3Authoring, endpoint, {
        filename: restored.filename ?? slot.filename,
        base64: restored.base64,
      });
      if (result.ok) form.h3Authoring = result.state;
      else failed += 1;
    };
    commit("firstFrame", boundaries.firstFrame, wantedFirst);
    commit("lastFrame", boundaries.lastFrame, wantedLast);
    if (failed > 0) {
      toasts.push(
        "Couldn't restore the original frame media — the file wasn't found on any connected host. Reattach it to generate.",
        "error",
      );
    }
    return;
  }
  const editRestore = attachmentMode ? await restoreEditImages(metadata, deps) : null;
  const restored = attachmentMode ? null : await restoreSourceImage(metadata, deps);
  // The lookups can take seconds (cold gallery, cross-host fetch). Bail if
  // this restore was superseded: a newer prefill or ⌘N bumped the epoch, the
  // user attached their own source, the model changed under us, or the new
  // family can't take an image at all.
  if (epoch !== restoreEpoch || form.model !== modelAtStart) return;
  if (!caps.value.supportsImg2img) return;
  if (attachmentMode ? form.imageAttachments.length > 0 : Boolean(form.sourceImage)) return;
  if (attachmentMode && editRestore?.images.length) {
    // The strip ceiling is the RECIPE's, never a client constant.
    const max = caps.value.referenceImages?.max ?? null;
    const restoredImages = max === null ? editRestore.images : editRestore.images.slice(0, max);
    const omitted = editRestore.images.length - restoredImages.length;
    preserveRestoredSourceCanvas(restoredImages[0]!);
    form.imageAttachments = restoredImages;
    form.exclusiveWell = "references";
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
  } else if (!attachmentMode && restored) {
    const generationWidth = metadata.generation_width ?? metadata.width;
    const generationHeight = metadata.generation_height ?? metadata.height;
    preserveRestoredSourceCanvas(restored.base64);
    form.sourceImage = restored.base64;
    form.sourceImageName = restored.filename;
    form.exclusiveWell = "source";
    await nextTick();
    if (
      epoch !== restoreEpoch ||
      form.model !== modelAtStart ||
      form.sourceImage !== restored.base64
    )
      return;
    form.sourceImageWidth = restored.width ?? null;
    form.sourceImageHeight = restored.height ?? null;
    form.width = generationWidth;
    form.height = generationHeight;
    const fit = parseSourceFitPolicy(metadata.source_fit);
    if (fit) form.sourceFit = fit;
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
  draft.bindSequenceModel(form.model);

  const notes = [sequenceReuseNote(clips.length, plan.lossy)];
  if (raised > 0) {
    notes.push(sequenceReuseClampNote(modelDisplayNameForId(form.model, installedModels.value)));
  }
  sequenceReuseNotice.value = notes.join(" · ");
  void loadChainLimits();
}

/**
 * Simple | Scenes — the two ways of making a clip, from the toolbar toggle and
 * from the palette. The sub-mode is a remembered preference on the draft; the
 * switch itself moves the draft's OUTPUT, keeping the clip style either way.
 *
 * Nothing is destroyed in either direction. Going to Scenes seeds scene 1 from
 * the words and the length already on the composer when no scene has been
 * written yet; coming back parks the scenes exactly as they are — the
 * timeline's own Clear the clip is the only eraser.
 */
function setClipMode(mode: ClipMode) {
  draft.clipMode = mode;
  if (mode === "simple") {
    if (!isSequence.value) return;
    draft.setOutput(
      "single",
      { getPrompt: () => form.prompt, setPrompt: (value) => (form.prompt = value) },
      sequenceDefaultFrames.value,
    );
    return;
  }
  if (isSequence.value) return;
  const seedPrompt = form.prompt.trim();
  const seedFrames = form.frames;
  // Mirror the inspector's model rule, and swap BEFORE seeding clips: a style
  // that cannot join scenes is parked and replaced by the first one that can,
  // so the new clips default their length from the model that will render them.
  const current = selectedEntry.value;
  if (!current || !sequenceCapableModels.value.some((m) => m.name === current.name)) {
    draft.lastSingleModel = form.model || null;
    const pick = lastUsed.pick("clip", sequenceCapableModels.value);
    if (pick) formStore.applyModel(pick);
  }
  const unwritten = draft.clips.every((clip) => !clip.prompt.trim());
  draft.setOutput(
    "sequence",
    { getPrompt: () => form.prompt, setPrompt: (value) => (form.prompt = value) },
    sequenceDefaultFrames.value,
  );
  if (!unwritten) return;
  const first = draft.clips[0];
  if (!first) return;
  if (seedPrompt) first.prompt = seedPrompt;
  const seededLength = sceneLengthFor(seedFrames);
  if (seededLength !== null) first.frames = seededLength;
}

/**
 * The composer's length as ONE scene's length: the longest offered scene that
 * does not exceed it. A simple clip may be longer than any single scene — that
 * is the auto-chained one-shot — so it becomes the longest scene there is
 * rather than a length the chain validator refuses. `null` = shorter than the
 * shortest scene, so scene 1 keeps the model's own default.
 */
function sceneLengthFor(frames: number): number | null {
  if (!Number.isFinite(frames) || frames <= 0) return null;
  const options = sequenceFrameOptions(
    sequenceClipFrameCap(selectedSequenceEntry.value, chainLimits.value),
    sequenceMotionTail.value,
    selectedSequenceEntry.value?.family,
  );
  const fits = options.filter((option) => option <= frames);
  return fits.length > 0 ? fits[fits.length - 1]! : null;
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
  if (!on) {
    sequenceReuseNotice.value = null;
    // No bench is mounted in one-shot mode, and a still reserves more than
    // twice a clip's canvas floor — clamping here measured the timeline
    // against a floor it was never under and shrank a height the user dragged.
    return;
  }
  // The bench floor is mode-aware; a persisted one-shot height may sit below
  // the sequence floor and would otherwise scroll the composer.
  clampBenchToViewport();
});

/**
 * Act on a shell intent exactly once, whether it was raised while this view
 * was mounted or on the way here. The palette and the native menu raise the
 * intent and then navigate, so a plain watcher registered on the already
 * incremented tick would drop every cross-workspace command silently.
 */
function onCreateIntent(intent: CreateIntent, tick: () => number, run: () => void) {
  watch(
    tick,
    () => {
      if (ui.consumeIntent(intent)) run();
    },
    { immediate: true },
  );
}

// ⌘N — clear the composer for a fresh generation, keeping the model.
onCreateIntent(
  "newGeneration",
  () => ui.newGenerationTick,
  () => {
    invalidateRetainedRestore();
    preparationGuard.invalidate();
    preparedBatch.value = null;
    expansionRunning.value = false;
    expansionError.value = null;
    expansionMissingModel.value = null;
    expansionAttemptHostLabel.value = null;
    quickExpansionOriginal.value = null;
    quickExpansionSnapshot.value = null;
    submissionGuard.invalidate();
    formStore.clearComposer();
    void nextTick(() => composerRef.value?.focus?.());
  },
);

// A clear control may live several components below this view. Fence retained
// gallery authority whenever the user removes any staged private-media slot;
// additions and replacements keep the current reuse draft intact.
watch(
  () =>
    Number(Boolean(form.sourceImage)) +
    form.imageAttachments.length +
    Number(Boolean(form.identityImage)) +
    Number(Boolean(form.maskImage)) +
    Number(Boolean(form.controlImage)) +
    Number(Boolean(form.audioFile)) +
    Number(Boolean(form.sourceVideo)) +
    Number(Boolean(form.extendVideo)) +
    form.keyframes.length,
  (count, previous) => {
    if (count < previous && !authoritativeReuseApply) invalidateRetainedRestore();
  },
);

// ⌘R — randomize the seed.
onCreateIntent(
  "randomizeSeed",
  () => ui.randomizeSeedTick,
  () => {
    form.seed = String(randomSeed());
  },
);

// Palette — take the clip apart into scenes.
onCreateIntent(
  "clipScenes",
  () => ui.clipScenesTick,
  () => setClipMode("scenes"),
);

// Palette's Make a short clip and File ▸ New Clip — the Short clip door,
// exactly as the toolbar's segment opens it (remembered way, remembered
// style), whether or not New image was already open.
const outputKindDoor = useOutputKindDoor(
  () => form,
  (mode) => inspectorRef.value?.setOutputMode(mode),
);
onCreateIntent(
  "shortClip",
  () => ui.shortClipTick,
  () => outputKindDoor.setOutputKind("clip"),
);

// ⌥↩ / palette — the picture on the canvas, made four more times.
onCreateIntent(
  "makeVariations",
  () => ui.makeVariationsTick,
  () => makeVariations(),
);

// Menu ▸ Generate / Expand Prompt reuse the composer actions. In sequence
// output the same intent submits the sequence.
onCreateIntent(
  "generate",
  () => ui.generateTick,
  () => (isSequence.value ? void generateSequence() : void generate()),
);
onCreateIntent(
  "expand",
  () => ui.expandTick,
  () => {
    if (refusePromptTransform()) return;
    composerRef.value?.expand?.();
  },
);

onMounted(() => {
  // Hydrate cached prompt history even when no engine is reachable. A later
  // ready-host transition refreshes and replaces each live host's slice.
  if (!import.meta.env.TEST) void loadPromptHistory();
  if (!import.meta.env.TEST) liveActivity.start();
  window.addEventListener("resize", clampBenchToViewport);
  clampBenchToViewport();
  observeComposerHeight();
  void listenForNativeImageDrops();
  // The persisted sequence draft wins on ordinary visits; a ?output=sequence
  // deep-link is consumed once and stripped.
  draft.hydrate();
  consumeOutputQuery();
});

onBeforeUnmount(() => {
  clearSelectedQueueRender();
  promptHistoryCoordinator.invalidate();
  if (!import.meta.env.TEST) liveActivity.stop();
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  sequenceSubmissionGuard.invalidate();
  const sequenceCancellation = sequenceCancellationRequest;
  sequenceCancellationRequest = null;
  sequenceAmendInFlight = false;
  if (sequenceCancellation) {
    void confirmCancellation(sequenceCancellation).catch(() => {});
  }
  clearSequenceStageMedia();
  nativeImageDropUnmounted = true;
  stopNativeImageDrop?.();
  stopBenchResize();
  window.removeEventListener("resize", clampBenchToViewport);
  composerObserver?.disconnect();
  composerObserver = null;
});
</script>

<template>
  <StarterCards v-if="showStarterCards" @browse="router.push('/models')" />

  <div v-else data-test="generate-layout" class="relative flex h-full min-h-0 overflow-hidden">
    <div
      v-if="nativeImageDragOver"
      data-test="native-image-drop-overlay"
      class="pointer-events-none absolute inset-3 z-40 flex items-center justify-center rounded-window border-2 border-dashed border-accent bg-bg-deep/90 text-base text-accent shadow-md"
    >
      Drop image to load settings and use as source
    </div>

    <!-- Main column: header / canvas / activity / composer -->
    <div class="flex min-w-0 flex-1 flex-col">
      <CreateHeader
        :form="form"
        @open-tab="inspectorTab = $event"
        @set-output="inspectorRef?.setOutputMode($event)"
      />
      <ClipModeStrip :form="form" @set-clip-mode="setClipMode" />

      <div
        ref="workbenchRef"
        data-test="generate-workbench"
        class="relative flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        <!-- Canvas. Its floor is BOUND, not a utility: the bench clamp
             reserves the same number, and a hard-coded class is how the two
             drifted to 320 and 144 and clipped the composer away. -->
        <div
          data-test="generate-canvas"
          class="flex flex-1 items-center justify-center overflow-hidden bg-bg-crust p-7"
          :style="{ minHeight: `${canvasFloor}px` }"
        >
          <!-- Prepared variations review takes the canvas while it is open. -->
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
          <div
            v-else-if="job || selectedQueueRender"
            class="flex h-full w-full min-h-0 flex-col items-center"
          >
            <div
              data-test="preview-region"
              class="grid min-h-0 w-full flex-1 place-items-center self-stretch overflow-hidden [container-type:size]"
            >
              <div
                class="relative max-h-full w-full max-w-full overflow-hidden border border-border-control bg-media-bed [container-name:preview-frame] [container-type:inline-size]"
                data-test="preview-frame"
                :style="previewFrameStyle"
                @contextmenu="job ? contextMenu.open($event, canvasMenu()) : undefined"
              >
                <!-- A mesh is checked before everything else: it carries
                     neither frames nor samples, so every arm below would draw
                     its binary glTF as a picture. The poster is its still. -->
                <MeshViewer
                  v-if="resultMeshSrc"
                  :key="`${resultMeshSrc}#${durableMeshAttempt}`"
                  class="absolute inset-0"
                  data-test="preview-mesh"
                  :src="resultMeshSrc"
                  :poster="resultMeshPoster"
                  :alt="job?.prompt || 'Generated 3-D mesh'"
                  auto-rotate
                  expandable
                  @ready="onResultMeshReady"
                  @fail="onResultMeshFail"
                />
                <!-- Audio is checked next: an audio print has no frames, so
                     the video probe falls through and the <img> below renders
                     a WAV — a broken canvas at the end of a render that
                     actually succeeded. The waveform is the visual; the
                     transport sits over it. -->
                <div
                  v-else-if="job?.resultUrl && isAudioResult(job)"
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
                  v-if="
                    selectedQueuePreviewSrc || (job && job.status !== 'complete' && job.previewUrl)
                  "
                  data-test="develop-preview"
                  :src="selectedQueuePreviewSrc ?? job?.previewUrl ?? ''"
                  alt=""
                  class="absolute inset-0 h-full w-full object-contain"
                  :style="{
                    filter: `blur(${Math.max(2, 14 - 12 * (selectedQueuePreviewFraction ?? (job ? jobProgress(job) : 0)))}px)`,
                  }"
                />
                <!-- The grain canvas paints edge-to-edge (temperature wash), so
                     once previews exist it thins out with progress to reveal
                     the forming print underneath. -->
                <DevelopCanvas
                  v-if="selectedQueueRender || (job && job.status !== 'complete')"
                  :seed="job?.visualSeed ?? 'selected-queue-render'"
                  :progress="selectedQueuePreviewFraction ?? (job ? jobProgress(job) : 0)"
                  :phase="job ? jobPhase(job) : 'developing'"
                  class="absolute inset-0"
                  :style="{
                    opacity:
                      selectedQueuePreviewSrc || job?.previewUrl
                        ? String(
                            Math.max(
                              0.18,
                              1 -
                                (selectedQueuePreviewFraction ?? (job ? jobProgress(job) : 0)) *
                                  0.9,
                            ),
                          )
                        : '1',
                  }"
                />
                <!-- Grain is the signature; the ring overlays it until the
                     first latent preview arrives, after which the forming
                     print itself takes over. The status stays below the frame
                     where the grain cannot obscure it. -->
                <div
                  v-if="
                    (selectedQueueRender && !selectedQueuePreviewSrc) ||
                    (job && job.status !== 'complete' && !job.previewUrl)
                  "
                  data-test="develop-progress"
                  class="pointer-events-none absolute inset-0 flex items-center justify-center"
                >
                  <ProgressRing
                    :value="
                      selectedQueuePreviewFraction !== null
                        ? selectedQueuePreviewFraction * 100
                        : job
                          ? jobProgress(job) * 100
                          : 0
                    "
                    :size="96"
                    show-label
                  />
                </div>

                <!-- Caption strip (README §04): the result's actions live in
                     the image's own caption, never as a pill overlapping it.
                     Plain words first, the mono truth beside them. -->
                <div
                  data-test="canvas-caption"
                  class="absolute inset-x-0 bottom-0 flex items-center gap-3 border-t border-border bg-bg-crust py-1.5 pr-1.5 pl-2.5"
                  @contextmenu.stop
                >
                  <span
                    v-if="liveGenerationStatus"
                    data-test="generation-live-status"
                    class="min-w-0 flex-1 truncate font-mono text-micro text-accent"
                  >
                    {{ liveGenerationStatus }}
                  </span>
                  <span
                    v-else
                    data-test="generation-edge-code"
                    class="min-w-0 flex-1 truncate font-mono text-micro text-fg-2"
                    :title="edgeCode"
                  >
                    {{ edgeCode }}
                  </span>
                  <span
                    v-if="captionMeta"
                    data-test="generation-caption-meta"
                    class="shrink-0 font-mono text-micro text-fg-dim whitespace-nowrap"
                  >
                    {{ captionMeta }}
                  </span>
                  <template v-if="job && job.status === 'complete'">
                    <span class="h-4 w-px shrink-0 bg-border" aria-hidden="true" />
                    <button
                      v-if="canSaveCanvasResult(job)"
                      type="button"
                      data-test="canvas-save"
                      class="caption-action caption-action--word"
                      @click="saveCanvasResult(job)"
                    >
                      <Icon name="save" :size="14" />
                      Save
                    </button>
                    <button
                      v-if="canMakeVariations(job)"
                      type="button"
                      data-test="canvas-variations"
                      class="caption-action caption-action--word"
                      @click="makeVariations()"
                    >
                      Make 4 variations
                    </button>
                    <button
                      v-if="canUpscaleCanvasResult(job)"
                      type="button"
                      data-test="canvas-upscale"
                      class="caption-action caption-action--word"
                      @click="openCanvasUpscale(job)"
                    >
                      Make bigger
                    </button>
                    <button
                      type="button"
                      data-test="canvas-more"
                      class="caption-action caption-action--icon"
                      title="More"
                      aria-label="More actions"
                      @click="contextMenu.open($event, canvasMenu())"
                    >
                      <Icon name="more" :size="14" />
                    </button>
                  </template>
                </div>
              </div>
            </div>

            <!-- Batch dots -->
            <div v-if="siblings.length > 1" class="mt-2 flex items-center gap-1.5">
              <span
                v-for="(s, i) in siblings"
                :key="i"
                class="font-mono text-sm"
                :class="siblingDot(s)"
                :title="`Variation ${i + 1} of ${siblings.length}: ${s.status}${s.error ? `. ${s.error}` : ''}`"
                :aria-label="`Variation ${i + 1} of ${siblings.length}: ${s.status}${s.error ? `. ${s.error}` : ''}`"
              >
                {{ s.status === "complete" ? "◉" : s.status === "error" ? "◉" : "◎" }}
              </span>
              <span class="font-mono text-micro text-fg-dim whitespace-nowrap ml-1">
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
              class="min-h-0 w-full flex-1 rounded-inner border border-border-control bg-media-bed object-contain"
              autoplay
              controls
              loop
              playsinline
              @timeupdate="onSequenceTimeUpdate"
            />
            <div
              class="absolute left-3 top-3 flex items-center gap-2 rounded-control border border-border bg-bg/90 px-2 py-1.5 shadow-md backdrop-blur"
            >
              <span class="font-mono text-micro text-fg-dim whitespace-nowrap"
                >Scene {{ (playingSequenceStage ?? 0) + 1 }}</span
              >
              <button
                type="button"
                data-test="sequence-return-live"
                class="rounded-control border border-border px-2 py-1 text-micro text-fg-2 hover:text-fg"
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
            <span class="font-mono text-micro whitespace-nowrap text-accent">
              clip {{ (chains.live.activeStage ?? watchedSequence.current_stage) + 1 }}/{{
                watchedSequence.stage_count
              }}
              · developing…
            </span>
          </div>

          <!-- Settled sequence: the canvas holds the result, because the
               queue rail never did. -->
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
                class="relative max-h-full w-full max-w-full overflow-hidden rounded-inner border border-border-control bg-media-bed"
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
              <span class="font-mono text-micro text-fg-dim whitespace-nowrap"
                >saved to My images</span
              >
            </div>

            <div
              class="font-mono text-micro text-fg-dim whitespace-nowrap mt-2 max-w-full truncate"
              :title="settledSequenceCaption"
            >
              {{ settledSequenceCaption }}
            </div>
            <div class="mt-2 flex items-center gap-2">
              <button
                v-if="settledSequenceError"
                type="button"
                data-test="sequence-resume"
                class="rounded-control bg-error px-3 py-1 text-sm font-semibold text-on-accent transition-colors hover:brightness-105 active:translate-y-px"
                @click="resumeSettledSequence"
              >
                Resume
              </button>
              <button
                type="button"
                data-test="sequence-edit"
                class="border-border-control rounded-control border px-3 py-1 text-sm text-fg-2 transition-colors hover:text-fg"
                @click="editSettledSequence"
              >
                Edit clip
              </button>
              <button
                type="button"
                data-test="sequence-show-in-library"
                class="border-border-control rounded-control border px-3 py-1 text-sm text-fg-2 transition-colors hover:text-fg"
                @click="showSettledSequenceInLibrary"
              >
                Show in My images
              </button>
            </div>
          </div>

          <!-- Empty -->
          <EmptyStateBlock
            v-else
            data-test="empty-canvas"
            brand
            icon="image"
            headline="Your picture appears here"
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
              class="rounded-control bg-error px-3 py-1.5 text-sm font-semibold text-on-accent transition-colors hover:brightness-105 active:translate-y-px disabled:opacity-50"
              :disabled="expansionRunning || preparedSubmitting"
              @click="reexpandAndGenerate"
            >
              Re-expand for {{ currentModelLabel }} and generate
            </button>
            <button
              type="button"
              data-test="generate-expanded-anyway"
              class="border-error/50 rounded-control border px-3 py-1.5 text-sm font-medium text-error transition-colors hover:bg-error/10 active:translate-y-px disabled:opacity-50"
              :disabled="expansionRunning || preparedSubmitting"
              @click="generateExpandedAnyway"
            >
              Generate expanded prompt anyway
            </button>
            <button
              type="button"
              data-test="restore-expanded-original"
              class="rounded-control px-3 py-1.5 text-sm text-fg-2 transition-colors hover:text-fg active:translate-y-px"
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
          v-if="isSequence"
          data-test="create-bench-resizer"
          class="group relative z-10 flex h-3 shrink-0 cursor-row-resize touch-none items-center justify-center border-y border-border bg-bg-deep/80"
          role="separator"
          aria-label="Resize Activity and sequence editor"
          aria-orientation="horizontal"
          :aria-valuenow="benchHeight"
          :aria-valuemin="minBenchHeight()"
          :aria-valuemax="maxBenchHeight()"
          tabindex="0"
          title="Drag to resize · double-click to reset"
          @pointerdown="startBenchResize"
          @keydown="onBenchResizeKeydown"
          @dblclick="setBenchHeight(DEFAULT_SEQUENCE_BENCH_HEIGHT)"
        >
          <span
            class="h-1 w-12 rounded-inner bg-fg-dim/50 transition-colors group-hover:bg-accent group-focus-visible:bg-accent"
            aria-hidden="true"
          />
        </div>

        <div
          v-if="isSequence"
          data-test="create-bottom-panel"
          class="flex min-h-0 shrink-0 flex-col overflow-hidden bg-bg-deep"
          :style="{
            height: `${benchHeight}px`,
            containerType: 'size',
            containerName: 'create-bench',
          }"
        >
          <p
            v-if="isSequence && sequenceReuseNotice"
            data-test="sequence-reuse-note"
            class="font-mono text-micro text-fg-dim whitespace-nowrap shrink-0 px-1 pt-1.5"
          >
            {{ sequenceReuseNotice }}
          </p>

          <!-- The clip timeline sits above the composer, never instead of it -->
          <EmptyStateBlock
            v-if="showSequenceEmpty"
            data-test="sequence-empty"
            class="flex-1 py-6"
            icon="image"
            headline="A short clip needs a video style"
            guidance="Get an LTX Video or distilled LTX-2 style that can chain, then tell the story one scene at a time."
          >
            <template #action>
              <button
                type="button"
                data-test="sequence-browse-models"
                class="rounded-control bg-accent px-3 py-1.5 text-sm font-semibold text-on-accent"
                @click="
                  router.push('/models?tab=discover&type=video&kind=checkpoint&intent=sequence')
                "
              >
                Browse video styles
              </button>
            </template>
          </EmptyStateBlock>
          <!-- Protect the timeline's hard chrome floor from Activity. The
               timeline itself stays min-h-0 so its lane can squash, but this
               parent flex item makes Activity yield before the transport or
               the readout can clip at the supported bench floor. -->
          <div
            v-else-if="isSequence"
            data-test="generate-sequence-shell"
            class="flex min-h-[228px] flex-[1_0_228px] overflow-hidden"
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
              :elapsed-seconds="sequenceClipElapsed"
              :target="sequenceTarget"
              @duplicate="duplicateSequenceAsNew"
              @play-clip="playSequenceClip"
              @update:blocked-reason="timelineBlockedReason = $event"
              @update:confirmation="sequenceConfirmation = $event"
            />
          </div>
        </div>
        <!-- One composer for both modes: in clip mode it carries the selected
             scene's words and Generate submits the whole chain. -->
        <ComposerCard
          ref="composerRef"
          data-test="generate-composer"
          class="shrink-0"
          :form="form"
          :effective-batch-size="effectiveBatchSize"
          :expansion-running="expansionRunning"
          :expansion-host-label="expansionHostLabel"
          :can-undo="quickExpansionOriginal !== null"
          :prepared-blocked="!!preparedBatch && effectiveBatchSize === 1"
          :disabled="composerLocked"
          :disabled-reason="composerRefusal"
          :warning-reason="isSequence ? null : composerWarningReason"
          :submitting="composerSubmitting"
          :button-label="buttonLabel"
          :queued-note="isSequence ? null : queuedNote"
          :estimate-request="isSequence ? null : estimateRequest"
          :estimate-target="estimateTarget"
          :preprocessing-status="isSequence ? null : submissionStatus"
          :history="isSequence ? [] : promptHistory"
          :remix-source="remixSource"
          :batch-locked="batchLocked"
          :prompt-value="composerPrompt"
          :placeholder="composerPlaceholder"
          :show-count="!isSequence"
          :length-frames="form.frames"
          :length-fps="form.fps"
          :length-contract="composerLengthContract"
          :show-expand="!isSequence"
          @open-shape="inspectorTab = 'settings'"
          @prompt-authored="onPromptAuthored"
          @update:prompt-value="writeScenePrompt"
          @update:length-frames="form.frames = $event"
          @generate="composerGenerate"
          @cancel="composerCancel"
          @expand="expandForCurrentBatch()"
          @remix="remixForCurrentPrompt()"
          @update:remix-source="remixSource = $event"
          @restore="restoreQuickExpansion"
        >
          <!-- The one style selector in the app. -->
          <template #style>
            <StylePicker :form="form" @pull-missing-model="offerPullForSelectedModel" />
          </template>
        </ComposerCard>

        <ConfirmDialog
          :open="sequenceConfirmation !== null"
          :title="sequenceConfirmation?.title ?? ''"
          :message="sequenceConfirmation?.message ?? ''"
          :confirm-label="sequenceConfirmation?.confirmLabel ?? 'Confirm'"
          danger
          @confirm="resolveSequenceConfirmation('confirm')"
          @cancel="resolveSequenceConfirmation('cancel')"
        />
      </div>
    </div>

    <!-- Inspector (persisted, left-edge resizable width) -->
    <InspectorPanel
      ref="inspectorRef"
      :form="form"
      :tab="inspectorTab"
      :recent="recentPrints"
      :last-seed="generation.lastSeedUsed"
      :chain-limits="chainLimits"
      :canvas-intent="canvasIntent"
      @append-word="appendPromptWord"
      @canvas-intent="setCanvasIntent"
      @reset-settings="invalidateRetainedRestore"
      @update:tab="inspectorTab = $event"
      @load-template="loadTemplate"
      @reuse-print="reuseRecentPrint"
    />

    <UpscaleDialog
      v-model="upscaleModel"
      :open="!!upscaleEntry"
      kind="image"
      :source-name="upscaleEntry?.item.filename ?? ''"
      :models="models.upscalers"
      :busy="upscaleBusy"
      :error="upscaleError || null"
      @confirm="startCanvasUpscale"
      @close="closeCanvasUpscale"
    />

    <DownloadTargetDialog
      v-if="missingModelTargets"
      :model-name="missingModelTargets.model"
      :targets="missingModelTargets.targets"
      @select="chooseMissingModelHost"
      @close="missingModelTargets = null"
    />
    <MissingModelDialog
      v-if="missingModel"
      :model="missingModel.model"
      :host-label="missingModelHostLabel"
      :size-gb="missingModelSizeGb"
      :models="hostModels.unionInstalled"
      :resume-after-pull="missingModel.resumeAfterPull !== false"
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

<style scoped>
.caption-action {
  display: inline-flex;
  cursor: pointer;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 10px;
  flex-shrink: 0;
  border-radius: var(--mold-radius-1);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  white-space: nowrap;
  color: var(--mold-text);
}
.caption-action:hover {
  background: var(--mold-surface);
}
.caption-action--icon {
  width: 28px;
  padding: 0;
  justify-content: center;
  color: var(--mold-text-2);
}

/* The caption lives INSIDE the aspect-fitted frame, and the frame is only as
   wide as the print: an 832x1216 portrait at the minimum window gives it
   ~287px, where Save + Make 4 variations + Make bigger + the meta need ~420
   and the last of them paint outside the frame's `overflow-hidden` — clipped,
   with no scroll and no cue. Below that width the word actions stand down and
   the ... menu, which carries every one of them, answers instead. */
@container preview-frame (max-width: 440px) {
  .caption-action--word {
    display: none;
  }
}
</style>

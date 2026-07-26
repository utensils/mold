<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import {
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "../lib/generateModels";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import ProgressRing from "@ui/components/ProgressRing.vue";
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
import { normalizeTargetHost, readyHostSignature } from "../lib/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useConnectionStore } from "../stores/connection";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  needsHostRoute,
  type BatchRequestOptions,
  type Job,
} from "../stores/generation";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { copyBase64ImageToClipboard } from "../lib/clipboard";
import { useUiStore } from "../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import {
  buildGenerationEstimateRequest,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
  type ChainRoutingDecision,
} from "../lib/chainRouting";
import { applyPrefillToForm, buildRequest, cloneGenerateForm } from "../lib/generateForm";
import { composeStyle, mergeStyleNegative, styleHint } from "../lib/stylePresets";
import { frames8n1Error } from "../lib/chain";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  guidanceValidationError,
  resolutionValidationError,
  stepsValidationError,
} from "../lib/generateValidation";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { coerceSourceFitForMaskless } from "@studio/lib/sourceFit";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { upscaleImage } from "../lib/api/upscale";
import { expandPrompt } from "../lib/api/expand";
import type { HostRoute } from "../stores/hosts";
import { formatTemplateMediaReferences, type GenerationTemplate } from "../lib/generationTemplates";
import { fetchHistoryAll, type HistoryHostTarget } from "../lib/api/history";
import { randomSeed } from "../stores/generation";
import type { GenerateRequest, OutputMetadata } from "../lib/api/types";
import {
  metadataReferencesSource,
  restoreSourceImage,
  sha256HexOfBase64,
} from "../lib/sourceRestore";
import { isMissingModelError } from "../lib/generateErrors";
import { startCatalogDownload } from "../lib/api/catalog";
import { computeEtaSeconds, useDownloadsStore, type DownloadsState } from "../stores/downloads";
import { usePullResumeStore } from "../stores/pullResume";
import { modelDisplayNameForId } from "../lib/models";
import { localMediaPath, mediaPath } from "../lib/gallery/media";
import { ApiError, apiFetch, apiFetchTo } from "../lib/api/client";
import { blobToBase64 } from "../lib/image";
import { ipc } from "../lib/ipc";
import { applyDesktopImageDrop } from "../lib/desktopImageDrop";
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
// Multi-host gallery — source-image restore looks up prints across hosts.
const hostGallery = useGalleryStore();
const downloads = useDownloadsStore();
const pullResume = usePullResumeStore();

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
    } else {
      toasts.push(String(err), "error");
    }
  }
}

// Store-backed so the model, prompt, and params survive navigating away and
// back — this view unmounts on every route change.
const formStore = useGenerateFormStore();
const form = formStore.form;
const composerRef = ref<InstanceType<typeof ComposerCard> | null>(null);
const templatesOpen = ref(false);
const templatesEl = ref<HTMLDivElement | null>(null);
/** Recent prompts for the composer's ↑/↓ history cycling. */
const promptHistory = ref<string[]>([]);
const nativeImageDragOver = ref(false);
const preparedBatch = ref<PreparedExpansionBatchState | null>(null);
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
const preparationGuard = new PreparationRequestGuard();
const submissionGuard = new PreparationRequestGuard();

let stopNativeImageDrop: (() => void) | null = null;
let nativeImageDropUnmounted = false;

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
    if (payload.type === "enter" || payload.type === "over") {
      nativeImageDragOver.value = true;
      return;
    }
    nativeImageDragOver.value = false;
    if (payload.type !== "drop") return;
    const path = payload.paths.find((candidate) => /\.(png|jpe?g)$/i.test(candidate));
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

const job = computed(() => generation.active);
const siblings = computed(() => generation.siblings);
const caps = computed(() => generationCapabilitiesForFamily(form.family));
const formValidationError = computed(
  () =>
    resolutionValidationError(form.width, form.height) ??
    stepsValidationError(form.steps) ??
    guidanceValidationError(form.guidance) ??
    (caps.value.supportsVideo ? frames8n1Error(form.frames) : null) ??
    (caps.value.supportsVideo ? fpsValidationError(form.fps) : null) ??
    cameraControlValidationError(form) ??
    audioOutputValidationError(form) ??
    advancedVideoValidationError(form),
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
  mergeInstalledModels(models.installed, hostModels.unionInstalled),
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
  caps.value.forcesBatchSizeOne ? 1 : Math.max(1, Math.floor(form.batchSize)),
);

/** Expansion always resolves a concrete host, even in the one-host case. */
const currentExpansionRoute = computed<HostRoute | null>(() =>
  hosts.resolveRoute(stickyTarget.value, form.model || null),
);
const expansionHostLabel = computed(() => currentExpansionRoute.value?.label ?? null);

const preparedStaleReasons = computed(() => {
  const batch = preparedBatch.value;
  if (!batch) return [];
  return preparedExpansionStaleReasons(batch, {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
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

const buttonLabel = computed(() =>
  generation.pending.length > 0 ? `Generate (+${generation.pending.length} queued)` : "Generate",
);

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

function loadTemplate(template: GenerationTemplate) {
  // Base64 media was stripped on save; buildRequest's pruneRequestForFamily
  // still guards anything the (possibly different) family can't use.
  Object.assign(form, template.form);
  templatesOpen.value = false;
  if (form.model && !findInstalledModel(installedModels.value, form.model)) {
    toasts.push(`Model "${form.model}" isn't installed — settings applied anyway.`);
  }
  if (template.mediaReferences.length > 0) {
    toasts.push(`Re-add media: ${formatTemplateMediaReferences(template.mediaReferences)}.`);
  }
}

function siblingDot(s: Job): string {
  if (s.status === "complete") return "text-ink"; // ◉ developed
  if (s.status === "error") return "text-stop";
  return "text-ink-3"; // ◎ pending
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
      disabled: !j.result || !!j.result.video_frames,
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
    { separator: true },
    {
      label: "Show in Gallery",
      disabled: j.status !== "complete",
      action: () => void router.push("/gallery"),
    },
  ];
}

function expansionInputs(count: number): PreparedExpansionInputs {
  return {
    sourcePrompt: form.prompt.trim(),
    model: form.model,
    family: form.family,
    requestedCount: count,
    stylePreset: form.stylePreset || null,
    selectedHostPolicy: stickyTarget.value,
  };
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
  const draftCaps = generationCapabilitiesForFamily(draft.family);
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

async function generate() {
  if (!form.prompt.trim() || !form.model || chainReject.value || preparedSubmitting.value) return;
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
    return;
  }

  const preparedSubmission = prepared
    ? {
        batchId: prepared.batchId,
        batch: prepared.prompts.length,
        promptIds: prepared.prompts.map((prompt) => prompt.id),
        prompts: prepared.prompts.map((prompt) => prompt.text.trim()),
        originalPrompt: prepared.sourcePrompt,
        route: { ...prepared.route, target: { ...prepared.route.target } },
      }
    : null;
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
  const submitToken = submissionGuard.begin();
  preparedSubmitting.value = preparedSubmission !== null;
  try {
    const draft = cloneGenerateForm(form);
    const draftCaps = generationCapabilitiesForFamily(draft.family);
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
    // A pinned host that went away is an error, not a reroute. Resolved
    // BEFORE source preprocessing so an upscale-then-fit cache miss hits the same host.
    let route: HostRoute | null = preparedSubmission?.route ?? quickSubmission?.route ?? null;
    if (!preparedSubmission && !quickSubmission && routeRequired.value) {
      route = hosts.resolveRoute(
        appPrefs.settings?.generateTargetHost ?? null,
        draft.model || null,
      );
      if (!route) {
        toasts.push("The selected host isn't reachable. Pick another host.", "error");
        return;
      }
    }
    if (!(await preprocessSourceFit(route, draft))) return;
    if (!submissionGuard.isCurrent(submitToken)) return;
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
    // Stash the exact source bytes by sha (the hash the server records as
    // source_image_sha256) so Reuse settings can restore uploads and fitted
    // sources later. Fire-and-forget — never blocks the submit.
    if (request.source_image) {
      const sourceB64 = request.source_image;
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
        }
      : {};
    const { settled } = generation.submitBatch(request, batch, route, chainRouting, requestOptions);
    if (preparedSubmission) {
      preparationGuard.invalidate();
      preparedBatch.value = null;
      expansionError.value = null;
      expansionMissingModel.value = null;
      void nextTick(() => composerRef.value?.focus?.());
    }
    if (
      !quickSubmission ||
      quickExpansionSnapshot.value?.requestToken === quickSubmission.requestToken
    ) {
      quickExpansionSnapshot.value = null;
    }
    composerRef.value?.record?.(preparedSubmission?.originalPrompt ?? request.prompt);
    const done = await settled;
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
      if (isMissingModelError(failed.error) && !hostSaysInstalled) {
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
      } else {
        toasts.push(failed.error, "error");
      }
    }
  } finally {
    preparedSubmitting.value = false;
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
  // Gallery reuse ships full metadata (full-fidelity restore); palette /
  // history / jobs keep the legacy scalar copy.
  applyPrefillToForm(form, prefill, installedModels.value);
  if ("metadata" in prefill && prefill.metadata) {
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
  if (!caps.value.supportsImg2img || caps.value.sourceImageMode !== "single") return;
  const modelAtStart = form.model;
  const restored = await restoreSourceImage(metadata, {
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
  });
  // The lookups can take seconds (cold gallery, cross-host fetch). Bail if
  // this restore was superseded: a newer prefill or ⌘N bumped the epoch, the
  // user attached their own source, the model changed under us, or the new
  // family can't take an image at all.
  if (epoch !== restoreEpoch || form.sourceImage || form.model !== modelAtStart) return;
  if (!caps.value.supportsImg2img || caps.value.sourceImageMode !== "single") return;
  if (restored) {
    form.sourceImage = restored.base64;
    form.sourceImageName = restored.filename;
  } else {
    toasts.push(
      "Couldn't restore the source image — the original file wasn't found on any connected host.",
      "error",
    );
  }
}

// Apply a prefill whenever one arrives (Reuse settings, history, "Generate
// with <model>"), including one already queued before this view mounted.
watch(() => composer.prefill, applyPrefill, { immediate: true });

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

// Menu ▸ Generate / Expand Prompt reuse the composer actions.
watch(
  () => ui.generateTick,
  () => void generate(),
);
watch(
  () => ui.expandTick,
  () => composerRef.value?.expand?.(),
);

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  void listenForNativeImageDrops();
});

onBeforeUnmount(() => {
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  nativeImageDropUnmounted = true;
  stopNativeImageDrop?.();
  document.removeEventListener("pointerdown", onDocumentPointerDown);
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
        data-test="generate-workbench"
        class="relative flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        <!-- Templates popover (relocated from the inspector) -->
        <div ref="templatesEl" class="absolute right-3 top-3 z-20">
          <button
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
        <div class="flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-desk p-7">
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
            @regenerate="expandForCurrentBatch(true)"
            @refresh="expandForCurrentBatch(true)"
            @discard="discardPreparedBatch"
            @pull="pullExpansionModel"
            @retry-expansion="retryExpansionAfterPull"
            @generate="generate"
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
                <video
                  v-if="job?.resultUrl && job.result?.video_frames"
                  :src="job.resultUrl"
                  class="absolute inset-0 h-full w-full object-contain"
                  autoplay
                  loop
                  controls
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
                  class="absolute inset-0 h-full w-full object-cover"
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
                <!-- Grain is the signature; the ring overlays it with the
                     percent + stage until the first latent preview arrives,
                     after which the forming print itself takes over. -->
                <div
                  v-if="job && job.status !== 'complete' && !job.previewUrl"
                  data-test="develop-progress"
                  class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3"
                >
                  <ProgressRing :value="jobProgress(job) * 100" :size="96" show-label />
                  <span class="edge-code text-safelight">
                    {{
                      job.status === "finishing"
                        ? `fixing — ${job.stage ?? "finishing"}…`
                        : job.status === "loading"
                          ? "loading model…"
                          : "developing…"
                    }}
                  </span>
                </div>
                <div
                  v-if="job && (job.status === 'denoising' || job.status === 'finishing')"
                  class="edge-code absolute bottom-2 left-3"
                >
                  <template v-if="job.status === 'denoising'">
                    {{ job.step }}/{{ job.total }}
                  </template>
                  <template v-else>Fixing — {{ job.stage ?? "finishing" }}…</template>
                </div>
              </div>
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
              :message="job.error ?? 'Generation failed for an unknown reason.'"
            />
          </div>

          <!-- Empty -->
          <EmptyStateBlock
            v-else
            data-test="empty-canvas"
            brand
            icon="image"
            headline="Your print develops here"
            guidance="Describe an image below, pick a look, and press Generate. Everything runs on your own machine."
          />
        </div>

        <!-- Batch-1 expansion status (prepared batches carry their own inline
             surfaces; these are the composer-level ones). -->
        <GenerateErrorNotice
          v-if="quickStaleReasons.length && !preparedBatch"
          data-test="quick-expansion-stale"
          class="mx-5 mb-2"
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

        <ActivityStrip />

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
          :has-prepared="!!preparedBatch"
          :chain-reject="chainReject"
          :button-label="buttonLabel"
          :estimate-request="estimateRequest"
          :estimate-target="estimateTarget"
          :preprocessing-status="preprocessingStatus"
          :history="promptHistory"
          @generate="generate"
          @expand="expandForCurrentBatch()"
          @restore="restoreQuickExpansion"
        />
      </div>
    </div>

    <!-- Inspector (persisted, left-edge resizable width) -->
    <InspectorPanel
      :form="form"
      :last-seed="generation.lastSeedUsed"
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
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "../lib/generateModels";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import ParamPanel from "../components/generate/ParamPanel.vue";
import LoraStack from "../components/generate/LoraStack.vue";
import TemplatesPanel from "../components/generate/TemplatesPanel.vue";
import SourceImageWell from "../components/generate/SourceImageWell.vue";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import ExpansionPullStatus from "../components/generate/ExpansionPullStatus.vue";
import PreparedExpansionBatch from "../components/generate/PreparedExpansionBatch.vue";
import HostSelector from "../components/generate/HostSelector.vue";
import MissingModelDialog from "../components/generate/MissingModelDialog.vue";
import SourceGlyph from "../components/generate/SourceGlyph.vue";
import PanelResizeHandle from "../components/shell/PanelResizeHandle.vue";
import { modelSource } from "../lib/modelSource";
import { modelAvailabilityTag, normalizeTargetHost } from "../lib/hosts";
import { dragWidth } from "../lib/panelResize";
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
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { coerceSourceFitForMaskless } from "../lib/sourceFit";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { upscaleImage } from "../lib/api/upscale";
import { expandPrompt } from "../lib/api/expand";
import type { HostRoute } from "../stores/hosts";
import { formatTemplateMediaReferences, type GenerationTemplate } from "../lib/generationTemplates";
import { autoGrowRows } from "../lib/autogrow";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "../lib/promptCycler";
import { fetchHistoryAll, type HistoryHostTarget } from "../lib/api/history";
import { formatGB } from "../lib/format";
import { randomSeed } from "../stores/generation";
import type { GenerateRequest, ModelEntry, OutputMetadata } from "../lib/api/types";
import {
  metadataReferencesSource,
  restoreSourceImage,
  sha256HexOfBase64,
} from "../lib/sourceRestore";
import { isMissingModelError } from "../lib/generateErrors";
import { startCatalogDownload } from "../lib/api/catalog";
import { computeEtaSeconds, useDownloadsStore, type DownloadsState } from "../stores/downloads";
import { usePullResumeStore } from "../stores/pullResume";
import { localMediaPath, mediaPath } from "../lib/gallery/media";
import { ApiError, apiFetch, apiFetchTo } from "../lib/api/client";
import { blobToBase64 } from "../lib/image";
import { ipc } from "../lib/ipc";
import { applyDesktopImageDrop } from "../lib/desktopImageDrop";
import { useGalleryStore } from "../stores/gallery";
import { fitAspectRatio } from "../lib/fitAspectRatio";
import { primaryModifierPressed, shortcutLabel } from "../lib/platform";
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
const promptEl = ref<HTMLTextAreaElement | null>(null);
const previewRegion = ref<HTMLDivElement | null>(null);
const previewFrameSize = ref({ width: 0, height: 0 });
const expandControl = ref<InstanceType<typeof ExpandControl> | null>(null);
const pickerEl = ref<HTMLDivElement | null>(null);
const pickerOpen = ref(false);
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
  if (!pickerOpen.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) pickerOpen.value = false;
}

// Live inspector width while dragging its left-edge handle; null follows the
// persisted preference (appPrefs.generateParamsWidth). Persist only on commit.
const draftAsideWidth = ref<number | null>(null);
const asideWidth = computed(() => draftAsideWidth.value ?? appPrefs.generateParamsWidth);

function onAsideResize(dx: number) {
  draftAsideWidth.value = dragWidth("generateParams", appPrefs.generateParamsWidth, dx, "left");
}

async function onAsideCommit() {
  const width = draftAsideWidth.value;
  if (width === null) return;
  if (width !== appPrefs.generateParamsWidth) await appPrefs.update({ generateParamsWidth: width });
  draftAsideWidth.value = null;
}

function onAsideReset() {
  draftAsideWidth.value = null;
  void appPrefs.update({ generateParamsWidth: null });
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
 *  ParamPanel shows the reason under Frames; this blocks the submit. */
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
// Falls back to the union entry so a model that only exists on an extra host
// still populates params/defaults after being picked.
const selectedModel = computed<ModelEntry | null>(() =>
  findInstalledModel(installedModels.value, form.model),
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

/**
 * What the picker renders: the all-host union under Auto / Most capable,
 * narrowed to the sticky host's installed set when one is picked. The current
 * `form.model` is left alone — `stickyHostMissingModel` already warns that a
 * generate there will auto-pull the weights.
 */
/** The sticky pick as the Host selector shows it — a ghost host id (removed
 *  or never reconnected) reads as Auto so filtering and tag suppression can
 *  never disagree with the selector (Copilot on #436). */
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
    selectedHostPolicy: stickyTarget.value,
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
    ...currentHostSnapshot(),
  });
});

const pickerModels = computed<ModelEntry[]>(() => {
  const target = stickyTarget.value;
  const fetched = target && target !== "capable" && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  return filterModelsForTarget(
    installedModels.value,
    target,
    fetched ? new Set(hostModels.installedOn(target).map((m) => m.name)) : null,
  );
});

/**
 * The picker's list: the primary's installed models merged with every model
 * installed on an extra host, grouped by family (primary entries win the
 * dedup so their defaults are used).
 */
const pickerFamilies = computed<Map<string, ModelEntry[]>>(() => {
  const byName = new Map<string, ModelEntry>();
  for (const m of pickerModels.value) byName.set(m.name, m);
  const groups = new Map<string, ModelEntry[]>();
  for (const m of byName.values()) {
    const list = groups.get(m.family) ?? [];
    list.push(m);
    groups.set(m.family, list);
  }
  return groups;
});

/** Subtle per-row tag for models that live only on non-primary hosts. */
function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost) return null;
  // With a sticky host every rendered row is on that host — tags are noise.
  const target = stickyTarget.value;
  if (target && target !== "capable") return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

/**
 * The sticky target host's label when it lacks the selected model (per the
 * last availability snapshot) — the job will auto-pull the weights there.
 */
const stickyHostMissingModel = computed<string | null>(() => {
  const sel = stickyTarget.value;
  if (!sel || sel === "capable" || !form.model) return null;
  const host = hosts.all.find((h) => h.id === sel);
  if (!host) return null;
  const ids = hostModels.hostsFor(form.model);
  if (ids.length === 0 || ids.includes(sel)) return null;
  return host.label;
});

// Availability data is demand-driven: fetch on mount / when the set of ready
// hosts changes, and force-fresh whenever the picker opens (a model pulled on
// an extra host by another client shows up the moment the user looks). No
// global timers.
watch(pickerOpen, (open) => {
  if (open) void hostModels.refresh(true);
});
watch(
  () =>
    hosts.all
      .filter((h) => h.status === "ready")
      .map((h) => h.id)
      .join("\n"),
  () => void hostModels.refresh(),
  // immediate: routing must be model-aware on the FIRST Generate click, not
  // only after the picker has been opened once (peer review on #390).
  { immediate: true },
);

const buttonLabel = computed(() =>
  generation.pending.length > 0 ? `Generate (+${generation.pending.length} queued)` : "Generate",
);

const previewWidth = computed(() => job.value?.width ?? form.width);
const previewHeight = computed(() => job.value?.height ?? form.height);
const previewFrameStyle = computed(() => ({
  aspectRatio: `${previewWidth.value} / ${previewHeight.value}`,
  width: `${previewFrameSize.value.width}px`,
  height: `${previewFrameSize.value.height}px`,
}));

let previewResizeObserver: ResizeObserver | null = null;

function resizePreview(width?: number, height?: number) {
  const rect = previewRegion.value?.getBoundingClientRect();
  previewFrameSize.value = fitAspectRatio(
    width ?? rect?.width ?? 0,
    height ?? rect?.height ?? 0,
    previewWidth.value,
    previewHeight.value,
  );
}

watch([previewWidth, previewHeight], () => resizePreview());

const edgeCode = computed(() => {
  const j = job.value;
  if (!j) return "";
  const name = j.model.toUpperCase().replace(":", "·");
  const s = j.result ? `S ${j.result.seed_used}` : `S ${j.visualSeed.slice(0, 12)}`;
  const stepPart = `${j.status === "complete" ? j.total : j.step}/${j.total}`;
  const size = j.result ? `${j.result.width}×${j.result.height}` : `${j.width}×${j.height}`;
  const time = j.result ? `${(j.result.generation_time_ms / 1000).toFixed(1)}s` : "";
  return [name, s, stepPart, size, time].filter(Boolean).join("  ");
});

function pickModel(m: ModelEntry) {
  formStore.applyModel(m);
  pickerOpen.value = false;
}

function loadTemplate(template: GenerationTemplate) {
  // Base64 media was stripped on save; buildRequest's pruneRequestForFamily
  // still guards anything the (possibly different) family can't use.
  Object.assign(form, template.form);
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
    const response = await expandPrompt(
      inputs.sourcePrompt,
      {
        variations: count,
        ...(inputs.family ? { modelFamily: inputs.family } : {}),
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
        current.selectedHostPolicy !== inputs.selectedHostPolicy ||
        !hostStillReady
      ) {
        expansionError.value =
          "The prompt or generation host changed while expansion was running. Expand again to use the current inputs.";
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
        selectedHostPolicy: inputs.selectedHostPolicy,
        route: { ...route, target: { ...route.target } },
      };
      if (replacePrepared) {
        const active = document.activeElement;
        const shouldRestoreFocus =
          replacementOwnedFocus &&
          (active === document.body || (!!active && preparedSection?.contains(active)));
        preparedBatch.value = null;
        if (shouldRestoreFocus) void nextTick(() => promptEl.value?.focus());
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

function restoreQuickExpansion() {
  const original = quickExpansionOriginal.value;
  if (original === null) return;
  submissionGuard.invalidate();
  form.prompt = original;
  form.originalPrompt = null;
  quickExpansionOriginal.value = null;
  quickExpansionSnapshot.value = null;
  expansionError.value = null;
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
  void nextTick(() => promptEl.value?.focus());
}

function discardPreparedBatch() {
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  preparedBatch.value = null;
  expansionRunning.value = false;
  expansionError.value = null;
  expansionMissingModel.value = null;
  expansionAttemptHostLabel.value = null;
  void nextTick(() => promptEl.value?.focus());
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

/**
 * Apply the source-fit policy to the attached source (and mask) before the
 * request is built: canvas-fit a mismatched source, generate the pad mask
 * for pad-repaint, and for upscale-then-fit run the source through
 * `POST /api/upscale/stream` first. `route` is the ALREADY-RESOLVED
 * generation host so the upscaler model auto-downloads on the same machine
 * the job will run on. Returns false when the submit must abort.
 */
async function preprocessSourceFit(
  route: HostRoute | null,
  draft: ReturnType<typeof cloneGenerateForm>,
): Promise<boolean> {
  const draftCaps = generationCapabilitiesForFamily(draft.family);
  if (!draftCaps.supportsImg2img || draftCaps.sourceImageMode !== "single") return true;
  if (!draft.sourceImage) return true;
  const originalSource = draft.sourceImage;
  const originalMask = draft.maskImage;
  const originalSourceFit = JSON.stringify(draft.sourceFit);
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
    draft.sourceImage = result.source;
    draft.maskImage = result.mask;
    // Keep the visible well in sync only if the user has not moved the live
    // composer to another model/source while preprocessing was in flight.
    if (
      form.model === draft.model &&
      form.family === draft.family &&
      form.sourceImage === originalSource &&
      form.maskImage === originalMask &&
      JSON.stringify(form.sourceFit) === originalSourceFit &&
      form.width === draft.width &&
      form.height === draft.height
    ) {
      form.sourceImage = result.source;
      form.maskImage = result.mask;
    }
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
    expansionError.value = `${quickStaleReasons.value.join(" ")} Restore or expand again before generating.`;
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
    const batch = preparedSubmission
      ? preparedSubmission.batch
      : draftCaps.forcesBatchSizeOne
        ? 1
        : draft.batchSize;
    // With multiple live hosts — or a dead primary while another host can
    // serve — route the batch (sticky pick, Auto = least busy, or Most
    // capable) — model-aware, so hosts that already have the weights win.
    // A pinned host that went away is an error, not a reroute. Resolved
    // BEFORE source preprocessing so upscale-then-fit hits the same host.
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
      void nextTick(() => promptEl.value?.focus());
    }
    if (
      !quickSubmission ||
      quickExpansionSnapshot.value?.requestToken === quickSubmission.requestToken
    ) {
      quickExpansionSnapshot.value = null;
    }
    cycler.record(preparedSubmission?.originalPrompt ?? request.prompt);
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

// ↑/↓ cycle recent prompts (shell-history style) when the caret is on the
// composer's first/last line, so multi-line editing keeps native arrows.
const cycler = new PromptCycler();

/** The composer grows with its content (capped) instead of scrolling at 2 rows. */
function growPrompt() {
  if (promptEl.value) autoGrowRows(promptEl.value);
}
// Programmatic prompt changes (history cycling, expand, templates) resize too.
watch(
  () => form.prompt,
  () => void nextTick(growPrompt),
  { flush: "post" },
);
onMounted(() => growPrompt());

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
    cycler.setEntries(history.entries.map((entry) => entry.prompt));
  } catch {
    // No history API (older engine / DB off) — arrows just move the caret.
  }
}

function cycleHistory(direction: "prev" | "next") {
  const replacement = direction === "prev" ? cycler.prev(form.prompt) : cycler.next();
  if (replacement === null) return;
  form.prompt = replacement;
  void nextTick(() => {
    const el = promptEl.value;
    el?.setSelectionRange(el.value.length, el.value.length);
  });
}

function onComposerKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && primaryModifierPressed(e)) {
    e.preventDefault();
    void generate();
  } else if ((e.key === "e" || e.key === "E") && primaryModifierPressed(e)) {
    e.preventDefault();
    expandControl.value?.expand();
  } else if (e.key === "ArrowUp" && promptEl.value && caretOnFirstLine(promptEl.value)) {
    e.preventDefault();
    cycleHistory("prev");
  } else if (e.key === "ArrowDown" && promptEl.value && caretOnLastLine(promptEl.value)) {
    e.preventDefault();
    cycleHistory("next");
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
  void nextTick(() => promptEl.value?.focus());
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
    submissionGuard.invalidate();
    formStore.clearComposer();
    void nextTick(() => promptEl.value?.focus());
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
  () => expandControl.value?.expand(),
);

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  promptEl.value?.focus();
  if (previewRegion.value && typeof ResizeObserver !== "undefined") {
    previewResizeObserver = new ResizeObserver(([entry]) => {
      if (entry) resizePreview(entry.contentRect.width, entry.contentRect.height);
    });
    previewResizeObserver.observe(previewRegion.value);
  }
  resizePreview();
  void listenForNativeImageDrops();
});

onBeforeUnmount(() => {
  preparationGuard.invalidate();
  submissionGuard.invalidate();
  nativeImageDropUnmounted = true;
  stopNativeImageDrop?.();
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  previewResizeObserver?.disconnect();
});
</script>

<template>
  <StarterCards v-if="showStarterCards" @browse="router.push('/models')" />

  <div
    v-else
    data-test="generate-layout"
    class="relative grid h-full min-h-0 overflow-hidden"
    :style="{ gridTemplateColumns: `1fr ${asideWidth}px` }"
  >
    <div
      v-if="nativeImageDragOver"
      data-test="native-image-drop-overlay"
      class="pointer-events-none absolute inset-3 z-40 flex items-center justify-center rounded-chrome border-2 border-dashed border-safelight bg-bath/90 text-body-lg text-safelight shadow-raised"
    >
      Drop image to load settings and use as source
    </div>

    <!-- Canvas + composer -->
    <div data-test="generate-workbench" class="flex min-h-0 min-w-0 flex-col overflow-hidden p-6">
      <div class="flex min-h-0 flex-1 flex-col">
        <div
          ref="previewRegion"
          data-test="preview-region"
          class="flex min-h-0 flex-1 items-center justify-center overflow-hidden"
        >
          <div
            class="relative w-full overflow-hidden rounded-media border border-control-edge"
            :class="job ? 'bg-print-surface' : 'bg-empty-surface'"
            data-test="preview-frame"
            :style="previewFrameStyle"
            @contextmenu="job && contextMenu.open($event, canvasMenu())"
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
                opacity: job.previewUrl ? String(Math.max(0.18, 1 - jobProgress(job) * 0.9)) : '1',
              }"
            />
            <div
              v-if="!job"
              data-test="empty-canvas"
              class="absolute inset-0 flex items-center justify-center p-6 text-center"
            >
              <div class="flex max-w-64 flex-col items-center">
                <div
                  class="border-halide/40 mb-4 flex h-20 w-24 items-center justify-center rounded-media border bg-[color-mix(in_srgb,var(--halide)_7%,transparent)]"
                  aria-hidden="true"
                >
                  <svg
                    viewBox="0 0 48 40"
                    class="h-10 w-12 text-halide/70"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.5"
                  >
                    <rect x="4" y="4" width="40" height="32" rx="1" />
                    <circle cx="33" cy="13" r="3" />
                    <path d="m9 31 10-11 7 7 5-5 8 9" />
                  </svg>
                </div>
                <div class="font-display text-display-sm font-semibold text-ink">No print yet</div>
                <p class="mt-1 text-caption text-ink-2">
                  Choose a model, describe your print, then generate.
                </p>
              </div>
            </div>
            <div
              v-if="job && (job.status === 'denoising' || job.status === 'finishing')"
              class="edge-code absolute bottom-2 left-3"
            >
              <template v-if="job.status === 'denoising'">{{ job.step }}/{{ job.total }}</template>
              <template v-else>Fixing — {{ job.stage ?? "finishing" }}…</template>
            </div>
          </div>
        </div>

        <div v-if="job" class="edge-code mt-2 truncate" :title="edgeCode">{{ edgeCode }}</div>

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
            {{ siblings.filter((s) => s.status === "complete").length }} of {{ siblings.length }}
          </span>
        </div>

        <p v-if="job?.status === 'error'" class="mt-2 text-caption text-stop">{{ job.error }}</p>
      </div>

      <!-- Composer -->
      <div
        data-test="generate-composer"
        class="mt-4 shrink-0 rounded-chrome border border-control-edge bg-bench p-3 transition-colors duration-100 focus-within:border-safelight"
      >
        <textarea
          ref="promptEl"
          v-model="form.prompt"
          data-selectable
          rows="2"
          aria-label="Prompt"
          placeholder="Describe the print — a lighthouse at dusk, kodak portra…"
          class="w-full resize-none overflow-x-hidden bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          @keydown="onComposerKeydown"
          @input="
            cycler.reset();
            growPrompt();
          "
        />
        <div class="mt-2 flex min-w-0 flex-wrap items-center justify-between gap-2">
          <ExpandControl
            ref="expandControl"
            :prompt="form.prompt"
            :batch-size="effectiveBatchSize"
            :running="expansionRunning"
            :host-label="expansionHostLabel"
            :can-undo="quickExpansionOriginal !== null"
            :blocked="!!preparedBatch && effectiveBatchSize === 1"
            @expand="expandForCurrentBatch"
            @restore="restoreQuickExpansion"
          />
          <div class="flex min-w-0 items-center gap-3">
            <span
              v-if="preprocessingStatus"
              class="truncate text-caption text-ink-3"
              data-test="preprocessing-status"
            >
              {{ preprocessingStatus }}
            </span>
            <EstimateBadge :request="estimateRequest" :target="estimateTarget" />
            <button
              v-if="effectiveBatchSize === 1"
              type="button"
              data-test="generate-button"
              class="h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-60"
              :disabled="!form.prompt.trim() || !form.model || chainReject || !!preparedBatch"
              @click="generate"
            >
              {{ buttonLabel }}
              <kbd class="kbd-hint ml-1.5 opacity-80">{{ shortcutLabel("↩") }}</kbd>
            </button>
          </div>
        </div>
        <div
          v-if="expansionError && !preparedBatch && !expansionMissingModel"
          role="alert"
          class="border-stop/45 mt-3 flex flex-wrap items-center justify-between gap-2 rounded-control border bg-stop/10 px-2.5 py-2 text-caption text-stop"
        >
          <span>{{ expansionError }}</span>
        </div>
        <ExpansionPullStatus
          v-if="expansionError && expansionMissingModel && expansionPullStatus && !preparedBatch"
          :model="expansionMissingModel.model"
          :host-label="expansionMissingModel.route.label"
          :error="expansionError"
          :status="expansionPullStatus"
          :eta-seconds="expansionPullEtaSeconds"
          @pull="pullExpansionModel"
          @retry-expansion="retryExpansionAfterPull"
        />
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
      </div>
    </div>

    <!-- Inspector. overflow-x-hidden is load-bearing: with only overflow-y
         set, any child momentarily wider than the 320px column during a
         window resize computes overflow-x to auto and grows a horizontal
         scrollbar under the prompt area. -->
    <aside class="border-edge overflow-x-hidden overflow-y-auto border-l bg-bench p-4">
      <HostSelector />
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">Model</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>
      <div ref="pickerEl" class="relative">
        <button
          type="button"
          :aria-expanded="pickerOpen"
          class="border-edge flex min-h-9 w-full items-center justify-between gap-2 rounded-control border bg-bath px-2 py-1.5 text-body text-ink"
          @click="pickerOpen = !pickerOpen"
        >
          <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
            selectedModel?.name ?? "Choose a model"
          }}</span>
          <span v-if="selectedModel?.disk_usage_bytes" class="data-mono shrink-0 text-ink-3">
            {{ formatGB(selectedModel.disk_usage_bytes) }}
          </span>
        </button>
        <div
          v-if="pickerOpen"
          data-test="model-picker-menu"
          class="border-edge absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-chrome border bg-bench shadow-raised"
        >
          <template v-for="[family, list] in pickerFamilies" :key="family">
            <div class="edge-code px-2 pt-2 pb-1">{{ family.toUpperCase() }}</div>
            <button
              v-for="m in list"
              :key="m.name"
              type="button"
              class="flex w-full items-start gap-2 px-2 py-1.5 text-left text-body text-ink-2 hover:bg-bath hover:text-ink"
              @click="pickModel(m)"
            >
              <SourceGlyph :source="modelSource(m)" class="mt-0.5 shrink-0 text-ink-3" />
              <span class="min-w-0 flex-1">
                <span
                  data-test="model-option-name"
                  class="block break-all text-ink"
                  :title="m.name"
                >
                  {{ m.name }}
                </span>
                <span
                  v-if="availabilityTag(m)"
                  data-test="model-availability"
                  class="edge-code mt-0.5 block break-all whitespace-normal"
                >
                  {{ availabilityTag(m) }}
                </span>
              </span>
              <span
                class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full"
                :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
                :title="m.is_loaded ? 'On GPU' : ''"
              />
            </button>
          </template>
          <button
            type="button"
            data-test="browse-catalog"
            class="border-edge flex w-full items-center border-t px-2 py-2 text-left text-body text-halide hover:bg-bath"
            @click="
              pickerOpen = false;
              void router.push(caps.supportsVideo ? '/models?type=video' : '/models');
            "
          >
            Browse all models →
          </button>
        </div>
      </div>
      <p v-if="stickyHostMissingModel" class="mt-1.5 text-caption text-ink-3">
        Not on {{ stickyHostMissingModel }} — will download there.
      </p>

      <ParamPanel
        :form="form"
        :last-seed="generation.lastSeedUsed"
        :upscalers="models.upscalers"
        class="mt-5"
      />
      <SourceImageWell :form="form" />
      <LoraStack
        v-if="caps.supportsLora"
        :form="form"
        :model="form.model"
        @append-word="appendPromptWord"
      />
      <TemplatesPanel :form="form" @load="loadTemplate" />
    </aside>

    <!-- The aside scrolls, so its resize handle lives on the (relative) grid
         container, pinned to the column boundary and straddling the border. -->
    <PanelResizeHandle
      class="absolute inset-y-0 z-10 translate-x-1/2"
      :style="{ right: `${asideWidth}px` }"
      label="Resize inspector"
      @resize="onAsideResize"
      @commit="onAsideCommit"
      @reset="onAsideReset"
    />

    <MissingModelDialog
      v-if="missingModel"
      :model="missingModel.model"
      :host-label="missingModelHostLabel"
      :size-gb="missingModelSizeGb"
      @confirm="pullMissingModel"
      @close="missingModel = null"
    />
  </div>
</template>

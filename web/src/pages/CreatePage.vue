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
import { useRoute } from "vue-router";
import ComposerCard from "../components/create/ComposerCard.vue";
import ResultCanvas from "../components/create/ResultCanvas.vue";
import ControlsAside from "../components/create/ControlsAside.vue";
import CreateModelPicker from "../components/create/CreateModelPicker.vue";
import AdvancedDrawer from "../components/create/AdvancedDrawer.vue";
import ActivityStrip from "../components/create/ActivityStrip.vue";
import EstimateBadge from "../components/create/EstimateBadge.vue";
import { advancedActiveCount } from "../components/create/advancedCount";
import { projectResolution } from "../components/create/resolutionProjection";
import ScriptComposer from "../components/ScriptComposer.vue";
import ChainJobCard from "../components/ChainJobCard.vue";
import ExpandModal from "../components/ExpandModal.vue";
import ImagePickerModal from "../components/ImagePickerModal.vue";
import MaskEditorModal from "../components/MaskEditorModal.vue";
import GenerationTemplatesPanel from "../components/GenerationTemplatesPanel.vue";
import ColdStartGuide from "../components/create/ColdStartGuide.vue";
import RecentGrid from "../components/create/RecentGrid.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import { defaultUpscaler } from "../components/create/advanced/upscalers";
import { blobToBase64 } from "../lib/base64";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { ASPECTS } from "@ui/lib/resolution";
import type { DevelopPhase } from "@ui/lib/grain";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { createUuid } from "@studio/lib/id";
import {
  createChainJob,
  deleteGalleryImage,
  expandPrompt,
  fetchPromptHistory,
  imageUrl,
  listGallery,
  upscaleStream,
} from "../api";
import {
  applyMetadataToForm,
  isQwenImageEditFamily,
  useGenerateForm,
} from "../composables/useGenerateForm";
import { mergeStyleNegative, styleHint } from "../lib/stylePresets";
import {
  activeCanvasJob,
  useGenerateStream,
  type Job,
} from "../composables/useGenerateStream";
import {
  completedSeedToLock,
  loadLastSeed,
  storeLastSeed,
} from "../lib/lastSeed";
import { useChainJobStream } from "../composables/useChainJobStream";
import { useQueue } from "../composables/useQueue";
import { decideGenerateRequestRouting } from "../lib/chainRouting";
import { isStandaloneGenerationModel } from "../lib/modelFilters";
import {
  maskPaddingRectangles,
  resolveSourceFitTransform,
} from "@studio/lib/sourceFit";
import { useStatusPoll } from "../composables/useStatusPoll";
import { useHostRouting } from "../composables/useHostRouting";
import { ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { generationCapabilitiesForFamily } from "../lib/generateCapabilities";
import { modelDisplayName } from "../lib/modelName";
import type { HostRoute } from "../lib/hostRouting";
import type {
  ChainRequestWire,
  ChainStageWire,
  ExpandFormState,
  GalleryImage,
  GenerateRequestWire,
  ModelInfoExtended,
  SourceFitPolicy,
  SourceImageState,
} from "../types";
import type { ChainScriptToml } from "../lib/chainToml";

type ComposerMode = "single" | "script";

function loadMuted(): boolean {
  try {
    return localStorage.getItem("mold.gallery.muted") !== "false";
  } catch {
    return true;
  }
}

const form = useGenerateForm();
const pageRoute = useRoute();
const { status } = useStatusPoll();
const queue = useQueue();
const routing = useHostRouting();
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
const showPicker = ref(false);
const showMask = ref(false);
const showAdvanced = ref(false);
const showTemplates = ref(false);
const templatesHost = ref<HTMLElement | null>(null);
const composerError = ref<string | null>(null);
const preprocessingStatus = ref<string | null>(null);
const submitStatus = computed(
  () => composerError.value ?? preprocessingStatus.value,
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
const expandRoute = ref<HostRoute | null>(null);
interface QuickPreparedExpansion {
  expandedPrompt: string;
  originalPrompt: string;
  model: string;
  family: string;
  selectedHostPolicy: string;
  route: HostRoute | null;
}
const quickPrepared = ref<QuickPreparedExpansion | null>(null);
interface PreparedWebBatch {
  batchId: string;
  sourcePrompt: string;
  model: string;
  family: string;
  requestedCount: number;
  selectedHostPolicy: string;
  baseRequest: GenerateRequestWire;
  decision: Exclude<
    ReturnType<typeof decideGenerateRequestRouting>,
    { kind: "reject" }
  >;
  route: HostRoute | null;
}
const preparedBatch = ref<PreparedWebBatch | null>(null);

function cloneRoute(route: HostRoute | null): HostRoute | null {
  return route ? { ...route, target: { ...route.target } } : null;
}

function sameRoute(
  frozen: HostRoute | null,
  current: HostRoute | null,
): boolean {
  if (!frozen || !current) return frozen === current;
  return (
    frozen.hostId === current.hostId &&
    frozen.target.baseUrl === current.target.baseUrl &&
    frozen.target.apiKey === current.target.apiKey
  );
}

function preparedStaleReasons(batch: PreparedWebBatch): string[] {
  const reasons: string[] = [];
  if (form.state.value.prompt.trim() !== batch.sourcePrompt)
    reasons.push("Source prompt changed after these variations were prepared.");
  if (form.state.value.model !== batch.model)
    reasons.push(
      `Model changed from "${batch.model}" to "${form.state.value.model}".`,
    );
  if (currentFamily.value !== batch.family)
    reasons.push(
      `Model family changed from "${batch.family}" to "${currentFamily.value}".`,
    );
  if (form.state.value.batchSize !== batch.requestedCount)
    reasons.push(
      `Batch changed from ${batch.requestedCount} to ${form.state.value.batchSize}.`,
    );
  if (routing.targetId.value !== batch.selectedHostPolicy)
    reasons.push(
      "The Run on selection changed after these variations were prepared.",
    );
  const currentRoute = routing.multiHost.value
    ? routing.resolve(batch.model)
    : null;
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
      `Model changed from "${snapshot.model}" to "${form.state.value.model}".`,
    );
  if (currentFamily.value !== snapshot.family)
    reasons.push(
      `Model family changed from "${snapshot.family}" to "${currentFamily.value}".`,
    );
  if (routing.targetId.value !== snapshot.selectedHostPolicy)
    reasons.push(
      "The Run on selection changed after this prompt was expanded.",
    );
  const currentRoute = routing.multiHost.value
    ? routing.resolve(snapshot.model)
    : null;
  if (!sameRoute(snapshot.route, currentRoute))
    reasons.push(
      `${snapshot.route?.label ?? "This server"} is no longer the prepared generation route.`,
    );
  return reasons;
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
  if (!policy || policy.mode === "upscale-then-fit") {
    return { mode: "pad-repaint" };
  }
  return policy;
}

function loadComposerMode(): ComposerMode {
  try {
    return localStorage.getItem("mold.composer.mode") === "script"
      ? "script"
      : "single";
  } catch {
    return "single";
  }
}
const composerMode = ref<ComposerMode>(loadComposerMode());
function setComposerMode(v: ComposerMode) {
  composerMode.value = v;
  try {
    localStorage.setItem("mold.composer.mode", v);
  } catch {
    /* ignore */
  }
}
watch(
  () => pageRoute.query.mode,
  (mode) => {
    if (mode === "sequence") setComposerMode("script");
  },
  { immediate: true },
);

const expandStageIndex = ref<number | null>(null);
const expandStagePrompt = ref("");
// The composer's style chip steers the main-prompt expansion as natural
// language. Chain stages are their own text — the style row never touches them.
const expandStyleDirective = computed(() =>
  expandStageIndex.value !== null
    ? null
    : styleHint(form.state.value.stylePreset ?? ""),
);
const scriptComposerRef = ref<InstanceType<typeof ScriptComposer> | null>(null);

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
const CHAIN_JOB_STORAGE_KEY = "mold.create.chain-job";
function loadSubmittedChainJobId(): string | null {
  try {
    return localStorage.getItem(CHAIN_JOB_STORAGE_KEY);
  } catch {
    return null;
  }
}
const submittedChainJobId = ref<string | null>(loadSubmittedChainJobId());
const submittedChainJob = useChainJobStream(submittedChainJobId);
const submittedChainJobDetail = submittedChainJob.detail;
watch(submittedChainJobId, (id) => {
  try {
    if (id) localStorage.setItem(CHAIN_JOB_STORAGE_KEY, id);
    else localStorage.removeItem(CHAIN_JOB_STORAGE_KEY);
  } catch {
    // Storage is advisory; the durable server job keeps running regardless.
  }
});

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

/** Human title for the selected model — falls back to the raw form value. */
const currentModelLabel = computed(() =>
  currentModel.value
    ? modelDisplayName(currentModel.value)
    : form.state.value.model,
);

const currentFamily = computed(
  () => currentModel.value?.family ?? form.state.value.modelFamily,
);

const capabilities = computed(() =>
  generationCapabilitiesForFamily(currentFamily.value),
);

// Chain (Sequence) generation is only meaningful for the video families that
// stitch multiple clips. For everything else the Sequence tab stays
// discoverable but explains itself instead of offering a dead Generate button.
const supportsChain = computed(() => capabilities.value.supportsVideo);

const gpuListForPlacement = computed(
  () =>
    status.value?.gpus?.map((g) => ({
      ordinal: g.ordinal,
      name: `GPU ${g.ordinal}`,
    })) ?? [],
);

const chainDecision = computed(() =>
  decideGenerateRequestRouting(
    {
      frames: form.state.value.frames,
      model: form.state.value.model,
      temporal_upscale: form.state.value.temporalUpscale,
    },
    currentModel.value?.family ?? null,
  ),
);

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
    if (current && installedModels.value.some((m) => m.name === current))
      return;
    form.applyModelDefaults(first);
  },
  { immediate: true },
);

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
  form.state.value.maskImage = null;
  form.state.value.controlImage = null;
  variations.value = [];
  preparedBatch.value = null;
  quickPrepared.value = null;
  prevPrompt.value = null;
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
  advancedActiveCount({
    negativePrompt: capabilities.value.supportsNegativePrompt
      ? form.state.value.negativePrompt
      : "",
    hasSource: form.state.value.imageAttachments.length > 0,
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
    controlNet:
      capabilities.value.supportsControlNet &&
      form.state.value.controlImage != null,
    videoSuite:
      capabilities.value.supportsVideo &&
      (form.state.value.gifPreview ||
        form.state.value.enableAudio === false ||
        form.state.value.pipeline != null ||
        form.state.value.audioFile != null ||
        form.state.value.audioFilePath.trim() !== "" ||
        form.state.value.sourceVideo != null ||
        form.state.value.sourceVideoPath.trim() !== "" ||
        form.state.value.keyframes.length > 0 ||
        form.state.value.retakeRange != null ||
        form.state.value.spatialUpscale != null ||
        form.state.value.temporalUpscale != null),
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
const runningJob = computed(() => activeCanvasJob(stream.jobs.value));
const latestDone = computed(() => {
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
  stream.jobs.value.find((j) => j.state === "error"),
);

const canvasMode = computed<
  "empty" | "generating" | "result" | "error" | "variations"
>(() => {
  if (variations.value.length) return "variations";
  if (runningJob.value) return "generating";
  if (
    latestError.value &&
    (!latestDone.value ||
      latestError.value.startedAt > latestDone.value.startedAt)
  )
    return "error";
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
  if (r.video_thumbnail) return `data:image/png;base64,${r.video_thumbnail}`;
  return `data:image/${r.format};base64,${r.image}`;
});
const resultCaption = computed(() => {
  const job = latestDone.value;
  const r = job?.result;
  if (!r) return "";
  const secs = Math.round(r.generation_time_ms / 1000);
  // Name the machine that actually rendered it — an unrouted job ran here.
  const where = job?.hostLabel ?? "this server";
  return `${r.model} · seed ${r.seed_used} · ${secs}s · ${where}`;
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
): Promise<PreparedStillSource | false> {
  let source = form.state.value.imageAttachments[0] ?? null;
  const mask = form.state.value.maskImage;
  if (!source) return { source, mask };

  const outerPolicy = form.state.value.sourceFitPolicy;
  if (outerPolicy?.mode === "upscale-then-fit") {
    const model = outerPolicy.upscalerModel || form.state.value.upscaleModel;
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

  const family = currentModel.value?.family ?? form.state.value.modelFamily;
  if (isQwenImageEditFamily(family) || form.state.value.frames)
    return { source, mask };
  const target = {
    width: form.state.value.width,
    height: form.state.value.height,
  };
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
  const qwenImageEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  if (qwenImageEdit && form.state.value.imageAttachments.length === 0) {
    composerError.value = "Qwen image edit needs a target image.";
    return false;
  }
  if (
    !qwenImageEdit &&
    form.state.value.maskImage &&
    form.state.value.imageAttachments.length === 0
  ) {
    composerError.value = "Mask image needs a source image.";
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

async function onSubmit() {
  // The route is settled first, and before source preprocessing, for two
  // reasons: an unreachable pinned machine is the real complaint (its model
  // list is empty, so a model check first would blame the model instead), and
  // an upscale cache miss has to land on the same machine as the render.
  const quick = quickPrepared.value;
  if (quick) {
    const stale = quickStaleReasons(quick);
    if (stale.length) {
      composerError.value = `${stale.join(" ")} Undo or expand again before generating.`;
      return;
    }
  }
  const route = quick ? cloneRoute(quick.route) : resolveSubmitRoute();
  if (route === false) return;
  if (!validateSubmit()) return;
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  const preparedSource = await prepareStillSourceToRequest(route);
  if (preparedSource === false) return;
  const req = form.toRequest(currentModel.value);
  if (quick) req.original_prompt = quick.originalPrompt;
  if ("source_image" in req) {
    req.source_image = preparedSource.source?.base64 ?? null;
    if (preparedSource.mask) req.mask_image = preparedSource.mask.base64;
    else delete req.mask_image;
  }
  stream.submit(req, decision, route);
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

// Sequences are submitted as durable chain jobs and followed with an
// EventSource, which cannot carry a per-host `x-api-key` — so they stay on this
// server. Say so rather than letting the Run on chip imply otherwise.
const sequenceIgnoresRouting = computed(
  () =>
    routing.multiHost.value &&
    (routing.resolve(form.state.value.model || null)?.hostId ??
      ORIGIN_HOST_ID) !== ORIGIN_HOST_ID,
);

async function onSubmitScript(script: ChainScriptToml) {
  const stages: ChainStageWire[] = script.stage.map((s) => ({
    prompt: s.prompt,
    frames: s.frames,
    transition: s.transition,
    fade_frames: s.fade_frames,
    negative_prompt: s.negative_prompt,
    seed_offset: s.seed_offset,
    source_image: s.source_image_b64 ?? null,
  }));
  const req: ChainRequestWire = {
    model: script.chain.model,
    stages,
    motion_tail_frames: script.chain.motion_tail_frames,
    width: script.chain.width,
    height: script.chain.height,
    fps: script.chain.fps,
    seed: script.chain.seed ?? null,
    steps: script.chain.steps,
    guidance: script.chain.guidance,
    strength: script.chain.strength,
    output_format: script.chain.output_format,
    enable_audio: script.chain.enable_audio,
  };
  try {
    const { job_id } = await createChainJob(req);
    submittedChainJobId.value = job_id;
  } catch (e) {
    composerError.value = e instanceof Error ? e.message : String(e);
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
  if (form.state.value.batchSize > 1) {
    const route = resolveSubmitRoute();
    if (route === false || !validateSubmit()) return;
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
    const style = styleHint(form.state.value.stylePreset ?? "");
    composerError.value = null;
    try {
      const response = await expandPrompt(
        {
          prompt: sourcePrompt,
          model_family: family,
          variations: count,
          ...(style ? { style } : {}),
        },
        undefined,
        route?.target,
      );
      variations.value = validateExpandedPrompts(response.expanded, count);
      preparedBatch.value = {
        batchId: createUuid(),
        sourcePrompt,
        model,
        family,
        requestedCount: count,
        selectedHostPolicy,
        baseRequest,
        decision,
        route: cloneRoute(route),
      };
    } catch (error) {
      composerError.value =
        error instanceof Error ? error.message : String(error);
    }
    return;
  }
  // batch = 1: server enrichment via the Expand modal, applied in place.
  const route = resolveSubmitRoute();
  if (route === false) return;
  expandRoute.value = cloneRoute(route);
  showExpand.value = true;
}

function onExpandStage(stageIndex: number, prompt: string) {
  const route = resolveSubmitRoute();
  if (route === false) return;
  expandRoute.value = cloneRoute(route);
  expandStageIndex.value = stageIndex;
  expandStagePrompt.value = prompt;
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
  if (expandStageIndex.value !== null) {
    scriptComposerRef.value?.setStagePrompt(expandStageIndex.value, v);
    return;
  }
  prevPrompt.value = form.state.value.prompt;
  quickPrepared.value = {
    expandedPrompt: v,
    originalPrompt: form.state.value.prompt.trim(),
    model: form.state.value.model,
    family: currentFamily.value,
    selectedHostPolicy: routing.targetId.value,
    route: cloneRoute(expandRoute.value),
  };
  form.state.value.prompt = v;
  // Same bake-and-clear as the desktop app: the rewrite absorbed the look, so
  // leaving the chip lit would apply it twice at submit.
  bakeStyleAndClear();
}

function undoExpand() {
  if (prevPrompt.value === null) return;
  form.state.value.prompt = prevPrompt.value;
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
  prevStyle.value = null;
  quickPrepared.value = null;
}

// ── Variations review (batch > 1) ─────────────────────────────────────
function useVariation(index: number) {
  const v = variations.value[index];
  if (v == null) return;
  form.state.value.prompt = v;
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
  variations.value = [];
  preparedBatch.value = null;
  for (const [index, prompt] of list.entries()) {
    // Each variation already carries the style extras, so it is the final
    // prompt — override the base request's prompt rather than re-appending.
    // Each is one print; the batch size drove the variation count, not the
    // per-job image count.
    stream.submit(
      {
        ...prepared.baseRequest,
        prompt,
        batch_size: 1,
        original_prompt: prepared.sourcePrompt,
        batch_id: prepared.batchId,
        batch_index: index + 1,
        batch_count: list.length,
      },
      prepared.decision,
      prepared.route,
    );
  }
}

// ── Source image handling (preserved) ─────────────────────────────────
async function onClearSource() {
  if (form.state.value.maskImage && !(await resolveMaskSourceConflict()))
    return;
  form.state.value.imageAttachments = [];
}

async function onPickSource(v: SourceImageState[]) {
  const qwenEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  if (
    !qwenEdit &&
    form.state.value.maskImage &&
    form.state.value.imageAttachments.length > 0 &&
    v.length > 0 &&
    !(await resolveMaskSourceConflict())
  ) {
    return;
  }
  form.state.value.imageAttachments = qwenEdit
    ? [...form.state.value.imageAttachments, ...v]
    : v.slice(0, 1);
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
  form.state.value = applyMetadataToForm(form.state.value, item.metadata, {
    format: item.format,
    models: models.value,
  });
}

function openJob(job: Job) {
  const r = job.result;
  if (!r) return;
  const match = galleryEntries.value.find(
    (e) => e.metadata.seed === r.seed_used && e.metadata.model === r.model,
  );
  if (match) openItem(match);
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
    class="mx-auto max-w-[1600px] px-4 pb-24 pt-5"
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
          :jobs="stream.jobs.value"
          @cancel="stream.cancel"
          @dismiss="stream.remove"
          @open="openJob"
        />

        <div class="flex items-center gap-2">
          <SegmentedControl
            :model-value="composerMode"
            :options="[
              { value: 'single', label: 'Single' },
              { value: 'script', label: 'Sequence' },
            ]"
            label="Composer mode"
            data-test="composer-mode"
            @update:model-value="setComposerMode"
          />
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
          v-if="composerMode === 'script' && sequenceIgnoresRouting"
          class="rounded-control bg-halide/10 px-3 py-1.5 text-xs text-halide"
          data-test="sequence-origin-note"
        >
          Sequences render on this server — the Run on choice applies to single
          prints.
        </div>

        <div
          v-if="composerMode === 'script' && !supportsChain"
          class="rounded-card-lg border border-edge bg-bench p-6 shadow-[inset_0_1px_0_var(--card-hi)]"
          data-test="chain-unsupported"
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
                <span class="text-ink-2">{{ currentModelLabel }}</span>
                renders single prints only. pick an ltx-video or ltx-2 model to
                chain clips into a sequence.
              </p>
              <div class="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  class="rounded-control border border-ce px-3 py-1.5 font-mono text-[11px] text-ink-2 transition hover:border-safelight hover:text-rebate"
                  data-test="chain-back-to-single"
                  @click="setComposerMode('single')"
                >
                  back to single
                </button>
                <router-link
                  to="/models"
                  class="rounded-control border border-ce px-3 py-1.5 font-mono text-[11px] text-ink-2 transition hover:border-safelight hover:text-rebate"
                >
                  browse video models →
                </router-link>
              </div>
            </div>
          </div>
        </div>

        <ScriptComposer
          v-else-if="composerMode === 'script'"
          ref="scriptComposerRef"
          :model="form.state.value.model"
          :width="form.state.value.width"
          :height="form.state.value.height"
          :fps="form.state.value.fps ?? 24"
          @submit="onSubmitScript"
          @expand="(idx: number, p: string) => onExpandStage(idx, p)"
        />

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
            :expanded="expanded"
            :history="promptHistory"
            @submit="onSubmit"
            @expand="onExpand"
            @undo-expand="undoExpand"
          >
            <template v-if="isPhone" #mobile-controls>
              <div
                class="mt-3 flex flex-col gap-3"
                data-test="phone-create-controls"
              >
                <CreateModelPicker
                  :models="installedModels"
                  :model="form.state.value.model"
                  @select="selectModel"
                />
                <ControlsAside
                  v-model="form.state.value"
                  :family="currentFamily"
                  :adv-count="advCount"
                  :mobile="true"
                  :last-seed="lastSeedUsed"
                  @open-advanced="openAdvanced"
                  @reset-settings="onResetSettings"
                />
              </div>
            </template>
          </ComposerCard>
          <EstimateBadge :request="estimateRequest" :target="estimateTarget" />

          <div
            v-if="submitStatus"
            class="rounded-control bg-stop/10 px-3 py-1.5 text-xs text-stop"
            data-test="composer-submit-error"
          >
            {{ submitStatus }}
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
            v-if="canvasMode === 'empty' && showColdStart"
            class="flex min-h-[300px] items-center justify-center rounded-card-lg border border-edge bg-bench p-6 shadow-[inset_0_1px_0_var(--card-hi)]"
          >
            <ColdStartGuide />
          </div>
          <ResultCanvas
            v-else
            :mode="canvasMode"
            :progress="genProgress"
            :stage="genStage"
            :preview-src="runningJob?.previewUrl ?? undefined"
            :progress-fraction="genProgress / 100"
            :develop-seed="runningJob?.seedVisual"
            :develop-phase="developPhase"
            :print-width="runningJob?.request.width"
            :print-height="runningJob?.request.height"
            :result-src="resultSrc"
            :result-caption="resultCaption"
            :error="latestError?.error ?? undefined"
            :variations="variations"
            @update:variations="variations = $event"
            @use-variation="useVariation"
            @discard="discardVariations"
            @queue="queueVariations"
            @click="canvasMode === 'result' ? openLatestResult() : undefined"
          />
        </template>

        <ChainJobCard
          v-if="submittedChainJobDetail"
          :job="submittedChainJobDetail"
          @updated="submittedChainJobId = submittedChainJobDetail?.id ?? null"
        />

        <section>
          <div class="mb-2 flex items-center justify-between">
            <span class="font-display text-[15px] font-semibold text-rebate"
              >Recent</span
            >
            <span class="font-mono text-[11px] text-ink-3"
              >{{ galleryEntries.length }} prints</span
            >
          </div>
          <RecentGrid :entries="galleryEntries" @open="openItem" />
        </section>
      </main>

      <!-- Right: controls region — model + basics + inline Advanced (spec §06
           v0.12 surface split). On phones the Advanced column collapses into
           the Advanced sheet, opened from the button inside ControlsAside. -->
      <div v-if="!isPhone" class="flex min-w-0 flex-col gap-4">
        <CreateModelPicker
          :models="installedModels"
          :model="form.state.value.model"
          @select="selectModel"
        />
        <ControlsAside
          v-model="form.state.value"
          :family="currentFamily"
          :adv-count="advCount"
          :mobile="false"
          :last-seed="lastSeedUsed"
          @open-advanced="openAdvanced"
          @reset-settings="onResetSettings"
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
          @open-picker="showPicker = true"
          @clear-source="onClearSource"
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
        @close="showAdvanced = false"
        @open-picker="showPicker = true"
        @clear-source="onClearSource"
        @open-mask="showMask = true"
        @append-prompt="form.appendPromptPhrase"
      />
    </div>

    <ExpandModal
      :open="showExpand"
      :prompt="
        expandStageIndex !== null ? expandStagePrompt : form.state.value.prompt
      "
      :expand="form.state.value.expand"
      :current-model="currentModel"
      :queue-busy="!!runningJob"
      :style-directive="expandStyleDirective"
      :target="expandRoute?.target"
      @update:expand="(v: ExpandFormState) => (form.state.value.expand = v)"
      @apply-prompt="applyExpandedPrompt"
      @close="
        showExpand = false;
        expandStageIndex = null;
        expandRoute = null;
      "
    />
    <ImagePickerModal
      :open="showPicker"
      :title="
        currentFamily === 'qwen-image-edit' ? 'Edit images' : 'Source image'
      "
      :multiple="currentFamily === 'qwen-image-edit'"
      @pick="onPickSource"
      @close="showPicker = false"
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

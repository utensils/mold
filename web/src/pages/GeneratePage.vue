<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { requestChoice, toast } from "../lib/toasts";
import ComposerCard from "../components/create/ComposerCard.vue";
import ResultCanvas from "../components/create/ResultCanvas.vue";
import ControlsAside from "../components/create/ControlsAside.vue";
import CreateModelPicker from "../components/create/CreateModelPicker.vue";
import AdvancedDrawer from "../components/create/AdvancedDrawer.vue";
import ActivityStrip from "../components/create/ActivityStrip.vue";
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
import { blobToBase64 } from "../lib/base64";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { ASPECTS } from "@ui/lib/resolution";
import {
  createChainJob,
  deleteGalleryImage,
  fetchModels,
  fetchPromptHistory,
  imageUrl,
  listGallery,
  upscaleStream,
} from "../api";
import {
  applyMetadataToForm,
  isQwenImageEditFamily,
  promptWithStyle,
  useGenerateForm,
} from "../composables/useGenerateForm";
import {
  angleForIndex,
  mergeStyleNegative,
  styleHint,
} from "../lib/stylePresets";
import { useGenerateStream, type Job } from "../composables/useGenerateStream";
import { useChainJobStream } from "../composables/useChainJobStream";
import { useQueue } from "../composables/useQueue";
import { decideGenerateRequestRouting } from "../lib/chainRouting";
import { isStandaloneGenerationModel } from "../lib/modelFilters";
import {
  maskPaddingRectangles,
  resolveSourceFitTransform,
} from "../lib/sourceFit";
import { useStatusPoll } from "../composables/useStatusPoll";
import { generationCapabilitiesForFamily } from "../lib/generateCapabilities";
import type {
  ChainRequestWire,
  ChainStageWire,
  ExpandFormState,
  GalleryImage,
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
const { status } = useStatusPoll();
const queue = useQueue();
const models = ref<ModelInfoExtended[]>([]);
// Gates the cold-start guide (spec §08 G10): only after the first models load
// resolves so we never flash the guide while the list is still in flight.
const modelsLoaded = ref(false);
const galleryEntries = ref<GalleryImage[]>([]);
const promptHistory = ref<string[]>([]);
const muted = ref(loadMuted());

const showExpand = ref(false);
const showPicker = ref(false);
const showMask = ref(false);
const showAdvanced = ref(false);
// Which advanced section to reveal when the drawer opens. "+ Add LoRA" jumps
// straight to the LoRA picker; the plain Advanced button leaves it null so the
// drawer opens on its first available section.
const advOpenTo = ref<"lora" | null>(null);
const showTemplates = ref(false);
const composerError = ref<string | null>(null);
const preprocessingStatus = ref<string | null>(null);
const submitStatus = computed(
  () => composerError.value ?? preprocessingStatus.value,
);

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

// Phone surface → the Advanced sheet instead of the side drawer.
const isPhone = ref(false);
let phoneQuery: MediaQueryList | null = null;
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

// Drawer state (mirrors GalleryPage).
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

const stream = useGenerateStream();
const submittedChainJobId = ref<string | null>(null);
const submittedChainJob = useChainJobStream(submittedChainJobId);
const submittedChainJobDetail = submittedChainJob.detail;

async function refreshModels() {
  try {
    models.value = await fetchModels();
  } catch (e) {
    console.error(e);
  } finally {
    modelsLoaded.value = true;
  }
}

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
let modelsTimer: ReturnType<typeof setInterval> | null = null;

function startAutoRefresh() {
  stopAutoRefresh();
  galleryTimer = setInterval(() => {
    if (!document.hidden) void refreshGallery();
  }, 10_000);
  modelsTimer = setInterval(() => {
    if (!document.hidden) void refreshModels();
  }, 15_000);
}

function stopAutoRefresh() {
  if (galleryTimer) {
    clearInterval(galleryTimer);
    galleryTimer = null;
  }
  if (modelsTimer) {
    clearInterval(modelsTimer);
    modelsTimer = null;
  }
}

const currentModel = computed(
  () => models.value.find((m) => m.name === form.state.value.model) ?? null,
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
  prevPrompt.value = null;
  prevStyle.value = null;
  composerError.value = null;
  preprocessingStatus.value = null;
  void nextTick(() => composerCardRef.value?.focus?.());
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

const runningJob = computed(() =>
  stream.jobs.value.find((j) => j.state === "running"),
);
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

const canvasMode = computed<"empty" | "generating" | "result" | "variations">(
  () => {
    if (variations.value.length) return "variations";
    if (runningJob.value) return "generating";
    if (latestDone.value) return "result";
    return "empty";
  },
);

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

const resultSrc = computed(() => {
  const r = latestDone.value?.result;
  if (!r) return "";
  if (r.video_thumbnail) return `data:image/png;base64,${r.video_thumbnail}`;
  return `data:image/${r.format};base64,${r.image}`;
});
const resultCaption = computed(() => {
  const r = latestDone.value?.result;
  if (!r) return "";
  const secs = Math.round(r.generation_time_ms / 1000);
  return `${r.model} · seed ${r.seed_used} · ${secs}s · this server`;
});

function openLatestResult() {
  if (latestDone.value) openJob(latestDone.value);
}

// ── Source preprocessing / fitting (preserved) ────────────────────────
async function preprocessSourceIfNeeded(): Promise<boolean> {
  const policy = form.state.value.sourceFitPolicy;
  const source = form.state.value.imageAttachments[0];
  if (policy?.mode !== "upscale-then-fit" || !source) return true;
  const model = policy.upscalerModel || form.state.value.upscaleModel;
  if (!model) return true;
  preprocessingStatus.value = `Preprocessing source with ${model}`;
  let completed = false;
  await upscaleStream(
    { model, image: source.base64, output_format: "png" },
    {
      onProgress: (evt) => {
        if (evt.type === "stage_start") preprocessingStatus.value = evt.name;
        if (evt.type === "stage_done")
          preprocessingStatus.value = `${evt.name} (done)`;
        if (evt.type === "info") preprocessingStatus.value = evt.message;
      },
      onComplete: (evt) => {
        form.state.value.imageAttachments = [
          {
            ...source,
            filename: source.filename.replace(/(\.[^.]+)?$/, "-prefit.png"),
            base64: evt.image,
            width: undefined,
            height: undefined,
            mime: "image/png",
          },
        ];
        completed = true;
      },
      onError: (err) => {
        composerError.value =
          err.kind === "http"
            ? `Source preprocessing failed: ${err.body}`
            : `Source preprocessing failed: ${err.message}`;
      },
    },
  );
  preprocessingStatus.value = null;
  return completed;
}

async function fitStillSourceToRequest(): Promise<void> {
  const family = currentModel.value?.family ?? form.state.value.modelFamily;
  if (isQwenImageEditFamily(family) || form.state.value.frames) return;
  const source = form.state.value.imageAttachments[0];
  if (!source) return;
  const target = {
    width: form.state.value.width,
    height: form.state.value.height,
  };
  const sourceImg = await loadHtmlImage(source);
  if (
    sourceImg.naturalWidth === target.width &&
    sourceImg.naturalHeight === target.height
  ) {
    return;
  }
  const policy = drawableFitPolicy(form.state.value.sourceFitPolicy);
  const transform = resolveSourceFitTransform(
    { width: sourceImg.naturalWidth, height: sourceImg.naturalHeight },
    target,
    policy,
  );
  const canvas = document.createElement("canvas");
  canvas.width = transform.outputWidth;
  canvas.height = transform.outputHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(
    sourceImg,
    transform.offsetX,
    transform.offsetY,
    transform.drawWidth,
    transform.drawHeight,
  );
  form.state.value.imageAttachments = [
    canvasToSourceImage(canvas, source, "-fit"),
  ];

  if (policy.mode !== "pad-repaint" && !form.state.value.maskImage) return;
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = transform.outputWidth;
  maskCanvas.height = transform.outputHeight;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return;
  maskCtx.fillStyle = "black";
  maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  if (form.state.value.maskImage) {
    const maskImg = await loadHtmlImage(form.state.value.maskImage);
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
  form.state.value.maskImage = canvasToSourceImage(
    maskCanvas,
    form.state.value.maskImage ?? {
      kind: source.kind,
      filename: "pad-mask.png",
      base64: "",
    },
    "-fit-mask",
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

async function onSubmit() {
  if (!validateSubmit()) return;
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  if (!(await preprocessSourceIfNeeded())) return;
  await fitStillSourceToRequest();
  const req = form.toRequest();
  stream.submit(req, decision);
  // Push to history immediately so ↑ recalls it before the server round-trips.
  composerCardRef.value?.record(req.prompt);
  if (
    form.state.value.seedMode === "increment" &&
    form.state.value.seed !== null
  ) {
    form.state.value.seed += Math.max(1, form.state.value.batchSize);
  }
}

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
function onExpand() {
  if (form.state.value.batchSize > 1) {
    // Fan out into editable variations, reviewed in the canvas before queue.
    const baseExpanded =
      promptWithStyle(form.state.value).trim() || "a quiet landscape";
    variations.value = Array.from(
      { length: form.state.value.batchSize },
      (_, i) => `${baseExpanded}, ${angleForIndex(i)}`,
    );
    return;
  }
  // batch = 1: server enrichment via the Expand modal, applied in place.
  showExpand.value = true;
}

function onExpandStage(stageIndex: number, prompt: string) {
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
}

function discardVariations() {
  variations.value = [];
}

async function queueVariations() {
  if (!validateSubmit()) return;
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    toast("error", decision.reason);
    return;
  }
  const list = variations.value.slice();
  variations.value = [];
  const base = form.toRequest();
  for (const prompt of list) {
    // Each variation already carries the style extras, so it is the final
    // prompt — override the base request's prompt rather than re-appending.
    // Each is one print; the batch size drove the variation count, not the
    // per-job image count.
    stream.submit({ ...base, prompt, batch_size: 1 }, decision);
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

async function onLightboxUseSource(item: GalleryImage) {
  try {
    const res = await fetch(imageUrl(item.filename));
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const base64 = await blobToBase64(await res.blob());
    form.state.value.imageAttachments = [
      { kind: "gallery", filename: item.filename, base64 },
    ];
    closeDrawer();
  } catch (err) {
    toast("error", err instanceof Error ? err.message : String(err));
  }
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
  advOpenTo.value = null;
  showAdvanced.value = true;
}

onMounted(async () => {
  if (typeof window.matchMedia === "function") {
    phoneQuery = window.matchMedia("(max-width: 639px)");
    syncPhone();
    phoneQuery.addEventListener?.("change", syncPhone);
  }
  await refreshModels();
  try {
    galleryEntries.value = await listGallery();
  } catch (e) {
    console.error(e);
  }
  void refreshHistory();
  if (!form.state.value.model) {
    const first = installedModels.value[0];
    if (first) form.applyModelDefaults(first);
  }
  window.addEventListener("mold:new-print", onNewPrint);
  startAutoRefresh();
});

onBeforeUnmount(() => {
  stopAutoRefresh();
  phoneQuery?.removeEventListener?.("change", syncPhone);
  window.removeEventListener("mold:new-print", onNewPrint);
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
        <ActivityStrip
          :jobs="stream.jobs.value"
          @cancel="stream.cancel"
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
          <div class="relative">
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
              <GenerationTemplatesPanel v-model="form.state.value" />
            </div>
          </div>
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
                <span class="text-ink-2">{{ form.state.value.model }}</span>
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
          />

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
            :result-src="resultSrc"
            :result-caption="resultCaption"
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
      <div class="flex min-w-0 flex-col gap-4">
        <CreateModelPicker
          :models="installedModels"
          :model="form.state.value.model"
          @select="selectModel"
        />
        <ControlsAside
          v-model="form.state.value"
          :family="currentFamily"
          :adv-count="advCount"
          :mobile="isPhone"
          @open-advanced="openAdvanced"
        />
        <!-- Tablet+ : inline, always-visible Advanced column. -->
        <AdvancedDrawer
          v-if="!isPhone"
          :mobile="false"
          v-model="form.state.value"
          :family="currentFamily"
          :adv-count="advCount"
          :open-to="advOpenTo"
          :placement-gpus="gpuListForPlacement"
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
        :open-to="advOpenTo"
        :placement-gpus="gpuListForPlacement"
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
      @update:expand="(v: ExpandFormState) => (form.state.value.expand = v)"
      @apply-prompt="applyExpandedPrompt"
      @close="
        showExpand = false;
        expandStageIndex = null;
      "
    />
    <ImagePickerModal
      :open="showPicker"
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
      @upscale="onLightboxUseSource"
      @delete="handleDelete"
    />
  </div>
</template>

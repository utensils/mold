<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { requestChoice, toast } from "../lib/toasts";
import ComposerCard from "../components/create/ComposerCard.vue";
import ResultCanvas from "../components/create/ResultCanvas.vue";
import ControlsAside from "../components/create/ControlsAside.vue";
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
import ResourceStrip from "../components/ResourceStrip.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import { blobToBase64 } from "../lib/base64";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { ASPECTS } from "@ui/lib/resolution";
import {
  createChainJob,
  deleteGalleryImage,
  fetchModels,
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
import { angleForIndex } from "../lib/stylePresets";
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
const galleryEntries = ref<GalleryImage[]>([]);
const muted = ref(loadMuted());

const showExpand = ref(false);
const showPicker = ref(false);
const showMask = ref(false);
const showAdvanced = ref(false);
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
  }
}

async function refreshGallery() {
  try {
    galleryEntries.value = await listGallery();
  } catch {
    /* ignore */
  }
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
    if (added) void refreshGallery();
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

function selectModel(model: ModelInfoExtended) {
  form.applyModelDefaults(model);
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

function applyExpandedPrompt(v: string) {
  if (expandStageIndex.value !== null) {
    scriptComposerRef.value?.setStagePrompt(expandStageIndex.value, v);
    return;
  }
  prevPrompt.value = form.state.value.prompt;
  form.state.value.prompt = v;
}

function undoExpand() {
  if (prevPrompt.value === null) return;
  form.state.value.prompt = prevPrompt.value;
  prevPrompt.value = null;
}

// ── Variations review (batch > 1) ─────────────────────────────────────
function useVariation(index: number) {
  const v = variations.value[index];
  if (v == null) return;
  form.state.value.prompt = v;
  form.state.value.stylePreset = null; // extras are already baked into `v`
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

function openAdvancedLora() {
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
  if (!form.state.value.model) {
    const first = installedModels.value[0];
    if (first) form.applyModelDefaults(first);
  }
  startAutoRefresh();
});

onBeforeUnmount(() => {
  stopAutoRefresh();
  phoneQuery?.removeEventListener?.("change", syncPhone);
  queue.stop();
});
</script>

<template>
  <div
    data-test="generate-shell"
    class="mx-auto max-w-[1600px] px-4 pb-24 pt-5"
  >
    <div class="mb-4 lg:hidden">
      <ResourceStrip variant="chip" />
    </div>

    <div
      data-test="generate-workspace"
      class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_296px] xl:grid-cols-[238px_minmax(0,1fr)_296px]"
    >
      <!-- Left rail: model list + LoRA card -->
      <aside class="hidden flex-col gap-3.5 xl:flex">
        <div class="rounded-card border border-edge bg-bench p-4">
          <div
            class="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-3"
          >
            Model
          </div>
          <div class="mt-3 flex flex-col gap-1">
            <button
              v-for="m in installedModels"
              :key="m.name"
              type="button"
              class="flex items-center gap-2 rounded-control px-2.5 py-2 text-left font-mono text-xs text-ink-2 transition hover:bg-white/5"
              :class="{ 'text-rebate': m.name === form.state.value.model }"
              :data-test="`model-row-${m.name}`"
              @click="selectModel(m)"
            >
              <span class="min-w-0 flex-1 truncate">{{ m.name }}</span>
              <span
                v-if="m.name === form.state.value.model"
                class="font-extrabold text-safelight"
                >✓</span
              >
            </button>
            <router-link
              to="/models"
              class="mt-1 px-2.5 py-1 text-left font-mono text-[11px] text-ink-3 hover:text-safelight"
              >All models →</router-link
            >
          </div>
        </div>

        <div class="rounded-card border border-edge bg-bench p-4">
          <div class="flex items-center justify-between">
            <span
              class="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-3"
              >LoRA stack</span
            >
            <span class="font-mono text-[11px] text-ink-3">{{
              form.state.value.loras.length
            }}</span>
          </div>
          <button
            type="button"
            class="mt-3 w-full rounded-control border border-dashed border-ce p-2.5 text-xs text-ink-2 hover:bg-white/5"
            data-test="add-lora"
            @click="openAdvancedLora"
          >
            + Add LoRA
          </button>
        </div>
      </aside>

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

        <ScriptComposer
          v-if="composerMode === 'script'"
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
            v-model:prompt="form.state.value.prompt"
            v-model:style-preset="form.state.value.stylePreset"
            :aspect-label="aspectLabel"
            :width="form.state.value.width"
            :height="form.state.value.height"
            :steps="form.state.value.steps"
            :batch-size="form.state.value.batchSize"
            :expanded="expanded"
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

          <ResultCanvas
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
          <div class="max-h-[52rem] overflow-y-auto pr-1">
            <GalleryFeed
              :entries="galleryEntries"
              :loading="false"
              :view="'grid'"
              :muted="muted"
              :show-recreate="true"
              @open="openItem"
              @recreate="recreateFromGallery"
            />
          </div>
        </section>
      </main>

      <!-- Right: controls -->
      <ControlsAside
        v-model="form.state.value"
        :family="currentFamily"
        :adv-count="advCount"
        @open-advanced="showAdvanced = true"
      />
    </div>

    <AdvancedDrawer
      :open="showAdvanced"
      v-model="form.state.value"
      :family="currentFamily"
      :adv-count="advCount"
      :mobile="isPhone"
      :placement-gpus="gpuListForPlacement"
      @close="showAdvanced = false"
      @open-picker="showPicker = true"
      @clear-source="onClearSource"
      @open-mask="showMask = true"
      @append-prompt="form.appendPromptPhrase"
    />

    <ExpandModal
      :open="showExpand"
      :prompt="
        expandStageIndex !== null ? expandStagePrompt : form.state.value.prompt
      "
      :expand="form.state.value.expand"
      :current-model="currentModel"
      :queue-busy="!!runningJob"
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

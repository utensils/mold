<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  GenerateFormState,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  LoraSelection,
  ModelInfoExtended,
  Scheduler,
  SourceImageState,
  SourceMediaState,
} from "../types";
import {
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
  familySupportsAudio,
  supportsLora,
} from "../types";
import LoraPicker from "./LoraPicker.vue";
import ModelPicker from "./ModelPicker.vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";
import {
  isQwenImageEditFamily,
  outputFormatsForFamily,
} from "../composables/useGenerateForm";
import { blobToBase64 } from "../lib/base64";
import GenerationTemplatesPanel from "./GenerationTemplatesPanel.vue";

const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    models: ModelInfoExtended[];
    forceExpanded?: boolean;
    showModelPicker?: boolean;
    showLoras?: boolean;
    title?: string;
  }>(),
  {
    forceExpanded: false,
    showModelPicker: true,
    showLoras: true,
    title: "Parameters",
  },
);
const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
}>();

function patch<K extends keyof GenerateFormState>(
  key: K,
  value: GenerateFormState[K],
) {
  emit("update:modelValue", { ...props.modelValue, [key]: value });
}

const currentModel = computed(
  () => props.models.find((m) => m.name === props.modelValue.model) ?? null,
);
const family = computed(
  () => currentModel.value?.family ?? props.modelValue.modelFamily,
);

const showNegative = computed(() => !NO_CFG_FAMILIES.includes(family.value));
const showScheduler = computed(() =>
  UNET_SCHEDULER_FAMILIES.includes(family.value),
);
const showVideo = computed(() => VIDEO_FAMILIES.includes(family.value));
const showLtx2 = computed(
  () => family.value === "ltx2" || family.value === "ltx-2",
);
const showCfgPlus = computed(
  () => family.value === "sd3" || family.value === "sd3.5",
);
const showStrength = computed(
  () =>
    props.modelValue.imageAttachments.length > 0 &&
    !isQwenImageEditFamily(family.value),
);
const currentSourceImage = computed(
  () => props.modelValue.imageAttachments[0] ?? null,
);

function selectModel(m: ModelInfoExtended) {
  const next: GenerateFormState = {
    ...props.modelValue,
    model: m.name,
    modelFamily: m.family,
    width: m.default_width,
    height: m.default_height,
    steps: m.default_steps,
    guidance: m.default_guidance,
    // Clear the LoRA stack on every model change. LoRAs target a
    // specific tensor layout; even FLUX → FLUX swaps may not be
    // compatible (different finetune shapes), so the safest default is
    // to start from empty and let the user re-pick deliberately.
    loras: [],
  };
  if (VIDEO_FAMILIES.includes(m.family)) {
    next.frames ??= 25;
    next.fps ??= 24;
  } else {
    next.frames = null;
    next.fps = null;
  }
  // Reset output format to the family's default when it's no longer valid
  // (e.g. switching from FLUX → LTX-2 leaves `png` selected, which the
  // server would reject).
  const formats = outputFormatsForFamily(m.family);
  if (!formats.includes(next.outputFormat)) {
    next.outputFormat = formats[0];
  }
  // Audio toggle follows the family. AV-capable (LTX-2 / LTX-2.3) → on by
  // default; anything else → null so the server's chain endpoint doesn't
  // reject a stale `enable_audio: true` carried over from a prior AV
  // session. Mirrors `applyModelDefaults` in useGenerateForm.
  next.enableAudio = familySupportsAudio(m.family) ? true : null;
  if (isQwenImageEditFamily(m.family)) {
    next.batchSize = 1;
    next.maskImage = null;
    next.controlImage = null;
    next.controlModel = "";
  } else if (next.imageAttachments.length > 1) {
    next.imageAttachments = next.imageAttachments.slice(0, 1);
  }
  emit("update:modelValue", next);
}

function onLorasChange(loras: LoraSelection[]) {
  patch("loras", loras);
}

/// Append a trigger phrase to the active prompt with comma-separation
/// when the prompt is non-empty. Mirrors `useGenerateForm.appendPromptPhrase`
/// — duplicated here because the panel owns the form value via v-model
/// rather than calling into the composable directly.
function onAppendPromptPhrase(phrase: string) {
  const trimmed = phrase.trim();
  if (!trimmed) return;
  const current = props.modelValue.prompt;
  const next = current.trim() ? `${current.trimEnd()}, ${trimmed}` : trimmed;
  patch("prompt", next);
}

const familySupportsLora = computed(() => supportsLora(family.value));

// frames must be 8n+1 (9, 17, 25, 33, ...)
function clampFrames(n: number): number {
  if (!Number.isFinite(n)) return 25;
  const rounded = Math.max(9, Math.round((n - 1) / 8) * 8 + 1);
  return rounded;
}

// Video length is a pure UI convenience — the backend only consumes frames.
const videoLength = computed(() => {
  const frames = props.modelValue.frames ?? 25;
  const fps = props.modelValue.fps ?? 24;
  return fps > 0 ? frames / fps : 0;
});

function onChangeFrames(raw: string) {
  const n = clampFrames(Number(raw));
  patch("frames", n);
}

function onChangeLength(raw: string) {
  const secs = Number(raw);
  if (!Number.isFinite(secs) || secs <= 0) return;
  const fps = props.modelValue.fps ?? 24;
  patch("frames", clampFrames(secs * fps));
}

function onChangeFps(raw: string) {
  const nextFps = Number(raw);
  if (!Number.isFinite(nextFps) || nextFps <= 0) return;
  // Changing fps keeps length steady and adjusts frames.
  const length = videoLength.value;
  emit("update:modelValue", {
    ...props.modelValue,
    fps: nextFps,
    frames: clampFrames(length * nextFps),
  });
}

const advancedOpen = ref(false);
const showMaskEditor = ref(false);
const showMaskBasePicker = ref(false);
const maskEditorSource = ref<SourceImageState | null>(null);

const sizePresets = [512, 768, 1024] as const;
const batchChips = [1, 2, 3, 4] as const;

const schedulerOptions: Scheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "unipc",
];

const pipelineOptions: Ltx2PipelineMode[] = [
  "one-stage",
  "two-stage",
  "two-stage-hq",
  "distilled",
  "ic-lora",
  "keyframe",
  "a2vid",
  "retake",
];
const spatialOptions: Ltx2SpatialUpscale[] = ["x1-5", "x2"];
const temporalOptions: Ltx2TemporalUpscale[] = ["x2"];

// ── Expand / collapse persistence ────────────────────────────────────────
const EXPANDED_KEY = "mold.generate.params.expanded";

function loadExpanded(): boolean {
  try {
    return localStorage.getItem(EXPANDED_KEY) === "true";
  } catch {
    return false;
  }
}

const expanded = ref(loadExpanded());
const bodyOpen = computed(() => props.forceExpanded || expanded.value);

function setExpanded(v: boolean) {
  expanded.value = v;
  try {
    localStorage.setItem(EXPANDED_KEY, String(v));
  } catch {
    /* ignore */
  }
}

function toggleExpanded() {
  setExpanded(!expanded.value);
}

// ── Dirty marker — any field that deviates from the model's defaults ───
const paramsDirty = computed(() => {
  const s = props.modelValue;
  const m = currentModel.value;
  if (!m) return false;
  return (
    s.width !== m.default_width ||
    s.height !== m.default_height ||
    s.steps !== m.default_steps ||
    Math.abs(s.guidance - m.default_guidance) > 0.001 ||
    s.batchSize !== 1 ||
    s.seedMode !== "random" ||
    s.seed !== null ||
    s.negativePrompt.length > 0 ||
    s.loras.length > 0 ||
    s.cfgPlus ||
    s.maskImage !== null ||
    s.controlImage !== null ||
    s.controlModel.length > 0 ||
    s.upscaleModel.length > 0 ||
    s.gifPreview ||
    s.audioFile !== null ||
    s.audioFilePath.length > 0 ||
    s.sourceVideo !== null ||
    s.sourceVideoPath.length > 0 ||
    s.keyframes.length > 0 ||
    s.pipeline !== null ||
    s.retakeRange !== null ||
    s.spatialUpscale !== null ||
    s.temporalUpscale !== null
  );
});

/// Force-expand from a parent (e.g. when `onSubmit` is called with no
/// model selected — the user can't pick one until the panel is open).
defineExpose({ setExpanded });

// ── Summary line shown when collapsed ────────────────────────────────────
const summary = computed(() => {
  const s = props.modelValue;
  const seedTxt =
    s.seedMode === "random" ? "random" : `${s.seedMode} ${s.seed ?? "—"}`;
  const modelName = s.model || "—";
  return `${modelName} · ${s.width}×${s.height} · ${s.steps} · g ${s.guidance.toFixed(1)} · seed ${seedTxt}`;
});

function patchSeedMode(mode: GenerateFormState["seedMode"]) {
  const nextSeed =
    mode === "random" ? props.modelValue.seed : (props.modelValue.seed ?? 1);
  emit("update:modelValue", {
    ...props.modelValue,
    seedMode: mode,
    seed: nextSeed,
  });
}

async function readFile(event: Event): Promise<File | null> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = "";
  return file;
}

async function readImageState(event: Event): Promise<SourceImageState | null> {
  const file = await readFile(event);
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}

async function readMediaState(event: Event): Promise<SourceMediaState | null> {
  const file = await readFile(event);
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}

async function setMaskImage(event: Event) {
  patch("maskImage", await readImageState(event));
}

function openMaskFromSource() {
  if (!currentSourceImage.value) return;
  maskEditorSource.value = currentSourceImage.value;
  showMaskEditor.value = true;
}

function openMaskBasePicker() {
  showMaskBasePicker.value = true;
}

function onPickMaskBase(images: SourceImageState[]) {
  const image = images[0] ?? null;
  showMaskBasePicker.value = false;
  if (!image) return;
  maskEditorSource.value = image;
  showMaskEditor.value = true;
}

function onApplyMask(mask: SourceImageState) {
  patch("maskImage", mask);
  showMaskEditor.value = false;
}

async function setControlImage(event: Event) {
  patch("controlImage", await readImageState(event));
}

async function setAudioFile(event: Event) {
  const audioFile = await readMediaState(event);
  if (!audioFile) return;
  emit("update:modelValue", {
    ...props.modelValue,
    audioFile,
    audioFilePath: "",
  });
}

async function setSourceVideo(event: Event) {
  const sourceVideo = await readMediaState(event);
  if (!sourceVideo) return;
  emit("update:modelValue", {
    ...props.modelValue,
    sourceVideo,
    sourceVideoPath: "",
  });
}

async function addKeyframe(event: Event) {
  const image = await readImageState(event);
  if (!image) return;
  const last = props.modelValue.keyframes.at(-1);
  const frame = last ? last.frame + 24 : 0;
  patch("keyframes", [...props.modelValue.keyframes, { frame, image }]);
}

function updateKeyframeFrame(index: number, raw: string) {
  const frame = Math.max(0, Math.round(Number(raw) || 0));
  const keyframes = props.modelValue.keyframes.slice();
  const item = keyframes[index];
  if (!item) return;
  keyframes[index] = { ...item, frame };
  patch("keyframes", keyframes);
}

function removeKeyframe(index: number) {
  const keyframes = props.modelValue.keyframes.slice();
  keyframes.splice(index, 1);
  patch("keyframes", keyframes);
}
</script>

<template>
  <div class="glass rounded-2xl p-4">
    <button
      v-if="!forceExpanded"
      type="button"
      class="flex w-full items-center gap-3 text-left"
      :aria-expanded="bodyOpen"
      data-test="params-summary-toggle"
      @click="toggleExpanded"
    >
      <span
        class="flex-1 truncate text-sm text-slate-200"
        data-test="params-summary"
      >
        {{ summary }}
      </span>
      <span
        v-if="paramsDirty"
        class="h-2 w-2 rounded-full bg-brand-400"
        data-test="params-dirty-dot"
        aria-label="Parameters differ from model defaults"
        title="Parameters differ from model defaults"
      ></span>
      <span class="text-slate-400">{{ bodyOpen ? "▾" : "▸" }}</span>
    </button>
    <div v-else class="flex items-center justify-between">
      <h2 class="text-sm font-semibold text-slate-100">
        {{ title ?? "Parameters" }}
      </h2>
      <span
        v-if="paramsDirty"
        class="rounded-full bg-brand-500/20 px-2 py-0.5 text-[11px] text-brand-100"
      >
        custom
      </span>
    </div>

    <div v-if="bodyOpen" data-test="params-body" class="mt-3">
      <section v-if="showModelPicker !== false">
        <ModelPicker
          :models="models"
          :model-value="modelValue.model"
          @update:model-value="(v: string) => patch('model', v)"
          @select="selectModel"
        />
      </section>

      <GenerationTemplatesPanel
        :model-value="modelValue"
        @update:model-value="
          (v: GenerateFormState) => emit('update:modelValue', v)
        "
      />

      <section class="mt-4">
        <label class="text-xs uppercase text-slate-400">Size</label>
        <div class="mt-1 flex flex-wrap gap-2">
          <button
            v-for="n in sizePresets"
            :key="n"
            type="button"
            class="rounded-full px-3 py-1 text-sm"
            :class="
              modelValue.width === n && modelValue.height === n
                ? 'bg-brand-500 text-white'
                : 'bg-slate-900/60 text-slate-200'
            "
            @click="
              emit('update:modelValue', {
                ...modelValue,
                width: n,
                height: n,
              })
            "
          >
            {{ n }}×{{ n }}
          </button>
          <div class="flex items-center gap-2 text-sm">
            <input
              type="number"
              :value="modelValue.width"
              class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @input="
                patch(
                  'width',
                  Number(($event.target as HTMLInputElement).value) ||
                    modelValue.width,
                )
              "
            />
            ×
            <input
              type="number"
              :value="modelValue.height"
              class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @input="
                patch(
                  'height',
                  Number(($event.target as HTMLInputElement).value) ||
                    modelValue.height,
                )
              "
            />
          </div>
        </div>
      </section>

      <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label class="text-xs uppercase text-slate-400"
            >Steps — {{ modelValue.steps }}</label
          >
          <input
            type="range"
            min="1"
            max="100"
            :value="modelValue.steps"
            class="w-full"
            @input="
              patch('steps', Number(($event.target as HTMLInputElement).value))
            "
          />
        </div>
        <div>
          <label class="text-xs uppercase text-slate-400"
            >Guidance — {{ modelValue.guidance.toFixed(1) }}</label
          >
          <input
            type="range"
            min="0"
            max="20"
            step="0.1"
            :value="modelValue.guidance"
            class="w-full"
            @input="
              patch(
                'guidance',
                Number(($event.target as HTMLInputElement).value),
              )
            "
          />
        </div>
      </section>

      <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label class="text-xs uppercase text-slate-400">Seed</label>
          <div class="mt-1 grid grid-cols-3 gap-1">
            <button
              v-for="mode in ['random', 'static', 'increment']"
              :key="mode"
              type="button"
              class="rounded-lg px-2 py-1 text-xs capitalize"
              :class="
                modelValue.seedMode === mode
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200'
              "
              @click="patchSeedMode(mode as GenerateFormState['seedMode'])"
            >
              {{ mode }}
            </button>
          </div>
          <div class="mt-2 flex items-center gap-2">
            <input
              type="number"
              :value="modelValue.seed ?? ''"
              placeholder="random"
              class="flex-1 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              :disabled="modelValue.seedMode === 'random'"
              @input="
                patch(
                  'seed',
                  ($event.target as HTMLInputElement).value === ''
                    ? null
                    : Number(($event.target as HTMLInputElement).value),
                )
              "
            />
            <button
              type="button"
              class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm"
              @click="
                emit('update:modelValue', {
                  ...modelValue,
                  seedMode: 'random',
                  seed: null,
                })
              "
            >
              🎲
            </button>
          </div>
        </div>
        <div>
          <label class="text-xs uppercase text-slate-400">Batch</label>
          <div class="mt-1 flex gap-1">
            <button
              v-for="n in batchChips"
              :key="n"
              type="button"
              class="rounded-full px-3 py-1 text-sm"
              :disabled="isQwenImageEditFamily(family)"
              :class="[
                modelValue.batchSize === n
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
                isQwenImageEditFamily(family)
                  ? 'cursor-not-allowed opacity-50'
                  : '',
              ]"
              @click="patch('batchSize', n)"
            >
              {{ n }}
            </button>
          </div>
        </div>
      </section>

      <section v-if="showNegative" class="mt-4">
        <label class="text-xs uppercase text-slate-400">Negative prompt</label>
        <textarea
          :value="modelValue.negativePrompt"
          rows="2"
          class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
          placeholder="e.g. blurry, low quality, watermark"
          @input="
            patch(
              'negativePrompt',
              ($event.target as HTMLTextAreaElement).value,
            )
          "
        />
      </section>

      <section v-if="showStrength" class="mt-4">
        <label class="text-xs uppercase text-slate-400"
          >Strength — {{ modelValue.strength.toFixed(2) }}</label
        >
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          :value="modelValue.strength"
          class="w-full"
          @input="
            patch('strength', Number(($event.target as HTMLInputElement).value))
          "
        />
      </section>

      <section v-if="!isQwenImageEditFamily(family)" class="mt-4 space-y-3">
        <label class="text-xs uppercase text-slate-400">Image inputs</label>
        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label class="rounded-lg bg-slate-900/40 p-2 text-xs text-slate-300">
            Mask image
            <input
              type="file"
              accept="image/png,image/jpeg"
              class="mt-1 block w-full text-[11px]"
              data-test="mask-upload"
              @change="setMaskImage"
            />
            <span v-if="modelValue.maskImage" class="mt-1 block truncate">
              {{ modelValue.maskImage.filename }}
              <button
                type="button"
                class="ml-1 text-slate-400 hover:text-rose-300"
                @click.prevent="patch('maskImage', null)"
              >
                clear
              </button>
            </span>
            <div class="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                class="rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
                :disabled="!currentSourceImage"
                data-test="edit-source-mask"
                @click.prevent="openMaskFromSource"
              >
                Edit source
              </button>
              <button
                type="button"
                class="rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-700"
                data-test="pick-mask-base"
                @click.prevent="openMaskBasePicker"
              >
                Pick base
              </button>
            </div>
          </label>
          <label class="rounded-lg bg-slate-900/40 p-2 text-xs text-slate-300">
            Control image
            <input
              type="file"
              accept="image/png,image/jpeg"
              class="mt-1 block w-full text-[11px]"
              @change="setControlImage"
            />
            <span v-if="modelValue.controlImage" class="mt-1 block truncate">
              {{ modelValue.controlImage.filename }}
              <button
                type="button"
                class="ml-1 text-slate-400 hover:text-rose-300"
                @click.prevent="patch('controlImage', null)"
              >
                clear
              </button>
            </span>
          </label>
        </div>
        <div
          v-if="modelValue.controlImage"
          class="grid grid-cols-1 gap-3 sm:grid-cols-2"
        >
          <input
            :value="modelValue.controlModel"
            class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
            placeholder="controlnet-canny-sd15"
            @input="
              patch('controlModel', ($event.target as HTMLInputElement).value)
            "
          />
          <label class="text-xs text-slate-400">
            Control scale — {{ modelValue.controlScale.toFixed(2) }}
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              :value="modelValue.controlScale"
              class="w-full"
              @input="
                patch(
                  'controlScale',
                  Number(($event.target as HTMLInputElement).value),
                )
              "
            />
          </label>
        </div>
        <input
          :value="modelValue.upscaleModel"
          class="w-full rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
          placeholder="Optional upscaler model, e.g. real-esrgan-x4plus:fp16"
          @input="
            patch('upscaleModel', ($event.target as HTMLInputElement).value)
          "
        />
        <ImagePickerModal
          :open="showMaskBasePicker"
          title="Mask base image"
          :multiple="false"
          @pick="onPickMaskBase"
          @close="showMaskBasePicker = false"
        />
        <MaskEditorModal
          :open="showMaskEditor"
          :source-image="maskEditorSource"
          :initial-mask="modelValue.maskImage"
          @apply="onApplyMask"
          @close="showMaskEditor = false"
        />
      </section>

      <LoraPicker
        v-if="showLoras !== false && familySupportsLora"
        :family="family"
        :model-value="modelValue.loras"
        @update:model-value="onLorasChange"
        @append-prompt="onAppendPromptPhrase"
      />

      <section
        v-if="showVideo"
        class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3"
      >
        <div>
          <label class="text-xs uppercase text-slate-400">Frames (8n+1)</label>
          <input
            type="number"
            :value="modelValue.frames ?? 25"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            @change="onChangeFrames(($event.target as HTMLInputElement).value)"
          />
        </div>
        <div>
          <label class="text-xs uppercase text-slate-400">Length (s)</label>
          <input
            type="number"
            step="0.1"
            min="0.1"
            :value="videoLength.toFixed(2)"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            data-test="video-length"
            @change="onChangeLength(($event.target as HTMLInputElement).value)"
          />
        </div>
        <div>
          <label class="text-xs uppercase text-slate-400">FPS</label>
          <input
            type="number"
            :value="modelValue.fps ?? 24"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            @change="onChangeFps(($event.target as HTMLInputElement).value)"
          />
        </div>
      </section>

      <section v-if="showVideo" class="mt-4 space-y-3">
        <label class="flex items-center gap-2 text-xs text-slate-300">
          <input
            type="checkbox"
            :checked="modelValue.gifPreview"
            @change="
              patch('gifPreview', ($event.target as HTMLInputElement).checked)
            "
          />
          Generate GIF preview
        </label>

        <div v-if="showLtx2" class="space-y-3">
          <label class="text-xs uppercase text-slate-400">LTX-2 pipeline</label>
          <select
            :value="modelValue.pipeline ?? ''"
            class="w-full rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
            @change="
              patch(
                'pipeline',
                (($event.target as HTMLSelectElement).value ||
                  null) as Ltx2PipelineMode | null,
              )
            "
          >
            <option value="">auto</option>
            <option v-for="p in pipelineOptions" :key="p" :value="p">
              {{ p }}
            </option>
          </select>
          <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label
              class="rounded-lg bg-slate-900/40 p-2 text-xs text-slate-300"
            >
              Audio file
              <input
                type="file"
                class="mt-1 block w-full text-[11px]"
                @change="setAudioFile"
              />
              <span v-if="modelValue.audioFile" class="mt-1 block truncate">
                {{ modelValue.audioFile.filename }}
                <button
                  type="button"
                  class="ml-1 text-slate-400 hover:text-rose-300"
                  @click.prevent="patch('audioFile', null)"
                >
                  clear
                </button>
              </span>
            </label>
            <input
              :value="modelValue.audioFilePath"
              :disabled="modelValue.audioFile !== null"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100 disabled:opacity-50"
              placeholder="Server audio path"
              @input="
                patch(
                  'audioFilePath',
                  ($event.target as HTMLInputElement).value,
                )
              "
            />
            <label
              class="rounded-lg bg-slate-900/40 p-2 text-xs text-slate-300"
            >
              Source video
              <input
                type="file"
                accept="video/*"
                class="mt-1 block w-full text-[11px]"
                @change="setSourceVideo"
              />
              <span v-if="modelValue.sourceVideo" class="mt-1 block truncate">
                {{ modelValue.sourceVideo.filename }}
                <button
                  type="button"
                  class="ml-1 text-slate-400 hover:text-rose-300"
                  @click.prevent="patch('sourceVideo', null)"
                >
                  clear
                </button>
              </span>
            </label>
            <input
              :value="modelValue.sourceVideoPath"
              :disabled="modelValue.sourceVideo !== null"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100 disabled:opacity-50"
              placeholder="Server source video path"
              @input="
                patch(
                  'sourceVideoPath',
                  ($event.target as HTMLInputElement).value,
                )
              "
            />
          </div>
          <div class="grid grid-cols-2 gap-3">
            <input
              type="number"
              step="0.1"
              :value="modelValue.retakeRange?.start_seconds ?? ''"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
              placeholder="Retake start"
              @input="
                patch('retakeRange', {
                  start_seconds:
                    Number(($event.target as HTMLInputElement).value) || 0,
                  end_seconds: modelValue.retakeRange?.end_seconds ?? 1,
                })
              "
            />
            <input
              type="number"
              step="0.1"
              :value="modelValue.retakeRange?.end_seconds ?? ''"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
              placeholder="Retake end"
              @input="
                patch('retakeRange', {
                  start_seconds: modelValue.retakeRange?.start_seconds ?? 0,
                  end_seconds:
                    Number(($event.target as HTMLInputElement).value) || 1,
                })
              "
            />
          </div>
          <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <select
              :value="modelValue.spatialUpscale ?? ''"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
              @change="
                patch(
                  'spatialUpscale',
                  (($event.target as HTMLSelectElement).value ||
                    null) as Ltx2SpatialUpscale | null,
                )
              "
            >
              <option value="">spatial native</option>
              <option v-for="v in spatialOptions" :key="v" :value="v">
                spatial {{ v }}
              </option>
            </select>
            <select
              :value="modelValue.temporalUpscale ?? ''"
              class="rounded-lg bg-slate-900/60 px-2 py-1 text-sm text-slate-100"
              @change="
                patch(
                  'temporalUpscale',
                  (($event.target as HTMLSelectElement).value ||
                    null) as Ltx2TemporalUpscale | null,
                )
              "
            >
              <option value="">temporal native</option>
              <option v-for="v in temporalOptions" :key="v" :value="v">
                temporal {{ v }}
              </option>
            </select>
          </div>
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <label class="text-xs uppercase text-slate-400">Keyframes</label>
              <label
                class="rounded-md bg-slate-800 px-2 py-0.5 text-xs text-slate-200 hover:bg-slate-700"
              >
                Add
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  class="hidden"
                  @change="addKeyframe"
                />
              </label>
            </div>
            <div
              v-for="(keyframe, index) in modelValue.keyframes"
              :key="`${keyframe.image.filename}-${index}`"
              class="grid grid-cols-[4.5rem_1fr_auto] items-center gap-2 rounded-lg bg-slate-900/40 p-2 text-xs text-slate-300"
            >
              <input
                type="number"
                min="0"
                :value="keyframe.frame"
                class="rounded bg-slate-950/70 px-2 py-1 text-slate-100"
                @input="
                  updateKeyframeFrame(
                    index,
                    ($event.target as HTMLInputElement).value,
                  )
                "
              />
              <span class="truncate">{{ keyframe.image.filename }}</span>
              <button
                type="button"
                class="text-slate-400 hover:text-rose-300"
                @click="removeKeyframe(index)"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      </section>

      <section v-if="showScheduler || showCfgPlus" class="mt-4">
        <button
          type="button"
          class="text-xs uppercase tracking-wide text-slate-400 hover:text-slate-200"
          @click="advancedOpen = !advancedOpen"
        >
          {{ advancedOpen ? "▾" : "▸" }} Advanced
        </button>

        <div
          v-if="advancedOpen"
          class="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2"
        >
          <div>
            <label class="text-xs uppercase text-slate-400">Scheduler</label>
            <select
              :value="modelValue.scheduler ?? 'default'"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @change="
                patch(
                  'scheduler',
                  ($event.target as HTMLSelectElement).value as Scheduler,
                )
              "
            >
              <option v-for="s in schedulerOptions" :key="String(s)" :value="s">
                {{ s }}
              </option>
            </select>
          </div>
          <label
            v-if="showCfgPlus"
            class="flex items-center gap-2 text-sm text-slate-300"
          >
            <input
              type="checkbox"
              :checked="modelValue.cfgPlus"
              @change="
                patch('cfgPlus', ($event.target as HTMLInputElement).checked)
              "
            />
            CFG++
          </label>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
/*
 * Advanced drawer (Mold Studio Create) — the capability-gated accordion of
 * fine controls. One section open at a time (parent-held `openSection`). Each
 * section maps onto EXISTING form fields (see GenerateParamsPanel semantics);
 * sections a family doesn't support never render. DrawerPanel on wide screens
 * (≥640px, 560px), SheetPanel "full" on phones. Reset clears advanced fields
 * only — the prompt, model, shape, resolution, detail and seed survive.
 */
import { computed, ref } from "vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import Chip from "@ui/components/Chip.vue";
import LoraPicker from "../LoraPicker.vue";
import PlacementPanel from "../PlacementPanel.vue";
import type {
  DevicePlacement,
  GenerateFormState,
  LoraSelection,
  OutputFormat,
  Scheduler,
  SourceFitPolicy,
} from "../../types";
import { generationCapabilitiesForFamily } from "../../lib/generateCapabilities";
import { outputFormatsForFamily } from "../../composables/useGenerateForm";

type SectionKey =
  | "scheduler"
  | "negative"
  | "source"
  | "lora"
  | "upscale"
  | "output"
  | "video"
  | "placement";

const props = withDefaults(
  defineProps<{
    open: boolean;
    modelValue: GenerateFormState;
    family: string;
    advCount?: number;
    /** Phone surface → SheetPanel instead of DrawerPanel. */
    mobile?: boolean;
    /** GPUs for the placement section (empty → section hidden). */
    placementGpus?: { ordinal: number; name: string }[];
  }>(),
  { advCount: 0, mobile: false, placementGpus: () => [] },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  close: [];
  "open-picker": [];
  "clear-source": [];
  "open-mask": [];
  "append-prompt": [phrase: string];
}>();

const NEG_CHIPS = [
  "blurry",
  "extra fingers",
  "watermark",
  "low quality",
  "oversaturated",
];

const openSection = ref<SectionKey | null>("scheduler");
function toggle(section: SectionKey) {
  openSection.value = openSection.value === section ? null : section;
}

const caps = computed(() => generationCapabilitiesForFamily(props.family));
const formats = computed(() => outputFormatsForFamily(props.family));

const showScheduler = computed(
  () => caps.value.supportsScheduler || caps.value.supportsCfgPlus,
);
const showPlacement = computed(() => props.placementGpus.length > 0);

// The Scheduler type permits parameterized object variants; the drawer only
// surfaces the named string schedulers.
const schedulerChoices = computed<string[]>(() =>
  caps.value.schedulerOptions.flatMap((s) =>
    typeof s === "string" ? [s] : [],
  ),
);
const schedulerName = computed(() =>
  typeof props.modelValue.scheduler === "string"
    ? props.modelValue.scheduler
    : "default",
);
const schedulerSummary = computed(() =>
  caps.value.supportsScheduler ? schedulerName.value : "CFG++",
);

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

// ── Scheduler & sampling ──────────────────────────────────────────────
function setScheduler(value: string) {
  patch({ scheduler: value as Scheduler });
}

// ── Negative prompt ───────────────────────────────────────────────────
function addNegative(word: string) {
  const cur = props.modelValue.negativePrompt.trim();
  patch({ negativePrompt: cur ? `${cur}, ${word}` : word });
}

// ── Source image ──────────────────────────────────────────────────────
const hasSource = computed(() => props.modelValue.imageAttachments.length > 0);
const fitOptions = [
  { value: "pad-fit", label: "Contain" },
  { value: "crop-fill", label: "Cover" },
  { value: "pad-repaint", label: "Pad + repaint" },
] as const;
const fitMode = computed(
  () => props.modelValue.sourceFitPolicy?.mode ?? "pad-repaint",
);
function setFit(mode: string) {
  patch({ sourceFitPolicy: { mode } as SourceFitPolicy });
}

// ── Upscale ───────────────────────────────────────────────────────────
const upscaleOn = computed(() => props.modelValue.upscaleModel.trim() !== "");
function toggleUpscale(on: boolean) {
  patch({ upscaleModel: on ? props.modelValue.upscaleModel || "" : "" });
  if (on) openSection.value = "upscale";
}

// ── Output & seed ─────────────────────────────────────────────────────
const seedModes = [
  { value: "random", label: "Random" },
  { value: "static", label: "Fixed" },
  { value: "increment", label: "Increment" },
] as const;

// ── LoRA / placement passthrough ──────────────────────────────────────
function setLoras(loras: LoraSelection[]) {
  patch({ loras });
}
function setPlacement(placement: DevicePlacement | null) {
  patch({ placement });
}

// frames must be 8n+1 (9, 17, 25, 33, …) — ported from GenerateParamsPanel.
function clampFrames(n: number): number {
  if (!Number.isFinite(n)) return 25;
  return Math.max(9, Math.round((n - 1) / 8) * 8 + 1);
}

// ── Reset (advanced fields only — prompt/model/shape/seed survive) ────
function resetAdvanced() {
  patch({
    negativePrompt: "",
    scheduler: null,
    cfgPlus: false,
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    controlModel: "",
    strength: 0.75,
    sourceFitPolicy: { mode: "pad-repaint" },
    loras: [],
    upscaleModel: "",
  });
}
</script>

<template>
  <component
    :is="mobile ? SheetPanel : DrawerPanel"
    :open="open"
    :width="560"
    :variant="mobile ? 'full' : undefined"
    title="Advanced"
    @close="emit('close')"
  >
    <template v-if="!mobile" #header>
      <div class="adv__head" data-test="advanced-header">
        <div class="adv__title">Advanced</div>
        <div class="adv__subtitle">
          Fine controls, tucked away until you need them
        </div>
      </div>
      <BadgePill v-if="advCount > 0" data-test="advanced-active"
        >{{ advCount }} active</BadgePill
      >
    </template>

    <div class="adv__sections">
      <AccordionSection
        v-if="showScheduler"
        icon="scheduler"
        title="Scheduler & sampling"
        :summary="schedulerSummary"
        :open="openSection === 'scheduler'"
        data-test="section-scheduler"
        @toggle="toggle('scheduler')"
      >
        <div v-if="caps.supportsScheduler" class="adv__field">
          <label class="adv__label">Scheduler</label>
          <select
            class="adv__select"
            data-test="scheduler-select"
            :value="schedulerName"
            @change="setScheduler(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="s in schedulerChoices" :key="s" :value="s">
              {{ s }}
            </option>
          </select>
        </div>
        <div v-if="caps.supportsCfgPlus" class="adv__row">
          <span class="adv__label">CFG++</span>
          <SwitchToggle
            :model-value="modelValue.cfgPlus"
            label="CFG++"
            data-test="cfg-plus"
            @update:model-value="patch({ cfgPlus: $event })"
          />
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="caps.supportsNegativePrompt"
        icon="negative"
        title="Negative prompt"
        summary="What to steer away from"
        :open="openSection === 'negative'"
        data-test="section-negative"
        @toggle="toggle('negative')"
      >
        <textarea
          class="adv__textarea"
          data-test="negative-input"
          placeholder="blurry, low quality, deformed…"
          :value="modelValue.negativePrompt"
          @input="
            patch({
              negativePrompt: ($event.target as HTMLTextAreaElement).value,
            })
          "
        />
        <div class="adv__chips">
          <Chip
            v-for="word in NEG_CHIPS"
            :key="word"
            :data-test="`neg-chip-${word.replace(/\s+/g, '-')}`"
            @click="addNegative(word)"
            >+ {{ word }}</Chip
          >
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="caps.sourceImageMode === 'single'"
        icon="image"
        title="Source image"
        :summary="
          hasSource
            ? `1 image · denoise ${modelValue.strength.toFixed(2)}`
            : 'Image-to-image & inpainting'
        "
        :open="openSection === 'source'"
        data-test="section-source"
        @toggle="toggle('source')"
      >
        <button
          v-if="!hasSource"
          type="button"
          class="adv__dropzone"
          data-test="source-attach"
          @click="emit('open-picker')"
        >
          Drop an image or <span class="adv__accent">browse</span>
        </button>
        <div v-else>
          <div class="adv__source-row">
            <span class="adv__source-name">{{
              modelValue.imageAttachments[0]?.filename
            }}</span>
            <button
              type="button"
              class="adv__remove"
              data-test="source-remove"
              @click="emit('clear-source')"
            >
              Remove
            </button>
          </div>
          <SliderRow
            label="Denoise strength"
            :model-value="modelValue.strength"
            :min="0"
            :max="1"
            :step="0.01"
            :value-label="modelValue.strength.toFixed(2)"
            @update:model-value="patch({ strength: $event })"
          />
          <div class="adv__field">
            <label class="adv__label">Fit to canvas</label>
            <SegmentedControl
              :model-value="fitMode"
              :options="fitOptions"
              label="Fit to canvas"
              @update:model-value="setFit"
            />
          </div>
          <button
            v-if="caps.supportsMask"
            type="button"
            class="adv__mask"
            data-test="source-mask"
            @click="emit('open-mask')"
          >
            {{
              modelValue.maskImage ? "Mask applied · edit" : "Edit inpaint mask"
            }}
          </button>
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="caps.supportsLora"
        icon="layers"
        title="LoRA stack"
        :summary="`${modelValue.loras.length} active · style adapters`"
        :open="openSection === 'lora'"
        data-test="section-lora"
        @toggle="toggle('lora')"
      >
        <LoraPicker
          :family="family"
          :model-value="modelValue.loras"
          @update:model-value="setLoras"
          @append-prompt="emit('append-prompt', $event)"
        />
      </AccordionSection>

      <AccordionSection
        icon="upscale"
        title="Upscale after generate"
        :summary="upscaleOn ? 'Higher-res output' : 'Off'"
        :header-interactive="false"
        :open="openSection === 'upscale'"
        data-test="section-upscale"
      >
        <template #action>
          <SwitchToggle
            :model-value="upscaleOn"
            label="Upscale after generate"
            data-test="upscale-toggle"
            @update:model-value="toggleUpscale"
          />
        </template>
        <div v-if="upscaleOn" class="adv__field">
          <label class="adv__label">Upscaler model</label>
          <input
            class="adv__input"
            data-test="upscale-model"
            placeholder="e.g. real-esrgan-x4plus"
            :value="modelValue.upscaleModel"
            @input="
              patch({ upscaleModel: ($event.target as HTMLInputElement).value })
            "
          />
        </div>
      </AccordionSection>

      <AccordionSection
        icon="output"
        title="Output & seed"
        summary="Format and reproducibility"
        :open="openSection === 'output'"
        data-test="section-output"
        @toggle="toggle('output')"
      >
        <div class="adv__field">
          <label class="adv__label">File format</label>
          <SegmentedControl
            :model-value="modelValue.outputFormat"
            :options="
              formats.map((f) => ({ value: f, label: f.toUpperCase() }))
            "
            label="File format"
            @update:model-value="
              patch({ outputFormat: $event as OutputFormat })
            "
          />
        </div>
        <div class="adv__field">
          <label class="adv__label">Seed</label>
          <SegmentedControl
            :model-value="modelValue.seedMode"
            :options="seedModes"
            label="Seed mode"
            @update:model-value="patch({ seedMode: $event })"
          />
        </div>
        <input
          v-if="modelValue.seedMode !== 'random'"
          class="adv__input"
          data-test="output-seed"
          type="number"
          min="0"
          placeholder="Seed"
          :value="modelValue.seed ?? ''"
          @input="
            patch({
              seed: Number(($event.target as HTMLInputElement).value) || null,
            })
          "
        />
      </AccordionSection>

      <AccordionSection
        v-if="caps.supportsVideo"
        icon="video"
        title="Video"
        :summary="`${modelValue.frames ?? 25} frames · ${modelValue.fps ?? 24} fps`"
        :open="openSection === 'video'"
        data-test="section-video"
        @toggle="toggle('video')"
      >
        <div class="adv__field">
          <label class="adv__label">Frames (8n+1)</label>
          <input
            class="adv__input"
            data-test="video-frames"
            type="number"
            :value="modelValue.frames ?? 25"
            @change="
              patch({
                frames: clampFrames(
                  Number(($event.target as HTMLInputElement).value),
                ),
              })
            "
          />
          <p class="adv__hint">Frames must be 8n+1 — try 97.</p>
        </div>
        <div class="adv__field">
          <label class="adv__label">Frames per second</label>
          <input
            class="adv__input"
            data-test="video-fps"
            type="number"
            min="1"
            :value="modelValue.fps ?? 24"
            @change="
              patch({
                fps: Number(($event.target as HTMLInputElement).value) || 24,
              })
            "
          />
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="showPlacement"
        icon="machines"
        title="GPU placement"
        summary="Pin this job to a device"
        :open="openSection === 'placement'"
        data-test="section-placement"
        @toggle="toggle('placement')"
      >
        <PlacementPanel
          :model-value="modelValue.placement"
          :family="family"
          :model="modelValue.model"
          :gpus="placementGpus"
          @update:model-value="setPlacement"
        />
      </AccordionSection>
    </div>

    <div class="adv__footer">
      <button
        type="button"
        class="adv__reset"
        data-test="advanced-reset"
        @click="resetAdvanced"
      >
        Reset
      </button>
      <div class="adv__spacer" />
      <button
        type="button"
        class="adv__done"
        data-test="advanced-done"
        @click="emit('close')"
      >
        Done
      </button>
    </div>
  </component>
</template>

<style scoped>
.adv__head {
  flex: 1;
}
.adv__title {
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 700;
}
.adv__subtitle {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  margin-top: 1px;
}
.adv__sections {
  display: flex;
  flex-direction: column;
  gap: 11px;
}
.adv__field {
  margin-bottom: 14px;
}
.adv__field:last-child {
  margin-bottom: 0;
}
.adv__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.adv__label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.adv__select,
.adv__input {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
  font-family: var(--f-mono);
}
.adv__textarea {
  width: 100%;
  box-sizing: border-box;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13.5px;
  line-height: 1.45;
  min-height: 64px;
  resize: none;
  outline: none;
  padding: 11px 13px;
}
.adv__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
}
.adv__dropzone {
  width: 100%;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-card);
  padding: 26px;
  font-size: 13px;
  cursor: pointer;
}
.adv__accent {
  color: var(--safelight);
  font-weight: 600;
}
.adv__source-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.adv__source-name {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.adv__remove {
  border: 0;
  background: transparent;
  color: var(--stop);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.adv__mask {
  margin-top: 12px;
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control-lg);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.adv__hint {
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
}
.adv__footer {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--edge);
}
.adv__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px 16px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.adv__spacer {
  flex: 1;
}
.adv__done {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 11px 26px;
  border-radius: var(--radius-control-lg);
  font-size: 13.5px;
  font-weight: 700;
  cursor: pointer;
}
</style>

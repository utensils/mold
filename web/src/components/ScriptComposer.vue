<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { VueDraggable } from "vue-draggable-plus";
import Icon from "@ui/components/Icon.vue";
import {
  defaultSequenceStages,
  sequenceDuration,
  sequenceValidation,
  type SequenceStage,
} from "@studio/lib/sequence";
import StageCard from "./StageCard.vue";
import { toast } from "../lib/toasts";
import ImagePickerModal from "./ImagePickerModal.vue";
import {
  readChainScript,
  writeChainScript,
  type ChainScriptToml,
  type ChainStageToml,
} from "../lib/chainToml";
import { fetchChainLimits, type ChainLimits } from "../api";
import type { SourceImageState } from "../types";

const props = defineProps<{
  model: string;
  width: number;
  height: number;
  fps: number;
}>();

const emit = defineEmits<{
  (e: "submit", script: ChainScriptToml): void;
  (e: "expand", stageIndex: number, prompt: string): void;
}>();

const DRAFT_KEY = "mold.chain.draft.v2";

function blankStage(transition?: "smooth" | "cut" | "fade"): ChainStageToml {
  return { prompt: "", frames: 97, transition };
}

function newScript(): ChainScriptToml {
  return {
    schema: "mold.chain.v1",
    chain: {
      model: props.model,
      width: props.width,
      height: props.height,
      fps: props.fps,
      steps: 8,
      guidance: 3.0,
      strength: 1.0,
      motion_tail_frames: 25,
      output_format: "mp4",
    },
    stage: defaultSequenceStages().map((stage) => ({
      prompt: stage.prompt,
      frames: stage.frames,
      transition: stage.transition,
    })),
  };
}

const script = ref<ChainScriptToml>(newScript());
const limits = ref<ChainLimits | null>(null);
// Tracks whether the chain-limits probe has resolved for the current model, so
// the UI can tell "still checking" apart from "resolved: no chain support".
const limitsLoaded = ref(false);
const importFileInput = ref<HTMLInputElement | null>(null);

// Stage index whose image picker is open, or null if the modal is closed.
// Script mode uses a single modal instance shared across stages instead of
// rendering one per stage — keeps DOM light and matches the Single-mode UX.
const pickerStageIndex = ref<number | null>(null);

onMounted(async () => {
  const draft = localStorage.getItem(DRAFT_KEY);
  if (draft) {
    try {
      script.value = JSON.parse(draft);
    } catch {
      /* ignore corrupt draft */
    }
  }
  if (script.value.stage.length < 2) {
    script.value.stage.push(blankStage("smooth"));
  }
  limits.value = await fetchChainLimits(props.model).catch(() => null);
  limitsLoaded.value = true;
  // Normalise enable_audio against the resolved model on first mount.
  // The persisted draft might carry `enable_audio: true` from a prior
  // session that used an AV-capable model; if the current model isn't
  // AV-capable, leaving that stale value in place would make the chain
  // endpoint reject the submit. So: AV → default-on if unset, non-AV →
  // hard-clear regardless of what the draft said.
  if (limits.value?.supports_audio) {
    if (script.value.chain.enable_audio === undefined) {
      script.value.chain.enable_audio = true;
    }
  } else {
    delete script.value.chain.enable_audio;
  }
});

// Strip base64 image bytes before persisting to localStorage. A single 2K×2K
// PNG base64s to ~5 MB which blows the 5–10 MB quota after a few stages.
// Keep the filename so the user sees which image was attached after reload.
function persistableDraft(s: ChainScriptToml): ChainScriptToml {
  return {
    ...s,
    stage: s.stage.map(({ source_image_b64: _omit, ...rest }) => rest),
  };
}

watch(
  script,
  (s) => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(persistableDraft(s)));
    } catch {
      /* quota exceeded — drop the draft silently */
    }
  },
  { deep: true },
);

watch(
  () => props.model,
  async (m) => {
    script.value.chain.model = m;
    limitsLoaded.value = false;
    limits.value = await fetchChainLimits(m).catch(() => null);
    limitsLoaded.value = true;
    // Mirror the onMounted default: switching to an AV-capable model
    // turns audio on if the user hasn't explicitly toggled it; switching
    // to a non-AV model clears it so the wire stays clean and the server
    // doesn't reject the chain.
    if (limits.value?.supports_audio) {
      script.value.chain.enable_audio ??= true;
    } else {
      delete script.value.chain.enable_audio;
    }
  },
);

function addStage() {
  script.value.stage.push(blankStage("smooth"));
}
function updateStage(i: number, next: ChainStageToml) {
  script.value.stage[i] = next;
}
function deleteStage(i: number) {
  if (script.value.stage.length <= 1) return;
  script.value.stage.splice(i, 1);
}
function moveUp(i: number) {
  if (i === 0) return;
  const a = script.value.stage;
  [a[i - 1], a[i]] = [a[i], a[i - 1]];
}
function moveDown(i: number) {
  const a = script.value.stage;
  if (i >= a.length - 1) return;
  [a[i], a[i + 1]] = [a[i + 1], a[i]];
}
function duplicate(i: number) {
  script.value.stage.splice(i + 1, 0, { ...script.value.stage[i] });
}

function openPicker(i: number) {
  pickerStageIndex.value = i;
}
function closePicker() {
  pickerStageIndex.value = null;
}
function onPickImage(v: SourceImageState[]) {
  const i = pickerStageIndex.value;
  if (i === null) return;
  const current = script.value.stage[i];
  if (!current) return;
  const first = v[0];
  if (!first) return;
  script.value.stage[i] = {
    ...current,
    source_image: first.filename,
    source_image_b64: first.base64,
  };
  pickerStageIndex.value = null;
}
function clearImage(i: number) {
  const current = script.value.stage[i];
  if (!current) return;
  const next: ChainStageToml = { ...current };
  delete next.source_image;
  delete next.source_image_b64;
  script.value.stage[i] = next;
}

const authoringStages = computed<SequenceStage[]>(() =>
  script.value.stage.map((stage) => ({
    prompt: stage.prompt,
    frames: stage.frames,
    transition: stage.transition ?? "smooth",
    ...(stage.fade_frames == null ? {} : { fade_frames: stage.fade_frames }),
  })),
);
const duration = computed(() =>
  sequenceDuration(
    authoringStages.value,
    script.value.chain.fps,
    script.value.chain.motion_tail_frames,
  ),
);
const totalFrames = computed(() => duration.value.frames);

function exportToml(): string {
  return writeChainScript(script.value);
}

function copyToml() {
  void navigator.clipboard.writeText(exportToml());
}

function downloadToml() {
  const toml = exportToml();
  const blob = new Blob([toml], { type: "application/toml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chain.toml";
  a.click();
  URL.revokeObjectURL(url);
}

async function importToml(file: File) {
  const text = await file.text();
  try {
    script.value = readChainScript(text);
  } catch (err) {
    toast("error", String(err));
  }
}

function handleImportChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) importToml(file);
  input.value = "";
}

const framesPerClipCap = computed(
  () => limits.value?.frames_per_clip_cap ?? 97,
);
const fadeFramesMax = computed(() => limits.value?.fade_frames_max ?? 32);
const maxStages = computed(() => limits.value?.max_stages ?? 16);
const maxTotalFrames = computed(() => limits.value?.max_total_frames ?? 1552);

const overCap = computed(
  () =>
    totalFrames.value > maxTotalFrames.value ||
    script.value.stage.length > maxStages.value,
);

const canAddStage = computed(() => script.value.stage.length < maxStages.value);

// The model resolved to a chain-capable one but the limits probe came back
// empty (network failure, or a video model that doesn't chain). Distinct from
// the still-loading state so the status line reads honestly.
const chainUnavailable = computed(
  () => limitsLoaded.value && limits.value === null,
);
const sequenceUnsupported = computed(
  () => limits.value?.supports_sequence === false,
);
const validationErrors = computed(() =>
  sequenceValidation(authoringStages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: maxTotalFrames.value,
    motionTailFrames: script.value.chain.motion_tail_frames,
  }),
);

const canGenerate = computed(
  () =>
    limits.value !== null &&
    !sequenceUnsupported.value &&
    !overCap.value &&
    validationErrors.value.length === 0,
);

const stageCountLabel = computed(() => {
  const n = script.value.stage.length;
  return `${n} ${n === 1 ? "clip" : "clips"}`;
});

const summaryLine = computed(
  () =>
    `${stageCountLabel.value} · ${totalFrames.value} frames · ${(
      totalFrames.value / script.value.chain.fps
    ).toFixed(1)}s @ ${script.value.chain.fps}fps`,
);

function submit() {
  if (!canGenerate.value) return;
  emit("submit", script.value);
}

function getStagePrompt(i: number): string {
  return script.value.stage[i]?.prompt ?? "";
}

function setStagePrompt(i: number, v: string) {
  if (script.value.stage[i]) {
    script.value.stage[i].prompt = v;
  }
}

/** Open the source-image picker for a specific stage. Exposed so the
 * parent Composer's toolbar can route its global image button to stage 0
 * in script mode without having to round-trip through CreatePage. */
function openStagePicker(i: number) {
  if (i < 0 || i >= script.value.stage.length) return;
  pickerStageIndex.value = i;
}

defineExpose({ getStagePrompt, setStagePrompt, openStagePicker });
</script>

<template>
  <div class="flex flex-col gap-3">
    <div class="flex items-center gap-2">
      <div>
        <span class="font-display text-sm font-semibold text-rebate"
          >Sequence</span
        >
        <p class="mt-0.5 text-xs text-ink-3">
          Tell the story one clip at a time.
        </p>
      </div>
      <details class="ml-auto">
        <summary class="sc-tool cursor-pointer list-none">
          <Icon name="settings" :size="13" />
          Sequence tools
        </summary>
        <div class="mt-2 flex flex-wrap justify-end gap-2">
          <button
            type="button"
            class="sc-tool"
            @click="importFileInput?.click()"
          >
            <Icon name="upload" :size="13" />
            Import
          </button>
          <input
            ref="importFileInput"
            type="file"
            accept=".toml"
            class="hidden"
            @change="handleImportChange"
          />
          <button type="button" class="sc-tool" @click="downloadToml()">
            <Icon name="download" :size="13" />
            Export
          </button>
          <button type="button" class="sc-tool" @click="copyToml()">
            <Icon name="copy" :size="13" />
            Copy TOML
          </button>
        </div>
      </details>
    </div>

    <VueDraggable v-model="script.stage" handle=".drag-handle">
      <StageCard
        v-for="(stage, i) in script.stage"
        :key="i"
        :index="i"
        :is-first="i === 0"
        :stage="stage"
        :frames-per-clip-cap="framesPerClipCap"
        :fade-frames-max="fadeFramesMax"
        @update:stage="updateStage(i, $event)"
        @delete="deleteStage(i)"
        @move-up="moveUp(i)"
        @move-down="moveDown(i)"
        @duplicate="duplicate(i)"
        @expand="emit('expand', i, stage.prompt)"
        @pick-image="openPicker(i)"
        @clear-image="clearImage(i)"
      />
    </VueDraggable>

    <div
      class="flex items-center justify-between text-xs"
      :class="overCap ? 'text-stop' : 'text-ink-3'"
      :title="
        overCap
          ? 'Reduce frames or stages — server will reject this script'
          : undefined
      "
    >
      <span class="font-mono" data-test="script-summary">{{
        summaryLine
      }}</span>
      <button
        v-if="canAddStage"
        type="button"
        class="sc-tool"
        @click="addStage"
      >
        <Icon name="plus" :size="13" />
        Add stage
      </button>
      <span v-else class="font-mono text-ink-3">max stages reached</span>
    </div>

    <p
      v-if="validationErrors.length"
      data-test="sequence-validation"
      class="rounded-control border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning"
      role="alert"
    >
      {{ validationErrors[0] }}
    </p>

    <label
      v-if="limits?.supports_audio"
      class="flex cursor-pointer items-center gap-2 px-1 text-xs text-ink-2"
      title="Generate per-stage audio and mux it into the stitched MP4 (LTX-2 / LTX-2.3 only). Smooth/Cut/Fade transitions stitch audio to match the visual transition."
    >
      <input
        type="checkbox"
        data-test="script-composer-enable-audio"
        class="h-4 w-4 rounded border-ce bg-bench text-safelight focus:ring-safelight"
        :checked="script.chain.enable_audio === true"
        @change="
          script.chain.enable_audio = ($event.target as HTMLInputElement)
            .checked
            ? true
            : undefined
        "
      />
      Generate audio
    </label>

    <p
      v-if="!limitsLoaded && limits === null"
      class="text-center font-mono text-[11px] text-ink-3"
      data-test="chain-limits-pending"
    >
      checking sequence limits…
    </p>
    <p
      v-else-if="chainUnavailable"
      class="text-center font-mono text-[11px] text-warning"
      data-test="chain-unavailable"
    >
      this model can't chain sequences.
    </p>
    <p
      v-else-if="sequenceUnsupported"
      class="rounded border border-warning/30 bg-warning/10 px-3 py-2 text-center font-mono text-[11px] leading-relaxed text-warning"
      data-test="chain-pipeline-unsupported"
      role="alert"
    >
      {{
        limits?.sequence_unsupported_reason ??
        "This model's pipeline can't render sequences."
      }}
    </p>
    <button
      class="w-full rounded-control-lg py-2.5 text-sm font-semibold transition"
      :class="
        canGenerate
          ? 'bg-safelight text-on-accent hover:brightness-110'
          : 'cursor-not-allowed border border-edge bg-bath text-ink-3'
      "
      :disabled="!canGenerate"
      data-test="script-generate"
      @click="submit"
    >
      Generate sequence
    </button>

    <ImagePickerModal
      :open="pickerStageIndex !== null"
      @pick="onPickImage"
      @close="closePicker"
    />
  </div>
</template>

<style scoped>
.sc-tool {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 12px;
  border-radius: var(--radius-control);
  font-family: var(--f-mono);
  font-size: 11px;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.sc-tool:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
</style>

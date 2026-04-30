<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type {
  DevicePlacement,
  GenerateFormState,
  OutputFormat,
} from "../types";
import PlacementPanel from "./PlacementPanel.vue";
import ScriptComposer from "./ScriptComposer.vue";
import { outputFormatsForFamily } from "../composables/useGenerateForm";

import type { ChainRoutingDecision } from "../lib/chainRouting";
import type { ChainScriptToml } from "../lib/chainToml";

export type ComposerMode = "single" | "script";

const props = defineProps<{
  modelValue: GenerateFormState;
  mode: ComposerMode;
  queueDepth: number | null;
  queueCapacity: number | null;
  gpus: { ordinal: number; state: string }[] | null;
  expandActive: boolean;
  settingsDirty: boolean;
  // Agent C (model-ui-overhaul §3) ────────────────────────────────────
  family: string; // family of the currently-selected model
  placementGpus: { ordinal: number; name: string }[];
  // ──────────────────────────────────────────────────────────────────
  /** Chain routing decision for the current form settings. When `chain`,
   * the Composer shows a "will render as N clips" cue so users understand
   * why the request will take much longer than a single-clip submit. */
  chainDecision: ChainRoutingDecision;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
  (e: "update:mode", v: ComposerMode): void;
  (e: "submit"): void;
  (e: "submit-script", script: ChainScriptToml): void;
  (e: "open-settings"): void;
  (e: "open-expand"): void;
  (e: "open-expand-stage", stageIndex: number, prompt: string): void;
  (e: "open-image-picker"): void;
  (e: "clear-source"): void;
}>();

const textarea = ref<HTMLTextAreaElement | null>(null);

function updatePrompt(value: string) {
  emit("update:modelValue", { ...props.modelValue, prompt: value });
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (props.modelValue.prompt.trim().length === 0) return;
    emit("submit");
  }
}

// Auto-grow textarea up to ~8 lines of content.
watch(
  () => props.modelValue.prompt,
  async () => {
    await nextTick();
    const el = textarea.value;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 8 * 28)}px`;
  },
  { immediate: true },
);

const statusLine = computed(() => {
  const parts: string[] = [];
  if (props.queueDepth !== null && props.queueCapacity !== null) {
    parts.push(`queue ${props.queueDepth}/${props.queueCapacity}`);
  }
  if (props.gpus) {
    const pills = props.gpus.map(
      (g) => `GPU ${g.ordinal} ${g.state === "idle" ? "▯" : "▮"}`,
    );
    parts.push(pills.join(" · "));
  }
  return parts.join(" · ");
});

function updatePlacement(p: DevicePlacement | null) {
  emit("update:modelValue", { ...props.modelValue, placement: p });
}

const outputFormats = computed(() => outputFormatsForFamily(props.family));

function updateOutputFormat(v: string) {
  emit("update:modelValue", {
    ...props.modelValue,
    outputFormat: v as OutputFormat,
  });
}

const scriptComposerRef = ref<InstanceType<typeof ScriptComposer> | null>(null);

// The composer exposes a single "🖼️" image button above the mode toggle so
// users can attach a source image from either mode without hunting for a
// different control. In Single mode this opens the global picker (wires
// into form.sourceImage). In Script mode, the per-stage pickers are still
// the primary affordance, but users expect a persistent toolbar entry —
// we route the click to the first stage's picker so it has a sensible
// target and the user doesn't have to scroll to find stage 1's attach
// button.
function onImageButton() {
  if (props.mode === "script") {
    scriptComposerRef.value?.openStagePicker(0);
    return;
  }
  emit("open-image-picker");
}

defineExpose({ scriptComposerRef });
</script>

<template>
  <div class="glass flex flex-col gap-2 rounded-3xl p-4">
    <div class="mb-1 flex items-center gap-2">
      <div class="inline-flex rounded-full bg-slate-900/60 p-0.5">
        <button
          type="button"
          :class="mode === 'single' ? 'bg-brand-500/60' : ''"
          class="rounded-full px-3 py-1 text-xs text-slate-200"
          @click="emit('update:mode', 'single')"
        >
          Single
        </button>
        <button
          type="button"
          :class="mode === 'script' ? 'bg-brand-500/60' : ''"
          class="rounded-full px-3 py-1 text-xs text-slate-200"
          @click="emit('update:mode', 'script')"
        >
          Script
        </button>
      </div>

      <!-- Global icon toolbar. Always rendered so users find the image
           picker and settings in the same place regardless of mode. The
           per-stage prompt enhancer (✨) and per-stage image pickers in
           Script mode remain the primary affordances inside each stage;
           these toolbar buttons exist so users can act on the first
           stage without scrolling, and so Single/Script feel symmetric. -->
      <div class="ml-auto flex gap-1">
        <button
          type="button"
          class="icon-btn"
          aria-label="Attach source image"
          :title="
            mode === 'script'
              ? 'Attach source image to the first stage'
              : 'Attach source image'
          "
          @click="onImageButton"
        >
          🖼️
        </button>
        <button
          type="button"
          class="icon-btn relative"
          aria-label="Settings"
          title="Settings"
          @click="emit('open-settings')"
        >
          ⚙
          <span
            v-if="settingsDirty"
            class="absolute right-1 top-1 h-2 w-2 rounded-full bg-brand-400"
          ></span>
        </button>
      </div>
    </div>

    <ScriptComposer
      v-if="mode === 'script'"
      ref="scriptComposerRef"
      :model="modelValue.model"
      :width="modelValue.width"
      :height="modelValue.height"
      :fps="modelValue.fps ?? 24"
      @submit="emit('submit-script', $event)"
      @expand="
        (idx: number, prompt: string) => emit('open-expand-stage', idx, prompt)
      "
    />

    <template v-else>
      <div class="flex items-start gap-3">
        <!-- source image chip -->
        <div
          v-if="modelValue.sourceImage"
          class="relative flex-shrink-0 overflow-hidden rounded-xl"
        >
          <img
            :src="`data:image/png;base64,${modelValue.sourceImage.base64}`"
            class="h-12 w-12 object-cover"
            alt="Source"
          />
          <button
            type="button"
            class="absolute -right-1 -top-1 h-5 w-5 rounded-full bg-slate-900/90 text-xs text-slate-100"
            aria-label="Remove source image"
            @click="emit('clear-source')"
          >
            ✕
          </button>
        </div>

        <textarea
          ref="textarea"
          :value="modelValue.prompt"
          placeholder="Describe what to generate — Enter to submit, Shift+Enter for a newline"
          class="min-h-[2.5rem] flex-1 resize-none bg-transparent text-base text-slate-100 placeholder:text-slate-500 focus:outline-none"
          @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
          @keydown="onKeydown"
        />

        <div class="flex flex-shrink-0 flex-col gap-1 sm:flex-row">
          <label class="sr-only" for="composer-output-format"
            >Output format</label
          >
          <select
            id="composer-output-format"
            data-test="composer-output-format"
            :value="modelValue.outputFormat"
            class="h-9 rounded-full bg-slate-900/60 px-3 text-sm text-slate-100 focus:outline-none"
            :title="`Output format — default: ${outputFormats[0]}`"
            @change="
              updateOutputFormat(($event.target as HTMLSelectElement).value)
            "
          >
            <option v-for="f in outputFormats" :key="f" :value="f">
              {{ f }}
            </option>
          </select>
          <button
            type="button"
            class="icon-btn"
            :class="{ 'text-brand-400': expandActive }"
            aria-label="Prompt expansion"
            @click="emit('open-expand')"
          >
            ✨
          </button>
        </div>
      </div>

      <div v-if="statusLine" class="px-1 text-xs text-slate-500">
        {{ statusLine }}
      </div>

      <div
        v-if="chainDecision.kind === 'chain'"
        class="rounded-lg bg-brand-900/40 px-3 py-1.5 text-xs text-brand-200"
      >
        Will render as
        <span class="font-semibold">{{ chainDecision.stageCount }}</span>
        chained clips of {{ chainDecision.clipFrames }} frames (motion-tail
        {{ chainDecision.motionTail }}) — expect this to take substantially
        longer than a single clip.
      </div>
      <div
        v-else-if="chainDecision.kind === 'reject'"
        class="rounded-lg bg-red-900/40 px-3 py-1.5 text-xs text-red-200"
      >
        {{ chainDecision.reason }}
      </div>

      <!-- Agent C (model-ui-overhaul §3): device placement -->
      <PlacementPanel
        :model-value="modelValue.placement"
        :family="family"
        :model="modelValue.model"
        :gpus="placementGpus"
        @update:model-value="updatePlacement"
      />
    </template>
  </div>
</template>

<style scoped>
.icon-btn {
  display: inline-flex;
  height: 2.25rem;
  width: 2.25rem;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  color: rgb(226 232 240);
  background: rgba(15, 23, 42, 0.6);
  transition: background 150ms ease;
}
.icon-btn:hover {
  background: rgba(30, 41, 59, 0.9);
}
</style>

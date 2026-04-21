<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { DevicePlacement, GenerateFormState } from "../types";
import PlacementPanel from "./PlacementPanel.vue";

import type { ChainRoutingDecision } from "../lib/chainRouting";

const props = defineProps<{
  modelValue: GenerateFormState;
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
  (e: "submit"): void;
  (e: "open-settings"): void;
  (e: "open-expand"): void;
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
</script>

<template>
  <div class="glass flex flex-col gap-2 rounded-3xl p-4">
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
        <button
          type="button"
          class="icon-btn"
          aria-label="Attach source image"
          @click="emit('open-image-picker')"
        >
          🖼️
        </button>
        <button
          type="button"
          class="icon-btn"
          :class="{ 'text-brand-400': expandActive }"
          aria-label="Prompt expansion"
          @click="emit('open-expand')"
        >
          ✨
        </button>
        <button
          type="button"
          class="icon-btn relative"
          aria-label="Settings"
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
      {{ chainDecision.motionTail }}) — expect this to take substantially longer
      than a single clip.
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

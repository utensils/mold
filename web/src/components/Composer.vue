<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { DevicePlacement, GenerateFormState } from "../types";
import PlacementPanel from "./PlacementPanel.vue";

/*
 * Compose box — prompt textarea + action rail + runtime status.
 *
 * The chrome sits inside a `.gen-prompt-box` so it picks up the direction's
 * accent (studio gets a rounded glass pill; lab gets a sharp card with a
 * left accent stripe). Action buttons use `.gen-chip` so they re-colour
 * alongside the rest of the UI.
 */

const props = defineProps<{
  modelValue: GenerateFormState;
  queueDepth: number | null;
  queueCapacity: number | null;
  gpus: { ordinal: number; state: string }[] | null;
  expandActive: boolean;
  settingsDirty: boolean;
  family: string;
  placementGpus: { ordinal: number; name: string }[];
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
  <div
    class="gen-prompt-box"
    style="display: flex; flex-direction: column; gap: 10px"
  >
    <div style="display: flex; gap: 12px; align-items: flex-start">
      <div
        v-if="modelValue.sourceImage"
        style="
          position: relative;
          flex-shrink: 0;
          overflow: hidden;
          border-radius: 10px;
        "
      >
        <img
          :src="`data:image/png;base64,${modelValue.sourceImage.base64}`"
          style="width: 48px; height: 48px; object-fit: cover; display: block"
          alt="Source"
        />
        <button
          type="button"
          class="card-iconbtn"
          style="
            position: absolute;
            top: -6px;
            right: -6px;
            width: 20px;
            height: 20px;
          "
          aria-label="Remove source image"
          @click="emit('clear-source')"
        >
          <svg
            width="10"
            height="10"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.4"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M6 6l12 12" />
            <path d="M18 6 6 18" />
          </svg>
        </button>
      </div>

      <textarea
        ref="textarea"
        :value="modelValue.prompt"
        placeholder="Describe what to generate — Enter to submit, Shift+Enter for a newline"
        class="gen-prompt"
        style="flex: 1; min-height: 40px"
        @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
        @keydown="onKeydown"
      />

      <div
        style="
          display: flex;
          flex-shrink: 0;
          flex-direction: column;
          gap: 6px;
          align-items: stretch;
        "
      >
        <button
          type="button"
          class="gen-chip"
          aria-label="Attach source image"
          @click="emit('open-image-picker')"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="9" cy="11" r="1.6" />
            <path d="m4 19 6-6 4 4 3-3 3 3" />
          </svg>
          Image
        </button>
        <button
          type="button"
          class="gen-chip"
          :class="{ on: expandActive }"
          aria-label="Prompt expansion"
          @click="emit('open-expand')"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M12 3 13.5 9 20 10.5 13.5 12 12 18 10.5 12 4 10.5 10.5 9Z"
            />
          </svg>
          Expand
        </button>
        <button
          type="button"
          class="gen-chip"
          style="position: relative"
          aria-label="Settings"
          @click="emit('open-settings')"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="3" />
            <path
              d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33 1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82c.2.49.66.84 1.17 1H21a2 2 0 1 1 0 4h-.09c-.51.16-.97.51-1.17 1z"
            />
          </svg>
          Settings
          <span
            v-if="settingsDirty"
            style="
              position: absolute;
              right: 4px;
              top: 4px;
              width: 6px;
              height: 6px;
              border-radius: 999px;
              background: var(--accent);
            "
          />
        </button>
      </div>
    </div>

    <div
      v-if="statusLine"
      style="
        padding: 0 2px;
        font-size: 11.5px;
        color: var(--fg-3);
        font-variant-numeric: tabular-nums;
      "
    >
      {{ statusLine }}
    </div>

    <PlacementPanel
      :model-value="modelValue.placement"
      :family="family"
      :model="modelValue.model"
      :gpus="placementGpus"
      @update:model-value="updatePlacement"
    />
  </div>
</template>

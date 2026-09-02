<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { ApiTarget } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import { fetchHistoryFrom, type HistoryEntry } from "../lib/api/history";
import type { GenerateForm } from "../lib/generateForm";
import type { ExpandTask, ModelEntry, RemixDimension, RemixSourceKind } from "../lib/api/types";
import { remixDimensionsForTask } from "@studio/lib/promptTransform";
import { modelDisplayNameForId } from "../lib/models";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    target: ApiTarget;
    running: boolean;
    canUndo: boolean;
    blocked: boolean;
    models?: ModelEntry[];
    remixSource?: RemixSourceKind;
    remixDimensions?: RemixDimension[];
    task?: ExpandTask;
    /**
     * Why both transforms are unavailable for the selected recipe, or `null`
     * when they are available. Today the only reason is a recipe that IGNORES
     * the prompt (no text encoder anywhere in the family), where a rewrite
     * changes no pixel and the host answers with one advisory result.
     */
    blockedReason?: string | null;
  }>(),
  {
    remixSource: "original",
    remixDimensions: () => [],
    task: "text-to-image",
    blockedReason: null,
  },
);
const modelLabel = (name: string) => modelDisplayNameForId(name, props.models ?? []);

const emit = defineEmits<{
  expand: [];
  remix: [];
  undo: [];
  "update:remixSource": [value: RemixSourceKind];
  "update:remixDimensions": [value: RemixDimension[]];
}>();

const DIMENSION_LABELS: Record<RemixDimension, string> = {
  composition: "Composition",
  camera: "Camera",
  lighting: "Lighting",
  setting: "Setting",
  mood: "Mood",
  movement: "Movement",
  style: "Style",
};

const availableDimensions = computed(() =>
  remixDimensionsForTask(props.task, Boolean(props.form.stylePreset?.trim())),
);
const hasOriginal = computed(() => Boolean(props.form.originalPrompt?.trim()));
const sourceLabel = computed(() => {
  if (!hasOriginal.value) return "Remixing current prompt";
  return props.remixSource === "original" ? "Remixing original prompt" : "Remixing current prompt";
});

function toggleDimension(dimension: RemixDimension): void {
  const next = new Set(props.remixDimensions);
  if (next.has(dimension)) {
    if (next.size === 1) return;
    next.delete(dimension);
  } else next.add(dimension);
  emit(
    "update:remixDimensions",
    availableDimensions.value.filter((candidate) => next.has(candidate)),
  );
}

const historyOpen = ref(false);
const loadingHistory = ref(false);
const error = ref("");
const history = ref<HistoryEntry[]>([]);
let historyToken = 0;

async function loadHistory(): Promise<void> {
  const token = ++historyToken;
  loadingHistory.value = true;
  error.value = "";
  try {
    const entries = await fetchHistoryFrom(props.target, "", 20);
    if (token === historyToken) history.value = entries;
  } catch (cause) {
    if (token !== historyToken) return;
    history.value = [];
    error.value = describeTransportError(cause);
  } finally {
    if (token === historyToken) loadingHistory.value = false;
  }
}

async function toggleHistory(): Promise<void> {
  historyOpen.value = !historyOpen.value;
  if (historyOpen.value) await loadHistory();
}

function useHistory(entry: HistoryEntry): void {
  props.form.prompt = entry.prompt;
  props.form.originalPrompt = null;
  historyOpen.value = false;
}

watch([() => props.target.baseUrl, () => props.target.apiKey] as const, () => {
  historyToken += 1;
  loadingHistory.value = false;
  history.value = [];
  error.value = "";
  if (historyOpen.value) void loadHistory();
});
</script>

<template>
  <div class="mobile-prompt-tools">
    <div class="mobile-prompt-actions" aria-label="Prompt tools">
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-expand"
        :disabled="running || blocked || !!blockedReason || !form.prompt.trim()"
        @click="$emit('expand')"
      >
        <template v-if="form.batchSize > 1">
          {{
            running
              ? `Preparing ${form.batchSize} variations…`
              : `Prepare ${form.batchSize} variations`
          }}
        </template>
        <template v-else>{{ running ? "Expanding…" : "Expand" }}</template>
      </button>
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-remix"
        :disabled="running || blocked || !!blockedReason || !form.prompt.trim()"
        @click="$emit('remix')"
      >
        {{ running ? "Working…" : "Remix" }}
      </button>
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-recent"
        :aria-expanded="historyOpen"
        aria-controls="mobile-prompt-history-panel"
        @click="toggleHistory"
      >
        Recent
      </button>
      <button
        v-if="canUndo"
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-undo"
        @click="$emit('undo')"
      >
        Undo
      </button>
    </div>

    <p v-if="blockedReason" class="mobile-helper-text" data-test="mobile-prompt-transform-blocked">
      {{ blockedReason }}
    </p>

    <!-- A refused Remix has nothing to disclose: the collapsed source and
         dimensions row under a disabled button would only invite a tap that
         goes nowhere. -->
    <details v-if="!blockedReason" class="mobile-remix-options" data-test="mobile-remix-options">
      <summary>{{ sourceLabel }} · {{ remixDimensions.length }} dimensions</summary>
      <fieldset v-if="hasOriginal" class="mobile-remix-source" :disabled="!!blockedReason">
        <legend>Remix source</legend>
        <label>
          <input
            type="radio"
            name="mobile-remix-source"
            value="original"
            :checked="remixSource === 'original'"
            @change="emit('update:remixSource', 'original')"
          />
          Original
        </label>
        <label>
          <input
            type="radio"
            name="mobile-remix-source"
            value="current"
            :checked="remixSource === 'current'"
            @change="emit('update:remixSource', 'current')"
          />
          Current
        </label>
      </fieldset>
      <fieldset class="mobile-remix-dimensions" :disabled="!!blockedReason">
        <legend>Creative dimensions</legend>
        <label v-for="dimension in availableDimensions" :key="dimension">
          <input
            type="checkbox"
            :checked="remixDimensions.includes(dimension)"
            @change="toggleDimension(dimension)"
          />
          {{ DIMENSION_LABELS[dimension] }}
        </label>
      </fieldset>
    </details>

    <div v-if="historyOpen" id="mobile-prompt-history-panel" class="mobile-inline-panel">
      <p v-if="loadingHistory" class="mobile-helper-text" role="status">Loading recent prompts…</p>
      <p v-else-if="history.length === 0 && !error" class="mobile-helper-text">
        No recent prompts on this host.
      </p>
      <div v-else class="mobile-option-list">
        <button
          v-for="entry in history"
          :key="`${entry.used_at}-${entry.prompt}`"
          type="button"
          data-test="mobile-prompt-history-item"
          @click="useHistory(entry)"
        >
          <span>{{ entry.prompt }}</span>
          <span>{{ modelLabel(entry.model) }}</span>
        </button>
      </div>
    </div>

    <p
      v-if="error"
      class="mobile-helper-text error-text"
      role="alert"
      data-test="mobile-prompt-tools-error"
    >
      {{ error }}
    </p>
  </div>
</template>

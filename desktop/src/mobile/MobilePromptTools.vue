<script setup lang="ts">
import { ref, watch } from "vue";
import type { ApiTarget } from "../lib/api/client";
import { fetchHistoryFrom, type HistoryEntry } from "../lib/api/history";
import type { GenerateForm } from "../lib/generateForm";

const props = defineProps<{
  form: GenerateForm;
  target: ApiTarget;
  running: boolean;
  canUndo: boolean;
  blocked: boolean;
}>();

defineEmits<{ expand: []; undo: [] }>();

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
    error.value = cause instanceof Error ? cause.message : String(cause);
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

watch(
  () => [props.target.baseUrl, props.target.apiKey] as const,
  () => {
    historyToken += 1;
    loadingHistory.value = false;
    history.value = [];
    if (historyOpen.value) void loadHistory();
  },
);
</script>

<template>
  <div class="mobile-prompt-tools">
    <div class="mobile-prompt-actions" aria-label="Prompt tools">
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-expand"
        :disabled="running || blocked || !form.prompt.trim()"
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
          <span>{{ entry.model }}</span>
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

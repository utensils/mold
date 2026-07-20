<script setup lang="ts">
import { ref, watch } from "vue";
import type { ApiTarget } from "../lib/api/client";
import { expandPrompt } from "../lib/api/expand";
import { fetchHistoryFrom, type HistoryEntry } from "../lib/api/history";
import type { GenerateForm } from "../lib/generateForm";

const props = defineProps<{
  form: GenerateForm;
  target: ApiTarget;
}>();

const openPanel = ref<"options" | "history" | null>(null);
const running = ref(false);
const loadingHistory = ref(false);
const error = ref("");
const variations = ref<1 | 3 | 5>(3);
const candidates = ref<string[]>([]);
const candidateOriginal = ref("");
const history = ref<HistoryEntry[]>([]);
let expansionToken = 0;
let historyToken = 0;

function expansionOptions(count: number) {
  return {
    variations: count,
    ...(props.form.family ? { modelFamily: props.form.family } : {}),
  };
}

async function expandOnce(): Promise<void> {
  const original = props.form.prompt.trim();
  if (!original || running.value) return;
  const token = ++expansionToken;
  running.value = true;
  error.value = "";
  try {
    const response = await expandPrompt(original, expansionOptions(1), props.target);
    if (token !== expansionToken) return;
    const expanded = response.expanded[0]?.trim();
    if (!expanded) throw new Error("Expansion returned no prompt.");
    props.form.originalPrompt = original;
    props.form.prompt = expanded;
  } catch (cause) {
    if (token !== expansionToken) return;
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (token === expansionToken) running.value = false;
  }
}

async function preview(): Promise<void> {
  const original = props.form.prompt.trim();
  if (!original || running.value) return;
  const token = ++expansionToken;
  running.value = true;
  error.value = "";
  candidates.value = [];
  candidateOriginal.value = "";
  try {
    const response = await expandPrompt(original, expansionOptions(variations.value), props.target);
    if (token !== expansionToken) return;
    candidates.value = response.expanded.map((candidate) => candidate.trim()).filter(Boolean);
    candidateOriginal.value = original;
    if (candidates.value.length === 0) throw new Error("Expansion returned no prompts.");
  } catch (cause) {
    if (token !== expansionToken) return;
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (token === expansionToken) running.value = false;
  }
}

function applyCandidate(candidate: string): void {
  props.form.originalPrompt = candidateOriginal.value || props.form.prompt.trim();
  props.form.prompt = candidate;
  openPanel.value = null;
}

function undo(): void {
  if (props.form.originalPrompt === null) return;
  props.form.prompt = props.form.originalPrompt;
  props.form.originalPrompt = null;
  candidates.value = [];
  candidateOriginal.value = "";
}

function toggleOptions(): void {
  error.value = "";
  openPanel.value = openPanel.value === "options" ? null : "options";
}

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
  if (openPanel.value === "history") {
    openPanel.value = null;
    return;
  }
  openPanel.value = "history";
  await loadHistory();
}

function useHistory(entry: HistoryEntry): void {
  props.form.prompt = entry.prompt;
  props.form.originalPrompt = null;
  openPanel.value = null;
}

function invalidateExpansion(): void {
  expansionToken += 1;
  running.value = false;
  candidates.value = [];
  candidateOriginal.value = "";
  error.value = "";
}

// Expansion is advisory, not authoritative. If the user keeps typing or a
// model switch changes the expansion context while the host is working, the
// late response must never overwrite the newer form or leave stale choices
// available to tap.
watch(() => [props.form.prompt, props.form.family, props.form.model] as const, invalidateExpansion);

watch(
  () => [props.target.baseUrl, props.target.apiKey] as const,
  () => {
    invalidateExpansion();
    historyToken += 1;
    loadingHistory.value = false;
    history.value = [];
    if (openPanel.value === "history") void loadHistory();
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
        :disabled="running || !form.prompt.trim()"
        @click="expandOnce"
      >
        {{ running && openPanel !== "options" ? "Expanding…" : "Expand" }}
      </button>
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-options"
        :aria-expanded="openPanel === 'options'"
        aria-controls="mobile-prompt-options-panel"
        @click="toggleOptions"
      >
        Options
      </button>
      <button
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-recent"
        :aria-expanded="openPanel === 'history'"
        aria-controls="mobile-prompt-history-panel"
        @click="toggleHistory"
      >
        Recent
      </button>
      <button
        v-if="form.originalPrompt !== null"
        type="button"
        class="secondary-button"
        data-test="mobile-prompt-undo"
        @click="undo"
      >
        Undo
      </button>
    </div>

    <div
      v-if="openPanel === 'options'"
      id="mobile-prompt-options-panel"
      class="mobile-inline-panel"
    >
      <fieldset class="mobile-compact-fieldset">
        <legend>Variations</legend>
        <div class="mobile-segmented-control">
          <button
            v-for="count in [1, 3, 5] as const"
            :key="count"
            type="button"
            :aria-pressed="variations === count"
            :data-test="`mobile-prompt-variations-${count}`"
            @click="variations = count"
          >
            {{ count }}
          </button>
        </div>
      </fieldset>
      <button
        type="button"
        class="secondary-button mobile-full-button"
        data-test="mobile-prompt-preview"
        :disabled="running || !form.prompt.trim()"
        @click="preview"
      >
        {{ running ? "Expanding…" : "Preview candidates" }}
      </button>
      <div v-if="candidates.length" class="mobile-option-list">
        <button
          v-for="(candidate, index) in candidates"
          :key="`${index}-${candidate}`"
          type="button"
          data-test="mobile-prompt-candidate"
          @click="applyCandidate(candidate)"
        >
          <span>{{ candidate }}</span>
          <span aria-hidden="true">Use</span>
        </button>
      </div>
    </div>

    <div
      v-if="openPanel === 'history'"
      id="mobile-prompt-history-panel"
      class="mobile-inline-panel"
    >
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

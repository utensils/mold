<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { PreparedExpansionBatch } from "../lib/preparedExpansion";

const props = defineProps<{
  batch: PreparedExpansionBatch;
  staleReasons: string[];
  preparing: boolean;
  error: string;
  submitting: boolean;
}>();
const emit = defineEmits<{
  edit: [payload: { id: string; text: string }];
  remove: [id: string];
  collapse: [id: string];
  regenerate: [];
  refresh: [];
  discard: [];
  generate: [];
}>();

const root = ref<HTMLElement | null>(null);
const confirmButton = ref<HTMLButtonElement | null>(null);
const confirming = ref<{ batchId: string; promptId: string } | null>(null);
const stale = computed(() => props.staleReasons.length > 0);
const valid = computed(() => props.batch.prompts.every((prompt) => prompt.text.trim()));

function edit(id: string, event: Event): void {
  emit("edit", { id, text: (event.target as HTMLTextAreaElement).value });
}

async function remove(id: string): Promise<void> {
  if (props.submitting || props.preparing || confirming.value) return;
  if (props.batch.prompts.length === 2) {
    confirming.value = { batchId: props.batch.batchId, promptId: id };
    await nextTick();
    confirmButton.value?.focus();
    return;
  }
  emit("remove", id);
}

function requestReplacement(kind: "refresh" | "regenerate"): void {
  if (confirming.value) return;
  if (kind === "refresh") emit("refresh");
  else emit("regenerate");
}

async function keep(): Promise<void> {
  const id = confirming.value?.promptId;
  confirming.value = null;
  await nextTick();
  if (id) root.value?.querySelector<HTMLButtonElement>(`[data-remove-id="${id}"]`)?.focus();
}

function confirmCollapse(): void {
  const pending = confirming.value;
  if (
    !pending ||
    pending.batchId !== props.batch.batchId ||
    !props.batch.prompts.some((prompt) => prompt.id === pending.promptId)
  ) {
    confirming.value = null;
    return;
  }
  emit("collapse", pending.promptId);
}

watch(
  () => props.batch.batchId,
  () => {
    confirming.value = null;
  },
  { flush: "sync" },
);
</script>

<template>
  <section
    ref="root"
    class="mobile-prepared"
    data-test="mobile-prepared-expansion"
    aria-labelledby="mobile-prepared-title"
  >
    <header>
      <div>
        <h2 id="mobile-prepared-title">Review {{ batch.prompts.length }} variations</h2>
        <p>{{ batch.route.label }} · frozen route</p>
      </div>
      <button
        v-if="!stale"
        type="button"
        class="secondary-button mobile-touch-action"
        data-test="mobile-regenerate-prepared"
        :disabled="preparing || submitting || !!confirming"
        @click="requestReplacement('regenerate')"
      >
        {{ preparing ? "Regenerating…" : "Regenerate all" }}
      </button>
    </header>

    <div v-if="preparing" class="mobile-prepared-note" role="status" aria-live="polite">
      Preparing a replacement on {{ batch.route.label }}. Your current variations stay available.
    </div>
    <div v-if="stale" class="mobile-prepared-stale" role="alert">
      <strong>Prepared work is out of date</strong>
      <ul>
        <li v-for="reason in staleReasons" :key="reason">{{ reason }}</li>
      </ul>
      <button
        type="button"
        class="secondary-button mobile-touch-action"
        data-test="mobile-refresh-prepared"
        :disabled="preparing || submitting || !!confirming"
        @click="requestReplacement('refresh')"
      >
        Refresh variations
      </button>
    </div>
    <p v-if="error" class="error-text" role="alert">{{ error }}</p>

    <ol>
      <li v-for="(prompt, index) in batch.prompts" :key="prompt.id">
        <span aria-hidden="true">{{ index + 1 }}</span>
        <textarea
          class="control mobile-prepared-editor"
          rows="3"
          :value="prompt.text"
          :aria-label="`Variation ${index + 1} of ${batch.prompts.length}`"
          :aria-invalid="!prompt.text.trim() || undefined"
          :disabled="submitting || preparing || !!confirming"
          @input="edit(prompt.id, $event)"
        />
        <button
          type="button"
          class="secondary-button mobile-touch-action"
          data-test="mobile-prepared-remove"
          :data-remove-id="prompt.id"
          :aria-label="`Remove variation ${index + 1}`"
          :disabled="submitting || preparing || !!confirming"
          @click="remove(prompt.id)"
        >
          Remove
        </button>
      </li>
    </ol>

    <div
      v-if="confirming"
      class="mobile-collapse-confirm"
      role="alertdialog"
      aria-modal="false"
      aria-labelledby="mobile-collapse-title"
      @keydown.esc="keep"
    >
      <strong id="mobile-collapse-title">Continue with one prompt?</strong>
      <p>This ends prepared review and moves the remaining variation into ordinary Batch 1.</p>
      <button
        ref="confirmButton"
        type="button"
        class="primary-button mobile-touch-action"
        data-test="mobile-confirm-collapse"
        @click="confirmCollapse"
      >
        Continue with one prompt
      </button>
      <button
        type="button"
        class="secondary-button mobile-touch-action"
        data-test="mobile-keep-prepared"
        @click="keep"
      >
        Keep two variations
      </button>
    </div>

    <footer>
      <button
        type="button"
        class="secondary-button mobile-touch-action"
        data-test="mobile-discard-prepared"
        @click="emit('discard')"
      >
        Discard prepared batch
      </button>
      <button
        type="button"
        class="primary-button mobile-touch-action"
        data-test="mobile-develop-prepared"
        :disabled="preparing || submitting || stale || !valid || !!confirming"
        @click="emit('generate')"
      >
        {{ submitting ? "Queueing variations…" : `Develop ${batch.prompts.length} variations` }}
      </button>
    </footer>
  </section>
</template>

<style scoped>
.mobile-prepared {
  display: grid;
  gap: 12px;
  padding: 14px 0;
  border-block: 1px solid var(--edge);
}
.mobile-prepared header,
.mobile-prepared footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.mobile-prepared h2,
.mobile-prepared p {
  margin: 0;
}
.mobile-prepared h2 {
  font-size: var(--text-body-lg);
}
.mobile-prepared header p {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: var(--text-caption);
}
.mobile-prepared ol {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.mobile-prepared li {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}
.mobile-prepared li > span {
  padding-top: 12px;
  color: var(--halide);
  text-align: center;
}
.mobile-prepared li > button {
  grid-column: 2;
  justify-self: start;
}
.mobile-prepared-editor {
  min-height: 88px;
  box-sizing: border-box;
  font-size: 16px;
  resize: vertical;
}
.mobile-touch-action {
  min-height: 44px;
}
.mobile-prepared-note,
.mobile-prepared-stale,
.mobile-collapse-confirm {
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--control-edge);
  border-radius: var(--radius-control);
  background: var(--bench);
}
.mobile-prepared-stale ul {
  margin: 0;
  padding-left: 20px;
  color: var(--ink-2);
}
.mobile-collapse-confirm p {
  color: var(--ink-2);
}
.mobile-prepared footer > * {
  flex: 1 1 150px;
}
@media (max-width: 360px) {
  .mobile-prepared footer > * {
    flex-basis: 100%;
  }
}
</style>

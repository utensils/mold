<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { hostSelectionLabel, type PreparedExpansionBatch } from "../../lib/preparedExpansion";
import type { ExpansionPullView } from "../../lib/expansionPull";
import ExpansionPullStatus from "./ExpansionPullStatus.vue";
import type { DisplayableModel } from "../../lib/models";

const props = defineProps<{
  batch: PreparedExpansionBatch;
  staleReasons: string[];
  preparing: boolean;
  error: string | null;
  pullStatus: ExpansionPullView | null;
  pullModel: string | null;
  pullHostLabel: string | null;
  pullEtaSeconds: number | null;
  activeHostLabel: string | null;
  submitting: boolean;
  models?: DisplayableModel[] | undefined;
}>();

const emit = defineEmits<{
  (e: "edit", payload: { id: string; text: string }): void;
  (e: "remove", id: string): void;
  (e: "collapse", removedId: string): void;
  (e: "regenerate"): void;
  (e: "refresh"): void;
  (e: "discard"): void;
  (e: "pull"): void;
  (e: "retry-expansion"): void;
  (e: "generate"): void;
}>();

const confirmingRemovalId = ref<string | null>(null);
const confirmButton = ref<HTMLButtonElement | null>(null);
const sectionEl = ref<HTMLElement | null>(null);
const REVIEW_PAGE_SIZE = 8;
const reviewAll = ref(false);
const reviewPage = ref(0);
const hasStaleReasons = computed(() => props.staleReasons.length > 0);
const emptyPromptIndexes = computed(() =>
  props.batch.prompts.flatMap((prompt, index) => (prompt.text.trim() ? [] : [index])),
);
const canGenerate = computed(
  () =>
    !props.preparing &&
    !props.submitting &&
    !hasStaleReasons.value &&
    emptyPromptIndexes.value.length === 0,
);
const hasCompactReview = computed(() => props.batch.prompts.length > REVIEW_PAGE_SIZE);
const reviewPageCount = computed(() =>
  Math.max(1, Math.ceil(props.batch.prompts.length / REVIEW_PAGE_SIZE)),
);
const visiblePrompts = computed(() => {
  const start = reviewAll.value ? reviewPage.value * REVIEW_PAGE_SIZE : 0;
  return props.batch.prompts
    .slice(start, start + REVIEW_PAGE_SIZE)
    .map((prompt, offset) => ({ prompt, index: start + offset }));
});

const policyLabel = computed(() => {
  const labels = new Map([[props.batch.route.hostId, props.batch.route.label]]);
  return hostSelectionLabel(props.batch.selectedHostPolicy, labels);
});

function editorInput(id: string, event: Event) {
  if (props.submitting) return;
  emit("edit", { id, text: (event.target as HTMLTextAreaElement).value });
}

async function removePrompt(id: string, index: number) {
  if (props.submitting) return;
  if (props.batch.prompts.length === 2) {
    confirmingRemovalId.value = id;
    await nextTick();
    confirmButton.value?.focus();
    return;
  }
  emit("remove", id);
  await nextTick();
  const editors = Array.from(
    document.querySelectorAll<HTMLTextAreaElement>('[data-test="prepared-prompt"]'),
  );
  editors[Math.min(index, editors.length - 1)]?.focus();
}

async function keepPreparedBatch() {
  const id = confirmingRemovalId.value;
  confirmingRemovalId.value = null;
  await nextTick();
  sectionEl.value?.querySelector<HTMLButtonElement>(`[data-remove-id="${id}"]`)?.focus();
}

const replacementFocusPending = ref(false);
const replacementOwnedFocus = ref(false);
function requestReplacement(kind: "refresh" | "regenerate") {
  replacementFocusPending.value = true;
  replacementOwnedFocus.value =
    !!sectionEl.value &&
    !!document.activeElement &&
    sectionEl.value.contains(document.activeElement);
  if (kind === "refresh") emit("refresh");
  else emit("regenerate");
}

watch(
  () => props.batch.batchId,
  async () => {
    reviewAll.value = false;
    reviewPage.value = 0;
    if (!replacementFocusPending.value) return;
    replacementFocusPending.value = false;
    const active = document.activeElement;
    const shouldRestoreFocus =
      replacementOwnedFocus.value &&
      (active === document.body || (!!active && !!sectionEl.value?.contains(active)));
    replacementOwnedFocus.value = false;
    if (!shouldRestoreFocus) return;
    await nextTick();
    sectionEl.value?.querySelector<HTMLTextAreaElement>('[data-test="prepared-prompt"]')?.focus();
  },
  { flush: "post" },
);

watch(
  () => props.batch.prompts.length,
  () => {
    reviewPage.value = Math.min(reviewPage.value, reviewPageCount.value - 1);
    if (!hasCompactReview.value) reviewAll.value = false;
  },
);

function confirmCollapse() {
  const id = confirmingRemovalId.value;
  if (!id) return;
  confirmingRemovalId.value = null;
  emit("collapse", id);
}
</script>

<template>
  <section
    ref="sectionEl"
    data-test="prepared-expansion-batch"
    class="border-edge mt-3 flex max-h-[min(34rem,60vh)] flex-col overflow-hidden rounded-chrome border bg-bath/45 p-3"
    aria-labelledby="prepared-expansion-title"
  >
    <header class="flex flex-wrap items-start justify-between gap-2">
      <div>
        <h2 id="prepared-expansion-title" class="text-body font-semibold text-ink">
          Review {{ batch.prompts.length }} variations
        </h2>
        <p class="mt-0.5 text-caption text-ink-2">
          {{ batch.route.label }}
          <span aria-hidden="true"> · </span>
          <span>{{ policyLabel }} resolved</span>
        </p>
      </div>
      <button
        v-if="!hasStaleReasons"
        type="button"
        data-test="regenerate-prepared"
        class="border-edge min-h-7 rounded-control border px-2 text-caption text-ink-2 transition-colors duration-100 hover:text-ink disabled:opacity-50"
        :disabled="preparing || submitting"
        @click="requestReplacement('regenerate')"
      >
        {{ preparing ? "Regenerating…" : "Regenerate all" }}
      </button>
    </header>

    <div
      v-if="preparing"
      role="status"
      aria-live="polite"
      class="mt-3 rounded-control bg-[color-mix(in_srgb,var(--halide)_10%,transparent)] px-2.5 py-2 text-caption text-halide"
    >
      Expanding {{ batch.requestedCount }} prompts on {{ activeHostLabel ?? batch.route.label }}…
      The current set stays available until its replacement is ready.
    </div>

    <div
      v-if="hasStaleReasons"
      role="alert"
      class="border-halide/40 mt-3 rounded-control border bg-[color-mix(in_srgb,var(--halide)_8%,transparent)] p-2.5"
    >
      <p class="text-body font-semibold text-ink">Prepared work is out of date</p>
      <ul class="mt-1 list-disc space-y-0.5 pl-4 text-caption text-ink-2">
        <li v-for="reason in staleReasons" :key="reason">{{ reason }}</li>
      </ul>
      <button
        type="button"
        data-test="refresh-prepared"
        class="border-halide/50 mt-2 min-h-7 rounded-control border px-2 text-caption text-halide transition-colors duration-100 hover:border-halide hover:text-ink disabled:opacity-50"
        :disabled="preparing || submitting"
        @click="requestReplacement('refresh')"
      >
        {{ preparing ? "Refreshing variations…" : "Refresh variations" }}
      </button>
    </div>

    <div
      v-if="error && !pullStatus"
      role="alert"
      class="border-stop/45 mt-3 flex flex-wrap items-center justify-between gap-2 rounded-control border bg-stop/10 px-2.5 py-2 text-caption text-stop"
    >
      <span>{{ error }}</span>
    </div>

    <ExpansionPullStatus
      v-if="pullStatus && pullModel && pullHostLabel && error"
      :model="pullModel"
      :host-label="pullHostLabel"
      :error="error"
      :status="pullStatus"
      :eta-seconds="pullEtaSeconds"
      :models="models"
      @pull="emit('pull')"
      @retry-expansion="emit('retry-expansion')"
    />

    <div
      v-if="hasCompactReview && !reviewAll"
      data-test="compact-review-summary"
      class="border-edge mt-3 flex flex-wrap items-center justify-between gap-2 rounded-control border bg-bench px-2.5 py-2"
    >
      <p class="text-caption text-ink-2">
        Showing the first {{ REVIEW_PAGE_SIZE }} prompts.
        {{ batch.prompts.length - REVIEW_PAGE_SIZE }} more are ready to queue.
      </p>
      <button
        type="button"
        data-test="review-all-prompts"
        class="border-edge min-h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink"
        @click="reviewAll = true"
      >
        Review all
      </button>
    </div>

    <div
      v-else-if="hasCompactReview"
      data-test="review-pagination"
      class="mt-3 flex flex-wrap items-center justify-between gap-2"
    >
      <button
        type="button"
        data-test="compact-prompt-review"
        class="min-h-7 rounded-control px-2 text-caption text-ink-2 hover:text-ink"
        @click="
          reviewAll = false;
          reviewPage = 0;
        "
      >
        Compact review
      </button>
      <div class="flex items-center gap-2 text-caption text-ink-2">
        <button
          type="button"
          data-test="previous-prompt-page"
          class="border-edge min-h-7 rounded-control border px-2 disabled:opacity-50"
          :disabled="reviewPage === 0"
          @click="reviewPage -= 1"
        >
          Previous
        </button>
        <span>Page {{ reviewPage + 1 }} of {{ reviewPageCount }}</span>
        <button
          type="button"
          data-test="next-prompt-page"
          class="border-edge min-h-7 rounded-control border px-2 disabled:opacity-50"
          :disabled="reviewPage >= reviewPageCount - 1"
          @click="reviewPage += 1"
        >
          Next
        </button>
      </div>
    </div>

    <ol class="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
      <li
        v-for="{ prompt, index } in visiblePrompts"
        :key="prompt.id"
        class="border-edge grid grid-cols-[2rem_minmax(0,1fr)_auto] items-start gap-2 rounded-control border bg-bench p-2 max-[560px]:grid-cols-[2rem_minmax(0,1fr)]"
      >
        <span class="data-mono pt-1.5 text-halide" aria-hidden="true">
          {{ String(index + 1).padStart(2, "0") }}
        </span>
        <div class="min-w-0">
          <textarea
            data-test="prepared-prompt"
            data-selectable
            rows="2"
            :value="prompt.text"
            :aria-label="`Variation ${index + 1} of ${batch.prompts.length}`"
            :aria-invalid="!prompt.text.trim() || undefined"
            :disabled="submitting"
            class="border-edge min-h-16 w-full resize-y rounded-control border bg-bath px-2.5 py-2 text-body text-ink outline-none transition-colors duration-100 placeholder:text-ink-3 focus:border-safelight"
            @input="editorInput(prompt.id, $event)"
          />
          <p v-if="!prompt.text.trim()" class="mt-1 text-caption text-stop">
            Variation {{ index + 1 }} needs a prompt before generation.
          </p>
        </div>
        <button
          v-if="batch.prompts.length > 1"
          type="button"
          class="min-h-7 rounded-control px-2 text-caption text-ink-3 transition-colors duration-100 hover:bg-stop/10 hover:text-stop max-[560px]:col-start-2 max-[560px]:justify-self-start"
          :aria-label="`Remove variation ${index + 1}`"
          :data-remove-id="prompt.id"
          :disabled="submitting"
          @click="removePrompt(prompt.id, index)"
        >
          Remove
        </button>
      </li>
    </ol>

    <div
      v-if="confirmingRemovalId"
      role="alertdialog"
      aria-modal="false"
      aria-labelledby="collapse-title"
      class="border-safelight/45 mt-3 rounded-control border bg-[color-mix(in_srgb,var(--safelight)_8%,transparent)] p-3"
      @keydown.esc="keepPreparedBatch"
    >
      <p id="collapse-title" class="text-body font-semibold text-ink">Continue with one prompt?</p>
      <p class="mt-1 max-w-[65ch] text-caption text-ink-2">
        This ends the prepared batch and moves the remaining variation into the main composer.
      </p>
      <div class="mt-2 flex flex-wrap gap-2">
        <button
          ref="confirmButton"
          type="button"
          data-test="confirm-collapse"
          class="min-h-8 rounded-control bg-safelight px-3 text-caption font-semibold text-on-accent"
          @click="confirmCollapse"
        >
          Continue with one prompt
        </button>
        <button
          type="button"
          data-test="keep-prepared-batch"
          class="border-edge min-h-8 rounded-control border px-3 text-caption text-ink-2 hover:text-ink"
          @click="keepPreparedBatch"
        >
          Keep prepared batch
        </button>
      </div>
    </div>

    <footer
      class="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-edge pt-3"
    >
      <button
        type="button"
        data-test="discard-prepared"
        class="min-h-8 rounded-control px-2 text-caption text-ink-3 transition-colors duration-100 hover:bg-stop/10 hover:text-stop"
        @click="emit('discard')"
      >
        Discard prepared batch
      </button>
      <div class="flex min-w-0 flex-wrap items-center justify-end gap-2">
        <span class="text-caption text-ink-3">
          {{ batch.prompts.length }} {{ batch.prompts.length === 1 ? "prompt" : "prompts" }} linked
          to Batch
        </span>
        <button
          type="button"
          data-test="generate-prepared"
          class="min-h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-50"
          :disabled="!canGenerate"
          @click="emit('generate')"
        >
          {{
            submitting
              ? "Queueing variations…"
              : hasCompactReview
                ? `Queue all ${batch.prompts.length}`
                : `Generate ${batch.prompts.length} variations`
          }}
        </button>
      </div>
    </footer>
  </section>
</template>

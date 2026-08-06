<script setup lang="ts">
import { computed } from "vue";
import type { RemixDimension, RemixSourceKind } from "../lib/api/types";

export interface MobileRemixReviewVariant {
  id: string;
  prompt: string;
  dimensions: RemixDimension[];
  selected: boolean;
}

const props = defineProps<{
  sourceKind: RemixSourceKind;
  sourcePrompt: string;
  variants: MobileRemixReviewVariant[];
  hostLabel: string;
  staleReasons: string[];
  running: boolean;
  error: string;
}>();

const emit = defineEmits<{
  toggle: [id: string];
  edit: [payload: { id: string; prompt: string }];
  reremix: [];
  apply: [];
  restore: [];
  discard: [];
}>();

const selectedCount = computed(() =>
  props.variants.reduce((count, variant) => count + Number(variant.selected), 0),
);
const sourceLabel = computed(() =>
  props.sourceKind === "original" ? "Original prompt" : "Current prompt",
);

function edit(id: string, event: Event): void {
  emit("edit", { id, prompt: (event.target as HTMLTextAreaElement).value });
}
</script>

<template>
  <section
    class="mobile-remix-review"
    data-test="mobile-remix-review"
    aria-labelledby="mobile-remix-review-title"
  >
    <header>
      <div>
        <h2 id="mobile-remix-review-title">Review remix variants</h2>
        <p>{{ sourceLabel }} · {{ hostLabel }} · frozen route</p>
      </div>
      <button
        type="button"
        class="secondary-button mobile-touch-action"
        data-test="mobile-reremix"
        :disabled="running"
        @click="emit('reremix')"
      >
        {{ running ? "Re-remixing…" : "Re-remix" }}
      </button>
    </header>

    <details class="mobile-remix-source-preview">
      <summary>{{ sourceLabel }}</summary>
      <p>{{ sourcePrompt }}</p>
    </details>

    <div v-if="staleReasons.length" class="mobile-prepared-stale" role="alert">
      <strong>Remix is out of date</strong>
      <ul>
        <li v-for="reason in staleReasons" :key="reason">{{ reason }}</li>
      </ul>
      <p>Re-remix with the current inputs before applying these variants.</p>
    </div>
    <p v-if="error" class="error-text" role="alert">{{ error }}</p>

    <ol>
      <li v-for="(variant, index) in variants" :key="variant.id">
        <label class="mobile-remix-select">
          <input
            type="checkbox"
            :checked="variant.selected"
            :disabled="running || staleReasons.length > 0"
            :aria-label="`Select remix ${index + 1}`"
            @change="emit('toggle', variant.id)"
          />
          <span>{{ index + 1 }}</span>
        </label>
        <textarea
          class="control mobile-remix-editor"
          rows="4"
          :value="variant.prompt"
          :disabled="running || staleReasons.length > 0"
          :aria-label="`Remix ${index + 1} of ${variants.length}`"
          :aria-invalid="!variant.prompt.trim() || undefined"
          @input="edit(variant.id, $event)"
        />
        <ul
          class="mobile-remix-dimension-tags"
          :aria-label="`Dimensions varied in remix ${index + 1}`"
        >
          <li v-for="dimension in variant.dimensions" :key="dimension">{{ dimension }}</li>
        </ul>
      </li>
    </ol>

    <footer>
      <div class="mobile-remix-secondary-actions">
        <button
          type="button"
          class="secondary-button mobile-touch-action"
          data-test="mobile-remix-restore-source"
          :disabled="running"
          @click="emit('restore')"
        >
          Restore source
        </button>
        <button
          type="button"
          class="secondary-button mobile-touch-action"
          data-test="mobile-remix-discard"
          :disabled="running"
          @click="emit('discard')"
        >
          Keep prompt
        </button>
      </div>
      <button
        type="button"
        class="primary-button mobile-touch-action"
        data-test="mobile-remix-apply"
        :disabled="
          running ||
          staleReasons.length > 0 ||
          selectedCount === 0 ||
          variants.some((variant) => variant.selected && !variant.prompt.trim())
        "
        @click="emit('apply')"
      >
        {{
          selectedCount === 1
            ? "Apply selected remix"
            : selectedCount > 1
              ? `Prepare ${selectedCount} selected`
              : "Select a remix"
        }}
      </button>
    </footer>
  </section>
</template>

<style scoped>
.mobile-remix-review {
  display: grid;
  gap: 12px;
  padding: 14px 0;
  border-block: 1px solid var(--edge);
}
.mobile-remix-review header,
.mobile-remix-review footer,
.mobile-remix-secondary-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.mobile-remix-review h2,
.mobile-remix-review p {
  margin: 0;
}
.mobile-remix-review h2 {
  font-size: var(--text-body-lg);
}
.mobile-remix-review header p,
.mobile-remix-source-preview {
  color: var(--ink-3);
  font-size: var(--text-caption);
}
.mobile-remix-source-preview p {
  margin-top: 8px;
  color: var(--ink-2);
  overflow-wrap: anywhere;
}
.mobile-remix-review > ol {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.mobile-remix-review > ol > li {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}
.mobile-remix-select {
  display: grid;
  min-width: 44px;
  min-height: 44px;
  place-items: center;
}
.mobile-remix-select input {
  width: 20px;
  height: 20px;
}
.mobile-remix-select span {
  color: var(--ink-3);
  font-size: var(--text-caption);
}
.mobile-remix-editor {
  min-height: 104px;
  box-sizing: border-box;
  font-size: 16px;
  resize: vertical;
}
.mobile-remix-dimension-tags {
  grid-column: 2;
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.mobile-remix-dimension-tags li {
  border: 1px solid var(--control-edge);
  border-radius: 999px;
  padding: 3px 7px;
  color: var(--ink-3);
  font-size: var(--text-caption);
  text-transform: capitalize;
}
.mobile-remix-review footer > .primary-button {
  width: auto;
  flex: 1 1 180px;
}
</style>

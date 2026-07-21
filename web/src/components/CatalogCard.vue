<script setup lang="ts">
/*
 * Discover model card (spec WEB MODELS Discover, prototype lines 1607-1611):
 * a kind badge with a "Details ›" affordance, the mono model name, a
 * family · size line, and a full-width Pull button. The card body opens the
 * detail drawer; Pull enqueues the download straight into the shell.
 */
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import type { CatalogEntryWire } from "../types";

const props = defineProps<{ entry: CatalogEntryWire }>();
const emit = defineEmits<{ open: []; pull: [] }>();

function formatGB(bytes: number | null): string {
  if (!bytes) return "—";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

// engine_phase > 5 marks a catalog package mold can't run yet.
const supported = computed(() => props.entry.engine_phase <= 5);
const pullLabel = computed(() => {
  if (!supported.value) return "Unsupported";
  return props.entry.installed ? "Repair" : "Pull";
});
</script>

<template>
  <article class="card" data-test="discover-card">
    <div class="card__top">
      <span class="card__kind">{{ props.entry.kind }}</span>
      <span
        v-if="props.entry.installed"
        class="card__installed"
        title="Already on disk"
      >
        installed
      </span>
      <button
        type="button"
        class="card__details"
        data-test="details-btn"
        @click="emit('open')"
      >
        Details ›
      </button>
    </div>

    <button
      type="button"
      class="card__nameline"
      data-test="card-open"
      @click="emit('open')"
    >
      <span class="card__name">{{ props.entry.name }}</span>
      <span class="card__meta">
        {{ props.entry.family }} · {{ formatGB(props.entry.size_bytes) }}
      </span>
    </button>

    <button
      type="button"
      class="card__pull"
      data-test="pull-btn"
      :disabled="!supported"
      :title="supported ? undefined : 'Unsupported catalog package'"
      @click="emit('pull')"
    >
      <Icon v-if="supported" name="download" :size="13" />
      {{ pullLabel }}
    </button>
  </article>
</template>

<style scoped>
.card {
  display: flex;
  flex-direction: column;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 16px;
  transition: border-color var(--dur-quick) var(--ease);
}

.card:hover {
  border-color: var(--ce);
}

.card__top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}

.card__kind {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--halide);
  border: 1px solid color-mix(in srgb, var(--halide) 40%, transparent);
  padding: 1px 7px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}

.card__installed {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--success);
  border: 1px solid color-mix(in srgb, var(--success) 40%, transparent);
  padding: 1px 7px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}

.card__details {
  margin-left: auto;
  border: 0;
  background: transparent;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  cursor: pointer;
  padding: 2px 0;
  transition: color var(--dur-quick) var(--ease);
}

.card__details:hover {
  color: var(--rebate);
}

.card__nameline {
  border: 0;
  background: transparent;
  text-align: left;
  padding: 0;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.card__name {
  font-family: var(--f-mono);
  font-size: 13.5px;
  font-weight: 600;
  color: var(--rebate);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card__meta {
  font-size: 11px;
  color: var(--ink-3);
  margin-bottom: 14px;
}

.card__pull {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 9px;
  border-radius: var(--radius-control);
  font-family: var(--f-body);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.card__pull:hover:not(:disabled) {
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 8%, transparent);
}

.card__pull:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.card__pull:focus-visible,
.card__details:focus-visible,
.card__nameline:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>

<script setup lang="ts">
/*
 * Command palette (⌘K) — presentational overlay. The host owns open state,
 * query, and the filtered item list; this component renders them and reports
 * close / update:query / run(id). ArrowUp/Down move the highlight (default
 * first), Enter runs it, Esc closes. Clicking the scrim closes.
 */
import { nextTick, ref, watch } from "vue";
import Icon from "./Icon.vue";

import type { PaletteItem } from "./types";

export type { PaletteItem };

const props = withDefaults(
  defineProps<{
    open: boolean;
    query: string;
    items: readonly PaletteItem[];
    emptyText?: string;
    /** A slower source (the live model catalog) is still resolving. */
    busy?: boolean;
    busyText?: string;
  }>(),
  { emptyText: "No matches", busy: false, busyText: "Searching models…" },
);

const emit = defineEmits<{
  close: [];
  "update:query": [value: string];
  run: [id: string];
}>();

const inputEl = ref<HTMLInputElement | null>(null);
const highlighted = ref(0);

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    highlighted.value = 0;
    await nextTick();
    inputEl.value?.focus();
  },
);

watch(
  () => props.query,
  () => {
    highlighted.value = 0;
  },
);

function onInput(event: Event) {
  emit("update:query", (event.target as HTMLInputElement).value);
}

function move(delta: number) {
  const count = props.items.length;
  if (count === 0) return;
  highlighted.value = (highlighted.value + delta + count) % count;
  const el = document.getElementById(optionId(highlighted.value));
  el?.scrollIntoView?.({ block: "nearest" });
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") {
    event.preventDefault();
    emit("close");
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    move(1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    move(-1);
  } else if (event.key === "Enter") {
    event.preventDefault();
    const item = props.items[highlighted.value];
    if (item) emit("run", item.id);
  }
}

function optionId(index: number) {
  const item = props.items[index];
  return item ? `ms-palette-opt-${item.id}` : "";
}
</script>

<template>
  <div v-if="open" class="ms-palette" @click="emit('close')">
    <div
      class="ms-palette__panel ms-fade-up"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      @click.stop
    >
      <div class="ms-palette__head">
        <Icon class="ms-palette__search" name="search" :size="17" />
        <input
          ref="inputEl"
          class="ms-palette__input"
          type="text"
          autofocus
          role="combobox"
          aria-expanded="true"
          aria-controls="ms-palette-list"
          :aria-activedescendant="optionId(highlighted) || undefined"
          aria-label="Search actions, models, settings"
          placeholder="Search actions, models, settings…"
          :value="query"
          @input="onInput"
          @keydown="onKeydown"
        />
        <span class="ms-palette__esc" aria-hidden="true">esc</span>
      </div>

      <div id="ms-palette-list" class="ms-palette__body" role="listbox">
        <button
          v-for="(item, index) in items"
          :id="optionId(index)"
          :key="item.id"
          type="button"
          class="ms-palette__row"
          role="option"
          :aria-selected="index === highlighted"
          :data-hl="index === highlighted ? 'true' : undefined"
          @mouseenter="highlighted = index"
          @click="emit('run', item.id)"
        >
          <span class="ms-palette__section">{{ item.section }}</span>
          <span class="ms-palette__label">{{ item.label }}</span>
          <span v-if="item.hint" class="ms-palette__hint">{{ item.hint }}</span>
          <Icon class="ms-palette__arrow" name="arrow-right" :size="14" />
        </button>
        <div
          v-if="busy"
          class="ms-palette__busy"
          role="status"
          aria-live="polite"
        >
          {{ busyText }}
        </div>
        <div v-else-if="items.length === 0" class="ms-palette__empty">
          {{ emptyText }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ms-palette {
  position: absolute;
  inset: 0;
  z-index: 120;
  background: rgba(6, 5, 10, 0.55);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 84px;
}

.ms-palette__panel {
  width: 560px;
  max-width: 86%;
  max-height: 440px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-card-lg);
  box-shadow: 0 40px 90px -20px rgba(0, 0, 0, 0.7);
}

.ms-palette__head {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--edge);
}

.ms-palette__search {
  flex: 0 0 auto;
  color: var(--ink-3);
}

.ms-palette__input {
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 15px;
  outline: none;
}

.ms-palette__input::placeholder {
  color: var(--ink-3);
}

.ms-palette__esc {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-sm);
  padding: 2px 6px;
}

.ms-palette__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 8px;
}

.ms-palette__row {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--rebate);
  padding: 11px 12px;
  border-radius: var(--radius-control);
  text-align: left;
  cursor: pointer;
  font-family: var(--f-body);
  transition: background var(--dur-quick) var(--ease);
}

.ms-palette__row[data-hl="true"] {
  background: var(--surface);
}

.ms-palette__row:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-palette__section {
  flex: 0 0 48px;
  width: 48px;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.ms-palette__label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13.5px;
}

.ms-palette__hint {
  flex: 0 0 auto;
  max-width: 40%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.04em;
  color: var(--ink-3);
}

.ms-palette__arrow {
  flex: 0 0 auto;
  color: var(--ce);
}

.ms-palette__busy {
  padding: 14px;
  text-align: center;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.ms-palette__empty {
  padding: 28px;
  text-align: center;
  color: var(--ink-3);
  font-size: 13px;
}
</style>

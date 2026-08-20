<script setup lang="ts">
/*
 * Tag chip editor (Library organization, V3 "Shelf"). Existing tags render
 * as removable chips; the trailing input adds a tag on Enter or comma, with
 * autocomplete from the merged cross-host tag vocabulary. The parent owns the
 * mutation (fan-out to every copy) — this component only reports intent.
 */
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import { normalizeTagName, tagKey } from "@studio/lib/libraryOrganization";
import type { TagCount } from "../../types";

const props = withDefaults(
  defineProps<{
    tags: readonly string[];
    suggestions?: readonly TagCount[];
    disabled?: boolean;
    placeholder?: string;
    /** Tags present on only part of a selection (bulk editor). */
    mixed?: readonly string[];
  }>(),
  {
    suggestions: () => [],
    disabled: false,
    placeholder: "Add tag…",
    mixed: () => [],
  },
);

const emit = defineEmits<{
  (e: "add", tag: string): void;
  (e: "remove", tag: string): void;
}>();

const draft = ref("");
const inputEl = ref<HTMLInputElement | null>(null);
const open = ref(false);

const present = computed(() => new Set(props.tags.map((tag) => tagKey(tag))));

const matches = computed(() => {
  const needle = tagKey(draft.value);
  return props.suggestions
    .filter((tag) => !present.value.has(tagKey(tag.name)))
    .filter((tag) => !needle || tagKey(tag.name).includes(needle))
    .slice(0, 6);
});

function commit(raw = draft.value) {
  const name = normalizeTagName(raw);
  draft.value = "";
  if (!name || present.value.has(tagKey(name))) return;
  emit("add", name);
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" || event.key === ",") {
    event.preventDefault();
    commit();
  } else if (
    event.key === "Backspace" &&
    draft.value.length === 0 &&
    props.tags.length > 0
  ) {
    emit("remove", props.tags[props.tags.length - 1]!);
  } else if (event.key === "Escape") {
    draft.value = "";
    open.value = false;
    inputEl.value?.blur();
  }
}

function focus() {
  inputEl.value?.focus();
}
defineExpose({ focus });
</script>

<template>
  <div class="te" :data-disabled="disabled ? 'true' : undefined">
    <div class="te__chips">
      <span
        v-for="tag in tags"
        :key="tag"
        class="te__chip"
        :data-mixed="mixed.includes(tag) ? 'true' : undefined"
        data-test="tag-chip"
      >
        <Icon name="tag" :size="11" />
        <span class="te__name">{{ tag }}</span>
        <button
          v-if="!disabled"
          type="button"
          class="te__x"
          :aria-label="`Remove tag ${tag}`"
          data-test="tag-remove"
          @click="emit('remove', tag)"
        >
          <Icon name="close" :size="11" :stroke-width="2.2" />
        </button>
      </span>
      <input
        v-if="!disabled"
        ref="inputEl"
        v-model="draft"
        class="te__input"
        type="text"
        :placeholder="placeholder"
        aria-label="Add tag"
        data-test="tag-input"
        @keydown="onKeydown"
        @focus="open = true"
        @blur="open = false"
      />
      <span v-else-if="tags.length === 0" class="te__empty">No tags</span>
    </div>
    <div
      v-if="!disabled && open && matches.length > 0"
      class="te__suggest"
      data-test="tag-suggestions"
    >
      <span class="te__suggest-k">suggested</span>
      <button
        v-for="tag in matches"
        :key="tag.name"
        type="button"
        class="te__sug"
        data-test="tag-suggestion"
        @mousedown.prevent="commit(tag.name)"
      >
        {{ tag.name }}<span class="te__sug-n">{{ tag.count }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.te {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.te__chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  min-height: 30px;
  padding: 4px 6px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
}
.te[data-disabled="true"] .te__chips {
  border-style: dashed;
}
.te__chips:focus-within {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}
.te__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 22px;
  padding: 0 4px 0 7px;
  border-radius: var(--radius-pill);
  background: var(--sel-bg);
  color: var(--sel-ink);
  font-size: 11.5px;
  font-weight: 600;
}
.te__chip[data-mixed="true"] {
  background: transparent;
  border: 1px dashed var(--sel-border);
}
.te__name {
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.te__x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.te__x:hover {
  background: color-mix(in srgb, var(--rebate) 12%, transparent);
}
.te__input {
  flex: 1;
  min-width: 90px;
  height: 22px;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  outline: none;
}
.te__input::placeholder {
  color: var(--ink-3);
}
.te__empty {
  font-size: 12px;
  font-style: italic;
  color: var(--ink-3);
}
.te__suggest {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
}
.te__suggest-k {
  font-family: var(--f-mono);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.te__sug {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 22px;
  padding: 0 8px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  background: transparent;
  color: var(--ink-2);
  font-size: 11.5px;
  cursor: pointer;
}
.te__sug:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.te__sug-n {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}
</style>

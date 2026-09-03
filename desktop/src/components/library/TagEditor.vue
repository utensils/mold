<script setup lang="ts">
/*
 * TagEditor — chip list + combobox input for a print's tags (Library V3
 * "Shelf", Lightbox aside and bulk bar). Pure props/emits: the parent owns
 * the tag set and the fan-out; this only edits a string[] and suggests from
 * the host-merged tag counts. Enter / comma adds, Backspace on an empty
 * input removes the last chip, Escape closes the suggestions, ↑/↓ move the
 * suggestion cursor. Suggestions match by `tagKey` (case-insensitive) and
 * never offer a tag already on the print.
 */
import { computed, ref, watch } from "vue";
import { normalizeTagName, tagKey } from "@studio/lib/libraryOrganization";
import type { TagCount } from "@studio/lib/api/galleryOrganization";

const props = withDefaults(
  defineProps<{
    modelValue: string[];
    suggestions?: TagCount[];
    disabled?: boolean;
    placeholder?: string;
    /** Max suggestions shown at once. */
    limit?: number;
    /** Accessible name for the input. */
    ariaLabel?: string;
  }>(),
  {
    suggestions: () => [],
    disabled: false,
    placeholder: "Add tag…",
    limit: 8,
    ariaLabel: "Add a tag",
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: string[]];
  add: [name: string];
  remove: [name: string];
}>();

const uid = `tag-editor-${Math.random().toString(36).slice(2, 8)}`;
const listboxId = `${uid}-listbox`;

const draft = ref("");
const open = ref(false);
const cursor = ref(-1);
const inputEl = ref<HTMLInputElement | null>(null);

const presentKeys = computed(() => new Set(props.modelValue.map(tagKey)));

/** Suggestions matching the draft (prefix-first, then substring), minus
 *  tags already present. An empty draft lists the most-used tags. */
const matches = computed<TagCount[]>(() => {
  const query = tagKey(draft.value);
  const candidates = props.suggestions.filter((tag) => {
    const key = tagKey(tag.name);
    return key.length > 0 && !presentKeys.value.has(key) && (query === "" || key.includes(query));
  });
  if (query) {
    candidates.sort((a, b) => {
      const ap = tagKey(a.name).startsWith(query) ? 0 : 1;
      const bp = tagKey(b.name).startsWith(query) ? 0 : 1;
      return ap - bp || b.count - a.count;
    });
  }
  return candidates.slice(0, props.limit);
});

watch(matches, (list) => {
  if (cursor.value >= list.length) cursor.value = list.length - 1;
});

const activeOptionId = computed(() =>
  open.value && cursor.value >= 0 ? `${uid}-option-${cursor.value}` : undefined,
);

function commit(raw: string) {
  const name = normalizeTagName(raw);
  if (!name) return;
  const key = tagKey(name);
  if (presentKeys.value.has(key)) {
    draft.value = "";
    return;
  }
  // Prefer the canonical casing a suggestion already carries.
  const canonical = props.suggestions.find((tag) => tagKey(tag.name) === key)?.name ?? name;
  emit("update:modelValue", [...props.modelValue, canonical]);
  emit("add", canonical);
  draft.value = "";
  cursor.value = -1;
}

function removeAt(index: number) {
  const name = props.modelValue[index];
  if (name === undefined) return;
  const next = props.modelValue.filter((_, i) => i !== index);
  emit("update:modelValue", next);
  emit("remove", name);
}

function onKeydown(event: KeyboardEvent) {
  if (props.disabled) return;
  switch (event.key) {
    case "Enter":
    case ",": {
      event.preventDefault();
      const picked = cursor.value >= 0 ? matches.value[cursor.value] : undefined;
      commit(picked ? picked.name : draft.value);
      break;
    }
    case "Backspace":
      if (draft.value.length === 0 && props.modelValue.length > 0) {
        event.preventDefault();
        removeAt(props.modelValue.length - 1);
      }
      break;
    case "Escape":
      if (open.value) {
        event.preventDefault();
        event.stopPropagation();
        open.value = false;
        cursor.value = -1;
      }
      break;
    case "ArrowDown":
      event.preventDefault();
      open.value = true;
      if (matches.value.length > 0) cursor.value = (cursor.value + 1) % matches.value.length;
      break;
    case "ArrowUp":
      event.preventDefault();
      if (matches.value.length > 0) {
        cursor.value = (cursor.value - 1 + matches.value.length) % matches.value.length;
      }
      break;
    default:
      break;
  }
}

function onInput() {
  open.value = true;
  cursor.value = -1;
}

function onBlur() {
  // Let a click on a suggestion land first.
  window.setTimeout(() => {
    open.value = false;
    cursor.value = -1;
  }, 120);
}

function focusInput() {
  inputEl.value?.focus();
}

defineExpose({ focusInput });
</script>

<template>
  <div
    class="border-border-control flex min-h-[34px] flex-wrap items-center gap-1.5 rounded-window border bg-bg-deep px-1.5 py-1"
    :class="disabled ? 'opacity-60' : 'cursor-text'"
    data-test="tag-editor"
    @click="focusInput"
  >
    <span
      v-for="(tag, index) in modelValue"
      :key="tag"
      class="border-border inline-flex h-[22px] items-center gap-1 rounded-control border bg-bg pr-1 pl-2 text-xs text-fg"
      data-test="tag-chip"
    >
      <span class="max-w-40 truncate">{{ tag }}</span>
      <button
        type="button"
        class="flex h-4 w-4 items-center justify-center rounded-inner text-xs leading-none text-fg-dim hover:bg-error/15 hover:text-error disabled:cursor-default disabled:hover:bg-transparent"
        :aria-label="`Remove tag ${tag}`"
        :disabled="disabled"
        data-test="tag-remove"
        @click.stop="removeAt(index)"
      >
        ×
      </button>
    </span>
    <span class="relative min-w-24 flex-1">
      <input
        ref="inputEl"
        v-model="draft"
        data-selectable
        type="text"
        class="h-[22px] w-full bg-transparent px-1 text-xs text-fg outline-none placeholder:text-fg-dim"
        role="combobox"
        aria-autocomplete="list"
        :aria-expanded="open && matches.length > 0"
        :aria-controls="listboxId"
        :aria-activedescendant="activeOptionId"
        :aria-label="ariaLabel"
        :placeholder="placeholder"
        :disabled="disabled"
        data-test="tag-input"
        @keydown="onKeydown"
        @input="onInput"
        @focus="open = true"
        @blur="onBlur"
      />
      <ul
        v-if="open && matches.length > 0"
        :id="listboxId"
        role="listbox"
        aria-label="Tag suggestions"
        class="border-border absolute top-full left-0 z-20 mt-1 max-h-48 w-56 overflow-y-auto rounded-window border bg-bg py-1 shadow-md"
        data-test="tag-suggestions"
      >
        <li
          v-for="(tag, i) in matches"
          :id="`${uid}-option-${i}`"
          :key="tag.name"
          role="option"
          :aria-selected="i === cursor"
          class="flex h-[26px] cursor-pointer items-center justify-between px-3 text-sm"
          :class="i === cursor ? 'bg-accent-tint text-fg' : 'text-fg-2'"
          @mouseenter="cursor = i"
          @mousedown.prevent="commit(tag.name)"
        >
          <span class="truncate">{{ tag.name }}</span>
          <span class="font-mono text-micro text-fg-dim">{{ tag.count }}</span>
        </li>
      </ul>
    </span>
  </div>
</template>

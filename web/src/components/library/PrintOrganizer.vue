<script setup lang="ts">
/*
 * Print organizer — the one editing surface for a print's organization
 * (Library V3 "Shelf", Lightbox aside). Editable title leads (click to edit,
 * Enter commits, Escape reverts; `validatePrintTitle` gates the commit), the
 * favorite heart sits beside it, the raw filename is demoted to a mono detail
 * line, then the tag editor and the "In collections" checklist. Read-only on
 * a host without `gallery.organize`. The page owns every mutation.
 */
import { computed, nextTick, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import {
  displayTitle,
  validatePrintTitle,
} from "@studio/lib/libraryOrganization";
import TagEditor from "./TagEditor.vue";
import CollectionPicker, {
  type CollectionPickerRow,
} from "./CollectionPicker.vue";
import type { GalleryImage, TagCount } from "../../types";

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    canOrganize: boolean;
    collections?: readonly CollectionPickerRow[];
    tagSuggestions?: readonly TagCount[];
  }>(),
  { collections: () => [], tagSuggestions: () => [] },
);

const emit = defineEmits<{
  (e: "rename", title: string | null): void;
  (e: "favorite", favorite: boolean): void;
  (e: "add-tag", tag: string): void;
  (e: "remove-tag", tag: string): void;
  (e: "set-collection", slug: string, member: boolean): void;
  (e: "new-collection"): void;
}>();

const editing = ref(false);
const draft = ref("");
const error = ref("");
const inputEl = ref<HTMLInputElement | null>(null);
const tagEditor = ref<InstanceType<typeof TagEditor> | null>(null);

const title = computed(() => props.item.title?.trim() ?? "");
const placeholder = computed(() => displayTitle(props.item));
const tags = computed(() => props.item.tags ?? []);

watch(
  () => props.item.filename,
  () => {
    editing.value = false;
    error.value = "";
  },
);

async function beginEdit() {
  if (!props.canOrganize) return;
  draft.value = title.value;
  error.value = "";
  editing.value = true;
  await nextTick();
  inputEl.value?.focus();
  inputEl.value?.select();
}

function commit() {
  if (!editing.value) return;
  const result = validatePrintTitle(draft.value);
  if (!result.ok) {
    error.value = result.reason;
    return;
  }
  editing.value = false;
  error.value = "";
  if ((result.value ?? "") !== title.value) emit("rename", result.value);
}

function revert() {
  editing.value = false;
  error.value = "";
  draft.value = title.value;
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Enter") {
    event.preventDefault();
    commit();
  } else if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    revert();
  }
}

function focusTags() {
  tagEditor.value?.focus();
}
defineExpose({ beginEdit, focusTags });
</script>

<template>
  <div class="po">
    <div class="po__title-row">
      <div class="po__title-wrap">
        <template v-if="editing">
          <input
            ref="inputEl"
            v-model="draft"
            class="po__input"
            type="text"
            maxlength="160"
            aria-label="Print title"
            :placeholder="placeholder"
            data-test="title-input"
            @keydown="onKeydown"
            @blur="commit"
          />
        </template>
        <button
          v-else-if="canOrganize"
          type="button"
          class="po__title"
          :class="{ 'po__title--empty': !title }"
          :title="title ? 'Rename' : 'Add a title'"
          aria-label="Edit title"
          data-test="title-edit"
          @click="beginEdit"
        >
          <span class="po__title-text" data-test="title-text">{{
            title || placeholder
          }}</span>
          <svg
            class="po__pen"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M4 20l4.2-1L19 8.2a1.6 1.6 0 000-2.3l-.9-.9a1.6 1.6 0 00-2.3 0L5 15.8z"
            />
            <path d="M14.5 6.3l3.2 3.2" />
          </svg>
        </button>
        <span
          v-else
          class="po__title po__title--static"
          :class="{ 'po__title--empty': !title }"
          data-test="title-text"
          >{{ title || placeholder }}</span
        >
        <p v-if="error" class="po__error" role="alert" data-test="title-error">
          {{ error }}
        </p>
      </div>
      <button
        v-if="canOrganize"
        type="button"
        class="po__fav"
        :data-on="item.favorite ? 'true' : undefined"
        :aria-pressed="!!item.favorite"
        :aria-label="
          item.favorite ? 'Remove from favorites' : 'Add to favorites'
        "
        :title="item.favorite ? 'Favorite (F)' : 'Add to favorites (F)'"
        data-test="favorite-toggle"
        @click="emit('favorite', !item.favorite)"
      >
        <Icon name="heart" :size="16" :stroke-width="1.9" />
      </button>
      <span
        v-else-if="item.favorite"
        class="po__fav po__fav--static"
        data-on="true"
        title="Favorite"
        ><Icon name="heart" :size="16" :stroke-width="1.9"
      /></span>
    </div>
    <p class="po__file" data-test="print-filename">{{ item.filename }}</p>

    <template v-if="canOrganize || tags.length > 0">
      <p class="po__k">Tags</p>
      <TagEditor
        ref="tagEditor"
        :tags="tags"
        :suggestions="tagSuggestions"
        :disabled="!canOrganize"
        @add="(tag) => emit('add-tag', tag)"
        @remove="(tag) => emit('remove-tag', tag)"
      />
    </template>

    <template v-if="canOrganize">
      <p class="po__k">In collections</p>
      <CollectionPicker
        :rows="collections"
        @toggle="(slug, member) => emit('set-collection', slug, member)"
        @new="emit('new-collection')"
      />
    </template>
  </div>
</template>

<style scoped>
.po {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 16px;
}
.po__title-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
.po__title-wrap {
  flex: 1;
  min-width: 0;
}
.po__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  min-height: 28px;
  margin: -2px 0 0 -4px;
  padding: 2px 4px;
  border: 0;
  border-radius: var(--radius-control-sm);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 600;
  line-height: 1.25;
  text-align: left;
  cursor: text;
}
.po__title:not(.po__title--static):hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}
.po__title--static {
  cursor: default;
}
.po__title--empty .po__title-text {
  color: var(--ink-3);
  font-weight: 500;
}
.po__title-text {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.po__pen {
  flex: 0 0 auto;
  color: var(--ink-3);
  opacity: 0;
  transition: opacity var(--dur-quick) var(--ease);
}
.po__title:hover .po__pen,
.po__title:focus-visible .po__pen {
  opacity: 1;
}
.po__title:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.po__input {
  width: 100%;
  min-height: 28px;
  margin: -2px 0 0 -4px;
  padding: 2px 4px;
  border: 0;
  border-bottom: 2px solid var(--safelight);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 600;
  outline: none;
}
.po__error {
  margin: 4px 0 0;
  font-size: 11.5px;
  color: var(--stop);
}
.po__fav {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  border: 1px solid var(--ce);
  border-radius: 50%;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  transition:
    color var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}
.po__fav:hover {
  color: var(--safelight);
  border-color: var(--safelight);
}
.po__fav[data-on="true"] {
  color: var(--safelight);
  border-color: var(--safelight);
}
.po__fav[data-on="true"] :deep(svg) {
  fill: currentColor;
}
.po__fav--static {
  cursor: default;
}
.po__fav:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.po__file {
  margin: 0;
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-3);
  overflow-wrap: anywhere;
}
.po__k {
  margin: 10px 0 0;
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
</style>

<script setup lang="ts">
/*
 * CollectionPicker — the "In collections" checklist (Lightbox aside) and the
 * bulk bar's "Add to collection" popover share this one shape (V3 "Shelf").
 * Pure props/emits: rows are the host-merged collections, `selected` holds
 * the slugs the print(s) are in, `mixed` the slugs only some of a
 * multi-selection are in (rendered indeterminate). "New collection…" turns
 * into an inline name input; Enter creates, Escape / empty cancels.
 */
import { computed, nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import type { MergedCollection } from "@studio/lib/libraryOrganization";

const props = withDefaults(
  defineProps<{
    collections: MergedCollection[];
    /** Slugs every selected print is in. */
    selected: string[];
    /** Slugs only some selected prints are in (indeterminate). */
    mixed?: string[];
    allowCreate?: boolean;
    disabled?: boolean;
    /** Host labels named under the list ("fans out to This Mac · plato"). */
    hostNote?: string | null;
    /** Accessible name for the group. */
    ariaLabel?: string;
    /** Optional per-slug count override (logical prints). */
    counts?: ((slug: string) => number) | null;
  }>(),
  {
    mixed: () => [],
    allowCreate: true,
    disabled: false,
    hostNote: null,
    ariaLabel: "Collections",
    counts: null,
  },
);

const emit = defineEmits<{
  toggle: [slug: string, checked: boolean];
  create: [name: string];
}>();

const selectedSet = computed(() => new Set(props.selected));
const mixedSet = computed(() => new Set(props.mixed));

function stateOf(slug: string): "checked" | "mixed" | "unchecked" {
  if (selectedSet.value.has(slug)) return "checked";
  if (mixedSet.value.has(slug)) return "mixed";
  return "unchecked";
}

function onToggle(slug: string) {
  if (props.disabled) return;
  // A mixed row resolves to "add to all", like Finder/Photos.
  emit("toggle", slug, stateOf(slug) !== "checked");
}

const creating = ref(false);
const draft = ref("");
const inputEl = ref<HTMLInputElement | null>(null);

async function startCreate() {
  if (props.disabled) return;
  creating.value = true;
  draft.value = "";
  await nextTick();
  inputEl.value?.focus();
}

function cancelCreate() {
  creating.value = false;
  draft.value = "";
}

function commitCreate() {
  const name = draft.value.trim();
  if (!name) {
    cancelCreate();
    return;
  }
  emit("create", name);
  creating.value = false;
  draft.value = "";
}

function countOf(collection: MergedCollection): number {
  return props.counts ? props.counts(collection.slug) : collection.count;
}

defineExpose({ startCreate });
</script>

<template>
  <div role="group" :aria-label="ariaLabel" data-test="collection-picker">
    <p
      v-if="collections.length === 0 && !creating"
      class="px-0.5 py-1 text-caption text-ink-3"
      data-test="collection-picker-empty"
    >
      No collections yet.
    </p>
    <button
      v-for="collection in collections"
      :key="collection.slug"
      type="button"
      role="checkbox"
      :aria-checked="
        stateOf(collection.slug) === 'checked'
          ? 'true'
          : stateOf(collection.slug) === 'mixed'
            ? 'mixed'
            : 'false'
      "
      class="flex h-7 w-full items-center gap-2 rounded-control px-0.5 text-left text-body text-ink transition-colors hover:bg-[color-mix(in_srgb,var(--safelight)_10%,transparent)] disabled:cursor-default disabled:opacity-60"
      :disabled="disabled"
      data-test="collection-row"
      :data-slug="collection.slug"
      @click="onToggle(collection.slug)"
    >
      <span
        class="border-ce flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border font-utility text-[11px] leading-none"
        :class="
          stateOf(collection.slug) === 'unchecked'
            ? 'bg-bath text-transparent'
            : 'border-safelight bg-safelight text-on-accent'
        "
        aria-hidden="true"
        data-test="collection-box"
      >
        {{ stateOf(collection.slug) === "mixed" ? "–" : "✓" }}
      </span>
      <span class="min-w-0 flex-1 truncate">{{ collection.name }}</span>
      <span class="shrink-0 font-utility text-[10px] text-ink-3">{{ countOf(collection) }}</span>
    </button>
    <template v-if="allowCreate">
      <button
        v-if="!creating"
        type="button"
        class="flex h-7 w-full items-center gap-2 rounded-control px-0.5 text-left text-body text-ink-2 transition-colors hover:bg-[color-mix(in_srgb,var(--safelight)_10%,transparent)] hover:text-ink disabled:cursor-default disabled:opacity-60"
        :disabled="disabled"
        data-test="collection-new"
        @click="startCreate"
      >
        <Icon name="plus" :size="14" class="shrink-0" />
        <span>New collection…</span>
      </button>
      <div v-else class="flex h-7 items-center gap-2 px-0.5" data-test="collection-new-form">
        <Icon name="plus" :size="14" class="shrink-0 text-ink-3" />
        <input
          ref="inputEl"
          v-model="draft"
          data-selectable
          type="text"
          placeholder="Collection name"
          aria-label="New collection name"
          class="border-edge h-6 min-w-0 flex-1 rounded-control border bg-transparent px-1.5 text-body text-ink outline-none focus:border-safelight"
          data-test="collection-new-input"
          @keydown.enter.prevent="commitCreate"
          @keydown.esc.prevent.stop="cancelCreate"
          @blur="commitCreate"
        />
      </div>
    </template>
    <p
      v-if="hostNote"
      class="mt-1 px-0.5 font-utility text-[9.5px] text-ink-3"
      data-test="collection-host-note"
    >
      {{ hostNote }}
    </p>
  </div>
</template>

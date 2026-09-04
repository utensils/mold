<script setup lang="ts">
/*
 * CollectionPicker — the "In albums" checklist (Lightbox aside) and the
 * bulk bar's "Add to album" popover share this one shape. Pure
 * props/emits: rows are the machine-merged albums, `selected` holds the
 * slugs the picture(s) are in, `mixed` the slugs only some of a
 * multi-selection are in (rendered indeterminate). "New album…" turns
 * into an inline name input; Enter creates, Escape / empty cancels.
 */
import { computed, nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import type { MergedCollection } from "@studio/lib/libraryOrganization";

const props = withDefaults(
  defineProps<{
    collections: MergedCollection[];
    /** Slugs every selected picture is in. */
    selected: string[];
    /** Slugs only some selected pictures are in (indeterminate). */
    mixed?: string[];
    allowCreate?: boolean;
    disabled?: boolean;
    /** Host labels named under the list ("fans out to This Mac · plato"). */
    hostNote?: string | null;
    /** Accessible name for the group. */
    ariaLabel?: string;
    /** Optional per-slug count override (logical pictures). */
    counts?: ((slug: string) => number) | null;
  }>(),
  {
    mixed: () => [],
    allowCreate: true,
    disabled: false,
    hostNote: null,
    ariaLabel: "Albums",
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
      class="px-0.5 py-1 text-micro text-fg-dim"
      data-test="collection-picker-empty"
    >
      No albums yet.
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
      class="flex h-7 w-full items-center gap-2 rounded-control px-0.5 text-left text-sm text-fg transition-colors hover:bg-accent-tint disabled:cursor-default disabled:opacity-60"
      :disabled="disabled"
      data-test="collection-row"
      :data-slug="collection.slug"
      @click="onToggle(collection.slug)"
    >
      <span
        class="border-border-control flex h-4 w-4 shrink-0 items-center justify-center rounded-inner border font-mono text-micro leading-none"
        :class="
          stateOf(collection.slug) === 'unchecked'
            ? 'bg-bg-deep text-transparent'
            : 'border-accent bg-accent text-on-accent'
        "
        aria-hidden="true"
        data-test="collection-box"
      >
        {{ stateOf(collection.slug) === "mixed" ? "–" : "✓" }}
      </span>
      <span class="min-w-0 flex-1 truncate">{{ collection.name }}</span>
      <span class="shrink-0 font-mono text-micro text-fg-dim">{{ countOf(collection) }}</span>
    </button>
    <template v-if="allowCreate">
      <button
        v-if="!creating"
        type="button"
        class="flex h-7 w-full items-center gap-2 rounded-control px-0.5 text-left text-sm text-fg-2 transition-colors hover:bg-accent-tint hover:text-fg disabled:cursor-default disabled:opacity-60"
        :disabled="disabled"
        data-test="collection-new"
        @click="startCreate"
      >
        <Icon name="plus" :size="14" class="shrink-0" />
        <span>New album…</span>
      </button>
      <div v-else class="flex h-7 items-center gap-2 px-0.5" data-test="collection-new-form">
        <Icon name="plus" :size="14" class="shrink-0 text-fg-dim" />
        <input
          ref="inputEl"
          v-model="draft"
          data-selectable
          type="text"
          placeholder="Album name"
          aria-label="New album name"
          class="border-border h-6 min-w-0 flex-1 rounded-control border bg-transparent px-1.5 text-sm text-fg outline-none focus:border-accent"
          data-test="collection-new-input"
          @keydown.enter.prevent="commitCreate"
          @keydown.esc.prevent.stop="cancelCreate"
          @blur="commitCreate"
        />
      </div>
    </template>
    <p
      v-if="hostNote"
      class="mt-1 px-0.5 font-mono text-micro text-fg-dim"
      data-test="collection-host-note"
    >
      {{ hostNote }}
    </p>
  </div>
</template>

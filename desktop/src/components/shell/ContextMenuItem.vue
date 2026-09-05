<script setup lang="ts">
/* One row of a context menu — the same button in the root list and in a
 * submenu, so the two can never drift apart. */
import { computed } from "vue";
import { hasChildren, type MenuEntry } from "../../stores/contextMenu";

type Item = Exclude<MenuEntry, { separator: true }>;

const props = withDefaults(
  defineProps<{
    entry: Item;
    highlighted: boolean;
    /** Reserve the tick column when any sibling in this list is checkable. */
    checkColumn: boolean;
    /** Root items only: a submenu may open from here. */
    expandable?: boolean;
    expanded?: boolean;
  }>(),
  { expandable: false, expanded: false },
);

const emit = defineEmits<{ enter: [MouseEvent]; activate: [MouseEvent] }>();

const submenu = computed(() => props.expandable && hasChildren(props.entry));

const tone = computed(() => [
  props.entry.disabled
    ? "cursor-default text-fg-dim"
    : props.entry.danger
      ? "text-error hover:bg-error/15"
      : "text-fg hover:bg-accent-tint",
  props.highlighted && !props.entry.disabled
    ? props.entry.danger
      ? "bg-error/15"
      : "bg-accent-tint"
    : "",
]);
</script>

<template>
  <button
    type="button"
    :role="entry.checked !== undefined ? 'menuitemcheckbox' : 'menuitem'"
    :aria-checked="entry.checked !== undefined ? entry.checked : undefined"
    :aria-disabled="entry.disabled || undefined"
    :aria-haspopup="submenu ? 'menu' : undefined"
    :aria-expanded="submenu ? expanded : undefined"
    class="flex h-[26px] w-full items-center gap-1.5 px-3 text-left text-xs transition-colors duration-75"
    :class="tone"
    :disabled="entry.disabled"
    @mouseenter="emit('enter', $event)"
    @click="emit('activate', $event)"
  >
    <span
      v-if="checkColumn"
      class="w-3.5 shrink-0 text-center font-mono text-micro leading-none"
      data-test="menu-check"
      aria-hidden="true"
      >{{ entry.checked ? "✓" : "" }}</span
    >
    <span class="min-w-0 flex-1 truncate">{{ entry.label }}</span>
    <span v-if="submenu" class="shrink-0 text-fg-dim" data-test="menu-chevron" aria-hidden="true"
      >›</span
    >
  </button>
</template>

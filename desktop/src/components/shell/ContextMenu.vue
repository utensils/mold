<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import {
  hasCheckColumn,
  hasChildren,
  isSeparator,
  MENU_WIDTH,
  useContextMenuStore,
  type MenuEntry,
} from "../../stores/contextMenu";

const menu = useContextMenuStore();

function onKeydown(e: KeyboardEvent) {
  if (!menu.visible) return;
  if (e.key === "Escape") {
    e.preventDefault();
    if (menu.submenu) menu.leaveSubmenu();
    else menu.close();
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    menu.move(1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    menu.move(-1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    if (!menu.submenu) menu.enterSubmenu();
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    menu.leaveSubmenu();
  } else if (e.key === "Enter") {
    e.preventDefault();
    menu.activateHighlighted();
  }
}

const closeIfOpen = () => {
  if (menu.visible) menu.close();
};

onMounted(() => {
  window.addEventListener("keydown", onKeydown, true);
  window.addEventListener("blur", closeIfOpen);
  window.addEventListener("resize", closeIfOpen);
  window.addEventListener("wheel", closeIfOpen, { passive: true });
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown, true);
  window.removeEventListener("blur", closeIfOpen);
  window.removeEventListener("resize", closeIfOpen);
  window.removeEventListener("wheel", closeIfOpen);
});

/** Hovering a root item: highlight it, open its submenu (anchored to the
 *  item's own box) or close the one belonging to another item. */
function onRootEnter(entry: MenuEntry, index: number, event: MouseEvent) {
  menu.highlighted = index;
  if (hasChildren(entry) && !entry.disabled) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    menu.openSubmenu(index, { left: rect.left, right: rect.right, top: rect.top });
  } else if (menu.submenu) {
    menu.closeSubmenu();
  }
}

function onRootClick(entry: MenuEntry, event: MouseEvent) {
  if (hasChildren(entry)) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    menu.activate(entry, { left: rect.left, right: rect.right, top: rect.top });
    return;
  }
  menu.activate(entry);
}

/** The submenu is named after its parent item. */
const submenuLabel = computed(() => {
  const parent = menu.submenu ? menu.entries[menu.submenu.parentIndex] : undefined;
  return parent && !isSeparator(parent) ? parent.label : "Submenu";
});

function itemClasses(entry: Exclude<MenuEntry, { separator: true }>, highlighted: boolean) {
  return [
    entry.disabled
      ? "cursor-default text-fg-dim"
      : entry.danger
        ? "text-error hover:bg-error/15"
        : "text-fg hover:bg-accent-tint",
    highlighted && !entry.disabled ? (entry.danger ? "bg-error/15" : "bg-accent-tint") : "",
  ];
}
</script>

<template>
  <Teleport to="body">
    <!-- Backdrop swallows the dismissing click so it can't activate what's underneath. -->
    <div
      v-if="menu.visible"
      class="fixed inset-0 z-40"
      @mousedown="menu.close()"
      @contextmenu.prevent="menu.close()"
    />
    <div
      v-if="menu.visible"
      class="fixed z-50 max-h-[calc(100vh-12px)] max-w-[calc(100vw-12px)] overflow-y-auto rounded-control border border-border bg-surface py-1 shadow-md"
      :style="{
        left: `${menu.x}px`,
        top: `${menu.y}px`,
        width: `min(${MENU_WIDTH}px, calc(100vw - 12px))`,
      }"
      role="menu"
      aria-orientation="vertical"
      aria-label="Context menu"
    >
      <template v-for="(entry, i) in menu.entries" :key="i">
        <div v-if="isSeparator(entry)" class="border-border mx-2 my-1 border-t" role="separator" />
        <button
          v-else
          type="button"
          :role="entry.checked !== undefined ? 'menuitemcheckbox' : 'menuitem'"
          :aria-checked="entry.checked !== undefined ? entry.checked : undefined"
          :aria-disabled="entry.disabled || undefined"
          :aria-haspopup="hasChildren(entry) ? 'menu' : undefined"
          :aria-expanded="hasChildren(entry) ? menu.submenu?.parentIndex === i : undefined"
          class="flex h-[26px] w-full items-center gap-1.5 px-3 text-left text-xs transition-colors duration-75"
          :class="itemClasses(entry, i === menu.highlighted)"
          :disabled="entry.disabled"
          @mouseenter="onRootEnter(entry, i, $event)"
          @click="onRootClick(entry, $event)"
        >
          <span
            v-if="hasCheckColumn(menu.entries)"
            class="w-3.5 shrink-0 text-center font-mono text-micro leading-none"
            data-test="menu-check"
            aria-hidden="true"
            >{{ entry.checked ? "✓" : "" }}</span
          >
          <span class="min-w-0 flex-1 truncate">{{ entry.label }}</span>
          <span
            v-if="hasChildren(entry)"
            class="shrink-0 text-fg-dim"
            data-test="menu-chevron"
            aria-hidden="true"
            >›</span
          >
        </button>
      </template>
    </div>
    <div
      v-if="menu.visible && menu.submenu"
      class="fixed z-50 max-h-[calc(100vh-12px)] max-w-[calc(100vw-12px)] overflow-y-auto rounded-control border border-border bg-surface py-1 shadow-md"
      :style="{
        left: `${menu.submenu.x}px`,
        top: `${menu.submenu.y}px`,
        width: `min(${MENU_WIDTH}px, calc(100vw - 12px))`,
      }"
      role="menu"
      aria-orientation="vertical"
      data-test="submenu"
      :aria-label="submenuLabel"
    >
      <template v-for="(entry, j) in menu.submenu.entries" :key="j">
        <div v-if="isSeparator(entry)" class="border-border mx-2 my-1 border-t" role="separator" />
        <button
          v-else
          type="button"
          :role="entry.checked !== undefined ? 'menuitemcheckbox' : 'menuitem'"
          :aria-checked="entry.checked !== undefined ? entry.checked : undefined"
          :aria-disabled="entry.disabled || undefined"
          class="flex h-[26px] w-full items-center gap-1.5 px-3 text-left text-xs transition-colors duration-75"
          :class="itemClasses(entry, j === menu.submenu.highlighted)"
          :disabled="entry.disabled"
          @mouseenter="menu.submenu.highlighted = j"
          @click="menu.activate(entry)"
        >
          <span
            v-if="hasCheckColumn(menu.submenu.entries)"
            class="w-3.5 shrink-0 text-center font-mono text-micro leading-none"
            data-test="menu-check"
            aria-hidden="true"
            >{{ entry.checked ? "✓" : "" }}</span
          >
          <span class="min-w-0 flex-1 truncate">{{ entry.label }}</span>
        </button>
      </template>
    </div>
  </Teleport>
</template>

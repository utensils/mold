<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import ContextMenuItem from "./ContextMenuItem.vue";
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
        <ContextMenuItem
          v-else
          :entry="entry"
          :highlighted="i === menu.highlighted"
          :check-column="hasCheckColumn(menu.entries)"
          expandable
          :expanded="menu.submenu?.parentIndex === i"
          @enter="onRootEnter(entry, i, $event)"
          @activate="onRootClick(entry, $event)"
        />
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
        <ContextMenuItem
          v-else
          :entry="entry"
          :highlighted="j === menu.submenu.highlighted"
          :check-column="hasCheckColumn(menu.submenu.entries)"
          @enter="menu.submenu.highlighted = j"
          @activate="menu.activate(entry)"
        />
      </template>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from "vue";
import { isSeparator, MENU_WIDTH, useContextMenuStore } from "../../stores/contextMenu";

const menu = useContextMenuStore();

function onKeydown(e: KeyboardEvent) {
  if (!menu.visible) return;
  if (e.key === "Escape") {
    e.preventDefault();
    menu.close();
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    menu.move(1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    menu.move(-1);
  } else if (e.key === "Enter" && menu.highlighted >= 0) {
    e.preventDefault();
    menu.activate(menu.entries[menu.highlighted]!);
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
      class="border-edge fixed z-50 max-h-[calc(100vh-12px)] max-w-[calc(100vw-12px)] overflow-y-auto rounded-chrome border bg-bench py-1 shadow-[0_4px_8px_rgba(0,0,0,0.5)]"
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
        <div v-if="isSeparator(entry)" class="border-edge mx-2 my-1 border-t" role="separator" />
        <button
          v-else
          type="button"
          role="menuitem"
          :aria-disabled="entry.disabled || undefined"
          class="flex h-[26px] w-full items-center px-3 text-left text-body transition-colors duration-75"
          :class="[
            entry.disabled
              ? 'cursor-default text-ink-3'
              : entry.danger
                ? 'text-stop hover:bg-[color-mix(in_srgb,var(--stop)_16%,transparent)]'
                : 'text-ink hover:bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]',
            i === menu.highlighted && !entry.disabled
              ? entry.danger
                ? 'bg-[color-mix(in_srgb,var(--stop)_16%,transparent)]'
                : 'bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]'
              : '',
          ]"
          :disabled="entry.disabled"
          @mouseenter="menu.highlighted = i"
          @click="menu.activate(entry)"
        >
          <span class="truncate">{{ entry.label }}</span>
        </button>
      </template>
    </div>
  </Teleport>
</template>

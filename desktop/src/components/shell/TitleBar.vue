<script setup lang="ts">
// Unified toolbar: macOS traffic lights occupy the leading edge, then a
// sidebar-collapse toggle, a centered workspace title, and the ⌘K search chip.
// The whole strip is a drag region except the interactive controls.
import { computed } from "vue";
import { useRoute } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import NotificationsCenter from "@studio/components/NotificationsCenter.vue";
import { useUiStore } from "../../stores/ui";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { PLATFORM_UI, shortcutLabel } from "../../lib/platform";

const route = useRoute();
const ui = useUiStore();
const appPrefs = useAppPrefsStore();

const workspace = computed(() => (route.meta.title as string | undefined) ?? "");

function toggleSidebar() {
  void appPrefs.update({ sidebarCollapsed: !appPrefs.sidebarCollapsed });
}
</script>

<template>
  <header
    data-tauri-drag-region
    class="flex items-center gap-3.5 border-b border-edge bg-bench pr-3"
    :class="PLATFORM_UI.isMacOS ? 'pl-[84px]' : 'pl-3'"
  >
    <button
      type="button"
      class="flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded-chrome text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink-2"
      title="Toggle sidebar"
      aria-label="Toggle sidebar"
      :aria-keyshortcuts="`${PLATFORM_UI.modifier}+\\`"
      @click="toggleSidebar"
    >
      <Icon name="sidebar" :size="16" />
    </button>

    <div
      data-tauri-drag-region
      class="flex-1 text-center font-utility text-[11px] tracking-[0.04em] text-ink-3 select-none"
    >
      Mold Studio<template v-if="workspace"> — {{ workspace }}</template>
    </div>

    <button
      type="button"
      class="flex items-center gap-1.5 rounded-chrome border border-edge px-2.5 py-1 font-utility text-[10px] text-ink-3 transition-colors duration-100 hover:text-ink-2"
      title="Command palette"
      aria-label="Open command palette"
      :aria-keyshortcuts="`${PLATFORM_UI.modifier}+K`"
      @click="ui.togglePalette()"
    >
      <span>Search</span>
      <span class="text-[11px]">{{ shortcutLabel("K") }}</span>
    </button>

    <NotificationsCenter />
  </header>
</template>

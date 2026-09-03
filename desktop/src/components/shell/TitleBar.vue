<script setup lang="ts">
// Unified toolbar (README §04): macOS traffic lights on the leading edge,
// then the sidebar toggle and back/forward, the view's mono title with a
// sans subtitle, the ⌘K search chip, and the notifications bell. The whole
// strip is a drag region except the interactive controls.
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import NotificationsCenter from "@studio/components/NotificationsCenter.vue";
import { useUiStore } from "../../stores/ui";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useShellSubtitle } from "../../composables/useShellSubtitle";
import { PLATFORM_UI, shortcutLabel } from "../../lib/platform";

const route = useRoute();
const router = useRouter();
const ui = useUiStore();
const appPrefs = useAppPrefsStore();
const subtitle = useShellSubtitle();

const title = computed(() => (route.meta.title as string | undefined) ?? "");

// Vue Router records back/forward availability on the history state it
// manages; the buttons mirror it so neither is ever a dead end.
const canGoBack = computed(() => {
  void route.fullPath;
  return typeof router.options.history.state.back === "string";
});
const canGoForward = computed(() => {
  void route.fullPath;
  return typeof router.options.history.state.forward === "string";
});

function toggleSidebar() {
  void appPrefs.update({ sidebarCollapsed: !appPrefs.sidebarCollapsed });
}
</script>

<template>
  <header
    data-tauri-drag-region
    class="flex h-[var(--mold-shell-toolbar-h)] shrink-0 items-center gap-3.5 border-b border-border bg-chrome pr-3"
    :class="PLATFORM_UI.isMacOS ? 'pl-[84px]' : 'pl-3'"
  >
    <div class="flex items-center gap-0.5">
      <button
        type="button"
        class="toolbar-icon"
        title="Toggle sidebar"
        aria-label="Toggle sidebar"
        :aria-keyshortcuts="`${PLATFORM_UI.modifier}+\\`"
        @click="toggleSidebar"
      >
        <Icon name="sidebar" :size="17" />
      </button>
      <button
        type="button"
        class="toolbar-icon"
        title="Back"
        aria-label="Back"
        :disabled="!canGoBack"
        @click="router.back()"
      >
        <Icon name="chevron-left" :size="17" />
      </button>
      <button
        type="button"
        class="toolbar-icon"
        title="Forward"
        aria-label="Forward"
        :disabled="!canGoForward"
        @click="router.forward()"
      >
        <Icon name="chevron-right" :size="17" />
      </button>
    </div>

    <div data-tauri-drag-region class="flex min-w-0 items-baseline gap-2.5 select-none">
      <span data-test="shell-title" class="font-mono text-base font-bold text-fg">{{ title }}</span>
      <span v-if="subtitle" data-test="shell-subtitle" class="truncate text-xs text-fg-dim">
        {{ subtitle }}
      </span>
    </div>

    <div data-tauri-drag-region class="flex-1" />

    <button
      type="button"
      class="flex h-7 min-w-[220px] items-center gap-2 rounded-control border border-border bg-bg px-2.5 text-fg-dim transition-colors duration-100 hover:border-border-focus"
      title="Command palette"
      aria-label="Open command palette"
      :aria-keyshortcuts="`${PLATFORM_UI.modifier}+K`"
      @click="ui.togglePalette()"
    >
      <Icon name="search" :size="14" />
      <span class="flex-1 text-left text-xs">Search or run a command…</span>
      <span class="font-mono text-micro font-bold text-accent">{{ shortcutLabel("K") }}</span>
    </button>

    <NotificationsCenter />
  </header>
</template>

<style scoped>
.toolbar-icon {
  display: inline-flex;
  width: 28px;
  height: 28px;
  align-items: center;
  justify-content: center;
  border-radius: var(--mold-radius-2);
  color: var(--mold-text-2);
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}
.toolbar-icon:hover:not(:disabled) {
  background: var(--mold-surface);
  color: var(--mold-text);
}
.toolbar-icon:disabled {
  color: var(--mold-text-faint);
}
</style>

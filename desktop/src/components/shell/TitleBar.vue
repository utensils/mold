<script setup lang="ts">
// Overlay titlebar: macOS traffic lights are inset over this strip
// (trafficLightPosition in tauri.conf.json), so it is all drag region.
import { useUiStore } from "../../stores/ui";
import { PLATFORM_UI, shortcutLabel } from "../../lib/platform";

const ui = useUiStore();
</script>

<template>
  <header
    data-tauri-drag-region
    class="border-edge flex items-center border-b bg-bench pr-3"
    :class="PLATFORM_UI.isMacOS ? 'pl-[84px]' : 'pl-3'"
  >
    <span
      data-tauri-drag-region
      class="brand-wordmark font-display text-display-sm font-bold tracking-[0.08em] text-ink-2 select-none"
      style="font-stretch: 90%"
    >
      MOLD
    </span>
    <div data-tauri-drag-region class="flex-1" />
    <button
      type="button"
      class="border-edge text-ink-3 hover:text-ink-2 flex h-7 items-center gap-2 rounded-control border px-3 text-caption transition-colors duration-100"
      title="Command palette"
      aria-label="Open command palette"
      :aria-keyshortcuts="`${PLATFORM_UI.modifier}+K`"
      @click="ui.togglePalette()"
    >
      <span>Search or run a command…</span>
      <kbd class="data-mono text-ink-3">{{ shortcutLabel("K") }}</kbd>
    </button>
  </header>
</template>

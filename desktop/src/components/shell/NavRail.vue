<script setup lang="ts">
import { RouterLink } from "vue-router";

const destinations = [
  { route: "/generate", label: "Generate", key: "⌘1" },
  { route: "/gallery", label: "Gallery", key: "⌘2" },
  { route: "/chains", label: "Chains", key: "⌘3" },
  { route: "/models", label: "Models", key: "⌘4" },
  { route: "/history", label: "History", key: "⌘5" },
];
</script>

<template>
  <nav class="border-edge flex w-[208px] flex-col border-r bg-bench pt-2 pb-2">
    <RouterLink
      v-for="d in destinations"
      :key="d.route"
      :to="d.route"
      class="group mx-2 flex h-7 items-center justify-between rounded-control px-2.5 text-body text-ink-2 transition-colors duration-100 hover:text-ink"
      active-class="!text-ink bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]"
    >
      <span class="font-medium">{{ d.label }}</span>
      <kbd
        class="data-mono text-ink-3 opacity-0 transition-opacity duration-100 group-hover:opacity-100"
      >
        {{ d.key }}
      </kbd>
    </RouterLink>

    <div class="mx-4 mt-4 mb-1 flex items-center gap-2">
      <span class="edge-code">JOBS</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>
    <!-- Live Develop chips for in-flight jobs land in M2. -->
    <p class="mx-4 text-caption text-ink-3">Nothing developing</p>

    <div class="flex-1" />

    <RouterLink
      to="/settings"
      class="mx-2 flex h-7 items-center justify-between rounded-control px-2.5 text-body text-ink-2 transition-colors duration-100 hover:text-ink"
      active-class="!text-ink bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]"
    >
      <span class="font-medium">Settings</span>
      <kbd class="data-mono text-ink-3">⌘,</kbd>
    </RouterLink>
  </nav>
</template>

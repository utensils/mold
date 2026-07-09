<script setup lang="ts">
import { RouterLink } from "vue-router";
import DevelopCanvas from "../../lib/develop/DevelopCanvas.vue";
import { useGenerationStore, jobPhase, jobProgress } from "../../stores/generation";

const generation = useGenerationStore();

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
    <RouterLink
      v-if="generation.active"
      to="/generate"
      class="mx-2 flex items-center gap-2 rounded-control px-2 py-1.5 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
    >
      <div
        class="h-8 w-8 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
      >
        <DevelopCanvas
          :seed="generation.active.visualSeed"
          :progress="jobProgress(generation.active)"
          :phase="jobPhase(generation.active)"
        />
      </div>
      <div class="min-w-0">
        <div class="truncate text-caption text-ink-2">{{ generation.active.model }}</div>
        <div class="edge-code">
          <template v-if="generation.active.status === 'denoising'">
            {{ generation.active.step }}/{{ generation.active.total }}
          </template>
          <template v-else-if="generation.active.status === 'complete'">FIXED</template>
          <template v-else-if="generation.active.status === 'error'">STOPPED</template>
          <template v-else>LATENT</template>
        </div>
      </div>
    </RouterLink>
    <p v-else class="mx-4 text-caption text-ink-3">Nothing developing</p>

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

<script setup lang="ts">
import { computed } from "vue";
import { useUpdaterStore } from "../../stores/updater";

const updater = useUpdaterStore();
const candidate = computed(() => updater.candidate);
</script>

<template>
  <aside
    v-if="updater.shouldNotify && candidate"
    data-test="update-banner"
    role="status"
    aria-live="polite"
    class="border-safelight/35 flex min-h-10 items-center gap-4 border-b bg-[color-mix(in_srgb,var(--safelight)_10%,var(--bench))] px-4"
  >
    <p class="min-w-0 flex-1 text-body text-ink">
      <span class="font-semibold">Mold {{ candidate.version }} is available.</span>
      <span class="ml-2 text-caption text-ink-2">Verified before installation.</span>
    </p>
    <button
      type="button"
      data-test="banner-install-update"
      class="h-7 shrink-0 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] hover:brightness-105"
      @click="updater.install()"
    >
      Update and restart
    </button>
    <button
      type="button"
      class="h-7 shrink-0 px-1 text-caption text-ink-2 hover:text-ink"
      aria-label="Dismiss update notification"
      @click="updater.dismissCandidate()"
    >
      Dismiss
    </button>
  </aside>
</template>

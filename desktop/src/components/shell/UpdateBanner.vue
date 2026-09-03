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
    class="border-accent/35 flex min-h-10 items-center gap-4 border-b bg-accent-tint px-4"
  >
    <p class="min-w-0 flex-1 text-sm text-fg">
      <span class="font-semibold">Mold {{ candidate.version }} is available.</span>
      <span class="ml-2 text-micro text-fg-2">Verified before installation.</span>
    </p>
    <button
      type="button"
      data-test="banner-install-update"
      class="ms-toolbar-button ms-toolbar-button--on font-semibold"
      @click="updater.install()"
    >
      Update and restart
    </button>
    <button
      type="button"
      class="shrink-0 text-micro text-fg-2 hover:text-fg"
      aria-label="Dismiss update notification"
      @click="updater.dismissCandidate()"
    >
      Dismiss
    </button>
  </aside>
</template>

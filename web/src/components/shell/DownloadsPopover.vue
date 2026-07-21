<script setup lang="ts">
/*
 * Downloads popover (spec §06, prototype web downloads panel). Wide viewports
 * (≥640px) get an anchored panel top-right under the nav; below 640px the same
 * body renders in a bottom SheetPanel. Opened by the shared
 * `mold:open-downloads` window event (App owns the state); the AppNav button
 * and ⌘K palette both dispatch it.
 */
import { onBeforeUnmount, onMounted, ref } from "vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import DownloadsBody from "./DownloadsBody.vue";
import type { DownloadJobWire } from "../../types";

defineProps<{
  open: boolean;
  active: DownloadJobWire[];
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  etaByJob: Record<string, number | null>;
  rateByJob: Record<string, number | null>;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

// Bottom-sheet only on phones; anchored panel everywhere else. Default to the
// anchored panel when matchMedia is unavailable (jsdom / SSR).
const isNarrow = ref(false);
let mql: MediaQueryList | null = null;
function sync() {
  isNarrow.value = mql?.matches ?? false;
}

onMounted(() => {
  if (typeof window.matchMedia === "function") {
    mql = window.matchMedia("(max-width: 639px)");
    sync();
    mql.addEventListener?.("change", sync);
  }
});
onBeforeUnmount(() => {
  mql?.removeEventListener?.("change", sync);
});
</script>

<template>
  <SheetPanel
    v-if="open && isNarrow"
    :open="open"
    variant="bottom"
    title="Downloads"
    @close="emit('close')"
  >
    <DownloadsBody
      :active="active"
      :queued="queued"
      :history="history"
      :eta-by-job="etaByJob"
      :rate-by-job="rateByJob"
      @cancel="emit('cancel', $event)"
      @retry="emit('retry', $event)"
    />
  </SheetPanel>

  <aside
    v-else-if="open"
    class="dl-pop"
    role="dialog"
    aria-label="Downloads"
    data-test="downloads-popover"
    @keydown.escape="emit('close')"
  >
    <header class="dl-pop__head">
      <span class="dl-pop__title">Downloads</span>
      <button
        type="button"
        class="dl-pop__close"
        aria-label="Close downloads"
        @click="emit('close')"
      >
        ✕
      </button>
    </header>
    <DownloadsBody
      :active="active"
      :queued="queued"
      :history="history"
      :eta-by-job="etaByJob"
      :rate-by-job="rateByJob"
      @cancel="emit('cancel', $event)"
      @retry="emit('retry', $event)"
    />
  </aside>
</template>

<style scoped>
@keyframes dl-fade-up {
  from {
    opacity: 0;
    transform: translateY(-6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

.dl-pop {
  position: absolute;
  top: 64px;
  right: 16px;
  z-index: 40;
  width: 340px;
  max-width: calc(100vw - 32px);
  max-height: min(440px, calc(100svh - 88px));
  overflow-y: auto;
  padding: 15px;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  box-shadow: 0 24px 60px -12px rgba(0, 0, 0, 0.6);
  animation: dl-fade-up var(--dur-base) var(--ease);
}

.dl-pop__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.dl-pop__title {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.dl-pop__close {
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  border-radius: 50%;
  border: 0;
  background: color-mix(in srgb, var(--rebate) 9%, transparent);
  color: var(--ink-2);
  font-size: 13px;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.dl-pop__close:hover {
  background: color-mix(in srgb, var(--rebate) 14%, transparent);
  color: var(--rebate);
}

.dl-pop__close:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>

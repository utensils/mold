<script setup lang="ts">
/*
 * Downloads popover (spec §06, prototype web downloads panel). Wide viewports
 * (≥640px) get an anchored panel top-right under the nav; below 640px the same
 * body renders in a bottom SheetPanel. Opened by the shared
 * `mold:open-downloads` window event (App owns the state); the AppNav button
 * and ⌘K palette both dispatch it.
 */
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import Icon from "@ui/components/Icon.vue";
import DownloadsBody from "./DownloadsBody.vue";
import type { DownloadJobWire, ModelInfoExtended } from "../../types";

const props = defineProps<{
  open: boolean;
  models?: ModelInfoExtended[];
  loaded?: boolean;
  loading?: boolean;
  error?: string | null;
  actionError?: string | null;
  actionBusy?: boolean;
  active: DownloadJobWire[];
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  etaByJob: Record<string, number | null>;
  rateByJob: Record<string, number | null>;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "refresh"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

// Dismissal. The Escape handler must live on the document, not on the panel:
// opening the popover doesn't move focus into it, so a keydown bound to the
// element itself never fires. Outside-click uses mousedown so a drag that ends
// outside doesn't count as "clicking away".
const panelEl = ref<HTMLElement | null>(null);

function onKeydown(e: KeyboardEvent) {
  if (isNarrow.value || e.defaultPrevented) return;
  const target = e.target instanceof Element ? e.target : null;
  if (target?.closest('[role="dialog"]') && !panelEl.value?.contains(target))
    return;
  if (e.key === "Escape") {
    e.preventDefault();
    emit("close");
  }
}
function onPointerDownOutside(e: MouseEvent) {
  if (isNarrow.value) return;
  const target = e.target as Node | null;
  if (target && panelEl.value?.contains(target)) return;
  emit("close");
}

function bindDismiss() {
  document.addEventListener("keydown", onKeydown);
  document.addEventListener("mousedown", onPointerDownOutside);
}
function unbindDismiss() {
  document.removeEventListener("keydown", onKeydown);
  document.removeEventListener("mousedown", onPointerDownOutside);
}

// Bottom-sheet only on phones; anchored panel everywhere else. Default to the
// anchored panel when matchMedia is unavailable (jsdom / SSR).
const isNarrow = ref(false);
let mql: MediaQueryList | null = null;
async function sync() {
  const wasNarrow = isNarrow.value;
  isNarrow.value = mql?.matches ?? false;
  if (props.open && wasNarrow && !isNarrow.value) {
    await nextTick();
    if (props.open && !isNarrow.value) panelEl.value?.focus();
  }
}

let returnFocus: HTMLElement | null = null;
watch(
  () => props.open,
  async (open) => {
    if (open) {
      await nextTick();
      if (!props.open) return;
      returnFocus =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      if (props.open && !isNarrow.value) panelEl.value?.focus();
    } else if (panelEl.value?.contains(document.activeElement)) {
      if (returnFocus?.isConnected) returnFocus.focus();
    }
  },
  { immediate: true },
);
watch(
  () => props.open,
  (open) => (open ? bindDismiss() : unbindDismiss()),
  { immediate: true },
);

onMounted(() => {
  if (typeof window.matchMedia === "function") {
    mql = window.matchMedia("(max-width: 639px)");
    sync();
    mql.addEventListener?.("change", sync);
  }
});
onBeforeUnmount(() => {
  unbindDismiss();
  mql?.removeEventListener?.("change", sync);
});
</script>

<template>
  <SheetPanel
    v-if="open && isNarrow"
    class="dl-sheet"
    :open="open"
    variant="bottom"
    title="Downloads"
    @close="emit('close')"
  >
    <button
      type="button"
      class="dl-sheet-close dl-retry"
      @click="emit('close')"
    >
      Close downloads
    </button>
    <DownloadsBody
      :action-error="actionError"
      :action-busy="actionBusy"
      :models="models"
      :loaded="loaded"
      :loading="loading"
      :error="error"
      @refresh="emit('refresh')"
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
    ref="panelEl"
    tabindex="-1"
  >
    <header class="dl-pop__head">
      <span class="dl-pop__title">Downloads</span>
      <button
        type="button"
        class="dl-pop__close"
        aria-label="Close downloads"
        @click="emit('close')"
      >
        <Icon name="close" :size="14" />
      </button>
    </header>
    <DownloadsBody
      :action-error="actionError"
      :action-busy="actionBusy"
      :models="models"
      :loaded="loaded"
      :loading="loading"
      :error="error"
      @refresh="emit('refresh')"
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
  font-size: 0.8125rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.dl-pop__close {
  width: 44px;
  height: 44px;
  flex: 0 0 44px;
  border-radius: 50%;
  border: 0;
  background: color-mix(in srgb, var(--rebate) 9%, transparent);
  color: var(--ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
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
.dl-sheet :deep(.ms-sheet__panel-bottom) {
  max-height: calc(100svh - 1rem);
  overflow-y: auto;
  min-height: 0;
}
.dl-sheet :deep(.ms-sheet__bottom-title) {
  font-size: 1rem;
}
.dl-sheet-close {
  min-height: 44px;
  padding: 0.5rem;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  color: var(--rebate);
  background: var(--bench);
  border: 1px solid var(--edge);
}
</style>

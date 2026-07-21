<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import HostSelector from "../generate/HostSelector.vue";
import type { GenerateForm } from "../../lib/generateForm";
import { aspectRatioLabel } from "../../lib/resolutions";
import { useHostsStore } from "../../stores/hosts";

/**
 * Create header (Mold Studio): the print title, a live summary pill, and the
 * host chip. When more than one host is connected the chip opens a popover
 * carrying the existing HostSelector — routing semantics are unchanged.
 */
const props = defineProps<{ form: GenerateForm }>();

const hosts = useHostsStore();

const summary = computed(() => {
  const { width, height, steps, family } = props.form;
  return `${aspectRatioLabel(width, height, family)} · ${width}×${height} · ${steps} steps`;
});

const primary = computed(() => hosts.primaryHost);
const hostLabel = computed(() => primary.value?.label ?? "This device");
const hostReady = computed(() => primary.value?.status === "ready");
const hostStatus = computed(() => {
  const status = primary.value?.status;
  if (status === "ready") return "ready";
  if (status === "error") return "offline";
  return status ?? "connecting";
});

const popoverEl = ref<HTMLDivElement | null>(null);
const popoverOpen = ref(false);
function toggle() {
  if (hosts.multiHost) popoverOpen.value = !popoverOpen.value;
}
function onPointerDown(event: PointerEvent) {
  if (!popoverOpen.value || !popoverEl.value) return;
  if (!event.composedPath().includes(popoverEl.value)) popoverOpen.value = false;
}
onMounted(() => document.addEventListener("pointerdown", onPointerDown));
onBeforeUnmount(() => document.removeEventListener("pointerdown", onPointerDown));
</script>

<template>
  <header data-test="create-header" class="ms-header">
    <span class="ms-header__title">Untitled print</span>
    <span class="ms-header__summary data-mono">{{ summary }}</span>
    <div class="ms-header__spacer" />
    <div ref="popoverEl" class="ms-header__host">
      <button
        type="button"
        class="ms-header__chip"
        :class="{ 'ms-header__chip--button': hosts.multiHost }"
        :aria-expanded="hosts.multiHost ? popoverOpen : undefined"
        :tabindex="hosts.multiHost ? 0 : -1"
        @click="toggle"
      >
        <span
          class="ms-header__dot"
          :class="hostReady ? 'ms-header__dot--ready' : 'ms-header__dot--wait'"
        />
        {{ hostLabel }} · {{ hostStatus }}
      </button>
      <div v-if="popoverOpen" class="ms-header__popover">
        <HostSelector />
      </div>
    </div>
  </header>
</template>

<style scoped>
.ms-header {
  height: 52px;
  flex: 0 0 52px;
  border-bottom: 1px solid var(--edge);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 22px;
}
.ms-header__title {
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 600;
}
.ms-header__summary {
  font-size: 10px;
  color: var(--ink-3);
  padding: 3px 8px;
  border: 1px solid var(--edge);
  border-radius: 20px;
}
.ms-header__spacer {
  flex: 1;
}
.ms-header__host {
  position: relative;
}
.ms-header__chip {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 0;
  padding: 4px 0;
}
.ms-header__chip--button {
  cursor: pointer;
  border: 1px solid var(--edge);
  border-radius: 20px;
  padding: 4px 10px;
}
.ms-header__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
.ms-header__dot--ready {
  background: #4ade80;
}
.ms-header__dot--wait {
  background: var(--ink-3);
}
.ms-header__popover {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  z-index: 20;
  width: 260px;
  padding: 14px 16px;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: 12px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
}
</style>

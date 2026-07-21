<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import type { GenerateForm } from "../../lib/generateForm";
import { aspectRatioLabel } from "../../lib/resolutions";
import { normalizeTargetHost } from "../../lib/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useHostsStore, type HostView } from "../../stores/hosts";

/**
 * Create header (Mold Studio): the print title, a live summary pill, and the
 * host chip. With more than one live host the chip toggles a routing menu —
 * Auto (least busy), Most capable ("capable" sentinel), or a sticky host —
 * writing the same persisted `generateTargetHost` contract as always.
 */
const props = defineProps<{ form: GenerateForm }>();

const hosts = useHostsStore();
const prefs = useAppPrefsStore();

const summary = computed(() => {
  const { width, height, steps, family } = props.form;
  return `${aspectRatioLabel(width, height, family)} · ${width}×${height} · ${steps} steps`;
});

/** Current routing pick; a persisted host that's gone reads as Auto. */
const target = computed(
  () => normalizeTargetHost(prefs.settings?.generateTargetHost ?? null, hosts.all) ?? "auto",
);

const targetHost = computed(() =>
  target.value === "auto" || target.value === "capable"
    ? null
    : (hosts.all.find((h) => h.id === target.value) ?? null),
);

/** The chip names the pick, not just the primary: sticky host → its label. */
const chipHost = computed(() => targetHost.value ?? hosts.primaryHost ?? null);
const chipLabel = computed(() => {
  if (target.value === "auto" && hosts.multiHost) return "Auto";
  if (target.value === "capable") return "Most capable";
  return chipHost.value?.label ?? "This device";
});
const chipReady = computed(() =>
  target.value === "auto" || target.value === "capable"
    ? hosts.all.some((h) => h.status === "ready")
    : chipHost.value?.status === "ready",
);
const chipStatus = computed(() => {
  if (target.value === "auto" || target.value === "capable") {
    return chipReady.value ? "ready" : "connecting";
  }
  const status = chipHost.value?.status;
  if (status === "ready") return "ready";
  if (status === "error") return "offline";
  return status ?? "connecting";
});

function pick(id: string) {
  void prefs.update({ generateTargetHost: id === "auto" ? null : id });
  popoverOpen.value = false;
}

function hostLine(host: HostView): string {
  if (host.status !== "ready") return "offline";
  return host.queueDepth !== null ? `queue ${host.queueDepth}` : "ready";
}

const popoverEl = ref<HTMLDivElement | null>(null);
const popoverOpen = ref(false);
function toggle() {
  if (hosts.multiHost) popoverOpen.value = !popoverOpen.value;
}
function onPointerDown(event: PointerEvent) {
  if (!popoverOpen.value || !popoverEl.value) return;
  if (!event.composedPath().includes(popoverEl.value)) popoverOpen.value = false;
}
function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") popoverOpen.value = false;
}
onMounted(() => {
  document.addEventListener("pointerdown", onPointerDown);
  document.addEventListener("keydown", onKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onPointerDown);
  document.removeEventListener("keydown", onKeydown);
});
</script>

<template>
  <header data-test="create-header" class="ms-header">
    <span class="ms-header__title">Untitled print</span>
    <span class="ms-header__summary data-mono">{{ summary }}</span>
    <div class="ms-header__spacer" />
    <div ref="popoverEl" class="ms-header__host">
      <button
        type="button"
        data-test="host-chip"
        class="ms-header__chip"
        :class="{ 'ms-header__chip--button': hosts.multiHost }"
        :aria-expanded="hosts.multiHost ? popoverOpen : undefined"
        :aria-haspopup="hosts.multiHost ? 'menu' : undefined"
        :tabindex="hosts.multiHost ? 0 : -1"
        @click="toggle"
      >
        <span
          class="ms-header__dot"
          :class="chipReady ? 'ms-header__dot--ready' : 'ms-header__dot--wait'"
        />
        {{ chipLabel }} · {{ chipStatus }}
        <Icon v-if="hosts.multiHost" name="chevron-down" :size="12" class="ms-header__chev" />
      </button>
      <div
        v-if="popoverOpen"
        data-test="host-menu"
        role="menu"
        aria-label="Generation host"
        class="ms-header__popover ms-fade-up"
      >
        <div class="ms-header__kicker data-mono">run on</div>
        <button
          type="button"
          role="menuitemradio"
          data-test="host-option-auto"
          class="ms-header__row"
          :aria-checked="target === 'auto'"
          @click="pick('auto')"
        >
          <span class="ms-header__row-label">Auto</span>
          <span class="ms-header__row-sub data-mono">least busy</span>
          <span v-if="target === 'auto'" class="ms-header__check">✓</span>
        </button>
        <button
          type="button"
          role="menuitemradio"
          data-test="host-option-capable"
          class="ms-header__row"
          :aria-checked="target === 'capable'"
          @click="pick('capable')"
        >
          <span class="ms-header__row-label">Most capable</span>
          <span class="ms-header__row-sub data-mono">strongest gpu</span>
          <span v-if="target === 'capable'" class="ms-header__check">✓</span>
        </button>
        <div class="ms-header__rule" />
        <button
          v-for="h in hosts.all"
          :key="h.id"
          type="button"
          role="menuitemradio"
          :data-test="`host-option-${h.id}`"
          class="ms-header__row"
          :disabled="h.status !== 'ready'"
          :aria-checked="target === h.id"
          @click="pick(h.id)"
        >
          <span
            class="ms-header__dot"
            :class="h.status === 'ready' ? 'ms-header__dot--ready' : 'ms-header__dot--wait'"
          />
          <span class="ms-header__row-label">{{ h.label }}</span>
          <span class="ms-header__row-sub data-mono">{{ hostLine(h) }}</span>
          <span v-if="target === h.id" class="ms-header__check">✓</span>
        </button>
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
  transition:
    color var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}
.ms-header__chip--button:hover {
  color: var(--rebate);
  border-color: var(--ce);
}
.ms-header__chev {
  color: var(--ink-3);
}
.ms-header__dot {
  width: 6px;
  height: 6px;
  flex: 0 0 6px;
  border-radius: 50%;
}
.ms-header__dot--ready {
  background: var(--success);
}
.ms-header__dot--wait {
  background: var(--ink-3);
}
/* Above every composer affordance (Templates sits at z-30 in the workbench). */
.ms-header__popover {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  z-index: 50;
  width: 264px;
  padding: 8px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-card);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}
.ms-header__kicker {
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
  padding: 4px 8px 6px;
}
.ms-header__row {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  color: var(--rebate);
  padding: 8px;
  border-radius: var(--radius-control);
  font-size: 12.5px;
  text-align: left;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}
.ms-header__row:hover:not(:disabled) {
  background: var(--surface);
}
.ms-header__row:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.ms-header__row-label {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-header__row-sub {
  font-size: 9.5px;
  color: var(--ink-3);
}
.ms-header__check {
  color: var(--safelight);
  font-size: 12px;
}
.ms-header__rule {
  height: 1px;
  background: var(--edge);
  margin: 4px 8px;
}
</style>

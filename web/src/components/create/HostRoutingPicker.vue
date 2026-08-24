<script setup lang="ts">
/*
 * Generation host picker (spec §08 multi-host) — the web twin of the desktop
 * Create header's host chip. With more than one machine registered the chip
 * opens a routing menu: Auto (model-aware least busy), Most capable (strongest
 * GPU), or a sticky explicit host. With only this server it collapses to an
 * honest one-line row that doorways into Machines rather than offering an
 * empty menu.
 *
 * Presentational on purpose: routing state comes in as props and the pick goes
 * out as an event, so the decisions stay in `lib/hostRouting.ts`.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  type RoutableHost,
} from "../../lib/hostRouting";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";

const props = defineProps<{
  hosts: RoutableHost[];
  /** Normalized pick: "auto", "capable", or a listed host id. */
  targetId: string;
}>();

const emit = defineEmits<{ select: [id: string]; "open-machines": [] }>();

const multiHost = computed(() => props.hosts.length > 1);

const pinnedHost = computed(() =>
  props.targetId === AUTO_TARGET_ID || props.targetId === CAPABLE_TARGET_ID
    ? null
    : (props.hosts.find((h) => h.id === props.targetId) ?? null),
);

const originHostEntry = computed(
  () => props.hosts.find((h) => h.id === ORIGIN_HOST_ID) ?? null,
);

const chipLabel = computed(() => {
  if (props.targetId === AUTO_TARGET_ID)
    return multiHost.value ? "Auto" : "this server";
  if (props.targetId === CAPABLE_TARGET_ID) return "Most capable";
  return (
    pinnedHost.value?.label ?? originHostEntry.value?.label ?? "this server"
  );
});

/** The chip's dot: green once the pick can actually take a job. */
const chipReady = computed(() => {
  if (
    props.targetId === AUTO_TARGET_ID ||
    props.targetId === CAPABLE_TARGET_ID
  ) {
    return props.hosts.some((h) => h.status === "ready" && !h.stale);
  }
  return pinnedHost.value?.status === "ready" && !pinnedHost.value.stale;
});

const chipStatus = computed(() => {
  if (
    props.targetId === AUTO_TARGET_ID ||
    props.targetId === CAPABLE_TARGET_ID
  ) {
    if (props.hosts.some((host) => host.status === "ready" && host.stale))
      return chipReady.value ? "ready" : "reconnecting";
    return chipReady.value ? "ready" : "connecting";
  }
  const status = pinnedHost.value?.status;
  if (pinnedHost.value?.stale) return "reconnecting";
  if (status === "ready") return "ready";
  if (status === "error") return "offline";
  return status ?? "connecting";
});

function hostLine(host: RoutableHost): string {
  if (host.stale) return "reconnecting";
  if (host.status === "error") return "offline";
  if (host.status === "connecting") return "connecting";
  return host.queueDepth !== null ? `queue ${host.queueDepth}` : "ready";
}

const open = ref(false);
const rootEl = ref<HTMLDivElement | null>(null);

function toggle() {
  open.value = !open.value;
}

function pick(id: string) {
  emit("select", id);
  open.value = false;
}

function onPointerDown(event: PointerEvent) {
  if (!open.value || !rootEl.value) return;
  if (!event.composedPath().includes(rootEl.value)) open.value = false;
}
function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") open.value = false;
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
  <div ref="rootEl" class="hostpick">
    <!-- Single machine: no routing to offer, so say what is true and point at
         Machines instead of dangling an empty menu. -->
    <button
      v-if="!multiHost"
      type="button"
      class="hostpick__row"
      data-test="controls-host"
      @click="emit('open-machines')"
    >
      <span
        class="hostpick__dot"
        :class="{ 'hostpick__dot--wait': !chipReady }"
      />
      <span class="hostpick__label">Run on {{ chipLabel }}</span>
      <span class="hostpick__hint">{{
        originHostEntry?.stale ? "reconnecting" : "add a machine"
      }}</span>
      <Icon name="chevron-right" :size="14" />
    </button>

    <template v-else>
      <button
        type="button"
        class="hostpick__row hostpick__row--chip"
        data-test="host-chip"
        :aria-expanded="open"
        aria-haspopup="menu"
        @click="toggle"
      >
        <span
          class="hostpick__dot"
          :class="{ 'hostpick__dot--wait': !chipReady }"
        />
        <span class="hostpick__label">Run on {{ chipLabel }}</span>
        <span class="hostpick__hint">{{ chipStatus }}</span>
        <Icon name="chevron-down" :size="14" />
      </button>

      <div
        v-if="open"
        class="hostpick__menu"
        role="menu"
        aria-label="Generation host"
        data-test="host-menu"
      >
        <div class="hostpick__kicker">run on</div>
        <button
          type="button"
          role="menuitemradio"
          class="hostpick__item"
          data-test="host-option-auto"
          :aria-checked="targetId === AUTO_TARGET_ID"
          @click="pick(AUTO_TARGET_ID)"
        >
          <span class="hostpick__item-label">Auto</span>
          <span class="hostpick__item-sub">least busy</span>
          <span v-if="targetId === AUTO_TARGET_ID" class="hostpick__check"
            >✓</span
          >
        </button>
        <button
          type="button"
          role="menuitemradio"
          class="hostpick__item"
          data-test="host-option-capable"
          :aria-checked="targetId === CAPABLE_TARGET_ID"
          @click="pick(CAPABLE_TARGET_ID)"
        >
          <span class="hostpick__item-label">Most capable</span>
          <span class="hostpick__item-sub">strongest gpu</span>
          <span v-if="targetId === CAPABLE_TARGET_ID" class="hostpick__check"
            >✓</span
          >
        </button>
        <div class="hostpick__rule" />
        <button
          v-for="host in hosts"
          :key="host.id"
          type="button"
          role="menuitemradio"
          class="hostpick__item"
          :data-test="`host-option-${host.id}`"
          :disabled="host.status !== 'ready'"
          :aria-checked="targetId === host.id"
          @click="pick(host.id)"
        >
          <span
            class="hostpick__dot"
            :class="{
              'hostpick__dot--wait': host.status !== 'ready' || host.stale,
            }"
          />
          <span class="hostpick__item-label">{{ host.label }}</span>
          <span class="hostpick__item-sub">{{ hostLine(host) }}</span>
          <span v-if="targetId === host.id" class="hostpick__check">✓</span>
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.hostpick {
  position: relative;
}

.hostpick__row {
  width: 100%;
  border: 0;
  border-top: 1px solid var(--edge);
  background: transparent;
  padding: 14px 0 0;
  margin-top: 2px;
  display: flex;
  align-items: center;
  gap: 9px;
  cursor: pointer;
  color: var(--ink-2);
  text-align: left;
}

.hostpick__row:hover .hostpick__label {
  color: var(--rebate);
}

.hostpick__row:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.hostpick__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--success);
  flex: 0 0 7px;
}

.hostpick__dot--wait {
  background: var(--halide);
}

.hostpick__label {
  font-size: 12px;
  color: var(--ink-2);
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hostpick__hint {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.hostpick__menu {
  position: absolute;
  right: 0;
  bottom: calc(100% + 6px);
  z-index: 30;
  min-width: 232px;
  padding: 7px;
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  background: var(--bench);
  box-shadow: var(--shadow-raised);
}

.hostpick__kicker {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  padding: 4px 8px 6px;
}

.hostpick__item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  border-radius: var(--radius-control);
  padding: 7px 8px;
  cursor: pointer;
  color: var(--ink-2);
  text-align: left;
}

.hostpick__item:hover:not(:disabled) {
  background: var(--bath);
}

.hostpick__item:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.hostpick__item-label {
  font-size: 12px;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hostpick__item-sub {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}

.hostpick__check {
  color: var(--safelight);
  font-size: 11px;
}

.hostpick__rule {
  height: 1px;
  background: var(--edge);
  margin: 6px 4px;
}
</style>

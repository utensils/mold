<script setup lang="ts">
/*
 * Machines overview card for a single host (spec §04 / §08 G4). Polls the
 * host's `/api/status`, then shows the status dot, name, GPU/address mono
 * line, a memory + queue mono row, and a memory bar. The primary origin uses
 * the accent memory tone; remotes use halide. While the first poll is in
 * flight the card is a shimmer skeleton; a transient failure shows an amber
 * reconnecting state with last-seen text and a Retry action while retaining
 * the last-good metrics.
 */
import { computed } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import StatusDot from "./StatusDot.vue";
import { useHostPoll } from "./hostClient";
import { deriveHostCardGpu, formatGb } from "./machineTelemetry";
import { HOST_RECONNECTING_LABEL } from "@studio/lib/hostConnectivity";
import type { HostEntry } from "../../lib/hostRegistry";

const props = defineProps<{ host: HostEntry; primary?: boolean }>();
const emit = defineEmits<{
  open: [id: string];
  reconnect: [id: string];
  contextMenu: [payload: { host: HostEntry; x: number; y: number }];
}>();

const disconnected = computed(() => props.host.connected === false);
const poll = useHostPoll(
  computed(() => (disconnected.value ? null : props.host)),
);

const showSkeleton = computed(
  () => !disconnected.value && poll.loading.value && !poll.status.value,
);
const reconnecting = computed(
  () =>
    !disconnected.value &&
    !showSkeleton.value &&
    (!poll.online.value || poll.stale.value),
);
const dotState = computed<"online" | "offline" | "unknown">(() => {
  if (disconnected.value || showSkeleton.value || reconnecting.value)
    return "unknown";
  return "online";
});

function hostAddress(url: string): string | null {
  try {
    return new URL(url).host;
  } catch {
    return null;
  }
}

const gpuLine = computed(() => {
  const status = poll.status.value;
  const name = deriveHostCardGpu(status)?.label ?? null;
  const secondary = props.primary
    ? (status?.hostname ?? null)
    : hostAddress(props.host.url);
  const parts = [name, secondary].filter((p): p is string => !!p);
  return parts.length ? parts.join(" · ") : "—";
});

const memory = computed<{ used: number; total: number } | null>(() => {
  const summary = deriveHostCardGpu(poll.status.value);
  return summary && summary.total > 0
    ? { used: summary.used, total: summary.total }
    : null;
});

const memLabel = computed(() =>
  memory.value
    ? `${formatGb(memory.value.used)} / ${formatGb(memory.value.total)} GB`
    : "—",
);
const memPct = computed(() =>
  memory.value ? (memory.value.used / memory.value.total) * 100 : 0,
);
const queueLabel = computed(() => {
  const depth = poll.status.value?.queue_depth;
  return `queue ${depth ?? 0}`;
});

const lastSeenLabel = computed(() => {
  if (poll.lastSeen.value == null) return "never reached";
  const secs = Math.round((Date.now() - poll.lastSeen.value) / 1000);
  if (secs < 60) return "last seen just now";
  const mins = Math.round(secs / 60);
  return `last seen ${mins}m ago`;
});

function open() {
  if (disconnected.value) return;
  emit("open", props.host.id);
}

function onKey(event: KeyboardEvent) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    open();
  }
}

function retry(event: Event) {
  event.stopPropagation();
  void poll.refresh();
}

function reconnect(event: Event) {
  event.stopPropagation();
  emit("reconnect", props.host.id);
}

function openContextMenu(event: MouseEvent) {
  emit("contextMenu", { host: props.host, x: event.clientX, y: event.clientY });
}
</script>

<template>
  <CardSurface v-if="showSkeleton" data-test="host-card-skeleton">
    <div class="ms-shimmer hc-skel hc-skel--line" />
    <div class="ms-shimmer hc-skel hc-skel--sub" />
    <div class="ms-shimmer hc-skel hc-skel--bar" />
  </CardSurface>

  <CardSurface v-else>
    <div
      class="hc"
      :role="disconnected ? undefined : 'button'"
      :tabindex="disconnected ? undefined : 0"
      data-test="host-card"
      :aria-label="disconnected ? undefined : `Open ${host.name}`"
      @click="open"
      @keydown="onKey"
      @contextmenu.prevent.stop="openContextMenu"
    >
      <div class="hc__head">
        <StatusDot :state="dotState" />
        <span class="hc__name" data-test="host-name">{{ host.name }}</span>
      </div>
      <div class="hc__gpu" data-test="host-gpu">{{ gpuLine }}</div>

      <template v-if="disconnected">
        <div class="hc__offline">
          <span class="hc__offline-text" data-test="host-disconnected"
            >disconnected</span
          >
          <button
            type="button"
            class="hc__retry"
            data-test="host-reconnect"
            @click="reconnect"
          >
            Connect
          </button>
        </div>
      </template>
      <template v-else>
        <div
          v-if="reconnecting"
          class="hc__offline"
          data-test="host-reconnecting-state"
        >
          <span class="hc__offline-text">
            {{ lastSeenLabel }}
            <!-- The card keeps polling, so the machine comes back on its own;
                 say so rather than implying Retry is the only way back. -->
            <span class="hc__reconnecting" data-test="host-reconnecting">{{
              HOST_RECONNECTING_LABEL
            }}</span>
          </span>
          <button
            type="button"
            class="hc__retry"
            data-test="host-retry"
            @click="retry"
          >
            Retry
          </button>
        </div>
        <div v-if="poll.status.value" class="hc__row">
          <span data-test="host-mem">{{ memLabel }}</span>
          <span data-test="host-queue">{{ queueLabel }}</span>
        </div>
        <ProgressBar
          v-if="poll.status.value"
          :value="memPct"
          :tone="primary ? 'accent' : 'info'"
          :label="`${host.name} memory`"
        />
      </template>
    </div>
  </CardSurface>
</template>

<style scoped>
.hc {
  cursor: pointer;
  color: var(--rebate);
  outline: none;
}

.hc:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 3px;
  border-radius: 8px;
}

.hc__head {
  display: flex;
  align-items: center;
  gap: 10px;
}

.hc__name {
  font-size: 14.5px;
  font-weight: 600;
}

.hc__gpu {
  margin-top: 6px;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}

.hc__row {
  display: flex;
  justify-content: space-between;
  margin: 14px 0 6px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.hc__offline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 16px;
}

.hc__offline-text {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.hc__reconnecting {
  display: block;
  color: var(--warning);
}

.hc__retry {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 13px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.hc__retry:hover {
  border-color: var(--ink-3);
}

.hc-skel {
  border-radius: 6px;
}

.hc-skel--line {
  height: 16px;
  width: 55%;
}

.hc-skel--sub {
  height: 10px;
  width: 40%;
  margin-top: 10px;
}

.hc-skel--bar {
  height: 6px;
  width: 100%;
  margin-top: 18px;
}
</style>

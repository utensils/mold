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
import Icon from "@ui/components/Icon.vue";
import CardSurface from "@ui/components/CardSurface.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import StatusDot from "@ui/components/StatusDot.vue";
import { useHostPoll } from "./hostClient";
import {
  deriveHostCardGpu,
  formatGb,
  hostGpuSnapshots,
} from "./machineTelemetry";
import { machineSentence } from "@studio/lib/machineSentence";
import { HOST_RECONNECTING_LABEL } from "@studio/lib/hostConnectivity";
import type { HostEntry } from "../../lib/hostRegistry";

const props = defineProps<{
  host: HostEntry;
  primary?: boolean;
  actionsOpen?: boolean;
}>();
const emit = defineEmits<{
  open: [id: string];
  reconnect: [id: string];
  contextMenu: [
    payload: { host: HostEntry; x: number; y: number; opener: HTMLElement },
  ];
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

/*
 * The one plain sentence about a machine — "4× L40S · CUDA · on your network
 * at plato:7680" — from `@studio/lib/machineSentence`, so the same box reads
 * identically here, in the desktop app's Machines list and in its machine
 * pane. The card used to build its own line and called a 4× L40S box "L40S".
 *
 * Every machine a browser can see is `remote`: the tab is not running ON any
 * of them, so "this device" would be a lie even for the serving origin. The
 * origin has no URL of its own, so its address is the hostname it reports.
 */
const gpuLine = computed(() => {
  const status = poll.status.value;
  const address = props.primary
    ? (status?.hostname ?? "")
    : (hostAddress(props.host.url) ?? "");
  const sentence = machineSentence(
    { kind: "remote", baseUrl: address },
    hostGpuSnapshots(status),
    { address: true },
  );
  return sentence || "—";
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

function retry(event: Event) {
  event.stopPropagation();
  void poll.refresh();
}

function reconnect(event: Event) {
  event.stopPropagation();
  emit("reconnect", props.host.id);
}

/*
 * A nested button's native Enter fires a `click` that bubbles to the card, so
 * every control inside the card stops both the click and the keydown that
 * produced it. Retry is Retry; it is not also "open the machine".
 */
function swallow(event: Event) {
  event.stopPropagation();
}

function openContextMenu(event: MouseEvent) {
  const opener = (event.currentTarget as HTMLElement)
    .closest("[data-test=host-card]")
    ?.querySelector<HTMLElement>("[data-test=host-actions]");
  if (!opener) return;
  const rect = opener.getBoundingClientRect();
  emit("contextMenu", {
    host: props.host,
    x: event.type === "contextmenu" ? event.clientX : rect.left,
    y: event.type === "contextmenu" ? event.clientY : rect.bottom,
    opener,
  });
}
</script>

<template>
  <CardSurface v-if="showSkeleton" data-test="host-card-skeleton">
    <div class="ms-shimmer hc-skel hc-skel--line" />
    <div class="ms-shimmer hc-skel hc-skel--sub" />
    <div class="ms-shimmer hc-skel hc-skel--bar" />
  </CardSurface>

  <CardSurface
    v-else
    class="hc-card"
    :class="{ 'hc-card--open': !disconnected }"
  >
    <!-- The whole card is the door, the way a print tile is (GalleryCard).
         A disconnected card has nothing to open, so it is not a control at
         all and carries no role, no tab stop and no hover affordance. -->
    <div
      class="hc"
      :class="{ 'hc--open': !disconnected }"
      data-test="host-card"
      :role="disconnected ? undefined : 'button'"
      :tabindex="disconnected ? undefined : 0"
      :aria-label="disconnected ? undefined : `Open ${host.name}`"
      @click="open"
      @keydown.enter.prevent="open"
      @keydown.space.prevent="open"
      @contextmenu.prevent.stop="openContextMenu"
    >
      <div class="hc__head">
        <StatusDot :state="dotState" />
        <button
          v-if="!disconnected"
          type="button"
          class="hc__name hc__open"
          data-test="host-open"
          tabindex="-1"
          :aria-label="`Open ${host.name}`"
          @click.stop="open"
        >
          <span data-test="host-name">{{ host.name }}</span>
        </button>
        <span v-else class="hc__name" data-test="host-name">{{
          host.name
        }}</span>
        <button
          type="button"
          class="hc__actions"
          data-test="host-actions"
          :aria-label="`Actions for ${host.name}`"
          aria-haspopup="menu"
          :aria-expanded="actionsOpen ?? false"
          @click.stop="openContextMenu"
          @keydown.enter.stop
          @keydown.space.stop
        >
          <Icon name="more" :size="18" />
        </button>
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
            @keydown.enter.stop="swallow"
            @keydown.space.stop="swallow"
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
            @keydown.enter.stop="swallow"
            @keydown.space.stop="swallow"
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
  color: var(--rebate);
}
.hc--open {
  cursor: pointer;
}
/* The tint belongs to the whole card, so it sits on the CardSurface rather
   than on `.hc`, which is inset by the card's own padding. */
.hc-card--open:hover {
  background: color-mix(in srgb, var(--rebate) 4%, transparent);
}
.hc:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.hc__head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.hc__actions,
.hc__open {
  min-height: 44px;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.hc__open {
  text-align: left;
  padding: 0;
}
.hc__actions {
  min-width: 44px;
  display: grid;
  place-items: center;
  margin-left: auto;
}
.hc__name {
  flex: 1;
  overflow-wrap: anywhere;
  min-width: 0;
  font-size: 0.90625rem;
  font-weight: 600;
}

.hc__gpu {
  min-width: 0;
  overflow-wrap: anywhere;
  margin-top: 6px;
  /* Plain words in sans; the meter's readouts below are the mono truth. */
  font-family: var(--f-body);
  font-size: 0.875rem;
  color: var(--ink-2);
}

.hc__row {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  margin: 14px 0 6px;
  font-family: var(--f-mono);
  font-size: 0.875rem;
  color: var(--ink-3);
}

.hc__offline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 16px;
}

.hc__offline-text {
  font-family: var(--f-mono);
  font-size: 0.875rem;
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
  min-height: 44px;
  border-radius: 8px;
  font-size: 0.875rem;
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

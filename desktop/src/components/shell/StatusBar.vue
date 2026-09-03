<script setup lang="ts">
/*
 * The status bar (README §04): answers "which machine, how full, how deep is
 * the queue" so those questions never cost a view change. Mono readouts on
 * the left, key hints on the right in accent-coloured keycaps.
 */
import { computed } from "vue";
import { useRouter } from "vue-router";
import { useDownloadsStore } from "../../stores/downloads";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useQueueActivity } from "../../composables/useQueueActivity";
import { useQueueCommands } from "../../composables/useQueueCommands";
import { queueSentence } from "../../lib/queueRows";
import { formatGB } from "../../lib/format";
import { shortcutLabel } from "../../lib/platform";

const router = useRouter();
const downloads = useDownloadsStore();
const hostStatus = useHostStatusStore();
const queue = useQueueActivity();
const commands = useQueueCommands();

const dotClass = computed(() => {
  switch (hostStatus.connection) {
    case "error":
      return "bg-error";
    case "connecting":
      return "bg-sapphire ms-pulse";
    case "idle":
      return "bg-fg-dim";
    default:
      return "bg-success";
  }
});

const machineLabel = computed(() => {
  const host = hostStatus.displayHost;
  if (!host) return "no machine";
  switch (hostStatus.connection) {
    case "error":
      return `${host.label} · offline`;
    case "connecting":
      return `${host.label} · connecting`;
    case "idle":
      return `${host.label} · engine off`;
    default:
      return host.label;
  }
});

const queueLine = computed(() =>
  queueSentence(queue.activeCount.value, queue.waitingCount.value, commands.paused.value),
);

const vramLine = computed(() =>
  hostStatus.gpus.length
    ? `vram ${formatGB(hostStatus.vramUsed)} / ${formatGB(hostStatus.vramTotal)}`
    : null,
);
const vramTone = computed(() => (hostStatus.vramCritical ? "text-error" : ""));

const ramTone = computed(() => {
  switch (hostStatus.hostMemoryPressure) {
    case "critical":
      return "text-error";
    case "warn":
      return "text-warning";
    default:
      return "";
  }
});
const ramLine = computed(() =>
  hostStatus.snapshot ? `ram ${formatGB(hostStatus.snapshot.system_ram.used)}` : null,
);

const downloading = computed(() => downloads.hostedInFlight.length);
</script>

<template>
  <footer
    data-test="status-bar"
    class="flex h-[var(--mold-shell-statusbar-h)] shrink-0 items-center gap-3.5 border-t border-border bg-chrome px-3 font-mono text-micro text-fg-dim select-none"
  >
    <button
      type="button"
      data-test="status-machine"
      class="flex items-center gap-1.5 hover:text-fg"
      :title="hostStatus.sentence"
      @click="
        router.push(
          hostStatus.displayHost?.primary === false
            ? `/machines/${hostStatus.displayHost.id}`
            : '/machines',
        )
      "
    >
      <span class="h-1.5 w-1.5 rounded-full" :class="dotClass" />
      <span>{{ machineLabel }}</span>
    </button>
    <span class="divider" />
    <button
      type="button"
      data-test="status-queue"
      class="hover:text-fg"
      @click="router.push('/queue')"
    >
      {{ queueLine }}
    </button>
    <template v-if="vramLine">
      <span class="divider" />
      <span data-test="status-vram" :class="vramTone">{{ vramLine }}</span>
    </template>
    <template v-if="ramLine">
      <span class="divider" />
      <span data-test="status-ram" :class="ramTone">{{ ramLine }}</span>
    </template>
    <template v-if="downloading > 0">
      <span class="divider" />
      <button type="button" class="text-warning hover:text-fg" @click="router.push('/models')">
        {{ downloading === 1 ? "1 style downloading" : `${downloading} styles downloading` }}
      </button>
    </template>
    <span class="flex-1" />
    <span class="keycap">{{ shortcutLabel("↩") }}</span
    ><span>Generate</span> <span class="keycap">{{ shortcutLabel("K") }}</span
    ><span>Search</span> <span class="keycap">{{ shortcutLabel("N") }}</span
    ><span>New image</span>
  </footer>
</template>

<style scoped>
.divider {
  width: var(--mold-bw);
  height: 12px;
  background: var(--mold-border);
}
.keycap {
  color: var(--mold-blue);
  font-weight: 700;
}
</style>

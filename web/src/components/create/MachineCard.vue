<script setup lang="ts">
/*
 * The Create rail's first card: which machine this tab is talking to.
 *
 * The connection used to be the LAST thing in the rail, under every setting,
 * with no explanation at all — so the one fact a browser surface has to state
 * (the pictures are made somewhere else, and closing the tab does not stop
 * them) was never said. It now leads the rail, with a Change link to the
 * Machines workspace beside it.
 *
 * Presentational: the page owns polling, routing and the picker. The card is
 * handed a name, a state, a sentence and the two numbers, and renders them.
 */
import { computed } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import StatusDot from "@ui/components/StatusDot.vue";
import { formatGb } from "../machines/machineTelemetry";

const props = withDefaults(
  defineProps<{
    name: string;
    status: "ready" | "connecting" | "error" | "offline";
    /** The card's one sentence. */
    sentence?: string;
    /** GPU memory in bytes; the meter needs both to mean anything. */
    used?: number | null;
    total?: number | null;
    /** Jobs waiting on this machine. Absent says nothing; 0 says "queue 0". */
    queue?: number | null;
    /** More than one machine is reachable, so where it runs is a choice. */
    multiHost?: boolean;
  }>(),
  {
    sentence:
      "This tab is talking to a machine on your network. Close the tab and it keeps working.",
    used: null,
    total: null,
    queue: null,
    multiHost: false,
  },
);

/** A machine still being reached is neither up nor down: it is unknown. */
const dotState = computed<"online" | "offline" | "unknown">(() => {
  if (props.status === "ready") return "online";
  if (props.status === "connecting") return "unknown";
  return "offline";
});

const memory = computed(() =>
  typeof props.used === "number" &&
  typeof props.total === "number" &&
  props.total > 0
    ? { used: props.used, total: props.total }
    : null,
);
const memoryLabel = computed(() =>
  memory.value
    ? `${formatGb(memory.value.used)} / ${formatGb(memory.value.total)} GB`
    : null,
);
const memoryPercent = computed(() =>
  memory.value ? (memory.value.used / memory.value.total) * 100 : 0,
);
</script>

<template>
  <section class="machine" data-test="machine-card">
    <header class="machine__head">
      <StatusDot :state="dotState" />
      <span class="machine__name" data-test="machine-card-name">{{
        name
      }}</span>
      <RouterLink
        class="machine__change"
        data-test="machine-card-change"
        to="/machines"
        >Change</RouterLink
      >
    </header>

    <p class="machine__sentence" data-test="machine-card-sentence">
      {{ sentence }}
    </p>

    <ProgressBar
      v-if="memory"
      class="machine__meter"
      :value="memoryPercent"
      :height="5"
      :label="`${name} memory`"
    />

    <div v-if="memoryLabel || queue !== null" class="machine__readouts">
      <span v-if="memoryLabel" data-test="machine-card-memory">{{
        memoryLabel
      }}</span>
      <span v-if="queue !== null" data-test="machine-card-queue"
        >queue {{ queue }}</span
      >
    </div>

    <div v-if="multiHost" class="machine__picker">
      <slot name="picker" />
    </div>
  </section>
</template>

<style scoped>
.machine {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-panel);
}

.machine__head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.machine__name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
  font-weight: 700;
  color: var(--mold-text);
}

.machine__change {
  flex-shrink: 0;
  font-size: var(--mold-fs-xs);
  color: var(--mold-blue);
  text-decoration: none;
}

.machine__change:hover {
  text-decoration: underline;
}

.machine__sentence {
  margin: 0;
  font-size: var(--mold-fs-xs);
  line-height: var(--mold-lh-body);
  color: var(--mold-text-dim);
}

.machine__meter {
  margin-top: 2px;
}

.machine__readouts {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.machine__picker {
  margin-top: 2px;
}
</style>

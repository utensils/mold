<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's MachineCard.vue.
/*
 * The rail's first card: which machine this tab is talking to. A browser has
 * no local GPU, so the connection is stated plainly with a way to change it —
 * the mock's Machine rule. The routing picker is injected through `#picker`
 * so the card never reaches a host list itself.
 */
import { computed } from "vue";
import { RouterLink } from "vue-router";
import StatusDot from "@ui/components/StatusDot.vue";

const props = withDefaults(
  defineProps<{
    name: string;
    status: "ready" | "connecting" | "reconnecting" | "error";
    sentence?: string;
    used?: number | null;
    total?: number | null;
    queue?: number | null;
    multiHost?: boolean;
  }>(),
  {
    sentence: "",
    used: null,
    total: null,
    queue: null,
    multiHost: false,
  },
);

const dotState = computed<"online" | "offline" | "unknown">(() =>
  props.status === "ready"
    ? "online"
    : props.status === "error"
      ? "offline"
      : "unknown",
);
const meterPercent = computed(() => {
  if (!props.total || props.used === null || props.used === undefined)
    return null;
  return Math.min(100, Math.max(0, (props.used / props.total) * 100));
});
</script>

<template>
  <section class="machine" data-test="create-machine-card">
    <div class="machine__head">
      <StatusDot :state="dotState" />
      <span class="sr-only">{{ status }}</span>
      <span class="machine__name" data-test="create-machine-name">{{
        name
      }}</span>
      <RouterLink
        to="/machines"
        class="machine__change"
        data-test="create-machine-change"
        >Change</RouterLink
      >
    </div>
    <p v-if="sentence" class="machine__sentence">{{ sentence }}</p>
    <div
      v-if="meterPercent !== null"
      class="machine__meter"
      data-test="create-machine-meter"
    >
      <span :style="{ width: `${meterPercent}%` }" />
    </div>
    <p
      v-if="queue !== null && queue > 0"
      class="machine__queue"
      data-test="create-machine-queue"
    >
      {{ queue }} waiting
    </p>
    <div v-if="multiHost" class="machine__picker">
      <slot name="picker" />
    </div>
    <div v-else class="machine__picker machine__picker--single">
      <slot name="picker" />
    </div>
  </section>
</template>

<style scoped>
.machine {
  background: var(--mold-surface);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  padding: 14px;
}
.machine__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.machine__name {
  font-family: var(--mold-font-mono);
  font-weight: 700;
  font-size: var(--mold-fs-sm);
  color: var(--mold-text);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.machine__change {
  margin-left: auto;
  font-size: var(--mold-fs-xs);
  color: var(--mold-blue);
  text-decoration: none;
}
.machine__sentence {
  margin: 9px 0 0;
  font-size: var(--mold-fs-xs);
  line-height: 1.45;
  color: var(--mold-text-dim);
}
.machine__meter {
  margin-top: 10px;
  height: 5px;
  background: var(--mold-bg-crust);
  overflow: hidden;
}
.machine__meter span {
  display: block;
  height: 100%;
  background: var(--mold-blue);
}
.machine__queue {
  margin: 8px 0 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.machine__picker {
  margin-top: 10px;
}
</style>

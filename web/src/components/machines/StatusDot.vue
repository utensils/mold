<script setup lang="ts">
/*
 * Small round status indicator for machine cards / host detail. Online is the
 * success dot, offline is the stop dot (spec §08 G4 offline card), and the
 * pre-first-poll "unknown" state is a muted dot.
 */
withDefaults(defineProps<{ state?: "online" | "offline" | "unknown" }>(), {
  state: "unknown",
});
</script>

<template>
  <span class="ms-dot" :data-state="state" aria-hidden="true" />
</template>

<style scoped>
.ms-dot {
  width: 8px;
  height: 8px;
  flex: 0 0 8px;
  border-radius: 50%;
  background: var(--ink-3);
}

.ms-dot[data-state="online"] {
  background: var(--success);
}

.ms-dot[data-state="offline"] {
  background: var(--stop);
}
</style>

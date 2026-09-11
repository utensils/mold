<script setup lang="ts">
/*
 * Small round status indicator for machine cards and the machine pane. Online
 * is the success dot, offline is the error dot (spec §08 G4 offline card), and
 * the pre-first-poll "unknown" state is a muted dot.
 *
 * Decorative: the dot never carries the only copy of a machine's state — the
 * words beside it do — so it is `aria-hidden`.
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
  background: var(--mold-text-dim);
}

.ms-dot[data-state="online"] {
  background: var(--mold-success);
}

.ms-dot[data-state="offline"] {
  background: var(--mold-error);
}
</style>

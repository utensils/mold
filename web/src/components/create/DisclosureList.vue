<script setup lang="ts">
/*
 * The bordered card the Create rail's disclosure rows sit in — one border
 * around the group and a hairline between rows, so six doors read as one
 * list rather than six cards.
 */
</script>

<template>
  <div class="disclosure-list" data-test="disclosure-list">
    <slot />
  </div>
</template>

<style scoped>
.disclosure-list {
  display: flex;
  flex-direction: column;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-panel);
  overflow: hidden;
}

.disclosure-list > :deep(* + *) {
  border-top: var(--mold-bw) solid var(--mold-border);
}
</style>

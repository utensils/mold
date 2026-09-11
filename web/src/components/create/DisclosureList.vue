<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's DisclosureList.vue.
/* The bordered card the rail's disclosure rows live in. */
</script>

<template>
  <div class="list" data-test="create-disclosures">
    <slot />
  </div>
</template>

<style scoped>
.list {
  background: var(--mold-surface);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  overflow: clip;
}
</style>

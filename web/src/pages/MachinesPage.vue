<script setup lang="ts">
/*
 * Machines workspace (spec §04) — local + remote GPUs as host cards with live
 * telemetry. W2 ships the placeholder shell only; the connect flow, host
 * cards, and detail view land in a later milestone (G1/G3).
 */
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
    <h1
      class="font-display text-2xl font-bold tracking-tight text-ink"
      data-test="machines-title"
    >
      Machines
    </h1>

    <div class="mt-10">
      <EmptyStateBlock
        icon="machines"
        headline="No machines yet — add one."
        guidance="Machines arrive in the next milestone."
      />
    </div>
  </div>
</template>

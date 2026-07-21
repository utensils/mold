<script setup lang="ts">
/*
 * Host detail (spec §04/§06) — telemetry, loaded/installed models, and the
 * set-as-target action for a single machine. W2 ships the placeholder shell;
 * the live detail view lands in a later milestone (G3).
 */
import { useRoute } from "vue-router";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";

const route = useRoute();
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-6 sm:px-6 lg:px-10">
    <router-link
      to="/machines"
      class="inline-flex items-center gap-1 font-mono text-xs text-safelight"
    >
      <span aria-hidden="true">&lsaquo;</span> Machines
    </router-link>
    <h1
      class="mt-3 font-display text-2xl font-bold tracking-tight text-ink"
      data-test="machine-detail-title"
    >
      {{ route.params.id }}
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

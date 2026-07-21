<script setup lang="ts">
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
</script>

<template>
  <main class="grid min-h-[calc(100svh-56px)] place-items-center p-6">
    <EmptyStateBlock
      icon="search"
      headline="Page not found"
      guidance="That Mold Studio destination does not exist."
    >
      <template #action>
        <RouterLink
          class="rounded-control bg-safelight px-4 py-2 text-on-accent"
          to="/create"
        >
          Go to Create
        </RouterLink>
      </template>
    </EmptyStateBlock>
  </main>
</template>

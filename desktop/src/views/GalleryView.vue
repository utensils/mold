<script setup lang="ts">
import { useRouter } from "vue-router";
import EmptyState from "../components/shell/EmptyState.vue";

const router = useRouter();
</script>

<template>
  <EmptyState
    headline="No prints yet"
    detail="Generate one and it lands here."
    action="Go to Generate"
    @action="router.push('/generate')"
  />
</template>

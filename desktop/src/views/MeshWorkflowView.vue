<script setup lang="ts">
import { computed } from "vue";
import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import { useConnectionStore } from "../stores/connection";

const connection = useConnectionStore();
const target = computed(() => ({
  baseUrl: connection.baseUrl ?? "",
  apiKey: connection.apiKey,
}));
</script>

<template>
  <MeshWorkflowStudio v-if="target.baseUrl" :target="target" />
  <div v-else class="grid h-full place-items-center text-fg-dim">
    Connect a machine to use 3-D workflows.
  </div>
</template>

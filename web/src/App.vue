<script setup lang="ts">
// ── Agent B (resource telemetry) ────────────────────────────────────────────
// `useResources` is mounted once at the App root and injected into pages
// that need it (/generate, and Agent C's PlacementPanel). This ensures a
// single shared EventSource instead of one per page navigation.
import { provide } from "vue";
import {
  useResources,
  RESOURCES_INJECTION_KEY,
} from "./composables/useResources";

const resources = useResources();
provide(RESOURCES_INJECTION_KEY, resources);
</script>

<template>
  <router-view />
</template>

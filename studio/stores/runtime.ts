import { defineStore } from "pinia";
import { computed, ref } from "vue";
import type { ApiTarget } from "../api/client";

export interface StudioHostRoute {
  id: string;
  label: string;
  target: ApiTarget;
}

export const useStudioRuntimeStore = defineStore("studio-runtime", () => {
  const routes = ref<StudioHostRoute[]>([]);
  const selectedHostId = ref<string | null>(null);
  const selectedRoute = computed(
    () =>
      routes.value.find((route) => route.id === selectedHostId.value) ?? null,
  );

  function replaceRoutes(next: StudioHostRoute[]) {
    routes.value = next;
    if (
      !selectedHostId.value ||
      !next.some((route) => route.id === selectedHostId.value)
    ) {
      selectedHostId.value = next[0]?.id ?? null;
    }
  }

  return { routes, selectedHostId, selectedRoute, replaceRoutes };
});

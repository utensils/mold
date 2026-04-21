<script setup lang="ts">
/**
 * Fixed bottom telemetry tray. Defaults collapsed; clicking the header
 * toggles the expanded body which hosts the full `<ResourceStrip>`.
 *
 * Mounted once at the App root so it persists across route changes. State
 * (`expanded` / `auto-hide`) lives in localStorage so a user who wants the
 * tray always visible doesn't have to re-expand it every reload.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import ResourceStrip from "./ResourceStrip.vue";
import type { ResourceSnapshot } from "../types";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";
import { inject } from "vue";
import type { ComputedRef, Ref } from "vue";

const STORAGE_KEY = "mold.resource-tray.expanded";

type UseResourcesShape = {
  snapshot: Ref<ResourceSnapshot | null>;
  gpuList: ComputedRef<unknown[]>;
  error: Ref<string | null>;
};

const injected = inject<UseResourcesShape | null>(
  RESOURCES_INJECTION_KEY,
  null,
);

function loadExpanded(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

const expanded = ref(loadExpanded());

function toggle() {
  expanded.value = !expanded.value;
  try {
    localStorage.setItem(STORAGE_KEY, expanded.value ? "1" : "0");
  } catch {
    /* ignore */
  }
}

const summary = computed(() => {
  const s = injected?.snapshot.value ?? null;
  if (!s) return "resources loading…";
  const parts: string[] = [];
  if (s.gpus.length > 0) {
    const g = s.gpus[0];
    const gb = (n: number) => (n / 1_000_000_000).toFixed(1);
    const util = g.gpu_utilization != null ? ` ${g.gpu_utilization}%` : "";
    const label = s.gpus.length > 1 ? `GPU0` : "GPU";
    parts.push(`${label} ${gb(g.vram_used)}/${gb(g.vram_total)}GB${util}`);
  }
  if (s.cpu) {
    parts.push(`CPU ${Math.round(s.cpu.usage_percent)}%`);
  }
  const gb = (n: number) => (n / 1_000_000_000).toFixed(1);
  parts.push(`RAM ${gb(s.system_ram.used)}/${gb(s.system_ram.total)}GB`);
  return parts.join(" · ");
});

function onKey(e: KeyboardEvent) {
  // Press `r` (not while focused in an input) to toggle the tray.
  if (e.key !== "r" || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target as HTMLElement | null;
  if (
    t &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)
  ) {
    return;
  }
  toggle();
}

onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>

<template>
  <div
    class="pointer-events-none fixed inset-x-0 bottom-0 z-30 flex justify-center px-2 pb-2 sm:px-4 sm:pb-3"
  >
    <div
      class="pointer-events-auto w-full max-w-[1800px] rounded-2xl border border-white/5 bg-slate-950/80 backdrop-blur-md"
    >
      <button
        type="button"
        class="flex w-full items-center justify-between gap-3 px-4 py-2 text-left text-[12px] text-ink-300 hover:text-ink-100"
        :aria-expanded="expanded"
        aria-controls="resource-tray-body"
        @click="toggle"
      >
        <span class="flex items-center gap-2">
          <span
            class="inline-block h-2 w-2 rounded-full"
            :class="
              injected?.snapshot.value
                ? 'bg-emerald-400'
                : 'bg-slate-500 animate-pulse'
            "
          />
          <span class="tabular-nums">{{ summary }}</span>
        </span>
        <span class="text-ink-400">{{ expanded ? "▾" : "▸" }}</span>
      </button>
      <div
        v-if="expanded"
        id="resource-tray-body"
        class="border-t border-white/5 px-4 py-3"
      >
        <ResourceStrip variant="full" />
      </div>
    </div>
  </div>
</template>

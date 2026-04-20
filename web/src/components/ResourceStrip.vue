<script setup lang="ts">
/**
 * Always-visible VRAM + system-RAM telemetry panel.
 *
 * Modes:
 *  - `variant="full"` (default) — docked at the bottom of the Composer column
 *    on /generate. One row per GPU plus one for system RAM, click-to-expand
 *    side sheet.
 *  - `variant="chip"` — compact single-line summary for the TopBar on
 *    narrow viewports (< lg).
 *
 * Data comes from the `useResources` singleton provided by App.vue.
 */
import { computed, inject, ref } from "vue";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";
import type { GpuSnapshot, ResourceSnapshot, RamSnapshot } from "../types";
import type { ComputedRef, Ref } from "vue";

type UseResourcesShape = {
  snapshot: Ref<ResourceSnapshot | null>;
  gpuList: ComputedRef<GpuSnapshot[]>;
  error: Ref<string | null>;
};

const props = withDefaults(
  defineProps<{
    variant?: "full" | "chip";
  }>(),
  { variant: "full" },
);
void props;

const injected = inject<UseResourcesShape | null>(
  RESOURCES_INJECTION_KEY,
  null,
);

const snapshot = computed<ResourceSnapshot | null>(
  () => injected?.snapshot.value ?? null,
);
const gpus = computed<GpuSnapshot[]>(() => injected?.gpuList.value ?? []);
const ram = computed<RamSnapshot | null>(
  () => snapshot.value?.system_ram ?? null,
);

const sheetOpen = ref(false);
function toggleSheet() {
  sheetOpen.value = !sheetOpen.value;
}
void sheetOpen;

function fmtGb(bytes: number): string {
  return (bytes / 1_000_000_000).toFixed(1);
}

function pct(used: number, total: number): number {
  if (total === 0) return 0;
  return Math.min(100, Math.round((used / total) * 100));
}

const chipSummary = computed(() => {
  const s = snapshot.value;
  if (!s) return "…";
  const ramStr = ram.value
    ? `${fmtGb(ram.value.used)} / ${fmtGb(ram.value.total)} GB`
    : "…";
  if (s.gpus.length === 0) return `RAM ${ramStr}`;
  const primary = s.gpus[0];
  return `GPU ${fmtGb(primary.vram_used)} / ${fmtGb(primary.vram_total)} · RAM ${ramStr}`;
});
</script>

<template>
  <div v-if="variant === 'chip'" class="inline-flex">
    <button
      data-test="resource-chip"
      type="button"
      class="inline-flex h-9 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-[13px] font-medium text-ink-200 transition hover:text-white"
      :title="snapshot?.hostname ?? ''"
      @click="toggleSheet"
    >
      <svg
        viewBox="0 0 24 24"
        class="h-3.5 w-3.5"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        aria-hidden="true"
      >
        <path
          d="M12 2v4M12 18v4M2 12h4M18 12h4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"
        />
      </svg>
      <span class="tabular-nums">{{ chipSummary }}</span>
    </button>
  </div>

  <aside
    v-else
    class="glass rounded-2xl px-4 py-3"
    role="region"
    aria-label="Resource telemetry"
  >
    <div v-if="!snapshot" class="text-[13px] text-ink-400">…</div>

    <div v-else class="flex flex-col gap-2 text-[13px]">
      <div
        v-for="gpu in gpus"
        :key="gpu.ordinal"
        data-test="resource-row"
        class="flex items-center gap-3"
      >
        <div class="w-28 shrink-0 font-medium text-ink-100">
          GPU {{ gpu.ordinal }} · {{ gpu.name }}
        </div>
        <div class="w-32 shrink-0 tabular-nums text-ink-200">
          {{ fmtGb(gpu.vram_used) }} / {{ fmtGb(gpu.vram_total) }} GB
        </div>
        <div
          class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5"
        >
          <div
            class="absolute inset-y-0 left-0 bg-brand-400/70"
            :style="{ width: `${pct(gpu.vram_used, gpu.vram_total)}%` }"
          />
        </div>
        <div
          v-if="
            gpu.vram_used_by_mold !== null && gpu.vram_used_by_other !== null
          "
          class="w-40 shrink-0 text-right text-[11px] text-ink-400 tabular-nums"
        >
          mold {{ fmtGb(gpu.vram_used_by_mold) }} · other
          {{ fmtGb(gpu.vram_used_by_other) }}
        </div>
      </div>

      <div
        v-if="ram"
        data-test="resource-row"
        class="flex items-center gap-3 border-t border-white/5 pt-2"
      >
        <div class="w-28 shrink-0 font-medium text-ink-100">RAM</div>
        <div class="w-32 shrink-0 tabular-nums text-ink-200">
          {{ fmtGb(ram.used) }} / {{ fmtGb(ram.total) }} GB
        </div>
        <div
          class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5"
        >
          <div
            class="absolute inset-y-0 left-0 bg-emerald-400/70"
            :style="{ width: `${pct(ram.used, ram.total)}%` }"
          />
        </div>
        <div
          class="w-40 shrink-0 text-right text-[11px] text-ink-400 tabular-nums"
        >
          mold {{ fmtGb(ram.used_by_mold) }} · other
          {{ fmtGb(ram.used_by_other) }}
        </div>
      </div>

      <div class="pt-1 text-[11px] text-ink-500">
        host {{ snapshot.hostname }} · updated
        {{ new Date(snapshot.timestamp).toLocaleTimeString() }}
      </div>
    </div>
  </aside>
</template>

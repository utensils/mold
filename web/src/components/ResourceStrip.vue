<script setup lang="ts">
/**
 * VRAM + GPU-utilization + system-RAM + CPU telemetry rows. Consumed by
 * the bottom tray (`ResourceTray.vue`) as the expanded body and by the
 * TopBar as a compact chip on narrow viewports.
 *
 * Modes:
 *  - `variant="full"` (default) — one row per GPU (with VRAM + core-util
 *    bars) plus system-RAM and CPU rows.
 *  - `variant="chip"` — compact single-line summary.
 *
 * Data comes from the `useResources` singleton provided by App.vue.
 */
import { computed, inject } from "vue";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";
import type {
  CpuSnapshot,
  GpuSnapshot,
  ResourceSnapshot,
  RamSnapshot,
} from "../types";
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
const cpu = computed<CpuSnapshot | null>(() => snapshot.value?.cpu ?? null);

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
  const parts: string[] = [];
  if (s.gpus.length > 0) {
    const g = s.gpus[0];
    const mem = `${fmtGb(g.vram_used)}/${fmtGb(g.vram_total)}`;
    const util = g.gpu_utilization != null ? ` ${g.gpu_utilization}%` : "";
    parts.push(`GPU ${mem}${util}`);
  }
  if (ram.value) {
    parts.push(`RAM ${fmtGb(ram.value.used)}/${fmtGb(ram.value.total)}`);
  }
  if (cpu.value) {
    parts.push(`CPU ${Math.round(cpu.value.usage_percent)}%`);
  }
  return parts.join(" · ");
});
</script>

<template>
  <div v-if="variant === 'chip'" class="inline-flex">
    <span
      data-test="resource-chip"
      class="inline-flex h-9 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-[13px] font-medium text-ink-200"
      :title="snapshot?.hostname ?? ''"
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
    </span>
  </div>

  <div v-else role="region" aria-label="Resource telemetry" class="text-[13px]">
    <div v-if="!snapshot" class="text-ink-400">…</div>

    <div v-else class="flex flex-col gap-2">
      <div
        v-for="gpu in gpus"
        :key="gpu.ordinal"
        data-test="resource-row"
        class="flex flex-col gap-1"
      >
        <div class="flex items-center gap-3">
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
              gpu.vram_used_by_mold !== null &&
              gpu.vram_used_by_other !== null &&
              gpu.vram_used_by_mold !== undefined
            "
            class="w-40 shrink-0 text-right text-[11px] text-ink-400 tabular-nums"
          >
            mold {{ fmtGb(gpu.vram_used_by_mold) }} · other
            {{ fmtGb(gpu.vram_used_by_other ?? 0) }}
          </div>
        </div>
        <div v-if="gpu.gpu_utilization != null" class="flex items-center gap-3">
          <div class="w-28 shrink-0 pl-3 text-[11px] uppercase text-ink-500">
            core load
          </div>
          <div class="w-32 shrink-0 tabular-nums text-ink-200">
            {{ gpu.gpu_utilization }}%
          </div>
          <div
            class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5"
          >
            <div
              class="absolute inset-y-0 left-0 bg-amber-400/70"
              :style="{ width: `${gpu.gpu_utilization}%` }"
            />
          </div>
          <div class="w-40 shrink-0"></div>
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

      <div v-if="cpu" data-test="resource-row" class="flex items-center gap-3">
        <div class="w-28 shrink-0 font-medium text-ink-100">
          CPU · {{ cpu.cores }} cores
        </div>
        <div class="w-32 shrink-0 tabular-nums text-ink-200">
          {{ cpu.usage_percent.toFixed(1) }}%
        </div>
        <div
          class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5"
        >
          <div
            class="absolute inset-y-0 left-0 bg-sky-400/70"
            :style="{ width: `${Math.min(100, cpu.usage_percent)}%` }"
          />
        </div>
        <div class="w-40 shrink-0"></div>
      </div>

      <div class="pt-1 text-[11px] text-ink-500">
        host {{ snapshot.hostname }} · updated
        {{ new Date(snapshot.timestamp).toLocaleTimeString() }}
      </div>
    </div>
  </div>
</template>

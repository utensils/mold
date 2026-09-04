<script setup lang="ts">
/*
 * Settings ▸ Styles & disk: the two directories the engine writes into, and
 * what the disk under them has left. The meter reads the primary's own
 * `/api/status.models_disk` — the only disk the shell is told about — so a
 * remote machine's storage stays on its own machine pane.
 */
import { computed } from "vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import { formatGB, percent } from "../../lib/format";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useSettingsConfigStore } from "../../stores/settingsConfig";

const config = useSettingsConfigStore();
const hostStatus = useHostStatusStore();

const disk = computed(() => hostStatus.status?.models_disk ?? null);
const usedBytes = computed(() =>
  disk.value ? Math.max(0, disk.value.total_bytes - disk.value.free_bytes) : 0,
);
const usedPercent = computed(() =>
  disk.value ? Math.round(percent(usedBytes.value, disk.value.total_bytes)) : 0,
);
</script>

<template>
  <div>
    <template v-if="config.available">
      <ConfigSettingRow schema-key="models_dir" />
      <ConfigSettingRow schema-key="output_dir" />
    </template>

    <div
      v-if="disk"
      class="flex min-h-[52px] items-center gap-3.5 border-b border-border px-3.5 py-3 last:border-b-0"
      data-test="settings-disk-meter"
    >
      <div class="flex min-w-0 flex-1 flex-col gap-0.5">
        <span class="text-sm font-medium text-fg">Disk for styles</span>
        <p class="max-w-md text-xs text-fg-dim">
          The whole disk the styles and pictures directories sit on.
        </p>
      </div>
      <div class="flex shrink-0 items-center gap-2.5">
        <span
          class="flex h-2 w-40 overflow-hidden border border-border"
          role="meter"
          aria-label="Disk for styles"
          :aria-valuenow="usedPercent"
          aria-valuemin="0"
          aria-valuemax="100"
        >
          <span class="bg-accent" :style="{ width: `${usedPercent}%` }" />
        </span>
        <span class="font-mono text-micro text-fg-dim">
          {{ formatGB(usedBytes) }} of {{ formatGB(disk.total_bytes) }}
        </span>
      </div>
    </div>
  </div>
</template>

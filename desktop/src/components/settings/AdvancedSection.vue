<script setup lang="ts">
import ConfigRowItem from "./ConfigRowItem.vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";
import { schemaFor } from "../../lib/settingsSchema";
import type { ConfigRow } from "../../lib/api/types";

defineProps<{ filter?: ((row: ConfigRow) => boolean) | undefined }>();

const config = useSettingsConfigStore();
const toasts = useToastStore();

async function save(row: ConfigRow, value: ConfigRow["value"]) {
  const error = await config.save(row.key, value);
  toasts.push(error ?? `Saved ${row.key}`, error ? "error" : "info");
}
async function reset(row: ConfigRow) {
  const error = await config.reset(row.key);
  toasts.push(error ?? `Reset ${row.key}`, error ? "error" : "info");
}
</script>

<template>
  <div>
    <ConfigSettingRow schema-key="server_port" />
    <p class="mt-4 mb-1 text-caption text-ink-3">
      Everything the engine exposes that has no curated control — including keys added by newer
      engines. Provenance: ⌂ database · ⛁ config.toml · ⚿ environment.
    </p>
    <template v-for="row in config.advancedRows" :key="row.key">
      <ConfigRowItem
        v-if="!schemaFor(row.key) && (!filter || filter(row))"
        :row="row"
        @save="(v) => save(row, v)"
        @reset="reset(row)"
      />
    </template>
  </div>
</template>

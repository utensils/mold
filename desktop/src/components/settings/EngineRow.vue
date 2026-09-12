<script setup lang="ts">
/*
 * One curated engine-config key on the desktop app.
 *
 * The row itself is the shared kit's `@studio/components/settings/ConfigSettingRow`,
 * which is store-free so web and desktop can hold their rows differently. This
 * is the whole desktop binding: the settings-config store supplies the row and
 * takes the writes, the toast store reports them, and the native folder picker
 * is handed down for `path` keys — a browser gets an editable field instead.
 */
import ConfigSettingRow from "@studio/components/settings/ConfigSettingRow.vue";
import type { ConfigValue } from "@studio/api/config";
import { ipc } from "../../lib/ipc";
import { schemaFor } from "@studio/lib/settingsSchema";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const props = defineProps<{
  /** Curated engine-config key (must exist in the schema). */
  schemaKey: string;
  /** Override select options (e.g. `default_model` gets installed styles). */
  options?: { value: string; label: string }[] | undefined;
}>();

const config = useSettingsConfigStore();
const toasts = useToastStore();

/** A row saves as it changes; only a failure is worth a toast, and it names
 *  the row in the words on screen, never the engine key. */
function labelFor(key: string): string {
  return schemaFor(key)?.label ?? key;
}

async function save(key: string, value: ConfigValue) {
  const error = await config.save(key, value);
  if (error) toasts.push(`${labelFor(key)} was not saved: ${error}`, "error");
}

async function reset(key: string) {
  const error = await config.reset(key);
  if (error) toasts.push(`${labelFor(key)} was not reset: ${error}`, "error");
}
</script>

<template>
  <ConfigSettingRow
    :schema-key="props.schemaKey"
    :row="config.row(props.schemaKey)"
    :options="props.options"
    :pick-directory="ipc.pickDirectory"
    @save="save"
    @reset="reset"
  />
</template>

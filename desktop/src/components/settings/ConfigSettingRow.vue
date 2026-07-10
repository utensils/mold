<script setup lang="ts">
import { computed } from "vue";
import SettingRow from "./SettingRow.vue";
import ToggleControl from "./ToggleControl.vue";
import SelectControl from "./SelectControl.vue";
import NumberControl from "./NumberControl.vue";
import TextControl from "./TextControl.vue";
import SliderControl from "./SliderControl.vue";
import PathControl from "./PathControl.vue";
import { schemaFor } from "../../lib/settingsSchema";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const props = defineProps<{
  /** Curated engine-config key (must exist in the schema). */
  schemaKey: string;
  /** Override select options (e.g. default_model gets installed models). */
  options?: { value: string; label: string }[];
}>();

const config = useSettingsConfigStore();
const toasts = useToastStore();

const schema = computed(() => schemaFor(props.schemaKey));
const row = computed(() => config.row(props.schemaKey));
const locked = computed(() => row.value?.source === "env");

async function save(value: string | number | boolean | null) {
  const error = await config.save(props.schemaKey, value);
  if (error) toasts.push(error, "error");
  else toasts.push(`Saved ${props.schemaKey}`);
}

async function reset() {
  const error = await config.reset(props.schemaKey);
  if (error) toasts.push(error, "error");
  else toasts.push(`Reset ${props.schemaKey}`);
}

const asBool = computed(() => row.value?.value === true || row.value?.value === "true");
const asText = computed(() => (row.value?.value == null ? "" : String(row.value.value)));
const asNumber = computed(() => {
  const v = row.value?.value;
  return typeof v === "number" ? v : v != null && v !== "" ? Number(v) : null;
});
</script>

<template>
  <SettingRow
    v-if="schema && row"
    :label="schema.label"
    :help="schema.help"
    :source="row.source"
    :locked="locked"
    :locked-by="row.env_var ?? undefined"
    :needs-engine-restart="schema.needsEngineRestart"
    resettable
    @reset="reset"
  >
    <ToggleControl
      v-if="schema.editor === 'toggle'"
      :model-value="asBool"
      :disabled="locked"
      @commit="save"
    />
    <SelectControl
      v-else-if="schema.editor === 'select'"
      :model-value="asText"
      :options="options ?? schema.options ?? []"
      :disabled="locked"
      @commit="(v) => save(v === '' ? null : v)"
    />
    <SliderControl
      v-else-if="schema.editor === 'slider'"
      :model-value="asNumber ?? schema.min ?? 0"
      :min="schema.min ?? 0"
      :max="schema.max ?? 1"
      :step="schema.step ?? 0.01"
      :disabled="locked"
      @commit="save"
    />
    <NumberControl
      v-else-if="schema.editor === 'number'"
      :model-value="asNumber"
      :min="schema.min"
      :max="schema.max"
      :step="schema.step"
      :disabled="locked"
      @commit="save"
    />
    <PathControl
      v-else-if="schema.editor === 'path'"
      :model-value="asText"
      :title="schema.label"
      :disabled="locked"
      @commit="save"
    />
    <TextControl
      v-else
      :model-value="asText"
      wide
      :disabled="locked"
      @commit="(v) => save(v === '' ? null : v)"
    />
  </SettingRow>
</template>

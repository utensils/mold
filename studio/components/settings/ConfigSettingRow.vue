<script setup lang="ts">
/*
 * One curated engine-config key, rendered from the schema and nothing else.
 *
 * Store-free on purpose: web and desktop hold their config rows in different
 * Pinia stores, so the row is handed in and the writes go back out as `save`
 * / `reset` naming the key. It autosaves — the ↺ is the only button — which
 * is why the controls below refuse to commit anything but a real change.
 */
import { computed } from "vue";
import SettingRow from "./SettingRow.vue";
import ToggleControl from "./ToggleControl.vue";
import SelectControl from "./SelectControl.vue";
import NumberControl from "./NumberControl.vue";
import TextControl from "./TextControl.vue";
import SliderControl from "./SliderControl.vue";
import PathControl from "./PathControl.vue";
import SecretControl from "./SecretControl.vue";
import {
  canResetConfig,
  secretValuePresent,
  type ConfigRow,
  type ConfigValue,
} from "../../api/config";
import { schemaFor } from "../../lib/settingsSchema";

const props = defineProps<{
  /** Curated engine-config key (must exist in the schema). */
  schemaKey: string;
  /** The host's row for that key, or null when it reports none. */
  row: ConfigRow | null;
  /** Override select options (e.g. `default_model` gets installed styles). */
  options?: { value: string; label: string }[] | undefined;
  /** Native folder picker for `path` keys; absent gives an editable field. */
  pickDirectory?: ((title: string) => Promise<string | null>) | undefined;
}>();
const emit = defineEmits<{
  (e: "save", key: string, value: ConfigValue): void;
  (e: "reset", key: string): void;
}>();

const schema = computed(() => schemaFor(props.schemaKey));

/** Why the row is read-only here, if it is — the environment wins over any
 *  stored value, and a startup-only key cannot move while the server runs. */
const lockedReason = computed(() => {
  if (schema.value?.liveReadOnly)
    return "Startup-only while the server is running. Use the CLI while stopped, then restart.";
  if (props.row?.source === "env")
    return `Locked by ${props.row.env_var ?? "the environment"} — unset it to edit here.`;
  return undefined;
});
const locked = computed(() => lockedReason.value !== undefined);

function save(value: ConfigValue) {
  emit("save", props.schemaKey, value);
}

const asBool = computed(
  () => props.row?.value === true || props.row?.value === "true",
);
const asText = computed(() =>
  props.row?.value == null ? "" : String(props.row.value),
);
const asNumber = computed(() => {
  const value = props.row?.value;
  const parsed =
    typeof value === "number"
      ? value
      : value != null && value !== ""
        ? Number(value)
        : null;
  // A non-numeric string from the engine must not feed NaN into an input.
  return parsed !== null && Number.isFinite(parsed) ? parsed : null;
});
</script>

<template>
  <SettingRow
    v-if="schema && row"
    :label="schema.label"
    :help="schema.help"
    :source="row.source"
    :locked-reason="lockedReason"
    :needs-engine-restart="schema.needsEngineRestart || row.restart_required"
    :resettable="canResetConfig(schemaKey)"
    @reset="emit('reset', schemaKey)"
  >
    <ToggleControl
      v-if="schema.editor === 'toggle'"
      :model-value="asBool"
      :disabled="locked"
      :aria-label="schema.label"
      @commit="save"
    />
    <SelectControl
      v-else-if="schema.editor === 'select'"
      :model-value="asText"
      :options="options ?? schema.options ?? []"
      :disabled="locked"
      :aria-label="schema.label"
      @commit="(v) => save(v === '' ? null : v)"
    />
    <SliderControl
      v-else-if="schema.editor === 'slider'"
      :model-value="asNumber ?? schema.min ?? 0"
      :min="schema.min ?? 0"
      :max="schema.max ?? 1"
      :step="schema.step ?? 0.01"
      :disabled="locked"
      :aria-label="schema.label"
      @commit="save"
    />
    <NumberControl
      v-else-if="schema.editor === 'number'"
      :model-value="asNumber"
      :min="schema.min"
      :max="schema.max"
      :step="schema.step"
      :disabled="locked"
      :aria-label="schema.label"
      @commit="save"
    />
    <PathControl
      v-else-if="schema.editor === 'path'"
      :model-value="asText"
      :title="schema.label"
      :disabled="locked"
      :pick="pickDirectory"
      @commit="save"
    />
    <SecretControl
      v-else-if="schema.editor === 'secret'"
      :present="secretValuePresent(row.value)"
      :busy="locked"
      :aria-label="schema.label"
      clearable
      @save="save"
      @clear="save(null)"
    />
    <TextControl
      v-else
      :model-value="asText"
      wide
      :disabled="locked"
      :aria-label="schema.label"
      @commit="(v) => save(v === '' ? null : v)"
    />
  </SettingRow>
</template>

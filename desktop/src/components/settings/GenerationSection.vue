<script setup lang="ts">
import { computed } from "vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import { useModelStore } from "../../stores/models";

const models = useModelStore();

const modelOptions = computed(() => [
  { value: "", label: "None (pick per session)" },
  ...models.installed.map((m) => ({ value: m.name, label: m.name })),
]);
</script>

<template>
  <div>
    <ConfigSettingRow schema-key="default_model" :options="modelOptions" />
    <ConfigSettingRow schema-key="default_width" />
    <ConfigSettingRow schema-key="default_height" />
    <ConfigSettingRow schema-key="default_steps" />
    <ConfigSettingRow schema-key="default_negative_prompt" />
    <ConfigSettingRow schema-key="embed_metadata" />
    <ConfigSettingRow schema-key="t5_variant" />
    <ConfigSettingRow schema-key="qwen3_variant" />
  </div>
</template>

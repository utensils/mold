<script setup lang="ts">
import { computed } from "vue";
import EngineRow from "./EngineRow.vue";
import { useModelStore } from "../../stores/models";

const models = useModelStore();

const modelOptions = computed(() => [
  { value: "", label: "None (pick per session)" },
  ...models.installed.map((m) => ({ value: m.name, label: m.name })),
]);
</script>

<template>
  <div>
    <EngineRow schema-key="default_model" :options="modelOptions" />
    <EngineRow schema-key="default_width" />
    <EngineRow schema-key="default_height" />
    <EngineRow schema-key="default_steps" />
    <EngineRow schema-key="default_negative_prompt" />
    <EngineRow schema-key="embed_metadata" />
    <EngineRow schema-key="t5_variant" />
    <EngineRow schema-key="umt5_variant" />
    <EngineRow schema-key="qwen3_variant" />
  </div>
</template>

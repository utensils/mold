<script setup lang="ts">
import EngineRow from "./EngineRow.vue";
</script>

<template>
  <div>
    <EngineRow schema-key="expand.enabled" />
    <EngineRow schema-key="expand.backend" />
    <EngineRow schema-key="expand.model" />
    <EngineRow schema-key="expand.api_model" />
    <EngineRow schema-key="expand.temperature" />
    <EngineRow schema-key="expand.top_p" />
    <EngineRow schema-key="expand.max_tokens" />
    <EngineRow schema-key="expand.thinking" />
  </div>
</template>

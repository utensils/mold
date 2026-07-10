<script setup lang="ts">
import ConfigSettingRow from "./ConfigSettingRow.vue";
</script>

<template>
  <div>
    <ConfigSettingRow schema-key="expand.enabled" />
    <ConfigSettingRow schema-key="expand.backend" />
    <ConfigSettingRow schema-key="expand.model" />
    <ConfigSettingRow schema-key="expand.api_model" />
    <ConfigSettingRow schema-key="expand.temperature" />
    <ConfigSettingRow schema-key="expand.top_p" />
    <ConfigSettingRow schema-key="expand.max_tokens" />
    <ConfigSettingRow schema-key="expand.thinking" />
  </div>
</template>

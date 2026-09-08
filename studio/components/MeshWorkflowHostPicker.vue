<script setup lang="ts">
defineProps<{
  hosts: Array<{
    id: string;
    label: string;
    status: "connecting" | "ready" | "error";
    stale?: boolean;
  }>;
  modelValue: string;
  automatic?: boolean;
  disabled?: boolean;
}>();

const emit = defineEmits<{ "update:modelValue": [id: string] }>();
</script>

<template>
  <label class="mesh-workflow-host-label">
    Run workflow on
    <select
      :value="modelValue"
      :disabled="disabled"
      data-test="mesh-workflow-host"
      aria-label="3-D workflow machine"
      @change="
        emit('update:modelValue', ($event.target as HTMLSelectElement).value)
      "
    >
      <option v-if="automatic" value="auto">Auto · least busy</option>
      <option v-if="automatic" value="capable">
        Most capable · strongest GPU
      </option>
      <option
        v-for="host in hosts"
        :key="host.id"
        :value="host.id"
        :disabled="host.status !== 'ready' && !host.stale"
      >
        {{ host.label }} ·
        {{
          host.stale || host.status === "connecting"
            ? "reconnecting"
            : host.status === "ready"
              ? "ready"
              : "offline"
        }}
      </option>
    </select>
  </label>
</template>

<style scoped>
.mesh-workflow-host-label {
  display: grid;
  gap: 6px;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  font-weight: 650;
}
.mesh-workflow-host-label select {
  min-width: 240px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-1);
  background: var(--mold-bg);
  color: var(--mold-text);
  padding: 9px 10px;
}
</style>

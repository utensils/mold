<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ipc } from "../../lib/ipc";
const emit = defineEmits<{ address: [value: string] }>();
const enabled = ref(false);
const endpoints = ref<string[]>([]);
const busy = ref(true);
const error = ref("");
function install(next: { enabled: boolean; endpoints: string[] }) {
  enabled.value = next.enabled;
  endpoints.value = next.endpoints;
  if (next.enabled && next.endpoints[0]) emit("address", next.endpoints[0]);
}
onMounted(async () => {
  try {
    install(await ipc.regularAccessGet());
  } catch (failure) {
    error.value = String(failure);
  } finally {
    busy.value = false;
  }
});
async function change(event: Event) {
  const input = event.target as HTMLInputElement;
  const selected = input.checked;
  busy.value = true;
  error.value = "";
  try {
    install(await ipc.regularAccessSet(selected));
  } catch (failure) {
    error.value = failure instanceof Error ? failure.message : String(failure);
  } finally {
    input.checked = enabled.value;
    busy.value = false;
  }
}
</script>
<template>
  <div class="mb-3 rounded-control border border-border bg-bg p-3">
    <label class="flex items-center gap-2"
      ><input type="checkbox" :checked="enabled" :disabled="busy" @change="change" />Allow Regular
      connections</label
    >
    <p class="mt-2 text-xs text-fg-2">
      Direct HTTP over LAN, Tailscale, or WAN relies on your network or VPN for encryption. The
      proxy requires Secure.
    </p>
    <p v-for="endpoint in endpoints" :key="endpoint" class="mt-2 font-mono text-xs">
      {{ endpoint }}
    </p>
    <p v-if="error" role="alert" class="mt-2 text-xs text-stop">{{ error }}</p>
  </div>
</template>

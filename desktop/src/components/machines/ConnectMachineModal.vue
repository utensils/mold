<script setup lang="ts">
import { ref } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import SecurePairingForm from "@studio/components/SecurePairingForm.vue";
import type { MobilePairingPayload, PairingClaim } from "@studio/api/pairing";
import type { DiscoveredHost } from "../../lib/ipc";
import { useHostsStore } from "../../stores/hosts";
defineProps<{ open: boolean; initialHost?: DiscoveredHost | null }>();
const emit = defineEmits<{ close: []; connected: [] }>();
const error = ref("");
async function paired({ payload }: { payload: MobilePairingPayload; claim: PairingClaim }) {
  try {
    await useHostsStore().connect(payload.base_url, null, payload.name || "Mold machine");
    emit("connected");
    emit("close");
  } catch (failure) {
    error.value =
      failure instanceof Error
        ? failure.message
        : "Pairing succeeded, but this machine is not answering yet. Reconnect it from Machines.";
  }
}
</script>
<template>
  <ModalPanel :open="open" :width="480" title="Connect a machine" @close="emit('close')">
    <p v-if="initialHost">
      {{ initialHost.name }} is nearby. Copy its pairing link to verify its identity and connect
      directly.
    </p>
    <SecurePairingForm
      v-if="open"
      client-name="Mold desktop"
      client-kind="desktop"
      @paired="paired"
    />
    <p v-if="error" role="alert">{{ error }}</p>
  </ModalPanel>
</template>

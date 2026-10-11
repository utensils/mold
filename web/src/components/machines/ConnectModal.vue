<script setup lang="ts">
import ModalPanel from "@ui/components/ModalPanel.vue";
import SecurePairingForm from "@studio/components/SecurePairingForm.vue";
import type { MobilePairingPayload, PairingClaim } from "@studio/api/pairing";
import { addHost, type HostEntry } from "../../lib/hostRegistry";
defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: []; added: [host: HostEntry] }>();
function paired({
  payload,
  claim,
}: {
  payload: MobilePairingPayload;
  claim: PairingClaim;
}) {
  const host = addHost({
    url: payload.base_url,
    name: payload.name || "Mold machine",
    instanceId: claim.instance_id,
  });
  emit("added", host);
  emit("close");
}
</script>
<template>
  <ModalPanel
    :open="open"
    title="Connect a machine"
    :width="480"
    @close="emit('close')"
  >
    <SecurePairingForm
      v-if="open"
      client-name="Mold Studio browser"
      client-kind="browser"
      @paired="paired"
    />
  </ModalPanel>
</template>

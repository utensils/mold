<script setup lang="ts">
import ModalPanel from "@ui/components/ModalPanel.vue";
import SecurePairingForm from "@studio/components/SecurePairingForm.vue";
import type { MobilePairingPayload, PairingClaim } from "@studio/api/pairing";
import type { RegularConnection } from "@studio/api/directConnection";
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
    connectionMode: payload.mode ?? "secure",
    ...(payload.mode === "regular" && claim.api_key
      ? { apiKey: claim.api_key }
      : {}),
  });
  emit("added", host);
  emit("close");
}
function regular(connection: RegularConnection) {
  const host = addHost({
    url: connection.baseUrl,
    name: connection.name,
    instanceId: connection.instanceId,
    connectionMode: "regular",
    ...(connection.apiKey ? { apiKey: connection.apiKey } : {}),
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
      @regular="regular"
    />
  </ModalPanel>
</template>

<script setup lang="ts">
import { ref } from "vue";
import {
  claimPairingSession,
  parseMobilePairingPayload,
  type MobilePairingPayload,
  type PairingClaim,
  type PairingClientIdentity,
} from "../api/pairing";
const props = defineProps<{
  clientName: string;
  clientKind: PairingClientIdentity["kind"];
}>();
const emit = defineEmits<{
  paired: [value: { payload: MobilePairingPayload; claim: PairingClaim }];
}>();
const invitation = ref("");
const busy = ref(false);
const error = ref("");
async function pair() {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    const payload = parseMobilePairingPayload(invitation.value.trim());
    const claim = await claimPairingSession(payload, {
      name: props.clientName,
      kind: props.clientKind,
    });
    invitation.value = "";
    emit("paired", { payload, claim });
  } catch (failure) {
    error.value =
      failure instanceof Error
        ? failure.message
        : "The machine could not be paired. Create a new link on the host and retry.";
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <form class="secure-pairing" @submit.prevent="pair">
    <p>
      On the machine you want to use, open Remote Access and copy a new pairing
      link. For a headless server, use <code>mold pairing create</code>.
    </p>
    <label for="secure-pairing-link">Pairing link</label>
    <textarea
      id="secure-pairing-link"
      v-model="invitation"
      rows="4"
      autocomplete="off"
      spellcheck="false"
      :disabled="busy"
      placeholder="Paste the pairing link or invitation JSON"
    />
    <p>
      Your connection stays encrypted on the local network and through the
      proxy. Pairing links work once and expire shortly.
    </p>
    <p v-if="error" role="alert">{{ error }}</p>
    <button type="submit" :disabled="busy || !invitation.trim()">
      {{ busy ? "Pairing…" : "Pair machine" }}
    </button>
  </form>
</template>
<style scoped>
.secure-pairing {
  display: grid;
  gap: 12px;
}
.secure-pairing textarea {
  width: 100%;
  padding: 12px;
  color: inherit;
  background: var(--mold-surface);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}
.secure-pairing button {
  min-height: 44px;
}
</style>
